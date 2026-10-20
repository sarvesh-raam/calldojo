"""CallSim: streaming voice loop (VAD + barge-in) with QA call scoring.
Run:  export GROQ_API_KEY=...   then   uvicorn server:app --reload
Open: http://localhost:8000
"""
import os, re, sys, json, time, base64, asyncio, traceback

# Redirected Windows consoles use cp1252; logging a Hindi/Tamil transcript must never crash a turn
sys.stdout.reconfigure(errors="backslashreplace")
sys.stderr.reconfigure(errors="backslashreplace")
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from groq import Groq, AsyncGroq
import edge_tts
from languages import LANGUAGES, DEFAULT_LANGUAGE, get_language
from limits import limits, DEMO_MODE

# The product the trainee is selling. The scorer fact-checks the agent's stated terms against it.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "product.json"), encoding="utf-8") as _f:
    PRODUCT = json.load(_f)

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    print("\n[!] GROQ_API_KEY environment variable is not set.")
    print("    Set it in a .env file (GROQ_API_KEY=your_key) or via environment variable.\n")

client = Groq(api_key=groq_api_key or "dummy_key")
# Async client for the WS stream: a sync stream iterator blocks the event loop,
# so barge-in messages can't be received and cancel() can't land mid-stream.
async_client = AsyncGroq(api_key=groq_api_key or "dummy_key")
app = FastAPI()

PERSONA = """You are Ravi, a 28-year-old first-time borrower who wants a small
business loan. You are nervous about hidden charges and missing an EMI.
You are talking to a loan agent on a phone call.
Rules: reply in 1-2 short spoken sentences. Never break character.
This is a voice call: use plain spoken words only, no emojis, symbols or markdown.
React to what the agent actually said: soften if they explain clearly,
push back if they dodge. Never invent loan terms; ask the agent instead."""


def system_prompt(lang: dict) -> dict:
    rule = lang["persona_rule"]
    return {"role": "system", "content": PERSONA + (f"\nLanguage: {rule}" if rule else "")}


# Only for the legacy /turn profiling endpoint. Live calls keep their own history per WebSocket.
turn_history = [{"role": "system", "content": PERSONA}]


def audio_filename(audio_bytes: bytes) -> str:
    """Name the upload by its real format. Hands-free sends WAV, Chrome/Android push-to-talk
    sends WebM, and iPhone Safari sends MP4; the API goes by the file extension."""
    if audio_bytes[:4] == b"RIFF":
        return "speech.wav"
    if audio_bytes[4:8] == b"ftyp":
        return "speech.mp4"
    if audio_bytes[:4] == b"OggS":
        return "speech.ogg"
    return "speech.webm"


def stt(audio_bytes: bytes, lang: dict = LANGUAGES[DEFAULT_LANGUAGE]) -> str:
    kw = {"language": lang["stt_language"]}
    if lang["stt_prompt"]:
        kw["prompt"] = lang["stt_prompt"]
    r = client.audio.transcriptions.create(
        file=(audio_filename(audio_bytes), audio_bytes), model=lang["stt_model"], **kw)
    return r.text.strip()


LLM_MODEL = os.environ.get("GROQ_LLM_MODEL", "qwen/qwen3.8-27b")

def llm(messages: list) -> str:
    r = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, max_tokens=120)
    return r.choices[0].message.content.strip()


async def tts(text: str, voice: str = LANGUAGES[DEFAULT_LANGUAGE]["voice"]) -> bytes:
    out = bytearray()
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":
            out += chunk["data"]
    return bytes(out)


import json
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

def extract_sentences(buffer: str):
    sentences = []
    abbrevs = ('rs.', 'mr.', 'mrs.', 'ms.', 'dr.', 'vs.', 'no.', 'e.g.', 'i.e.')
    while True:
        pos = -1
        for i, char in enumerate(buffer):
            if char in ('.', '!', '?', '।'):
                if i + 1 < len(buffer) and buffer[i+1] in (' ', '\n', '\t'):
                    candidate = buffer[:i+1].strip()
                    if any(candidate.lower().endswith(abbr) for abbr in abbrevs):
                        continue
                    if char == '.' and i > 0 and buffer[i-1].isdigit() and buffer[i+1].isdigit():
                        continue
                    pos = i + 1
                    break
        if pos != -1:
            sentences.append(buffer[:pos].strip())
            buffer = buffer[pos:].lstrip()
        else:
            break
    return sentences, buffer


# Emoji and pictographs: the TTS has nothing to say for them and fails on emoji-only chunks
NON_SPEECH = re.compile("[🀀-🫿☀-➿️‍]")


def speakable(text: str) -> str:
    return NON_SPEECH.sub("", text).strip()


FIRST_CLAUSE_MIN_CHARS = 20

def extract_first_clause(buffer: str):
    """Cut the first chunk at a clause break so audio starts before a long first
    sentence finishes. TTS time grows with text length (~3s for a long Tamil sentence)."""
    for i, char in enumerate(buffer):
        if i >= FIRST_CLAUSE_MIN_CHARS and char in (',', ';', '—') and i + 1 < len(buffer) and buffer[i+1] == ' ':
            return buffer[:i+1].strip(), buffer[i+1:].lstrip()
    return None, buffer


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    busy = limits.connect()
    if busy:
        await ws.send_json({"type": "limit", "message": busy, **limits.info()})
        await ws.close()
        return
    # Behind a hosting proxy the client address is the proxy's; the real one is in X-Forwarded-For
    visitor = (ws.headers.get("x-forwarded-for", "").split(",")[0].strip()
               or (ws.client.host if ws.client else "unknown"))
    print("[WebSocket] Client connected for streaming & VAD")
    active_turn_task = None
    turns_this_call = 0
    # One call per connection, so two tabs or trainees never share a conversation
    lang_code = DEFAULT_LANGUAGE
    lang = get_language(lang_code)
    history = [system_prompt(lang)]
    await ws.send_json({
        "type": "hello",
        "language": lang_code,
        "languages": [{"code": c, "label": l["label"]} for c, l in LANGUAGES.items()],
        "product": PRODUCT,
        **limits.info(),
    })

    async def run_turn(audio_bytes: bytes):
        # An unhandled error in a background task is otherwise silent: the client would wait forever
        try:
            await process_turn(audio_bytes)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            traceback.print_exc()
            try:
                await ws.send_json({"type": "error", "message": f"Turn failed: {e}"})
            except Exception:
                pass

    async def process_turn(audio_bytes: bytes):
        t0 = time.perf_counter()

        # 1. STT
        try:
            text = await asyncio.to_thread(stt, audio_bytes, lang)
        except Exception as e:
            # Log what arrived so device-specific recording bugs can be diagnosed from the logs
            print(f"[STT] Failed on {len(audio_bytes)} bytes, sent as {audio_filename(audio_bytes)}, "
                  f"first bytes {audio_bytes[:12]!r}: {e}")
            message = ("Couldn't read that recording. Hold the button and try again."
                       if "invalid_media_file" in str(e) else f"Speech-to-text error: {e}")
            await ws.send_json({"type": "error", "message": message})
            return

        t1 = time.perf_counter()
        stt_ms = round((t1 - t0) * 1000)

        if not text or not text.strip():
            await ws.send_json({"type": "notice", "message": "No speech detected. Please speak clearly."})
            return

        print(f"[WS Turn] You said: {text!r} (STT: {stt_ms}ms)")
        await ws.send_json({
            "type": "user_transcription",
            "text": text,
            "stt_ms": stt_ms
        })

        if history[-1]["role"] == "user":
            # Previous turn was interrupted before Ravi said anything audible:
            # merge so the LLM never sees two user turns in a row.
            history[-1]["content"] += " " + text
        else:
            history.append({"role": "user", "content": text})

        # 2. Stream LLM tokens & synthesize sentences
        try:
            stream = await async_client.chat.completions.create(
                model=lang["llm_model"],
                messages=history,
                stream=True,
                **lang["llm_kwargs"]
            )
        except Exception as e:
            await ws.send_json({"type": "error", "message": f"LLM error: {e}"})
            return

        buffer = ""
        full_reply = ""
        first_audio_sent = False
        ttfa_ms = None
        sentence_idx = 0

        async def send_sentence_audio(sentence: str):
            nonlocal first_audio_sent, ttfa_ms, sentence_idx
            clean_s = speakable(sentence)
            if not clean_s:
                return
            sentence_idx += 1
            t_tts_s = time.perf_counter()
            # Edge-TTS intermittently returns no audio; one bad sentence must not kill the whole turn
            audio_data = None
            for attempt in (1, 2):
                try:
                    audio_data = await tts(clean_s, lang["voice"])
                    break
                except edge_tts.exceptions.NoAudioReceived as e:
                    print(f"[TTS] Attempt {attempt} got no audio for {clean_s!r}: {e}")
            if not audio_data:
                return  # text is already on screen via llm_delta; it just won't be spoken
            t_tts_e = time.perf_counter()
            chunk_tts_ms = round((t_tts_e - t_tts_s) * 1000)

            if not first_audio_sent:
                first_audio_sent = True
                ttfa_ms = round((t_tts_e - t0) * 1000)
                await ws.send_json({
                    "type": "first_audio",
                    "time_to_first_audio_ms": ttfa_ms
                })

            await ws.send_json({
                "type": "audio_chunk",
                "index": sentence_idx,
                "text": clean_s,
                "audio_b64": base64.b64encode(audio_data).decode(),
                "tts_ms": chunk_tts_ms,
                "ttfa_ms": ttfa_ms
            })

        try:
            # Each await is a cancellation point for barge-in
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    buffer += delta
                    full_reply += delta
                    await ws.send_json({"type": "llm_delta", "delta": delta})

                    sentences, buffer = extract_sentences(buffer)
                    if not first_audio_sent and not sentences:
                        clause, buffer = extract_first_clause(buffer)
                        if clause:
                            sentences = [clause]
                    for s in sentences:
                        await send_sentence_audio(s)

            if buffer.strip():
                await send_sentence_audio(buffer.strip())
        except asyncio.CancelledError:
            await stream.close()
            raise
        except Exception as e:
            await ws.send_json({"type": "error", "message": f"Stream error: {e}"})
            return

        t_end = time.perf_counter()
        total_ms = round((t_end - t0) * 1000)
        history.append({"role": "assistant", "content": full_reply})
        print(f"[WS Turn] Ravi replied: {full_reply!r} | TTFA: {ttfa_ms}ms | Total: {total_ms}ms")

        await ws.send_json({
            "type": "turn_complete",
            "full_reply": full_reply,
            "timings": {
                "stt_ms": stt_ms,
                "time_to_first_audio_ms": ttfa_ms or total_ms,
                "total_ms": total_ms
            }
        })

    try:
        while True:
            data = await ws.receive()
            if data["type"] == "websocket.disconnect":
                # receive() reports a closed tab as a message, not an exception
                raise WebSocketDisconnect(data.get("code", 1000))
            if "bytes" in data and data["bytes"]:
                denied = limits.check_turn(visitor, turns_this_call, len(data["bytes"]))
                if denied:
                    await ws.send_json({"type": "limit", "message": denied, **limits.info()})
                    continue
                limits.record_turn(visitor)
                turns_this_call += 1
                # If a previous turn is still generating, cancel it
                if active_turn_task and not active_turn_task.done():
                    active_turn_task.cancel()
                active_turn_task = asyncio.create_task(run_turn(data["bytes"]))

            elif "text" in data and data["text"]:
                payload = json.loads(data["text"])
                action = payload.get("action")

                if action == "barge_in":
                    # User interrupted speaking! Cancel running task
                    if active_turn_task and not active_turn_task.done():
                        active_turn_task.cancel()
                        print("[Barge-in] Active turn task cancelled.")

                    # Truncate assistant history to only what the user actually heard
                    spoken_text = payload.get("spoken_text", "").strip()
                    if history[-1]["role"] == "assistant":
                        old_text = history[-1]["content"]
                        if spoken_text:
                            history[-1]["content"] = spoken_text
                            print(f"[Barge-in] Truncated Ravi history from {old_text!r} -> {spoken_text!r}")
                        else:
                            # Reply finished server-side but none of it was played
                            history.pop()
                            print(f"[Barge-in] Dropped unheard Ravi reply {old_text!r}")
                    elif spoken_text and history[-1]["role"] == "user":
                        # If barge-in happened before turn finished and assistant hasn't been added yet
                        history.append({"role": "assistant", "content": spoken_text})
                        print(f"[Barge-in] Appended partial heard response: {spoken_text!r}")

                    await ws.send_json({"type": "barge_in_ack", "truncated_to": spoken_text})

                elif action == "reset":
                    if active_turn_task and not active_turn_task.done():
                        active_turn_task.cancel()
                    del history[1:]
                    turns_this_call = 0
                    print("[WebSocket] Call history reset")
                    await ws.send_json({"type": "reset_ack", "ok": True})

                elif action == "set_language":
                    # Ravi's language is part of the persona, so switching starts a new call
                    if active_turn_task and not active_turn_task.done():
                        active_turn_task.cancel()
                    code = payload.get("language")
                    if code in LANGUAGES:
                        lang_code, lang = code, LANGUAGES[code]
                    history[:] = [system_prompt(lang)]
                    turns_this_call = 0
                    print(f"[WebSocket] Language set to {lang_code}; new call started")
                    await ws.send_json({"type": "language_ack", "language": lang_code})

                elif action == "score":
                    if active_turn_task and not active_turn_task.done():
                        active_turn_task.cancel()
                    denied = limits.check_score(visitor)
                    if denied:
                        await ws.send_json({"type": "scorecard", "error": denied, **limits.info()})
                        continue
                    result = await score_call(history, lang_code)
                    # Only count attempts that reached the scorer API (not "call too short")
                    if "scores" in result or result.get("error", "").startswith("Scoring failed"):
                        limits.record_score(visitor)
                    await ws.send_json({"type": "scorecard", **result})

    except WebSocketDisconnect:
        if active_turn_task and not active_turn_task.done():
            active_turn_task.cancel()
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
    finally:
        limits.disconnect()


@app.get("/")
def index():
    return FileResponse("index.html")


# For the uptime monitor that keeps the free host awake. Never touches Groq, so pings cost no quota.
# HEAD too: monitors often use HEAD, which a GET-only route would answer with 405.
@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"ok": True}


from datetime import datetime
from scorer import score_transcript

os.makedirs("transcripts", exist_ok=True)

def save_transcript(history: list, language: str) -> tuple:
    call_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"transcripts/call_{call_id}.json"
    call_data = {
        "call_id": call_id,
        "timestamp": datetime.now().isoformat(),
        "language": language,
        "transcript": list(history)
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(call_data, f, indent=2, ensure_ascii=False)
    print(f"[Stage 5] Saved transcript to {filepath}")
    return call_id, filepath


MIN_AGENT_TURNS = 3  # below this the rubric has nothing to judge (no objections, charges or close yet)

async def score_call(history: list, language: str) -> dict:
    agent_turns = sum(1 for m in history if m["role"] == "user")
    if agent_turns == 0:
        return {"error": "No call conversation to score yet. Speak to Ravi first."}
    if agent_turns < MIN_AGENT_TURNS:
        return {"error": f"Call too short to score ({agent_turns} of {MIN_AGENT_TURNS} agent turns). "
                         "Keep going: answer Ravi's concerns and close the call."}

    call_id, filepath = save_transcript(history, language)
    scorecard = await asyncio.to_thread(score_transcript, history, PRODUCT)
    scorecard["call_id"] = call_id
    scorecard["transcript_path"] = filepath
    return scorecard


@app.post("/turn")
async def turn(audio: UploadFile):
    if DEMO_MODE:
        # Unmetered profiling endpoint; never expose it on a public demo
        raise HTTPException(status_code=404)
    data = await audio.read()
    if len(data) == 0:
        return {"error": "No audio received"}
    
    t0 = time.perf_counter()
    try:
        text = await asyncio.to_thread(stt, data)
        t1 = time.perf_counter()

        if not text or not text.strip():
            return {
                "you_said": "(silence)",
                "customer_said": "I didn't catch that. Could you please repeat?",
                "timings": {"stt_ms": round((t1 - t0) * 1000), "llm_ms": 0, "tts_ms": 0, "total_ms": round((t1 - t0) * 1000)},
                "audio_b64": ""
            }

        print(f"[Turn] You said: {text!r}")
        turn_history.append({"role": "user", "content": text})
        reply = await asyncio.to_thread(llm, turn_history)
        print(f"[Turn] Ravi replied: {reply!r}")
        
        if not reply or not reply.strip():
            reply = "Could you please clarify that? I want to make sure I understand."
            
        turn_history.append({"role": "assistant", "content": reply})
        t2 = time.perf_counter()

        mp3 = await tts(reply)
        t3 = time.perf_counter()

        timings = {
            "stt_ms": round((t1 - t0) * 1000),
            "llm_ms": round((t2 - t1) * 1000),
            "tts_ms": round((t3 - t2) * 1000),
            "total_ms": round((t3 - t0) * 1000)
        }
        print(f"[Turn] Timings: {timings}")
        return {
            "you_said": text,
            "customer_said": reply,
            "timings": timings,
            "audio_b64": base64.b64encode(mp3).decode()
        }
    except Exception as e:
        print(f"Error processing turn: {e}")
        return {"error": str(e)}
