# CallSim / CallDojo: Voice AI Build & Learn Notes

Tracking progress, architectural learnings, and insights across all 6 stages.

---

## Stage Summary Table

| Stage | Focus Area | Key Metric / Deliverable | Status |
|---|---|---|---|
| **Stage 1** | Push-to-Talk Voice Loop (STT -> LLM -> TTS) | End-to-end loop running; State retention via `history` | Completed |
| **Stage 2** | Latency Profiling & Bottleneck Analysis | Latency breakdown table (`stt_ms`, `llm_ms`, `tts_ms`) | Completed |
| **Stage 3** | Streaming & Overlapping (WebSockets) | Sentence boundary chunking; `time_to_first_audio` (TTFA) | Completed |
| **Stage 4** | Conversational VAD & Barge-in | Natural turn-taking, interruption handling & state truncation | Completed |
| **Stage 5** | Call Evaluation & LLM Scorer | 4-criterion rubric; Human vs. LLM scorer agreement metric | Completed: 98.6% within 1 point on held-out test set |
| **Stage 6** | Multilingual & Local Inference Stretch | Hindi/Tamil benchmarks, Faster-Whisper, Architecture README | Hinglish/Tanglish + README done; local STT deferred |

---

## Stage-by-Stage Learnings

### Stage 1: Get the loop running
- **What I built:** Push-to-talk voice pipeline connecting Groq Whisper Large v3 Turbo (STT) -> Qwen 3.8 27B on Groq (LLM) -> Edge TTS `en-IN-NeerjaNeural` (TTS) over FastAPI with a responsive modern UI.
- **What surprised me:** In non-streaming mode, total turn latency is dominated heavily by TTS (Edge-TTS took ~1659ms out of 2775ms total, ~60% of the turn time). The sequential waterfall (Wait for STT -> Wait for LLM full generation -> Wait for TTS full generation) creates an unnatural ~2.8s pause before the user hears anything.
- **What I'd do differently:** Edge-TTS generates audio for the entire sentence/paragraph at once. Moving to token/sentence streaming (Stage 3) will be critical to cut the perceived waiting time.

---

### Stage 2: Learn where the time goes (Benchmarking Data)

| Turn | Agent Speech | STT (ms) | LLM (ms) | TTS (ms) | Total (ms) | TTS Share |
|---|---|---|---|---|---|---|
| **Turn 1** | Greeting & Loan Intro | 912 | 311 | 2900 | 4123 | **70.3%** |
| **Turn 2** | Transparency & Interest rate | 477 | 208 | 2544 | 3228 | **78.8%** |
| **Turn 3** | 2 Lakh EMI Calculation | 521 | 198 | 2867 | 3587 | **79.9%** |
| **Turn 4** | 5-day Grace Period Policy | 378 | 227 | 2212 | 2818 | **78.5%** |
| **Turn 5** | KYC Docs (Aadhaar / PAN) | 482 | 227 | 2525 | 3233 | **78.1%** |
| **AVG** | **Full 5-Turn Call** | **554 ms** | **234 ms** | **2609 ms** | **3398 ms** | **76.8%** |

- **What I built:** Latency profiling across a complete 5-turn borrower loan roleplay call.
- **What surprised me:** 
  - The LLM on Groq (`qwen/qwen3.8-27b`) is extremely fast (averaging **234ms**).
  - STT (`whisper-large-v3-turbo`) averages **554ms**.
  - **TTS is the overwhelming bottleneck**, taking **2609ms** on average (**76.8% of the entire delay!**).
- **Stage 2 Check Statement:**
  > *"My total turnaround is ~3400ms. Over 76% of it is the TTS stage (2609ms). The LLM only takes ~234ms. To make this feel like a real phone call, we must overlap the stages using sentence-level streaming so audio starts playing after the first sentence is generated rather than waiting for the entire response to complete."*

---

### Stage 3: Overlap the stages with streaming (WebSockets)
- **What I built:** Persistent WebSocket bidirectional streaming (`/ws`) with Groq `stream=True` token extraction, regex-based sentence boundary chunking, and frontend `AudioQueue` for sequential gapless playback.
- **Key Metric (Time to First Audio - TTFA):**
  - **Stage 2 (Non-streaming Turn 1):** User waited **4123ms** before hearing a single syllable.
  - **Stage 3 (Streaming Turn 1):** User heard audio in **2083ms** (a **2.04-second latency reduction!**).
- **What surprised me:** The total turn duration remained ~4.1s, but human perception of latency is bounded by *when the first sound starts playing*, not when the last token finishes. By overlapping sentence 1 playback while sentence 2 is generating, the perceived conversational response time was cut in half.
- **Stage 3 Check Statement:**
  > *"Time to first audio dropped from 4123ms to 2083ms. The total generation time is nearly the same, but the user experiences half the delay because sentence 1 begins playing in their ears while sentence 2 is being synthesized in the background over the WebSocket."*

---

### Stage 4: Make it feel like a call (VAD & Barge-in)
- **What I built:**
  - In-browser Voice Activity Detection using Silero VAD (`@ricky0123/vad-web`) with automatic speech segmentation and silence threshold (~700ms) for a hands-free phone call experience.
  - Client-side Barge-in detection: immediately pauses & clears `AudioQueue` when user starts speaking during AI playback.
  - Server-side cancellation & state truncation: `active_turn_task.cancel()` terminates generation instantly; trims `history[-1]["content"]` to only what the user actually heard before the interruption.
- **What surprised me:** If the LLM generates a 3-sentence answer, but the user interrupts at sentence 1, failing to truncate `history` causes the LLM to assume the user heard sentences 2 and 3. Truncating history to match reality is the difference between a natural conversation and a broken bot.
- **Stage 4 Check Statement:**
  > *"When the user interrupts mid-sentence, the client drops the remaining audio queue, sends `barge_in` over WebSocket, and the server cancels the task while slicing `history` down to the exact words the user heard. Consequently, Ravi's subsequent response is grounded in reality."*

---

### Stage 5: Add a scorer
- **What I built:**
  - `scorer.py`: a 4-criterion rubric (opening, handling objections, honesty about charges, closing) scored 1-5 by `openai/gpt-oss-20b`, with an explicit anchor for every score, not just 1/3/5.
  - Two-step scoring: the model first answers yes/no red-flag questions about the whole call (false claim, refused charges, hedged figures, pressure/rudeness, OTP request), then scores each criterion. **Code** applies caps from the flags (`FLAG_CAPS`), so a lie or an OTP request can never score well.
  - Strict validation: a failed or malformed response returns an error. It used to return a fabricated 4/5 scorecard, which also counted as "agreement" in the benchmark.
  - Benchmark: `benchmark/dev.json` (10 calls, used for tuning) and `benchmark/test.json` (20 calls written after the rubric was fixed, measurement only), run by `evaluate_scorer.py` with `--runs N` for run-to-run variance and a signed-bias column.
- **Results so far (within 1 point / exact match):**

  | Rubric | Split | Within 1 | Exact | What it showed |
  |---|---|---|---|---|
  | Original, 3 anchors | dev | 84.2% | 66.7% | Too harsh on honesty and closing (bias -0.87 each) |
  | Full anchors | dev | 96.2% | 80.0% | |
  | Full anchors | test | 86.7% | 71.5% | Honesty only 67.8% (bias +0.89): missed lies and pressure |
  | + red-flag caps | dev | 97.5% | 80.0% | |
  | **+ red-flag caps** | **test** | **98.6%** | **86.8%** | Honesty 67.8% -> 100% within 1; bias +0.89 -> +0.06. Same in both runs |

- **What surprised me:**
  - The original "30-transcript" benchmark was 10 calls plus 20 word-for-word copies. Copies inflate the sample size without adding evidence; the real n was 10.
  - Some disagreements were the *labels'* fault: near-identical calls had different labels (call_06 vs call_08 closing). Three labels were fixed to match the written rubric; each fix is recorded in `label_notes`.
  - Telling the model to "check red flags first" did not work. It scored each criterion from one quoted line and still gave "never any penalty, pay whenever" + an OTP request 4/5 on honesty. Asking for yes/no facts and enforcing the caps in code is more reliable than asking for a judgment.
  - Groq free tier: the per-minute limit allows about 4 scorings/minute, and the per-day limit (200k tokens) is about 75 scorings, shared by the benchmark and every trainee.
- **Red-flag caps fixed every earlier failure** on the held-out set, in both runs: the lying agent with an OTP request (test_05), the pressure close (test_14), the dodged late-fee question (test_08) and the hedged figures (test_02) now match their labels exactly. Two disagreements remain, both on opening/handling, not on safety.
- **New finding:** the longer red-flag output made the model return malformed JSON in 4 of 40 scorings (10%). In the app that means 1 in 10 trainees would see "Scoring failed". Fixed with one retry on malformed output.
- **What I'd do differently:** Get labels from real QA reviewers before tuning. Labels written by the same person who writes the rubric measure consistency, not human agreement. And budget API quota for the benchmark from the start.

---

### Stage 6: Stretch & Polish
- **What I built:**
  - **Hinglish and Tanglish** via `languages.py`: one entry per language (voice, Whisper model and prompt, LLM, persona rule), plus a language picker in the UI. Ravi writes Hindi/Tamil words in native script and English words in Latin letters, which is what the native voices pronounce correctly.
  - **One call per WebSocket connection.** History used to be a global, so two tabs shared one call and the scorer graded both.
  - **Robustness fixes found while testing:** the reply stream now uses `AsyncGroq`, because the sync iterator blocked the event loop so barge-in could not land mid-reply (barge-in ack now ~4 ms). Logging can no longer crash a turn on non-Latin text. A failed Edge-TTS sentence is retried, then skipped, instead of ending the turn. Calls under 3 agent turns are not scored.
  - **First-clause split:** if Ravi's first sentence is long, audio starts at its first comma.
  - `README.md` with architecture, protocol, results and limitations.
- **Model choices (from side-by-side probes):**

  | Problem | Evidence | Fix |
  |---|---|---|
  | `qwen3.8-27b` garbles Hindi/Tamil | Invented words; Cyrillic letters inside a Hindi word; also misquoted "1%" as "ten percent" in English | `gpt-oss-120b` (reasoning_effort=low) for hi/ta: clean output, ~+300 ms first token |
  | `whisper-large-v3-turbo` on Tamil | "12 percent, 1 percent" -> "22%, 11%"; switched into Devanagari mid-sentence | Full `whisper-large-v3` for hi/ta, similar latency |

- **TTFA after the first-clause split (4 turns each):** English 1,485-1,948 ms, Hindi 2,047-2,714 ms, Tamil 2,536-3,238 ms (was 3,296-4,161 ms).
- **What surprised me:** the language problems were invisible in English testing. Each one (wrong numbers from STT, garbled LLM text, a log line crashing a turn) only appeared once a real Hindi or Tamil utterance went through the whole pipeline.
- **Product-sheet fact check (grounding).** The trainee sells a fictional product (`product.json`). A second request, in parallel with the rubric, extracts every term the agent stated; code compares them with the sheet and caps honesty at 1 on any mismatch. The extraction prompt never contains the right values, so the model can't drift toward them. First version put the extraction inside the rubric request: accurate (8/8) but 3 of 11 scorings failed outright, so it became a separate request. Result on `grounding.json` (11 calls, English/Hindi/Tamil, with traps): **11/11 fully correct, 7/7 wrong terms caught, 0 false alarms.**
- **Model routing on the free tier.** Groq limits are per model, so the scorer falls back from `gpt-oss-120b` to `gpt-oss-20b` (own budget) on any rate limit; only the last model waits out a per-minute limit. A first version guessed "daily vs per-minute" from the retry-after length and misread long per-minute waits as daily; it now reads Groq's message ("tokens per day"). Low reasoning effort cut tokens per scoring and made scoring faster (about 1.2 s). `gpt-oss-120b` at low effort scored **100% within 1 point** on the held-out set with 0 failed scorings, so it became the main scorer.
- **Paid tier wasn't an option:** Groq's Developer tier upgrades were paused, and the alternative provider required a card, so the fallback design keeps the demo on free tiers.
- **Deferred: local faster-whisper.** The dev machine has no GPU. On CPU, `large-v3` (needed for Tamil) is ~10 s per utterance, and small models are English-only quality. An offline mode is realistic for English only (~1-2 s STT), so it was skipped for now.
- **Still to do:** run the held-out benchmark with red-flag caps; add Hindi/Tamil calls to the benchmark before trusting those scorecards; get QA-reviewer labels.
