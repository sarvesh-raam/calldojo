"""Per-language call settings. Adding a language = adding one entry here
(plus benchmark calls in that language before trusting its scorecards).

Model choices come from side-by-side probes (see notes.md, Stage 6):
- qwen/qwen3.8-27b garbles Hindi/Tamil (invented words, stray Cyrillic letters),
  so code-mixed languages use openai/gpt-oss-120b (~300ms slower to first token).
- whisper-large-v3-turbo misreads Tamil numbers ("12 percent" -> "22%"), so
  Hindi/Tamil use the full whisper-large-v3 at similar latency.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # GROQ_LLM_MODEL is read at import time, before server.py loads .env

DEFAULT_LANGUAGE = "en"

LANGUAGES = {
    "en": {
        "label": "English",
        "voice": "en-IN-PrabhatNeural",
        "stt_model": "whisper-large-v3-turbo",
        "stt_language": "en",
        # Whisper's prompt biases its vocabulary toward the product's words (without a prompt,
        # "Axis Bank" came out as "Access Bank"); keep it in step with product.json
        "stt_prompt": "DojoBank, Udaan business loan. Interest rate, processing fee, grace period, late fee, EMI, foreclosure, tenure. Aadhaar, PAN, bank statements, WhatsApp.",
        "llm_model": os.environ.get("GROQ_LLM_MODEL", "qwen/qwen3.8-27b"),
        "llm_kwargs": {"max_tokens": 120},
        "persona_rule": "",
    },
    "hi": {
        "label": "Hindi / Hinglish",
        "voice": "hi-IN-MadhurNeural",
        "stt_model": "whisper-large-v3",
        "stt_language": "hi",
        # Primes Whisper to keep English loanwords in Latin script, as people type Hinglish
        "stt_prompt": "मैं DojoBank से बात कर रही हूँ, आपके Udaan business loan के बारे में। Interest rate, processing fee, grace period, late fee, EMI. Aadhaar card, PAN card, bank statement, WhatsApp.",
        "llm_model": "openai/gpt-oss-120b",
        # max_tokens covers hidden reasoning tokens too, so it needs headroom
        "llm_kwargs": {"max_tokens": 400, "reasoning_effort": "low"},
        "persona_rule": (
            "Speak Hinglish the way a small shop owner in Delhi talks on the phone: mostly Hindi, with the "
            "English words people really use (loan, EMI, interest, processing fee, documents, bank). Write Hindi "
            "words in Devanagari and English words in Latin letters, e.g. 'मुझे loan चाहिए, लेकिन hidden charges तो नहीं हैं?' "
            "You are a man: always use masculine forms for yourself (मैं सुन रहा हूँ, मैं चाहता हूँ), never रही/चाहती."
        ),
    },
    "ta": {
        "label": "Tamil / Tanglish",
        "voice": "ta-IN-ValluvarNeural",
        "stt_model": "whisper-large-v3",
        "stt_language": "ta",
        "stt_prompt": "நான் DojoBank-ல இருந்து பேசறேன், உங்க Udaan business loan பத்தி. Interest rate, processing fee, grace period, late fee, EMI. Aadhaar card, PAN card, bank statement, WhatsApp.",
        "llm_model": "openai/gpt-oss-120b",
        "llm_kwargs": {"max_tokens": 400, "reasoning_effort": "low"},
        "persona_rule": (
            "Speak Tanglish the way a small shop owner in Chennai talks on the phone: mostly Tamil, with the "
            "English words people really use (loan, EMI, interest, processing fee, documents, bank). Write Tamil "
            "words in Tamil script and English words in Latin letters, e.g. 'எனக்கு loan வேணும், ஆனா hidden charges இருக்கா?' "
            "Address the agent by name or as madam/sir, never as அப்பா or அம்மா."
        ),
    },
}


def get_language(code: str) -> dict:
    return LANGUAGES.get(code) or LANGUAGES[DEFAULT_LANGUAGE]
