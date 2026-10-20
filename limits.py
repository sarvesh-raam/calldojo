"""Usage limits for a public demo. Off unless DEMO_MODE=1, so local use is unlimited.

A public link spends the owner's Groq quota on every visitor; without caps one curious
visitor (or bot) can exhaust the daily token limit and break the demo for everyone else.
Counters are in memory and reset each UTC day, which is enough for a single-process demo.
"""
import os
from collections import defaultdict
from datetime import datetime, timezone

DEMO_MODE = os.environ.get("DEMO_MODE") == "1"
DEMO_VIDEO_URL = os.environ.get("DEMO_VIDEO_URL", "")


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


TURNS_PER_CALL = _int("DEMO_TURNS_PER_CALL", 15)
TURNS_PER_VISITOR_DAY = _int("DEMO_TURNS_PER_VISITOR_DAY", 45)
SCORES_PER_VISITOR_DAY = _int("DEMO_SCORES_PER_VISITOR_DAY", 3)
TURNS_PER_DAY = _int("DEMO_TURNS_PER_DAY", 400)
# Each scoring is ~5,000 tokens (rubric + product fact check) of the scorer's 200k/day free-tier
# budget; 25 leaves room for the owner's own benchmark runs
SCORES_PER_DAY = _int("DEMO_SCORES_PER_DAY", 25)
MAX_CONNECTIONS = _int("DEMO_MAX_CONNECTIONS", 10)
MAX_AUDIO_BYTES = _int("DEMO_MAX_AUDIO_BYTES", 2_000_000)  # ~60 s of 16 kHz WAV


class DemoLimits:
    def __init__(self):
        self.day = None
        self.visitor_turns = defaultdict(int)
        self.visitor_scores = defaultdict(int)
        self.turns = 0
        self.scores = 0
        self.connections = 0

    def _roll_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self.day:
            self.day = today
            self.visitor_turns.clear()
            self.visitor_scores.clear()
            self.turns = self.scores = 0

    def _msg(self, text: str) -> str:
        return text + (" Watch the demo video instead." if DEMO_VIDEO_URL else "")

    # Each check returns None when allowed, or a message to show the visitor.
    def connect(self):
        if DEMO_MODE and self.connections >= MAX_CONNECTIONS:
            return self._msg("The demo is busy right now. Please try again in a few minutes.")
        self.connections += 1
        return None

    def disconnect(self):
        self.connections = max(0, self.connections - 1)

    def check_turn(self, visitor: str, turns_this_call: int, audio_bytes: int):
        if not DEMO_MODE:
            return None
        self._roll_day()
        if audio_bytes > MAX_AUDIO_BYTES:
            return "That recording was too long. Please keep each turn under a minute."
        if turns_this_call >= TURNS_PER_CALL:
            return (f"This demo call has reached its {TURNS_PER_CALL}-turn limit. "
                    "Click 'End call & get feedback', or restart the call.")
        if self.visitor_turns[visitor] >= TURNS_PER_VISITOR_DAY:
            return self._msg("You've reached today's demo limit. Thanks for trying CallDojo!")
        if self.turns >= TURNS_PER_DAY:
            return self._msg("Today's demo quota is used up.")
        return None

    def record_turn(self, visitor: str):
        self.visitor_turns[visitor] += 1
        self.turns += 1

    def check_score(self, visitor: str):
        if not DEMO_MODE:
            return None
        self._roll_day()
        if self.visitor_scores[visitor] >= SCORES_PER_VISITOR_DAY:
            return self._msg(f"You've used today's {SCORES_PER_VISITOR_DAY} feedback reports.")
        if self.scores >= SCORES_PER_DAY:
            return self._msg("Today's feedback quota is used up.")
        return None

    def record_score(self, visitor: str):
        self.visitor_scores[visitor] += 1
        self.scores += 1

    def info(self) -> dict:
        """What the page shows so visitors know the limits up front."""
        if not DEMO_MODE:
            return {"demo": False}
        return {
            "demo": True,
            "turns_per_call": TURNS_PER_CALL,
            "scores_per_day": SCORES_PER_VISITOR_DAY,
            "video_url": DEMO_VIDEO_URL,
        }


limits = DemoLimits()
