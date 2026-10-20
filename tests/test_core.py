"""Fast checks that run on every push (no API calls, no network).
Run locally: python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("GROQ_API_KEY", "test-key")

from fastapi.testclient import TestClient

import limits
import scorer
import server


class SentenceSplitting(unittest.TestCase):
    def test_splits_on_sentence_ends(self):
        done, rest = server.extract_sentences("Hello Priya. Is there a fee? Tell me")
        self.assertEqual(done, ["Hello Priya.", "Is there a fee?"])
        self.assertEqual(rest, "Tell me")

    def test_keeps_abbreviations_and_hindi_danda(self):
        done, _ = server.extract_sentences("Rs. 500 late fee. ठीक है। ok")
        self.assertEqual(done, ["Rs. 500 late fee.", "ठीक है।"])

    def test_first_clause_needs_min_length_and_ignores_number_commas(self):
        self.assertEqual(server.extract_first_clause("Hi Priya, thanks")[0], None)
        clause, rest = server.extract_first_clause("Loan of 1,50,000 rupees, okay")
        self.assertEqual(clause, "Loan of 1,50,000 rupees,")
        self.assertEqual(rest, "okay")

    def test_emoji_never_reaches_tts(self):
        self.assertEqual(server.speakable("🙏"), "")
        self.assertEqual(server.speakable("இருக்கா? 😅"), "இருக்கா?")


def model_output(score=5, **flags):
    return {
        "red_flags": {f: {"present": f in flags, "quote": flags.get(f, "")} for f in scorer.FLAG_CAPS},
        "criteria": {c: {"evidence": "e -> anchor 5", "reason": "r", "score": score} for c in scorer.CRITERIA},
        "summary_feedback": "x",
    }


class ScorerCaps(unittest.TestCase):
    def test_clean_call_is_not_capped(self):
        r = scorer._validate(model_output())
        self.assertEqual(r["overall_score"], 5.0)
        self.assertEqual(r["red_flags"], {})

    def test_lie_and_otp_force_ones(self):
        r = scorer._validate(model_output(false_or_implausible_claim="never any penalty",
                                          unsafe_request="tell me the OTP"))
        self.assertEqual(r["scores"]["honesty_about_charges"], 1)
        self.assertEqual(r["scores"]["handling_objections"], 1)
        self.assertEqual(r["scores"]["closing"], 1)
        self.assertEqual(r["scores"]["opening"], 5)
        self.assertIn("Capped 5->1", r["rationale"]["closing"])

    def test_lowest_cap_wins(self):
        r = scorer._validate(model_output(hedged_or_vague_figures="around 1-2%",
                                          refused_or_postponed_charges="later"))
        self.assertEqual(r["scores"]["honesty_about_charges"], 2)

    def test_malformed_output_raises(self):
        with self.assertRaises(ValueError):
            scorer._validate({"criteria": {}})
        bad = model_output()
        bad["criteria"]["closing"]["score"] = 6
        with self.assertRaises(ValueError):
            scorer._validate(bad)


class DemoLimits(unittest.TestCase):
    def setUp(self):
        self._mode = limits.DEMO_MODE
        limits.DEMO_MODE = True
        self.l = limits.DemoLimits()

    def tearDown(self):
        limits.DEMO_MODE = self._mode

    def test_turn_limit_per_call(self):
        self.assertIsNone(self.l.check_turn("ip", 0, 1000))
        self.assertIsNotNone(self.l.check_turn("ip", limits.TURNS_PER_CALL, 1000))

    def test_oversized_audio_rejected(self):
        self.assertIsNotNone(self.l.check_turn("ip", 0, limits.MAX_AUDIO_BYTES + 1))

    def test_score_limit_per_visitor(self):
        for _ in range(limits.SCORES_PER_VISITOR_DAY):
            self.assertIsNone(self.l.check_score("ip"))
            self.l.record_score("ip")
        self.assertIsNotNone(self.l.check_score("ip"))
        self.assertIsNone(self.l.check_score("other-ip"))

    def test_off_by_default_means_unlimited(self):
        limits.DEMO_MODE = False
        self.assertIsNone(self.l.check_turn("ip", 10_000, 10**9))


class Endpoints(unittest.TestCase):
    def test_health_get_and_head(self):
        client = TestClient(server.app)
        self.assertEqual(client.get("/health").json(), {"ok": True})
        self.assertEqual(client.head("/health").status_code, 200)

    def test_page_and_hello(self):
        client = TestClient(server.app)
        self.assertEqual(client.get("/").status_code, 200)
        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
        self.assertEqual(hello["type"], "hello")
        self.assertEqual({l["code"] for l in hello["languages"]}, {"en", "hi", "ta"})


if __name__ == "__main__":
    unittest.main()
