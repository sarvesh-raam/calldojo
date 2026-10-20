"""Fast checks that run on every push (no API calls, no network).
Run locally: python -m unittest discover -s tests -v
"""
import json
import os
import sys
import unittest
import unittest.mock

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


PRODUCT = {"terms": {
    "interest_rate": {"label": "Interest rate", "value": 12, "unit": "% per year"},
    "processing_fee": {"label": "Processing fee", "value": 1, "unit": "% of the loan"},
}}


class ProductFactCheck(unittest.TestCase):
    def check(self, claims):
        card = scorer._validate(model_output())
        scorer._apply_fact_check(card, claims, PRODUCT)
        return card

    def test_correct_terms_pass(self):
        r = self.check([{"term": "interest_rate", "number": 12, "quote": "12%"}])
        self.assertTrue(r["fact_check"][0]["ok"])
        self.assertEqual(r["scores"]["honesty_about_charges"], 5)
        self.assertNotIn("contradicts_product_sheet", r["red_flags"])

    def test_wrong_rate_caps_honesty_and_overall(self):
        r = self.check([{"term": "interest_rate", "number": 10, "quote": "just 10%"}])
        self.assertFalse(r["fact_check"][0]["ok"])
        self.assertEqual(r["scores"]["honesty_about_charges"], 1)
        self.assertEqual(r["overall_score"], 4.0)
        self.assertIn("sheet says 12", r["red_flags"]["contradicts_product_sheet"])

    def test_no_fee_claim_is_zero_and_wrong(self):
        r = self.check([{"term": "processing_fee", "number": 0, "quote": "no fee"}])
        self.assertFalse(r["fact_check"][0]["ok"])

    def test_unknown_terms_and_junk_numbers_are_ignored(self):
        r = self.check([{"term": "cashback", "number": 5}, {"term": "interest_rate", "number": "twelve"}])
        self.assertEqual(r["fact_check"], [])

    def test_missing_claims_is_malformed(self):
        with self.assertRaises(ValueError):
            scorer._parse_claims({})

    def test_rubric_request_is_unchanged_by_product(self):
        self.assertEqual(scorer._validate(model_output())["fact_check"], [])
        self.assertNotIn("product_claims", scorer.RUBRIC)

    def test_prompt_never_reveals_the_right_values(self):
        prompt = scorer.product_claims_prompt({"terms": {
            "interest_rate": {"label": "Interest rate", "value": 37.5, "unit": "% per year"}}})
        self.assertNotIn("37.5", prompt)


def rate_limited(per: str, retry_after: str = "0"):
    """A Groq 429 worded like the real one: '... on tokens per day (TPD)' or '... per minute (TPM)'."""
    import httpx
    from groq import RateLimitError
    req = httpx.Request("POST", "https://api.groq.com")
    return RateLimitError(f"Rate limit reached on tokens per {per}", response=httpx.Response(
        429, request=req, headers={"retry-after": retry_after}), body=None)


class FakeGroq:
    """Stands in for the Groq client: `script` maps model -> list of outcomes, consumed in order."""
    def __init__(self, script):
        self.script, self.calls = script, []
        self.chat = self
        self.completions = self

    def create(self, model, **kwargs):
        self.calls.append(model)
        outcome = self.script[model].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        msg = type("M", (), {"content": json.dumps(outcome)})
        return type("C", (), {"choices": [type("Ch", (), {"message": msg})]})


class ModelFallback(unittest.TestCase):
    def setUp(self):
        self._client, self._models = scorer.client, scorer.EVAL_MODELS
        scorer.EVAL_MODELS = ["main", "backup"]

    def tearDown(self):
        scorer.client, scorer.EVAL_MODELS = self._client, self._models

    def test_daily_limit_switches_to_backup_at_once(self):
        scorer.client = FakeGroq({"main": [rate_limited("day", "900")], "backup": [model_output()]})
        result, model = scorer._call_json("p", scorer._validate)
        self.assertEqual(model, "backup")
        self.assertEqual(scorer.client.calls, ["main", "backup"])

    def test_per_minute_limit_on_main_uses_backup_without_waiting(self):
        scorer.client = FakeGroq({"main": [rate_limited("minute", "45")], "backup": [model_output()]})
        with unittest.mock.patch.object(scorer.time, "sleep") as sleep:
            _, model = scorer._call_json("p", scorer._validate)
        self.assertEqual(model, "backup")
        sleep.assert_not_called()

    def test_malformed_output_retries_same_model(self):
        scorer.client = FakeGroq({"main": [{"criteria": {}}, model_output()], "backup": []})
        _, model = scorer._call_json("p", scorer._validate)
        self.assertEqual(model, "main")

    def test_both_out_of_quota_is_an_error_not_a_fake_score(self):
        scorer.client = FakeGroq({"main": [rate_limited("day", "900")], "backup": [rate_limited("day", "900")]})
        result = scorer.score_transcript([{"role": "user", "content": "hi"}])
        self.assertIn("error", result)

    def test_last_model_waits_out_per_minute_limit(self):
        # A long per-minute wait used to be mistaken for a daily limit; waits don't use up retries
        scorer.EVAL_MODELS = ["only"]
        scorer.client = FakeGroq({"only": [rate_limited("minute", "45"), rate_limited("minute", "45"),
                                           model_output()]})
        with unittest.mock.patch.object(scorer.time, "sleep") as sleep:
            _, model = scorer._call_json("p", scorer._validate)
        self.assertEqual(model, "only")
        self.assertEqual(sleep.call_count, 2)

    def test_last_model_daily_limit_is_final(self):
        scorer.EVAL_MODELS = ["only"]
        scorer.client = FakeGroq({"only": [rate_limited("day", "900")]})
        with self.assertRaises(Exception):
            scorer._call_json("p", scorer._validate)

    def test_connection_error_is_retried(self):
        import httpx
        from groq import APIConnectionError
        err = APIConnectionError(request=httpx.Request("POST", "https://api.groq.com"))
        scorer.client = FakeGroq({"main": [err, model_output()], "backup": []})
        with unittest.mock.patch.object(scorer.time, "sleep"):
            _, model = scorer._call_json("p", scorer._validate)
        self.assertEqual(model, "main")

    def test_scorecard_says_which_model_scored(self):
        scorer.client = FakeGroq({"main": [rate_limited("day", "900")], "backup": [model_output()]})
        result = scorer.score_transcript([{"role": "user", "content": "hi"}])
        self.assertEqual(result["scored_by"], "backup")


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
