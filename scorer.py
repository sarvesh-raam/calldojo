"""Stage 5: Evaluation & LLM Scorer for Loan Agent Transcripts.
Scores agent calls against a strict 4-criteria rubric (1 to 5 scale).
"""
import os, json, time
from concurrent.futures import ThreadPoolExecutor
from groq import Groq, APIConnectionError, BadRequestError, RateLimitError
from dotenv import load_dotenv

load_dotenv()

# Rate limits are handled in _call_json, not by the SDK: the SDK would sit out a *daily* limit
# with backoff before failing, when the right move is to switch to the backup model at once.
client = Groq(api_key=os.environ.get("GROQ_API_KEY", "dummy_key"), max_retries=0, timeout=60)

# Groq's free-tier limits are per model, so a backup model has its own daily budget. Both models
# must be benchmarked (python evaluate_scorer.py --model ...) before they're trusted to score.
# gpt-oss-120b at low reasoning effort: 100% within 1 point on the held-out set, 11/11 grounding.
EVAL_MODELS = [m for m in (
    os.environ.get("GROQ_EVAL_MODEL", "openai/gpt-oss-120b"),
    os.environ.get("GROQ_EVAL_FALLBACK_MODEL", "openai/gpt-oss-20b"),
) if m]
# gpt-oss "thinks" before answering and those hidden tokens count against the daily budget
REASONING_EFFORT = os.environ.get("GROQ_EVAL_REASONING_EFFORT", "low")
CRITERIA = ("opening", "handling_objections", "honesty_about_charges", "closing")
SCORE_ATTEMPTS = 2
MAX_RATE_LIMIT_WAIT = 60  # seconds; Groq's per-minute windows clear within a minute
MAX_RATE_LIMIT_WAITS = 3  # per-minute waits on the last model before giving up

# Red flag -> maximum score per criterion. Enforced in code because the model, scoring each
# criterion from one quoted line, reliably misses lies and pressure elsewhere in the call.
FLAG_CAPS = {
    "false_or_implausible_claim":         {"honesty_about_charges": 1, "handling_objections": 1},
    "refused_or_postponed_charges":       {"honesty_about_charges": 2, "handling_objections": 2},
    "hedged_or_vague_figures":            {"honesty_about_charges": 3, "handling_objections": 3},
    "pressure_or_rudeness":               {"handling_objections": 1},
    "pressure_or_rudeness_in_final_turn": {"closing": 1},
    "unsafe_request":                     {"closing": 1},
}

# Caps honesty when the agent misstates the product. Kept out of FLAG_CAPS: it is decided by
# comparing numbers in code (see _apply_fact_check), not by a yes/no answer from the model.
PRODUCT_MISMATCH_CAP = 1


def product_claims_prompt(product: dict) -> str:
    """A separate, small extraction request. It lists the terms and units but deliberately NOT the
    correct values: the model only extracts what the agent said, so it can't drift toward the right
    answer. Code compares the numbers with the sheet (see _apply_fact_check)."""
    terms = "\n".join(f'  - {k}: {t["label"]}, in {t["unit"]}' for k, t in product["terms"].items())
    return f"""
You extract facts from a loan sales call. List every product term the Loan Agent stated AS A FACT
(not questions, not the customer's words). Use only these terms, converting to the unit shown:
{terms}
"No processing fee" or "zero foreclosure charges" is a claim of 0. Numbers said in words count
("twelve percent" is 12). Skip ranges ("12-14%"). The call may be in English, Hindi or Tamil.

Return ONLY a JSON object:
{{"product_claims": [{{"term": "<one of the terms above>", "number": <number>, "quote": "<agent line>"}}]}}
Use an empty list if the agent stated none.
"""


RUBRIC = """
You are an expert Quality Assurance (QA) Evaluator for loan sales phone calls.
Evaluate the Loan Agent's performance based ONLY on the provided conversation transcript.
Score each criterion independently: a bad opening must not lower the closing score, etc.
Judge what the agent actually said, not what a perfect agent would have said in a longer call.
The call may be in English, Hindi/Hinglish or Tamil/Tanglish. Apply the same anchors in every language.
Quote agent lines in their original language, but write all evidence and feedback in English.

Rate each of the following 4 criteria on a 1 to 5 scale using these anchors:

1. opening — the agent's first turn.
   - 5: Greets the customer by name, names the bank, AND states a specific reason for the call
        (e.g. "regarding your business loan inquiry"). The agent's own name is a bonus, not required.
   - 4: Greets, names the bank and a specific reason, but misses the customer's name or is curt.
   - 3: Names the bank but the purpose is generic ("calling about your loan", "loan desk", "loan team").
   - 2: Does not identify the bank or themselves, but is not rude ("Hey, calling about your loan").
   - 1: Rude or pressuring opener, or so unidentified that the customer has to ask who is calling.

2. handling_objections — how the agent responds to the customer's worries and questions.
   - 5: Answers every concern directly with specific facts AND acknowledges or reassures the customer
        (even briefly: "I understand", "Don't worry Ravi").
   - 4: Answers every question directly with specific facts, but gives no reassurance. Use 4 only when
        the customer asked informational questions without voicing fear or worry.
   - 3: The customer voiced a fear or worry and the agent answered with facts but ignored the feeling;
        OR the agent left one question unanswered; OR the first answer was vague and only became specific
        when the customer pushed.
   - 2: Deflects or postpones a concern ("we'll discuss that later") without being dismissive overall.
   - 1: Dismissive, defensive, rude, pressuring, or answers a concern with a false claim.

3. honesty_about_charges — accuracy and specificity about money.
   - 5: States a specific interest rate AND at least one specific fee or penalty term (processing fee,
        late fee, grace period, foreclosure charge), with nothing vague or misleading.
   - 4: Everything stated is specific and plausible, but the interest rate is never stated
        OR no fee/penalty term is stated.
   - 3: Hedged or vague figures ("around 12-14%", "late fee is minimal"), or only general reassurance
        ("no hidden charges") without numbers.
   - 2: Avoids or postpones one direct question about charges, but makes no false claim.
   - 1: False or implausible claims ("zero fees ever, pay whenever you want"), refuses to disclose charges,
        or scam signals such as asking for an OTP.

4. closing — the agent's final turn(s). Check three elements:
     (a) required documents named (e.g. Aadhaar, PAN),
     (b) a concrete next step with a channel or time (WhatsApp, email, link, callback, "within 24 hours"),
     (c) courteous and pressure-free tone ("if you are comfortable", "I'll send", not barked orders).
   - 5: All three elements present.
   - 4: Exactly two elements present.
   - 3: Only one element present, or the next step is vague ("upload your docs on our site").
   - 2: No next step even though the customer asked how to proceed, but not rude.
   - 1: Rude, abrupt dismissal, high pressure, or an unsafe request (OTP, PIN, password).

STEP 1 - RED FLAGS. Read the WHOLE transcript and answer each yes/no question about the
Loan Agent's lines. Quote the agent line when the answer is yes, otherwise use "".
  - false_or_implausible_claim: a claim no real lender makes ("never any penalty", "pay whenever you
    want", "zero fees ever", guaranteed approval). "No hidden charges" or "zero foreclosure charges"
    are plausible and do NOT count.
  - refused_or_postponed_charges: the customer asked about a charge and the agent refused, brushed it
    off, or postponed it ("we'll discuss fees after approval", "explained in person").
  - hedged_or_vague_figures: money terms given as ranges or adjectives ("around 12-14%", "1 to 2
    percent, depends", "small", "very reasonable", "standard"). A precise number with "about"
    ("about 9,964 rupees") does NOT count.
  - pressure_or_rudeness: anywhere in the call the agent is rude, dismissive, defensive, or pushes the
    customer to decide or act now ("offer expires tonight", "do you want it or not").
  - pressure_or_rudeness_in_final_turn: the same, but in the agent's LAST line.
  - unsafe_request: the agent asks for an OTP, PIN, password or CVV.

STEP 2 - SCORES. Score each criterion with the anchors above.

Return ONLY a valid JSON object matching this schema. For each criterion, the evidence must name
the agent's line AND the anchor it matches, e.g. "'Interest is around 12-14%' is a hedged figure -> anchor 3".
Then write the reason for the trainee: one short sentence in plain English saying what they did well
or what was missing, with no anchor numbers and no quotes, e.g. "Gave the rate as a range instead of an exact figure."
{
  "red_flags": {
    "false_or_implausible_claim":         {"present": <bool>, "quote": "<agent line or empty>"},
    "refused_or_postponed_charges":       {"present": <bool>, "quote": "<agent line or empty>"},
    "hedged_or_vague_figures":            {"present": <bool>, "quote": "<agent line or empty>"},
    "pressure_or_rudeness":               {"present": <bool>, "quote": "<agent line or empty>"},
    "pressure_or_rudeness_in_final_turn": {"present": <bool>, "quote": "<agent line or empty>"},
    "unsafe_request":                     {"present": <bool>, "quote": "<agent line or empty>"}
  },
  "criteria": {
    "opening":               {"evidence": "<agent line -> anchor N, 1 sentence>", "reason": "<for the trainee>", "score": <int 1-5>},
    "handling_objections":   {"evidence": "<agent line -> anchor N, 1 sentence>", "reason": "<for the trainee>", "score": <int 1-5>},
    "honesty_about_charges": {"evidence": "<agent line -> anchor N, 1 sentence>", "reason": "<for the trainee>", "score": <int 1-5>},
    "closing":               {"evidence": "<agent line -> anchor N, 1 sentence>", "reason": "<for the trainee>", "score": <int 1-5>}
  },
  "summary_feedback": "<2-sentence actionable coaching advice for the agent>"
}
"""


def _is_daily_limit(e: RateLimitError) -> bool:
    # Groq names the limit in the message: "... on tokens per day (TPD)" vs "... per minute (TPM)".
    # Guessing from the retry-after length misread long per-minute waits as daily limits.
    return "per day" in str(e)


def _retry_after(e: RateLimitError) -> float:
    """Seconds Groq asks us to wait (capped, so a trainee never waits more than a minute)."""
    try:
        return min(float(e.response.headers.get("retry-after", "5")), MAX_RATE_LIMIT_WAIT)
    except (AttributeError, ValueError):
        return 5.0


def _call_json(prompt: str, parse):
    """One JSON request, returning (parsed result, model that answered).

    - Malformed output (~1 in 10: Groq rejects invalid JSON with a 400, or a field is missing) is
      random, so it's retried on the same model. So is a dropped connection.
    - Any rate limit moves straight to the next model, which has its own quota. Only the last
      model waits out a per-minute limit; a daily limit there is final.
    """
    last_error = None
    for i, model in enumerate(EVAL_MODELS):
        is_last = i == len(EVAL_MODELS) - 1
        failures = waits = 0
        while failures < SCORE_ATTEMPTS:
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    reasoning_effort=REASONING_EFFORT,
                )
                return parse(json.loads(completion.choices[0].message.content)), model
            except RateLimitError as e:
                last_error = e
                if not is_last or _is_daily_limit(e) or waits >= MAX_RATE_LIMIT_WAITS:
                    print(f"[Scorer] {model} is rate-limited; moving to the next model, if any")
                    break
                waits += 1  # waiting is not a failure of the request, so it doesn't use a retry
                time.sleep(_retry_after(e))
            except APIConnectionError as e:
                last_error = e
                failures += 1
                print(f"[Scorer] {model}: connection error, retrying")
                time.sleep(2)
            except (BadRequestError, ValueError) as e:  # JSONDecodeError is a ValueError
                last_error = e
                failures += 1
                print(f"[Scorer] {model} returned malformed output: {str(e)[:120]}")
    raise last_error


def _parse_claims(result: dict) -> list:
    claims = result.get("product_claims")
    if not isinstance(claims, list):
        raise ValueError("response missing 'product_claims'")
    return claims


def score_transcript(transcript: list, product: dict = None) -> dict:
    """Takes a list of dialogue turns [{"role": "user"|"assistant", "content": "..."}]
    and returns a rubric evaluation JSON. With a product sheet, a second, smaller request
    extracts the agent's stated terms (in parallel) and code checks them against the sheet.
    The rubric request itself is identical with or without a product sheet.
    """
    # Format dialogue into readable script:
    # Notice: In our roleplay, "user" is the Loan Agent (human trainee)
    # and "assistant" is Ravi (the customer).
    formatted_lines = []
    for turn in transcript:
        if turn["role"] == "system":
            continue
        speaker = "Loan Agent" if turn["role"] == "user" else "Ravi (Customer)"
        formatted_lines.append(f"{speaker}: {turn['content']}")

    transcript_text = "\n".join(formatted_lines)

    call = f"\n\n--- CONVERSATION TRANSCRIPT ---\n{transcript_text}\n--- END TRANSCRIPT ---"

    with ThreadPoolExecutor(max_workers=2) as pool:
        rubric_job = pool.submit(_call_json, RUBRIC + call, _validate)
        claims_job = pool.submit(_call_json, product_claims_prompt(product) + call, _parse_claims) if product else None

        try:
            scorecard, model = rubric_job.result()
            scorecard["scored_by"] = model
        except Exception as e:
            # Never fabricate a scorecard: callers check "error" and show/skip it.
            print(f"[Scorer Error] {e}")
            return {"error": f"Scoring failed: {e}"}

        if claims_job:
            try:
                _apply_fact_check(scorecard, claims_job.result()[0], product)
            except Exception as e:
                # The scorecard is still valid; say the fact check didn't run rather than lose it
                print(f"[Scorer] Fact check failed: {e}")
                scorecard["fact_check_error"] = "The fact check against the product sheet couldn't run for this call."
    return scorecard


def _fact_check(claims: list, product: dict) -> list:
    """Compare each number the agent stated with the product sheet."""
    checked = []
    for c in claims:
        if not isinstance(c, dict) or c.get("term") not in product["terms"]:
            continue  # an unknown term is the model's slip, not the agent's
        try:
            said = float(c.get("number"))
        except (TypeError, ValueError):
            continue
        term = product["terms"][c["term"]]
        expected = float(term["value"])
        checked.append({
            "term": c["term"],
            "label": term["label"],
            "unit": term["unit"],
            "said": said,
            "expected": expected,
            "ok": abs(said - expected) < 1e-6,
            "quote": c.get("quote", ""),
        })
    return checked


def _num(x: float) -> str:
    return f"{x:g}"


def _apply_fact_check(scorecard: dict, claims: list, product: dict):
    """Attach the fact check to a validated scorecard and cap honesty on any mismatch."""
    fact_check = _fact_check(claims, product)
    scorecard["fact_check"] = fact_check
    wrong = [f for f in fact_check if not f["ok"]]
    if not wrong:
        return
    scorecard["red_flags"]["contradicts_product_sheet"] = "; ".join(
        f'said {f["label"].lower()} {_num(f["said"])}, sheet says {_num(f["expected"])}' for f in wrong)
    scores, c = scorecard["scores"], "honesty_about_charges"
    if scores[c] > PRODUCT_MISMATCH_CAP:
        scorecard["rationale"][c] += f" [Capped {scores[c]}->{PRODUCT_MISMATCH_CAP}: contradicts product sheet]"
        scores[c] = PRODUCT_MISMATCH_CAP
        scorecard["overall_score"] = round(sum(scores.values()) / len(CRITERIA), 2)


def _validate(result: dict) -> dict:
    """Check the model's output, apply red-flag caps, and reshape it into the
    {scores, rationale, overall_score, summary_feedback, red_flags} format the UI expects."""
    criteria = result.get("criteria")
    if not isinstance(criteria, dict):
        raise ValueError("response missing 'criteria'")
    flags = result.get("red_flags")
    if not isinstance(flags, dict):
        raise ValueError("response missing 'red_flags'")

    scores, rationale, evidence = {}, {}, {}
    for c in CRITERIA:
        entry = criteria.get(c) or {}
        s = entry.get("score")
        if not isinstance(s, int) or not 1 <= s <= 5:
            raise ValueError(f"invalid score for {c!r}: {s!r}")
        scores[c] = s
        evidence[c] = entry.get("evidence", "")
        # The trainee-facing reason; older-format responses only have evidence
        rationale[c] = entry.get("reason") or evidence[c]

    raised = {}
    for flag, caps in FLAG_CAPS.items():
        f = flags.get(flag) or {}
        if f.get("present") is not True:
            continue
        raised[flag] = f.get("quote", "")
        for c, cap in caps.items():
            if scores[c] > cap:
                rationale[c] += f" [Capped {scores[c]}->{cap}: {flag.replace('_', ' ')}]"
                scores[c] = cap

    return {
        "scores": scores,
        # Derived from the criteria rather than trusting the model's arithmetic
        "overall_score": round(sum(scores.values()) / len(CRITERIA), 2),
        "rationale": rationale,
        "evidence": evidence,
        "summary_feedback": result.get("summary_feedback", ""),
        "red_flags": raised,
        "fact_check": [],
    }


if __name__ == "__main__":
    sample_call = [
        {"role": "user", "content": "Hello Ravi, I am calling from Axis Bank regarding your loan inquiry."},
        {"role": "assistant", "content": "Oh hi, I'm a bit nervous about hidden charges."},
        {"role": "user", "content": "Don't worry Ravi, our interest is 12% per year with zero hidden charges and a 5-day grace period."},
        {"role": "assistant", "content": "That eases my mind. What documents do I need to submit?"},
        {"role": "user", "content": "Just your Aadhaar and PAN card. I will send you the upload link on WhatsApp right now."}
    ]
    print(json.dumps(score_transcript(sample_call), indent=2))
