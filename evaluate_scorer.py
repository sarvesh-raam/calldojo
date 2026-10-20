"""Stage 5: Evaluation Script - Human vs. LLM Scorer Agreement
Measures how well the LLM evaluation agrees with labelled benchmark transcripts.
Computes:
1. Exact Match %
2. Within-1-Point Agreement % (Key Metric, target > 85%)
3. Mean Absolute Error (MAE) and signed bias (LLM - label) per criterion

Benchmark sets live in benchmark/*.json:
  dev.json  - used while tuning the rubric (expect optimistic numbers)
  test.json - held out; written after the rubric was fixed. Report this one.

  grounding.json - calls with correct/wrong product terms; checks the product-sheet fact check

Usage: python evaluate_scorer.py [--split dev|test|all|grounding] [--runs N]
"""
import argparse, json, os, sys, time
import scorer
from scorer import score_transcript, CRITERIA

# Windows consoles default to cp1252; model rationales can contain characters it can't encode
sys.stdout.reconfigure(errors="replace")

# Pause between calls; raise it (--delay) for models with small per-minute limits
DELAY = 1.0
BENCHMARK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark")


def load_split(split: str) -> list:
    names = ["dev", "test"] if split == "all" else [split]
    data = []
    for n in names:
        with open(os.path.join(BENCHMARK_DIR, f"{n}.json"), encoding="utf-8") as f:
            data += json.load(f)
    return data


def run_evaluation(dataset: list, verbose: bool = True) -> dict:
    exact_matches = {c: 0 for c in CRITERIA}
    within_one = {c: 0 for c in CRITERIA}
    total_diffs = {c: 0 for c in CRITERIA}
    signed_diffs = {c: 0 for c in CRITERIA}
    disagreements = []
    failed = []

    total_turns = len(dataset)
    for idx, item in enumerate(dataset):
        res = score_transcript(item["transcript"])
        if "error" in res:
            # Excluded rather than scored: a failure must not count as agreement
            print(f"[{idx+1}/{total_turns}] {item['id']}: SKIPPED ({res['error']})", flush=True)
            failed.append(item["id"])
            continue
        llm_scores = res["scores"]
        if verbose:
            print(f"[{idx+1}/{total_turns}] {item['id']}: Human={item['human_scores']} | LLM={llm_scores}", flush=True)
        time.sleep(DELAY)

        for c in CRITERIA:
            h = item["human_scores"][c]
            l = llm_scores[c]
            diff = abs(h - l)
            total_diffs[c] += diff
            signed_diffs[c] += l - h
            if diff == 0:
                exact_matches[c] += 1
            if diff <= 1:
                within_one[c] += 1
            else:
                disagreements.append({
                    "call_id": item["id"],
                    "criterion": c,
                    "human": h,
                    "llm": l,
                    "reason": res["rationale"].get(c, "")
                })

    scored = total_turns - len(failed)
    if scored == 0:
        print("No transcripts were scored successfully; no agreement metrics.")
        return {"scored": 0, "failed": failed}

    per_criterion = {
        c: {
            "exact_pct": exact_matches[c] / scored * 100,
            "within_1_pct": within_one[c] / scored * 100,
            "mae": total_diffs[c] / scored,
            "bias": signed_diffs[c] / scored,
        }
        for c in CRITERIA
    }
    return {
        "exact_match_pct": sum(exact_matches.values()) / (scored * 4) * 100,
        "within_1_point_pct": sum(within_one.values()) / (scored * 4) * 100,
        "per_criterion": per_criterion,
        "scored": scored,
        "failed": failed,
        "disagreements": disagreements,
    }


def run_grounding(dataset: list, product: dict) -> dict:
    """Did the fact check flag exactly the terms the agent got wrong?"""
    true_pos = false_pos = false_neg = 0
    failed, misses = [], []
    for idx, item in enumerate(dataset):
        res = score_transcript(item["transcript"], product)
        if "error" in res:
            print(f"[{idx+1}/{len(dataset)}] {item['id']}: SKIPPED ({res['error']})", flush=True)
            failed.append(item["id"])
            continue
        if res.get("fact_check_error"):
            # The rubric scored but the fact check didn't run: "nothing flagged" is not a pass
            print(f"[{idx+1}/{len(dataset)}] {item['id']}: SKIPPED (fact check didn't run)", flush=True)
            failed.append(item["id"])
            continue
        flagged = {f["term"] for f in res["fact_check"] if not f["ok"]}
        expected = set(item["expected_wrong"])
        true_pos += len(flagged & expected)
        false_pos += len(flagged - expected)
        false_neg += len(expected - flagged)
        ok = flagged == expected
        if expected and res["scores"]["honesty_about_charges"] > 1:
            ok = False  # a caught mismatch must also cap honesty
        mark = "ok  " if ok else "MISS"
        print(f"[{idx+1}/{len(dataset)}] {mark} {item['id']}: expected={sorted(expected)} flagged={sorted(flagged)} "
              f"honesty={res['scores']['honesty_about_charges']}", flush=True)
        if not ok:
            misses.append(item["id"])
        time.sleep(DELAY)

    scored = len(dataset) - len(failed)
    print("\n" + "=" * 72)
    print(f"GROUNDING: product-sheet fact check  (calls={len(dataset)}, scored={scored})")
    print("=" * 72)
    if scored:
        caught = true_pos / (true_pos + false_neg) * 100 if true_pos + false_neg else 100.0
        print(f"Calls fully correct:            {scored - len(misses)}/{scored}")
        print(f"Wrong terms caught (recall):    {true_pos}/{true_pos + false_neg} ({caught:.0f}%)")
        print(f"Correct terms flagged wrongly:  {false_pos}")
    if misses:
        print(f"Misses: {', '.join(misses)}")
    if failed:
        print(f"Skipped (scorer error): {', '.join(failed)}")
    return {"misses": misses, "failed": failed, "false_pos": false_pos, "false_neg": false_neg}


def print_report(results: list, split: str, n_calls: int):
    ok = [r for r in results if r.get("scored")]
    if not ok:
        return
    runs = len(ok)
    mean = lambda xs: sum(xs) / len(xs)

    print("\n" + "=" * 72)
    print(f"EVALUATION RESULTS: HUMAN VS. LLM SCORER  (split={split}, calls={n_calls}, runs={runs})")
    print("=" * 72)
    w1 = [r["within_1_point_pct"] for r in ok]
    em = [r["exact_match_pct"] for r in ok]
    spread = f"  [runs: {', '.join(f'{x:.1f}' for x in w1)}]" if runs > 1 else ""
    print(f"Overall Exact Match Agreement:       {mean(em):.1f}%")
    print(f"Overall Within-1-Point Agreement:    {mean(w1):.1f}%  (Target: >85%){spread}\n")

    print(f"{'Criterion':<25} | {'Exact Match':<11} | {'Within 1 Pt':<11} | {'MAE':<5} | Bias (LLM-label)")
    print("-" * 72)
    for c in CRITERIA:
        pc = [r["per_criterion"][c] for r in ok]
        print(f"{c:<25} | {mean([p['exact_pct'] for p in pc]):>10.1f}% | "
              f"{mean([p['within_1_pct'] for p in pc]):>10.1f}% | "
              f"{mean([p['mae'] for p in pc]):>5.2f} | {mean([p['bias'] for p in pc]):>+5.2f}")

    failed = sorted({f for r in results for f in r.get("failed", [])})
    if failed:
        print(f"\nSkipped (scorer error) in at least one run: {', '.join(failed)}")

    print("\nNotable Disagreements (Diff > 1):")
    seen = set()
    for r in ok:
        for d in r["disagreements"]:
            key = (d["call_id"], d["criterion"])
            if key in seen:
                continue
            seen.add(key)
            print(f"- {d['call_id']} on '{d['criterion']}': Human={d['human']} vs LLM={d['llm']}. Rationale: {d['reason']}")
    if not seen:
        print("- None! High alignment achieved across all criteria.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all", "grounding"], default="test")
    ap.add_argument("--runs", type=int, default=1, help="repeat to measure run-to-run variance")
    ap.add_argument("--model", help="benchmark this one model, with no fallback, so results "
                                    "are attributable (default: the live primary/backup chain)")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between calls")
    args = ap.parse_args()
    DELAY = args.delay
    if args.model:
        scorer.EVAL_MODELS[:] = [args.model]
    print(f"Scorer model(s): {', '.join(scorer.EVAL_MODELS)} | reasoning effort: {scorer.REASONING_EFFORT}")

    dataset = load_split(args.split)
    if args.split == "grounding":
        with open(os.path.join(os.path.dirname(BENCHMARK_DIR), "product.json"), encoding="utf-8") as f:
            run_grounding(dataset, json.load(f))
        sys.exit(0)
    results = []
    for run in range(args.runs):
        print(f"\nRun {run+1}/{args.runs}: evaluating {len(dataset)} transcripts ({args.split})...")
        results.append(run_evaluation(dataset))
    print_report(results, args.split, len(dataset))
