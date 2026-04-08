"""
stress_test.py — Reward & grader stress tester for pii_redaction_env.

Usage (local, run from project root):
    python stress_test.py

Usage (against live HF Space):
    python stress_test.py --url https://your-space.hf.space

Each test prints PASS / FAIL and a reason. Any FAIL is a real bug.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

# ── optional local imports (skip gracefully if running remote-only) ──────────
try:
    from env import PIIRedactionEnv
    from graders import grade_easy, grade_medium, grade_hard
    from models import PIIAction, PIIType, PIIReward, RedactionSpan
    from reward import compute_reward
    HAS_LOCAL = True
except ImportError:
    HAS_LOCAL = False


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"
results: list[tuple[str, bool, str]] = []


def report(name: str, passed: bool, detail: str = "") -> None:
    tag = PASS if passed else FAIL
    suffix = f"  →  {detail}" if detail else ""
    print(f"  {tag}  {name}{suffix}")
    results.append((name, passed, detail))


def skip(name: str, reason: str) -> None:
    print(f"  {SKIP}  {name}  →  {reason}")
    results.append((name, None, reason))


# ════════════════════════════════════════════════════════════════════════════
# LOCAL UNIT TESTS  (require local install)
# ════════════════════════════════════════════════════════════════════════════

def _span(start: int, end: int, pii_type: str = "PERSON", text: str = "x") -> "RedactionSpan":
    return RedactionSpan(start=start, end=end, pii_type=pii_type, text=text[:end-start] or text)


def test_reward_empty_vs_empty() -> None:
    """Empty predicted + empty gold must not produce 0.0 or 1.0."""
    r = compute_reward(predicted=[], gold=[], step_count=1)
    report(
        "reward: empty predicted + empty gold never 0.0/1.0",
        0.0 < r.total < 1.0,
        f"total={r.total}",
    )


def test_reward_exact_match() -> None:
    """Perfect prediction must not exceed 0.99."""
    gold = [_span(0, 5, "PERSON", "Alice"), _span(10, 20, "EMAIL", "x@test.com")]
    r = compute_reward(predicted=gold, gold=gold, step_count=1, terminal_score=0.99)
    report(
        "reward: perfect match clamped to <=0.99",
        r.total <= 0.99,
        f"total={r.total}",
    )


def test_reward_massive_false_positives() -> None:
    """1000 false-positive spans must not drive reward below 0.01 (clamp check)."""
    gold = [_span(0, 5, "PERSON", "Alice")]
    predicted = [_span(i * 10, i * 10 + 5, "EMAIL", "fakeX") for i in range(1, 1001)]
    r = compute_reward(predicted=predicted, gold=gold, step_count=1)
    report(
        "reward: 1000 false positives clamped to >=0.01",
        r.total >= 0.01,
        f"total={r.total}  penalty={r.false_positive_penalty:.2f}",
    )


def test_reward_terminal_score_zero() -> None:
    """terminal_score=0.0 is falsy — verify `or 0.0` doesn't corrupt terminal_score field."""
    gold = [_span(0, 5, "PERSON", "Alice")]
    # graders never return 0.0 but compute_reward can be called directly
    r = compute_reward(predicted=[], gold=gold, step_count=1, terminal_score=0.0)
    # total should come from terminal_score=0.0 path → clamped to 0.01
    report(
        "reward: terminal_score=0.0 → total clamped to 0.01 (not left at 0.0)",
        r.total >= 0.01,
        f"total={r.total}  terminal_score_field={r.terminal_score}",
    )


def test_reward_terminal_score_one() -> None:
    """terminal_score=1.0 must be clamped to 0.99."""
    gold = [_span(0, 5, "PERSON", "Alice")]
    r = compute_reward(predicted=gold, gold=gold, step_count=1, terminal_score=1.0)
    report(
        "reward: terminal_score=1.0 clamped to <=0.99",
        r.total <= 0.99,
        f"total={r.total}",
    )


def test_reward_negative_step_penalty() -> None:
    """1000 steps + no matches — total must still be >=0.01."""
    r = compute_reward(predicted=[], gold=[], step_count=1000)
    report(
        "reward: 1000 steps, no match → total >=0.01",
        r.total >= 0.01,
        f"total={r.total}  step_penalty={r.step_penalty:.2f}",
    )


def test_reward_piitype_total_field_writeable() -> None:
    """PIIReward.total has ge=-1.0, le=1.0 — writing out-of-bounds must raise."""
    r = PIIReward()
    raised = False
    try:
        r.total = 2.0  # should violate le=1.0
    except Exception:
        raised = True
    report(
        "reward: PIIReward.total rejects values >1.0",
        raised,
        "pydantic validator should block 2.0",
    )


def test_grader_easy_partial_miss() -> None:
    """grade_easy with one missing span must be strictly between 0 and 1."""
    gold = [_span(0, 5, "PERSON", "Alice"), _span(10, 15, "EMAIL", "b@bbb")]
    partial = gold[:1]  # missing one
    score = grade_easy(partial, gold)
    report(
        "grade_easy: partial prediction in (0, 1)",
        0.0 < score < 1.0,
        f"score={score}  ← returns 0.01 (binary grader, no partial credit)",
    )


def test_grader_easy_extra_span() -> None:
    """grade_easy: exact gold + one extra FP must not return 0.99."""
    gold = [_span(0, 5, "PERSON", "Alice")]
    predicted = gold + [_span(20, 25, "EMAIL", "extra")]
    score = grade_easy(predicted, gold)
    report(
        "grade_easy: extra FP drops score to 0.01 (expected — binary design)",
        score == 0.01,
        f"score={score}",
    )


def test_grader_medium_all_wrong_types() -> None:
    """grade_medium: correct positions but all wrong PII types must not score high."""
    gold = [_span(0, 5, "PERSON", "Alice"), _span(10, 15, "EMAIL", "b@bbb")]
    predicted = [_span(0, 5, "SSN", "Alice"), _span(10, 15, "PHONE", "b@bbb")]
    score = grade_medium(predicted, gold)
    report(
        "grade_medium: right positions, wrong types → low score",
        score < 0.5,
        f"score={score}",
    )


def test_grader_hard_quasi_bonus_cap() -> None:
    """grade_hard: perfect quasi recall should not push score above 0.99."""
    gold = [_span(0, 10, "QUASI_IDENTIFIER", "John 1980")]
    score = grade_hard(gold, gold)
    report(
        "grade_hard: perfect quasi recall clamped to <=0.99",
        score <= 0.99,
        f"score={score}",
    )


def test_grader_hard_empty_both() -> None:
    """grade_hard: empty predicted + empty gold must not produce 0.0 or 1.0."""
    score = grade_hard([], [])
    report(
        "grade_hard: empty vs empty in (0, 1)",
        0.0 < score < 1.0,
        f"score={score}",
    )


def test_env_step_without_reset() -> None:
    """Stepping without reset must not crash and must return valid reward."""
    env = PIIRedactionEnv()
    try:
        obs = env.step(PIIAction(spans=[], submit=True))
        ok = 0.0 < obs.reward < 1.0
        report("env: step() without reset returns valid reward", ok, f"reward={obs.reward}")
    except Exception as e:
        report("env: step() without reset returns valid reward", False, str(e))
    finally:
        env.close()


def test_env_step_after_done() -> None:
    """Step after done=True must auto-reset and return valid reward (not crash)."""
    env = PIIRedactionEnv()
    try:
        env.reset(seed=42, task_id="basic_pii_detection")
        env.step(PIIAction(spans=[], submit=True))  # sets done=True
        obs = env.step(PIIAction(spans=[], submit=True))  # should auto-reset
        ok = 0.0 < obs.reward < 1.0
        report("env: step() after done=True auto-resets and returns valid reward", ok, f"reward={obs.reward}")
    except Exception as e:
        report("env: step() after done=True auto-resets and returns valid reward", False, str(e))
    finally:
        env.close()


def test_env_concurrent_state_isolation() -> None:
    """Two threads calling reset+step on the same env instance must not corrupt each other."""
    env = PIIRedactionEnv()
    errors: list[str] = []

    def worker(task_id: str) -> None:
        for _ in range(5):
            try:
                env.reset(seed=42, task_id=task_id)
                obs = env.step(PIIAction(spans=[], submit=True))
                if not (0.0 < obs.reward < 1.0):
                    errors.append(f"{task_id}: invalid reward {obs.reward}")
            except Exception as e:
                errors.append(f"{task_id}: exception {e}")

    t1 = threading.Thread(target=worker, args=("basic_pii_detection",))
    t2 = threading.Thread(target=worker, args=("mixed_pii_redaction",))
    t1.start(); t2.start()
    t1.join(); t2.join()
    env.close()

    report(
        "env: concurrent reset+step on shared instance doesn't corrupt reward",
        len(errors) == 0,
        "; ".join(errors) if errors else "no corruption detected (NOTE: env is not thread-safe by design — this may pass by luck)",
    )


def test_env_all_seeds_produce_valid_rewards() -> None:
    """Seeds 0–19 across all tasks must all produce rewards in (0, 1)."""
    env = PIIRedactionEnv()
    bad: list[str] = []
    tasks = ["basic_pii_detection", "mixed_pii_redaction", "adversarial_quasi_identification"]
    for task_id in tasks:
        for seed in range(20):
            obs = env.reset(seed=seed, task_id=task_id)
            result = env.step(PIIAction(spans=[], submit=True))
            if not (0.0 < result.reward < 1.0):
                bad.append(f"{task_id}@seed={seed}: reward={result.reward}")
    env.close()
    report(
        "env: seeds 0-19 all tasks → reward always in (0, 1)",
        len(bad) == 0,
        "; ".join(bad) if bad else "all 60 combos valid",
    )


def test_env_duplicate_spans_no_double_reward() -> None:
    """Submitting the same span twice must not double the reward."""
    env = PIIRedactionEnv()
    env.reset(seed=42, task_id="basic_pii_detection")
    gold = list(env.state.ground_truth_spans)

    env.reset(seed=42, task_id="basic_pii_detection")
    r_single = env.step(PIIAction(spans=gold, submit=True)).reward

    env.reset(seed=42, task_id="basic_pii_detection")
    r_doubled = env.step(PIIAction(spans=gold + gold, submit=True)).reward

    env.close()
    report(
        "reward: submitting duplicate spans doesn't inflate reward",
        r_doubled <= r_single,
        f"single={r_single:.4f}  doubled={r_doubled:.4f}",
    )


# ════════════════════════════════════════════════════════════════════════════
# HTTP INTEGRATION TESTS  (require live server URL)
# ════════════════════════════════════════════════════════════════════════════

def http_get(base: str, path: str) -> requests.Response:
    return requests.get(f"{base}{path}", timeout=30)


def http_post(base: str, path: str, body: Any) -> requests.Response:
    return requests.post(f"{base}{path}", json=body, timeout=30)


def test_http_reset_never_returns_zero_or_one(base: str) -> None:
    for task_id in ["basic_pii_detection", "mixed_pii_redaction", "adversarial_quasi_identification"]:
        r = http_post(base, "/reset", {"seed": 42, "task_id": task_id})
        data = r.json()
        reward = data.get("reward")
        report(
            f"HTTP reset({task_id}): reward not 0.0 or 1.0",
            reward not in (0.0, 1.0),
            f"reward={reward}",
        )


def test_http_step_empty_spans_never_zero_or_one(base: str) -> None:
    for task_id in ["basic_pii_detection", "mixed_pii_redaction", "adversarial_quasi_identification"]:
        http_post(base, "/reset", {"seed": 42, "task_id": task_id})
        r = http_post(base, "/step", {"action": {"spans": [], "submit": True}})
        data = r.json()
        reward = data.get("reward")
        obs = data.get("observation", {})
        final = obs.get("final_score")
        details_total = obs.get("reward_details", {}).get("total")
        bad = any(v in (0.0, 1.0) for v in [reward, final, details_total] if v is not None)
        report(
            f"HTTP step({task_id}, empty): no field is exactly 0.0 or 1.0",
            not bad,
            f"reward={reward} final_score={final} details.total={details_total}",
        )


def test_http_step_flood_false_positives(base: str) -> None:
    """500 FP spans — reward must stay in (0, 1)."""
    http_post(base, "/reset", {"seed": 42, "task_id": "basic_pii_detection"})
    fake_spans = [
        {"start": i * 6, "end": i * 6 + 5, "pii_type": "EMAIL", "text": "fakeX"}
        for i in range(500)
    ]
    r = http_post(base, "/step", {"action": {"spans": fake_spans, "submit": True}})
    if r.status_code != 200:
        report("HTTP step(500 FP spans): reward in (0, 1)", False, f"HTTP {r.status_code}")
        return
    data = r.json()
    reward = data.get("reward")
    report(
        "HTTP step(500 FP spans): reward in (0, 1)",
        isinstance(reward, float) and 0.0 < reward < 1.0,
        f"reward={reward}",
    )


def test_http_concurrent_resets(base: str) -> None:
    """10 concurrent resets must all return valid rewards."""
    def do_reset(task_id: str) -> tuple[str, float | None]:
        r = http_post(base, "/reset", {"seed": 42, "task_id": task_id})
        return task_id, r.json().get("reward")

    tasks_cycle = ["basic_pii_detection", "mixed_pii_redaction", "adversarial_quasi_identification"] * 4
    bad: list[str] = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(do_reset, t) for t in tasks_cycle[:10]]
        for f in as_completed(futures):
            task_id, reward = f.result()
            if reward in (0.0, 1.0) or reward is None:
                bad.append(f"{task_id}: reward={reward}")
    report(
        "HTTP: 10 concurrent resets all return valid rewards",
        len(bad) == 0,
        "; ".join(bad) if bad else "all clean",
    )


def test_http_malformed_span(base: str) -> None:
    """Span with start > end must be rejected gracefully (not crash server)."""
    http_post(base, "/reset", {"seed": 42, "task_id": "basic_pii_detection"})
    bad_span = {"start": 100, "end": 5, "pii_type": "PERSON", "text": "x"}
    r = http_post(base, "/step", {"action": {"spans": [bad_span], "submit": True}})
    # Should be 422 validation error, not 500
    report(
        "HTTP step(start > end): returns 422, not 500",
        r.status_code == 422,
        f"status={r.status_code}",
    )


def test_http_unknown_task_id(base: str) -> None:
    r = http_post(base, "/reset", {"seed": 42, "task_id": "nonexistent_task"})
    report(
        "HTTP reset(unknown task_id): returns 4xx not 500",
        400 <= r.status_code < 500,
        f"status={r.status_code}",
    )


# ════════════════════════════════════════════════════════════════════════════
# RUNNER
# ════════════════════════════════════════════════════════════════════════════

def run_local_tests() -> None:
    print("\n━━━ LOCAL UNIT TESTS ━━━")
    test_reward_empty_vs_empty()
    test_reward_exact_match()
    test_reward_massive_false_positives()
    test_reward_terminal_score_zero()
    test_reward_terminal_score_one()
    test_reward_negative_step_penalty()
    test_reward_piitype_total_field_writeable()
    test_grader_easy_partial_miss()
    test_grader_easy_extra_span()
    test_grader_medium_all_wrong_types()
    test_grader_hard_quasi_bonus_cap()
    test_grader_hard_empty_both()
    test_env_step_without_reset()
    test_env_step_after_done()
    test_env_concurrent_state_isolation()
    test_env_all_seeds_produce_valid_rewards()
    test_env_duplicate_spans_no_double_reward()


def run_http_tests(base: str) -> None:
    print(f"\n━━━ HTTP INTEGRATION TESTS  ({base}) ━━━")
    test_http_reset_never_returns_zero_or_one(base)
    test_http_step_empty_spans_never_zero_or_one(base)
    test_http_step_flood_false_positives(base)
    test_http_concurrent_resets(base)
    test_http_malformed_span(base)
    test_http_unknown_task_id(base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None, help="Base URL of live HF Space (e.g. https://user-env.hf.space)")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--http-only", action="store_true")
    args = parser.parse_args()

    if not args.http_only:
        if HAS_LOCAL:
            run_local_tests()
        else:
            print("\n[WARN] Local imports unavailable — skipping unit tests. Run from project root.")

    if not args.local_only:
        if args.url:
            run_http_tests(args.url.rstrip("/"))
        else:
            print("\n[INFO] No --url provided — skipping HTTP tests. Pass --url https://your-space.hf.space")

    total = len([r for r in results if r[1] is not None])
    passed = len([r for r in results if r[1] is True])
    failed = len([r for r in results if r[1] is False])

    print(f"\n━━━ SUMMARY: {passed}/{total} passed", end="")
    if failed:
        print(f"  |  \033[31m{failed} FAILED\033[0m", end="")
    print(" ━━━\n")

    if failed:
        print("FAILED tests:")
        for name, ok, detail in results:
            if ok is False:
                print(f"  ✗  {name}")
                if detail:
                    print(f"       {detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
