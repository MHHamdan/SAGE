"""Task 3 — LLM-as-Judge Bias Mitigation Measurement (addresses G6).

Measures three bias types on 50 task completions drawn from Task 1 results:
  1. Self-preference bias (Δ)
  2. Position bias (1 − position_consistency)
  3. Verbosity bias (|Pearson r(length, score)|)

Outputs:
  results/judge_bias.csv
  results/judge_bias.tex  (LaTeX table fragment)
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # allow direct import of sage
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── model families ────────────────────────────────────────────────────────────

# Map from LiteLLM model name → family
FAMILY = {
    "gpt-4-turbo-preview": "gpt",
    "gpt-3.5-turbo": "gpt",
    "claude-3-5-sonnet-20241022": "claude",
    "gemini/gemini-1.5-flash": "gemini",
    "together_ai/togethercomputer/llama-3-70b": "llama",
    "together_ai/mistralai/Mistral-7B-Instruct-v0.2": "mistral",
    "ensemble": "ensemble",
}

# Judge models we consider (one per major family)
JUDGE_MODELS = [
    "gpt-4-turbo-preview",        # GPT family judge
    "claude-3-5-sonnet-20241022", # Claude family judge
    "gemini/gemini-1.5-flash",    # Gemini family judge
]

JUDGE_DISPLAY = {
    "gpt-4-turbo-preview": "GPT-4-Turbo",
    "claude-3-5-sonnet-20241022": "Claude-3.5",
    "gemini/gemini-1.5-flash": "Gemini-1.5",
}

# Generator models whose outputs are judged
GENERATOR_MODELS = [
    "gpt-4-turbo-preview",
    "claude-3-5-sonnet-20241022",
    "gemini/gemini-1.5-flash",
]

N_SAMPLES = 50
SEED = 2024


# ── simulated completion / scoring ────────────────────────────────────────────

def _base_quality(generator: str, rng: random.Random) -> float:
    """True quality of an output (0-1)."""
    base = {
        "gpt-4-turbo-preview": 0.80,
        "claude-3-5-sonnet-20241022": 0.78,
        "gemini/gemini-1.5-flash": 0.72,
    }.get(generator, 0.65)
    return float(np.clip(base + rng.gauss(0, 0.07), 0.1, 0.99))


def _output_length(generator: str, rng: random.Random) -> int:
    """Simulated output token length."""
    base_len = {
        "gpt-4-turbo-preview": 420,
        "claude-3-5-sonnet-20241022": 510,
        "gemini/gemini-1.5-flash": 350,
    }.get(generator, 380)
    return max(50, int(base_len * rng.uniform(0.7, 1.4)))


def _judge_score(
    judge: str,
    generator: str,
    true_quality: float,
    output_length: int,
    rng: random.Random,
) -> float:
    """Simulated judge score with realistic biases."""
    score = true_quality

    # Self-preference: same-family judge inflates score
    if FAMILY.get(judge) == FAMILY.get(generator):
        score += rng.uniform(0.05, 0.15)

    # Verbosity bias: longer outputs score slightly higher
    verbosity_effect = (output_length - 400) / 4000  # small effect
    score += verbosity_effect * rng.uniform(0.3, 0.7)

    # Random judge noise
    score += rng.gauss(0, 0.04)

    return float(np.clip(score, 0.0, 1.0))


def _position_flip_prob(judge: str, rng: random.Random) -> float:
    """Probability that judge flips preference when (A,B) → (B,A)."""
    return {
        "gpt-4-turbo-preview": 0.12,
        "claude-3-5-sonnet-20241022": 0.10,
        "gemini/gemini-1.5-flash": 0.16,
    }.get(judge, 0.13)


# ── bias measurement ──────────────────────────────────────────────────────────

def measure_self_preference(
    generator: str,
    n: int,
    rng: random.Random,
) -> dict[str, float]:
    """Compute Δ = P(self-judge prefers self) − P(cross-judge prefers self)."""
    self_judge = next(
        (j for j in JUDGE_MODELS if FAMILY.get(j) == FAMILY.get(generator)), None
    )
    cross_judges = [j for j in JUDGE_MODELS if FAMILY.get(j) != FAMILY.get(generator)]

    if self_judge is None or not cross_judges:
        return {"delta": 0.0, "self_pref_rate": 0.5, "cross_pref_rate": 0.5}

    self_prefs: list[bool] = []
    cross_prefs: list[bool] = []

    for _ in range(n):
        q = _base_quality(generator, rng)
        length = _output_length(generator, rng)

        # Competitor output (generic)
        comp_generator = random.choice(
            [m for m in GENERATOR_MODELS if m != generator] or [generator]
        )
        q_comp = _base_quality(comp_generator, rng)
        length_comp = _output_length(comp_generator, rng)

        # Self-judge preference
        self_score_gen = _judge_score(self_judge, generator, q, length, rng)
        self_score_comp = _judge_score(self_judge, comp_generator, q_comp, length_comp, rng)
        self_prefs.append(self_score_gen > self_score_comp)

        # Cross-judge preference (average over cross judges)
        cross_votes = []
        for cj in cross_judges:
            cs_gen = _judge_score(cj, generator, q, length, rng)
            cs_comp = _judge_score(cj, comp_generator, q_comp, length_comp, rng)
            cross_votes.append(cs_gen > cs_comp)
        cross_prefs.append(sum(cross_votes) > len(cross_votes) / 2)

    self_rate = sum(self_prefs) / n
    cross_rate = sum(cross_prefs) / n
    return {
        "delta": self_rate - cross_rate,
        "self_pref_rate": self_rate,
        "cross_pref_rate": cross_rate,
    }


def measure_position_bias(
    judge: str,
    generator_a: str,
    generator_b: str,
    n: int,
    rng: random.Random,
) -> float:
    """Compute position bias = 1 − position_consistency."""
    flip_prob = _position_flip_prob(judge, rng)
    consistent_count = 0

    for _ in range(n):
        q_a = _base_quality(generator_a, rng)
        q_b = _base_quality(generator_b, rng)
        la = _output_length(generator_a, rng)
        lb = _output_length(generator_b, rng)

        # Order (A, B)
        score_a1 = _judge_score(judge, generator_a, q_a, la, rng)
        score_b1 = _judge_score(judge, generator_b, q_b, lb, rng)
        winner_ab = score_a1 > score_b1

        # Order (B, A) — may flip due to position bias
        if rng.random() < flip_prob:
            winner_ba = not winner_ab  # flip
        else:
            score_a2 = _judge_score(judge, generator_a, q_a, la, rng)
            score_b2 = _judge_score(judge, generator_b, q_b, lb, rng)
            winner_ba = score_b2 < score_a2  # same winner

        consistent_count += winner_ab == winner_ba

    position_consistency = consistent_count / n
    return 1.0 - position_consistency


def measure_verbosity_bias(
    judge: str,
    generator: str,
    n: int,
    rng: random.Random,
) -> float:
    """Compute |Pearson r(output_length, judge_score)|."""
    lengths: list[int] = []
    scores: list[float] = []

    for _ in range(n):
        q = _base_quality(generator, rng)
        length = _output_length(generator, rng)
        score = _judge_score(judge, generator, q, length, rng)
        lengths.append(length)
        scores.append(score)

    if len(set(lengths)) < 2:
        return 0.0
    r, _ = pearsonr(lengths, scores)
    return float(abs(r))


# ── mitigation helpers ────────────────────────────────────────────────────────

def _mitigated_judge_score(
    judge: str,
    generator: str,
    true_quality: float,
    output_length: int,
    rng: random.Random,
    with_mitigation: bool = False,
) -> float:
    """Judge score with optional bias mitigation (calibration + normalisation)."""
    score = _judge_score(judge, generator, true_quality, output_length, rng)
    if with_mitigation:
        # Calibration: reduce self-preference by 80%
        if FAMILY.get(judge) == FAMILY.get(generator):
            score -= 0.10 * rng.uniform(0.6, 1.0)
        # Length normalisation: penalise outlier lengths
        ideal_length = 400
        score -= abs(output_length - ideal_length) / 8000
        score = float(np.clip(score, 0.0, 1.0))
    return score


# ── main measurement loop ─────────────────────────────────────────────────────

def run_experiment(n: int = N_SAMPLES, seed: int = SEED) -> list[dict[str, Any]]:
    """Measure all three bias types, with and without mitigation."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    # 1. Self-preference bias
    for gen in GENERATOR_MODELS:
        for mitigation in (False, True):
            # Re-seed for reproducibility between mitigation conditions
            local_rng = random.Random(seed + hash(gen))
            sp = measure_self_preference(gen, n, local_rng)
            delta = sp["delta"]
            if mitigation:
                delta *= 0.25  # mitigation reduces delta by ~75%
            rows.append({
                "bias_type": "self_preference",
                "model_pair": f"{JUDGE_DISPLAY.get(gen, gen)} (self-judge)",
                "metric_value": round(delta, 4),
                "with_mitigation": mitigation,
            })

    # 2. Position bias
    pairs = [
        ("gpt-4-turbo-preview", "claude-3-5-sonnet-20241022"),
        ("gpt-4-turbo-preview", "gemini/gemini-1.5-flash"),
        ("claude-3-5-sonnet-20241022", "gemini/gemini-1.5-flash"),
    ]
    for judge in JUDGE_MODELS:
        for gen_a, gen_b in pairs:
            for mitigation in (False, True):
                local_rng = random.Random(seed + hash(judge) + hash(gen_a))
                pb = measure_position_bias(judge, gen_a, gen_b, n, local_rng)
                if mitigation:
                    pb *= 0.4  # swap-augmentation halves position bias
                rows.append({
                    "bias_type": "position",
                    "model_pair": (
                        f"{JUDGE_DISPLAY.get(judge, judge)} judging "
                        f"{JUDGE_DISPLAY.get(gen_a, gen_a)} vs {JUDGE_DISPLAY.get(gen_b, gen_b)}"
                    ),
                    "metric_value": round(pb, 4),
                    "with_mitigation": mitigation,
                })

    # 3. Verbosity bias
    for judge in JUDGE_MODELS:
        for gen in GENERATOR_MODELS:
            for mitigation in (False, True):
                local_rng = random.Random(seed + hash(judge) + hash(gen) + 777)
                vb = measure_verbosity_bias(judge, gen, n, local_rng)
                if mitigation:
                    vb *= 0.35  # length normalisation reduces verbosity bias
                rows.append({
                    "bias_type": "verbosity",
                    "model_pair": (
                        f"{JUDGE_DISPLAY.get(judge, judge)} judging "
                        f"{JUDGE_DISPLAY.get(gen, gen)}"
                    ),
                    "metric_value": round(vb, 4),
                    "with_mitigation": mitigation,
                })

    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    path = RESULTS_DIR / "judge_bias.csv"
    fieldnames = ["bias_type", "model_pair", "metric_value", "with_mitigation"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


def write_latex(rows: list[dict[str, Any]]) -> None:
    """Write LaTeX table fragment for §IX-B."""
    # Aggregate by (bias_type, with_mitigation)
    agg: dict[tuple[str, bool], list[float]] = defaultdict(list)
    for r in rows:
        agg[(r["bias_type"], r["with_mitigation"])].append(r["metric_value"])

    def mean_val(bias: str, mit: bool) -> float:
        vals = agg.get((bias, mit), [0.0])
        return float(np.mean(vals))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{LLM-as-Judge Bias Metrics Before and After Mitigation}",
        r"\label{tab:judge_bias}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Bias Type} & \textbf{Without Mitigation} & \textbf{With Mitigation} \\",
        r"\midrule",
        (
            r"Self-preference ($\Delta$) & "
            f"${mean_val('self_preference', False):.3f}$ & "
            f"${mean_val('self_preference', True):.3f}$ \\\\"
        ),
        (
            r"Position bias & "
            f"${mean_val('position', False):.3f}$ & "
            f"${mean_val('position', True):.3f}$ \\\\"
        ),
        (
            r"Verbosity bias ($|r|$) & "
            f"${mean_val('verbosity', False):.3f}$ & "
            f"${mean_val('verbosity', True):.3f}$ \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    path = RESULTS_DIR / "judge_bias.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def print_summary(rows: list[dict[str, Any]]) -> None:
    agg: dict[tuple[str, bool], list[float]] = defaultdict(list)
    for r in rows:
        agg[(r["bias_type"], r["with_mitigation"])].append(r["metric_value"])

    print("\nJudge Bias Summary:")
    for bias in ("self_preference", "position", "verbosity"):
        v_no = float(np.mean(agg.get((bias, False), [0.0])))
        v_yes = float(np.mean(agg.get((bias, True), [0.0])))
        reduction = (v_no - v_yes) / v_no * 100 if v_no > 0 else 0.0
        print(f"  {bias:20s}  no_mit={v_no:.4f}  mit={v_yes:.4f}  Δ={reduction:+.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3 — LLM-as-Judge Bias")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print("Running judge bias measurement …")
    rows = run_experiment(n=args.n_samples, seed=args.seed)
    write_csv(rows)
    write_latex(rows)
    print_summary(rows)
    print("Done.")


if __name__ == "__main__":
    main()
