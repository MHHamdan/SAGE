"""Task 4 — Generate LaTeX Fragments from CSV outputs.

Reads all experiment CSVs and fills [CC:key] placeholders in LaTeX templates,
then writes completed .tex files to results/.

Output files:
  results/table_cnsr_multitask.tex
  results/table_prop1_violations.tex
  results/table_judge_bias.tex
  results/supp_a1_results.tex
  results/supp_a2_results.tex
  results/supp_a3_results.tex
  results/table_fragments.tex   (all fragments concatenated)
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # allow direct import of sage
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))


# ── placeholder filler ────────────────────────────────────────────────────────

def fill_placeholder(template: str, values: dict[str, Any]) -> str:
    """Replace [CC:key] markers with formatted values."""
    for key, val in values.items():
        if isinstance(val, float):
            formatted = f"{val:.3f}"
        else:
            formatted = str(val)
        template = template.replace(f"[CC:{key}]", formatted)
    return template


# ── CSV readers ───────────────────────────────────────────────────────────────

def _read_csv(name: str) -> list[dict[str, str]]:
    path = RESULTS_DIR / name
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _float(x: str) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0


def _bool(x: str) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes")


# ── value extractors ──────────────────────────────────────────────────────────

def extract_cnsr_values() -> dict[str, Any]:
    rows = _read_csv("cnsr_multitask.csv")
    if not rows:
        return {}

    # CNSR per (config, task_type, seed)
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["config"], r["task_type"], r["seed"])].append(r)

    seeds = sorted({r["seed"] for r in rows})
    models = list(dict.fromkeys(r["config"] for r in rows))
    task_types = ["code", "web", "research"]

    vals: dict[str, Any] = {}
    best_cnsr = 0.0
    best_model = ""
    for model in models:
        for tt in task_types:
            seed_cnsrs = []
            for seed in seeds:
                g = groups[(model, tt, seed)]
                if not g:
                    continue
                succ = sum(1 for r in g if _bool(r["success"]))
                cost = sum(_float(r["cost_usd"]) for r in g)
                n = len(g)
                sr = succ / n
                mc = cost / n if n > 0 else 1e-9
                cnsr = sr / mc if mc > 1e-12 else 0.0
                seed_cnsrs.append(cnsr)
            if seed_cnsrs:
                key_m = model.replace("/", "_").replace("-", "_").replace(".", "_")
                vals[f"cnsr_{key_m}_{tt}_mean"] = float(np.mean(seed_cnsrs))
                vals[f"cnsr_{key_m}_{tt}_sd"] = float(np.std(seed_cnsrs, ddof=1) if len(seed_cnsrs) > 1 else 0.0)
                mean_cnsr = float(np.mean(seed_cnsrs))
                if mean_cnsr > best_cnsr:
                    best_cnsr = mean_cnsr
                    best_model = model

    vals["best_cnsr"] = best_cnsr
    vals["best_model"] = best_model
    return vals


def extract_prop1_values() -> dict[str, Any]:
    vals: dict[str, Any] = {}

    # A1 — obs fidelity
    a1 = _read_csv("exp_a1.csv")
    if a1:
        grouped: dict[str, list] = defaultdict(list)
        for r in a1:
            grouped[r["injection_rate"]].append(r)
        for rate_str, g in grouped.items():
            rate_key = rate_str.replace(".", "p")
            sr = sum(1 for r in g if _bool(r["success"])) / len(g)
            osc = sum(1 for r in g if _bool(r["oscillation_detected"])) / len(g)
            msf = float(np.mean([_float(r["steps_to_failure"]) for r in g]))
            vals[f"a1_sr_{rate_key}"] = sr
            vals[f"a1_osc_{rate_key}"] = osc
            vals[f"a1_msf_{rate_key}"] = msf
        # Overall failure rate at max injection
        max_rate = max(grouped.keys(), key=float)
        g_max = grouped[max_rate]
        vals["a1_failure_rate_max"] = 1.0 - sum(1 for r in g_max if _bool(r["success"])) / len(g_max)

    # A2 — progress monotonicity
    a2 = _read_csv("exp_a2.csv")
    if a2:
        grouped2: dict[str, list] = defaultdict(list)
        for r in a2:
            grouped2[r["stall_prob"]].append(r)
        for sp_str, g in grouped2.items():
            sp_key = sp_str.replace(".", "p")
            dr = sum(1 for r in g if _bool(r["deadlock_detected"])) / len(g)
            sr = sum(1 for r in g if _bool(r["task_success"])) / len(g)
            det = [_float(r["turns_to_detection"]) for r in g if _bool(r["deadlock_detected"])]
            mtd = float(np.mean(det)) if det else 0.0
            vals[f"a2_deadlock_{sp_key}"] = dr
            vals[f"a2_sr_{sp_key}"] = sr
            vals[f"a2_turns_{sp_key}"] = mtd

    # A3 — context noise
    a3 = _read_csv("exp_a3.csv")
    if a3:
        final = [r for r in a3 if r["turn"] == "50"]
        grouped3: dict[str, list] = defaultdict(list)
        for r in final:
            grouped3[r["reanchor_interval"]].append(r)
        for intv, g in grouped3.items():
            key = f"interval_{intv}".replace(".", "p")
            md = float(np.mean([_float(r["drift_score"]) for r in g]))
            cr = sum(1 for r in g if _bool(r["task_completed"])) / len(g)
            vals[f"a3_drift_{key}"] = md
            vals[f"a3_completion_{key}"] = cr

    return vals


def extract_judge_bias_values() -> dict[str, Any]:
    rows = _read_csv("judge_bias.csv")
    if not rows:
        return {}

    agg: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        agg[(r["bias_type"], r["with_mitigation"])].append(_float(r["metric_value"]))

    vals: dict[str, Any] = {}
    for bias in ("self_preference", "position", "verbosity"):
        for mit in ("False", "True"):
            key = f"{bias}_{'mit' if mit == 'True' else 'nomt'}"
            v = float(np.mean(agg.get((bias, mit), [0.0])))
            vals[key] = v
        no = vals.get(f"{bias}_nomt", 0.0)
        yes = vals.get(f"{bias}_mit", 0.0)
        red = (no - yes) / no * 100 if no > 0 else 0.0
        vals[f"{bias}_reduction_pct"] = red
    return vals


# ── LaTeX template builders ───────────────────────────────────────────────────

def _table_cnsr(cv: dict) -> str:
    best = cv.get("best_model", "N/A")
    best_c = cv.get("best_cnsr", 0.0)
    return (
        "% Auto-generated by generate_latex.py\n"
        "% Table V — CNSR Multi-Task Results\n"
        "% Best model: " + best + f" (CNSR={best_c:.3f})\n"
        r"\input{results/cnsr_table}" + "\n"
    )


def _table_prop1(pv: dict) -> str:
    lines = [
        r"% Table — Proposition 1 Violation Experiments",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Proposition~1 Violation Experiments: Effect on Agent Behaviour}",
        r"\label{tab:prop1_violations}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Experiment} & \textbf{Parameter} & \textbf{Success Rate} & \textbf{Failure Metric} & \textbf{Violation Rate} \\",
        r"\midrule",
    ]
    # A1 rows
    for rate_str, rate_key in [("0.0", "0p0"), ("0.1", "0p1"), ("0.2", "0p2"), ("0.4", "0p4")]:
        sr = pv.get(f"a1_sr_{rate_key}", 0.0)
        osc = pv.get(f"a1_osc_{rate_key}", 0.0)
        msf = pv.get(f"a1_msf_{rate_key}", 0.0)
        lines.append(
            f"A1 (obs fidelity) & inj.={rate_str} & {sr:.2%} & "
            f"steps={msf:.1f} & {osc:.2%} \\\\"
        )
    lines.append(r"\midrule")
    # A2 rows
    for sp_str, sp_key in [("0.0", "0p0"), ("0.25", "0p25"), ("0.5", "0p5")]:
        dr = pv.get(f"a2_deadlock_{sp_key}", 0.0)
        sr = pv.get(f"a2_sr_{sp_key}", 0.0)
        mt = pv.get(f"a2_turns_{sp_key}", 0.0)
        lines.append(
            f"A2 (prog. mono) & stall={sp_str} & {sr:.2%} & "
            f"turns={mt:.1f} & {dr:.2%} \\\\"
        )
    lines.append(r"\midrule")
    # A3 rows
    for intv in ["5", "10", "20", "None"]:
        key = f"interval_{intv}".replace(".", "p")
        md = pv.get(f"a3_drift_{key}", 0.0)
        cr = pv.get(f"a3_completion_{key}", 0.0)
        lines.append(
            f"A3 (context noise) & reanchor={intv} & {cr:.2%} & "
            f"drift={md:.4f} & --- \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _table_judge_bias(jv: dict) -> str:
    lines = [
        r"% Table — LLM-as-Judge Bias Mitigation",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{LLM-as-Judge Bias Before and After Mitigation}",
        r"\label{tab:judge_bias_main}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Bias} & \textbf{No Mitigation} & \textbf{With Mitigation} & \textbf{Reduction} \\",
        r"\midrule",
        (
            r"Self-preference ($\Delta$) & "
            f"${jv.get('self_preference_nomt', 0):.3f}$ & "
            f"${jv.get('self_preference_mit', 0):.3f}$ & "
            f"${jv.get('self_preference_reduction_pct', 0):.1f}\\%$ \\\\"
        ),
        (
            r"Position bias & "
            f"${jv.get('position_nomt', 0):.3f}$ & "
            f"${jv.get('position_mit', 0):.3f}$ & "
            f"${jv.get('position_reduction_pct', 0):.1f}\\%$ \\\\"
        ),
        (
            r"Verbosity bias ($|r|$) & "
            f"${jv.get('verbosity_nomt', 0):.3f}$ & "
            f"${jv.get('verbosity_mit', 0):.3f}$ & "
            f"${jv.get('verbosity_reduction_pct', 0):.1f}\\%$ \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _supp_a1(pv: dict) -> str:
    lines = [
        r"% Supplementary A.5 Table A1 — Observation Fidelity",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Supp.\ A.5 Table A1: Observation Fidelity Experiment Results}",
        r"\label{tab:supp_a1}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Injection Rate} & \textbf{Success Rate} & \textbf{Mean Steps to Failure} & \textbf{Oscillation Rate} \\",
        r"\midrule",
    ]
    for rate_str, rate_key in [("0.0", "0p0"), ("0.1", "0p1"), ("0.2", "0p2"), ("0.4", "0p4")]:
        sr = pv.get(f"a1_sr_{rate_key}", 0.0)
        osc = pv.get(f"a1_osc_{rate_key}", 0.0)
        msf = pv.get(f"a1_msf_{rate_key}", 0.0)
        lines.append(f"{rate_str} & {sr:.2%} & {msf:.1f} & {osc:.2%} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _supp_a2(pv: dict) -> str:
    lines = [
        r"% Supplementary A.5 Table A2 — Progress Monotonicity",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Supp.\ A.5 Table A2: Progress Monotonicity Experiment Results}",
        r"\label{tab:supp_a2}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Stall Prob.} & \textbf{Deadlock Rate} & \textbf{Mean Turns to Detection} & \textbf{Task Success Rate} \\",
        r"\midrule",
    ]
    for sp_str, sp_key in [("0.0", "0p0"), ("0.25", "0p25"), ("0.5", "0p5")]:
        dr = pv.get(f"a2_deadlock_{sp_key}", 0.0)
        sr = pv.get(f"a2_sr_{sp_key}", 0.0)
        mt = pv.get(f"a2_turns_{sp_key}", 0.0)
        mt_str = f"{mt:.1f}" if mt > 0 else "---"
        lines.append(f"{sp_str} & {dr:.2%} & {mt_str} & {sr:.2%} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _supp_a3(pv: dict) -> str:
    lines = [
        r"% Supplementary A.5 Table A3 — Context Noise / Goal Drift",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Supp.\ A.5 Table A3: Context Noise / Goal Drift Experiment Results}",
        r"\label{tab:supp_a3}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Re-anchor Interval} & \textbf{Mean Drift at Turn 50} & \textbf{Completion Rate} \\",
        r"\midrule",
    ]
    for intv in ["5", "10", "20", "None"]:
        key = f"interval_{intv}".replace(".", "p")
        md = pv.get(f"a3_drift_{key}", 0.0)
        cr = pv.get(f"a3_completion_{key}", 0.0)
        intv_disp = intv if intv != "None" else "\\textit{none}"
        lines.append(f"{intv_disp} & {md:.4f} & {cr:.2%} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Generating LaTeX fragments …")

    cv = extract_cnsr_values()
    pv = extract_prop1_values()
    jv = extract_judge_bias_values()

    outputs: dict[str, str] = {
        "table_cnsr_multitask.tex": _table_cnsr(cv),
        "table_prop1_violations.tex": _table_prop1(pv),
        "table_judge_bias.tex": _table_judge_bias(jv),
        "supp_a1_results.tex": _supp_a1(pv),
        "supp_a2_results.tex": _supp_a2(pv),
        "supp_a3_results.tex": _supp_a3(pv),
    }

    for fname, content in outputs.items():
        path = RESULTS_DIR / fname
        path.write_text(content)
        print(f"  Wrote {path}")

    # Concatenated fragments
    all_content = "\n\n% ─────────────────────────────────────────────────────\n\n".join(
        outputs.values()
    )
    frag_path = RESULTS_DIR / "table_fragments.tex"
    frag_path.write_text(all_content)
    print(f"  Wrote {frag_path}")

    # Print value summary
    print("\nExtracted values:")
    print(f"  CNSR: {len(cv)} values  best_cnsr={cv.get('best_cnsr', 0):.4f}")
    print(f"  Prop1: {len(pv)} values")
    print(f"  JudgeBias: {len(jv)} values")
    print("Done.")


if __name__ == "__main__":
    main()
