"""Track 2 SMOKE — 5 tasks x 1 seed, real loop, agent=llama3.1:8b, temp=0.6.

Verifies (a) EM/F1 non-degenerate, (b) retrieval returns real paragraphs,
(c) monitor signals non-constant from real embeddings, (d) token counts captured.
Prints 3 example traces and STOPS. No scaling until approved.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.revision.ollama_harness import load_pool, run_episode  # noqa: E402

AGENT = "llama3.1:8b"
TEMP = 0.6          # 2B controller/monitor regime
SEED = 0
N = 5
DATA_DIR = ROOT / "results" / "ollama_real" / "data"


def main():
    items, split_hash = load_pool(N, DATA_DIR)
    print(f"[smoke] loaded {len(items)} HotpotQA-distractor items  split_hash={split_hash[:12]}")
    print(f"[smoke] agent={AGENT} temp={TEMP} seed={SEED}\n")

    results = []
    for i, it in enumerate(items):
        r = run_episode(it, AGENT, TEMP, SEED, k=4, max_turns=6, controller=None)
        results.append(r)
        print(f"  task {i}: EM={r['em']} F1={r['f1']:.2f} turns={r['turns_used']} "
              f"tok(in/out)={r['prompt_tokens']}/{r['completion_tokens']} "
              f"ans='{r['final_answer'][:50]}' gold='{r['gold'][:40]}'")

    # ── verification checks ──
    ems = [r["em"] for r in results]
    f1s = [r["f1"] for r in results]
    toks = [r["prompt_tokens"] + r["completion_tokens"] for r in results]
    all_sig = [(s["drift_score"], s["oscillation_score"], s["fidelity_score"])
               for r in results for s in r["signal_trace"]]
    arr = np.array(all_sig)
    print("\n[smoke] === VERIFICATION ===")
    print(f"  (a) EM/F1 non-degenerate: EM mean={np.mean(ems):.2f} (not all 0/1: {0 < sum(ems) < len(ems) or any(0<f<1 for f in f1s)}), "
          f"F1 range=[{min(f1s):.2f},{max(f1s):.2f}]")
    retr_ok = all(len(r["retrieved_titles"]) >= 3 and any(t in r["gold_titles"] for t in r["retrieved_titles"]) for r in results)
    print(f"  (b) retrieval real: every task retrieved >=3 titles; gold-title hit in retrieved for all={retr_ok}")
    print(f"      example retrieved titles[0]={results[0]['retrieved_titles']}")
    print(f"  (c) monitors non-constant (std over {len(arr)} turn-signals): "
          f"drift std={arr[:,0].std():.3f}, osc std={arr[:,1].std():.3f}, fid std={arr[:,2].std():.3f}")
    print(f"  (d) tokens captured: per-task total tokens={toks} (all>0: {all(t>0 for t in toks)})")

    # ── 3 example traces ──
    for idx in range(min(3, len(results))):
        r = results[idx]
        print(f"\n[smoke] ===== EXAMPLE TRACE {idx} (qid={r['qid']}) =====")
        print(f"  Q: {r['question']}")
        print(f"  GOLD: {r['gold']}   GOLD_TITLES: {r['gold_titles']}")
        print(f"  RETRIEVED (final): {r['retrieved_titles']}")
        for tr in r["turn_records"]:
            s = tr["signals"]
            print(f"   turn {tr['turn']}: action={tr['action']}[{tr['arg']}] valid={tr['valid']}")
            print(f"      thought: {tr['thought']}")
            print(f"      monitor: drift={s['drift_score']:.3f} osc={s['oscillation_score']:.3f} "
                  f"fid={s['fidelity_score']:.3f} conv={s['convergence_progress']:.3f}")
        print(f"  FINAL ANSWER: '{r['final_answer']}'   EM={r['em']} F1={r['f1']:.2f}")

    print("\n[smoke] DONE — review the 3 traces above. STOPPING (no scaling).")


if __name__ == "__main__":
    main()
