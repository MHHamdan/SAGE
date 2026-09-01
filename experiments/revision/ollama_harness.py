"""Track 2 — real HotpotQA + Ollama harness (shared by smoke / 2B / 2A).

Everything here is REAL: real HotpotQA distractor items, real embedding retrieval
over the provided paragraphs, real EM/F1 answer matching, monitor signals computed
from real model outputs + embeddings, and MEASURED Ollama token counts. No synthetic
drift-threshold success, no action_0..7 tokens.
"""
from __future__ import annotations

import hashlib
import json
import re
import string
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


# ── Ollama calls (measured tokens) ────────────────────────────────────────────

def ollama_generate(model: str, prompt: str, temperature: float, seed: int,
                    max_tokens: int = 400, system: Optional[str] = None) -> dict:
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": temperature, "seed": seed,
                           "num_predict": max_tokens}}
    if system:
        payload["system"] = system
    req = urllib.request.Request(f"{OLLAMA}/api/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return {"text": d.get("response", ""),
            "prompt_tokens": int(d.get("prompt_eval_count", 0)),
            "completion_tokens": int(d.get("eval_count", 0))}


_EMB_CACHE: dict[str, np.ndarray] = {}


def ollama_embed(texts: list[str], model: str = EMBED_MODEL) -> np.ndarray:
    out, need, idx = [None] * len(texts), [], []
    for i, t in enumerate(texts):
        key = hashlib.md5((model + "\x00" + t).encode()).hexdigest()
        if key in _EMB_CACHE:
            out[i] = _EMB_CACHE[key]
        else:
            need.append(t); idx.append((i, key))
    if need:
        req = urllib.request.Request(f"{OLLAMA}/api/embed",
                                     data=json.dumps({"model": model, "input": need}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            embs = json.loads(r.read())["embeddings"]
        for (i, key), e in zip(idx, embs):
            v = np.asarray(e, dtype=np.float64)
            _EMB_CACHE[key] = v; out[i] = v
    return np.vstack(out)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


# ── HotpotQA official EM / F1 ──────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


# ── Data pool (deterministic, split-hashed) ────────────────────────────────────

def load_pool(n: int, cache_dir: Path, offset: int = 0) -> tuple[list[dict], str]:
    """Return N distractor items [{qid,question,answer,paragraphs,gold_titles}] and a split hash."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"pool_n{n}_off{offset}.json"
    if cache.exists():
        items = json.loads(cache.read_text())
    else:
        import datasets
        ds = datasets.load_dataset("hotpot_qa", "distractor", split="validation",
                                   streaming=True)
        items = []
        for j, ex in enumerate(ds):
            if j < offset:
                continue
            if len(items) >= n:
                break
            ctx = ex["context"]
            paras = [[t, " ".join(s)] for t, s in zip(ctx["title"], ctx["sentences"])]
            items.append({"qid": ex["id"], "question": ex["question"],
                          "answer": ex["answer"], "paragraphs": paras,
                          "gold_titles": list(dict.fromkeys(ex["supporting_facts"]["title"]))})
        cache.write_text(json.dumps(items))
    split_hash = hashlib.md5("|".join(it["qid"] for it in items).encode()).hexdigest()
    return items, split_hash


# ── Retrieval (real embeddings over the provided paragraphs) ───────────────────

def retrieve(query: str, paragraphs: list[list[str]], k: int) -> list[list[str]]:
    q = ollama_embed([query])[0]
    texts = [f"{t}. {b}" for t, b in paragraphs]
    P = ollama_embed(texts)
    sims = P @ q / (np.linalg.norm(P, axis=1) * np.linalg.norm(q) + 1e-9)
    order = np.argsort(-sims)[:k]
    return [paragraphs[i] for i in order]


# ── Agent action parsing (fidelity signal) ─────────────────────────────────────

_ACTION_RE = re.compile(r"ACTION:\s*(SEARCH|ANSWER)\[(.*?)\]", re.IGNORECASE | re.DOTALL)
_THOUGHT_RE = re.compile(r"THOUGHT:\s*(.*?)(?:\nACTION:|$)", re.IGNORECASE | re.DOTALL)


def parse_action(text: str):
    m = _ACTION_RE.search(text)
    tm = _THOUGHT_RE.search(text)
    thought = tm.group(1).strip() if tm else text.strip()
    if not m:
        return thought, None, None  # malformed -> low fidelity
    return thought, m.group(1).upper(), m.group(2).strip()


AGENT_SYSTEM = (
    "You are a multi-hop question-answering agent. Use ONLY the provided context. "
    "Each step, reason then act. Output EXACTLY:\n"
    "THOUGHT: <your reasoning>\n"
    "ACTION: SEARCH[<sub-query>]   (only if you still lack a key fact)\n"
    "   or\n"
    "ACTION: ANSWER[<final answer>]\n"
    "Answer as SHORT as possible: a name, entity, number, or date. For yes/no "
    "questions answer exactly 'yes' or 'no'. Prefer to ANSWER as soon as the "
    "context supports it; do NOT repeat similar searches."
)

FORCE_ANSWER_HINT = (
    "\n\nThis is your FINAL step — you must answer now from the context above. "
    "Output ONLY: ACTION: ANSWER[<shortest possible answer; 'yes'/'no' for yes-no "
    "questions>]"
)


def build_prompt(question: str, retrieved: list[list[str]], thoughts: list[str],
                 force_answer: bool = False) -> str:
    ctx = "\n".join(f"[{t}] {b}" for t, b in retrieved)
    hist = "\n".join(f"(prev thought {i+1}) {th}" for i, th in enumerate(thoughts))
    base = (f"Question: {question}\n\nContext:\n{ctx}\n\n"
            + (f"Your prior reasoning:\n{hist}\n\n" if hist else ""))
    if force_answer:
        return base + FORCE_ANSWER_HINT.strip()
    return base + "Respond with THOUGHT and ACTION now."


# ── One real episode ───────────────────────────────────────────────────────────

def run_episode(item: dict, agent_model: str, temperature: float, seed: int,
                k: int = 4, max_turns: int = 6,
                controller: Optional[Callable] = None,
                force_final_answer: bool = True,
                generate: Callable = ollama_generate) -> dict:
    """Real ReAct multi-hop episode. Returns trace + EM/F1 + monitor signals + tokens.

    controller(signals, turn) -> optional intervention name (for 2B); None = NoControl.
    force_final_answer=False (P1 long-horizon regime): the agent is NOT rescued with a
    forced answer at the turn budget; if it never emits ANSWER it fails terminally
    (derailment), which is the failure mode the controller is meant to recover.
    """
    q = item["question"]
    goal_emb = ollama_embed([q])[0]
    retrieved = retrieve(q, item["paragraphs"], k)
    thoughts: list[str] = []
    prev_titles: set = set()
    fidelity_hist: list[float] = []
    signal_trace: list[dict] = []
    interventions: list[str] = []
    tokens_prompt = tokens_completion = 0
    best_align = 0.0
    final_answer = ""
    turn_records = []

    answered = False
    for turn in range(1, max_turns + 1):
        prompt = build_prompt(q, retrieved, thoughts,
                              force_answer=(force_final_answer and turn == max_turns))
        gen = generate(agent_model, prompt, temperature, seed + turn,
                        system=AGENT_SYSTEM)
        tokens_prompt += gen["prompt_tokens"]
        tokens_completion += gen["completion_tokens"]
        thought, act, arg = parse_action(gen["text"])
        valid = act is not None
        fidelity_hist.append(1.0 if valid else 0.4)
        thoughts.append(thought[:300])

        # ── monitor signals from REAL embeddings/outputs ──
        state_emb = ollama_embed([thought or gen["text"][:400]])[0]
        align = cos(goal_emb, state_emb)
        best_align = max(best_align, align)
        drift = float(np.clip((1.0 - align) / 2.0, 0, 1))
        cur_titles = {t for t, _ in retrieved}
        inter = len(cur_titles & prev_titles)
        osc = inter / max(len(cur_titles), 1) if turn > 1 else 0.0
        fid = float(np.mean(fidelity_hist))
        conv = float(np.clip((best_align + 1) / 2, 0, 1))
        signals = {"turn": turn, "drift_score": drift, "oscillation_score": osc,
                   "fidelity_score": fid, "convergence_progress": conv}
        signal_trace.append(signals)

        # ── controller hook (2B) ──
        if controller is not None:
            interv = controller(signals, turn)
            if interv:
                interventions.append(interv)
                if interv == "GoalReanchor":  # re-inject goal + fresh retrieval on the question
                    retrieved = retrieve(q, item["paragraphs"], k)
                    thoughts.append(f"[re-anchored to goal: {q}]")
                elif interv == "ForceReplan":  # reset accumulated reasoning
                    thoughts = thoughts[-1:]

        turn_records.append({"turn": turn, "raw": gen["text"][:400], "thought": thought[:200],
                             "action": act, "arg": (arg or "")[:120], "valid": valid,
                             "signals": signals})

        if act == "ANSWER":
            final_answer = arg or ""
            answered = True
            break
        if act == "SEARCH" and arg:
            more = retrieve(arg, item["paragraphs"], k)
            seen = {t for t, _ in retrieved}
            retrieved = retrieved + [p for p in more if p[0] not in seen]
            retrieved = retrieved[:k + 3]
        prev_titles = cur_titles

    if not answered and force_final_answer:
        # dedicated forced-answer call — never use a search query as the answer
        fp = build_prompt(q, retrieved, thoughts, force_answer=True)
        gen = generate(agent_model, fp, temperature, seed + 999,
                        system=AGENT_SYSTEM, max_tokens=64)
        tokens_prompt += gen["prompt_tokens"]
        tokens_completion += gen["completion_tokens"]
        _, act2, arg2 = parse_action(gen["text"])
        final_answer = (arg2 or gen["text"].strip())[:200]

    # derailed = ran out of the turn budget without ever committing an answer
    derailed = (not answered)
    em = exact_match(final_answer, item["answer"])
    f1 = f1_score(final_answer, item["answer"])
    return {
        "qid": item["qid"], "question": q, "gold": item["answer"],
        "final_answer": final_answer, "em": em, "f1": f1,
        "answered": answered, "derailed": derailed,
        "retrieved_titles": [t for t, _ in retrieved],
        "gold_titles": item["gold_titles"],
        "turns_used": len(turn_records), "interventions": interventions,
        "n_interventions": len(interventions),
        "prompt_tokens": tokens_prompt, "completion_tokens": tokens_completion,
        "signal_trace": signal_trace, "turn_records": turn_records,
    }
