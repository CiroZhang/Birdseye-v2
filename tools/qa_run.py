"""Answer + score the rally QA benchmark (tools/qa_build.py).

answerers:
  rule    deterministic reading of the 3D scene (what the reconstruction alone supports)
  claude  Claude Opus 5.5 reads the same scene JSON and answers in JSON
          (needs credentials: ANTHROPIC_API_KEY or `ant auth login`; costs money)
usage: python tools/qa_run.py outputs/qa/test.jsonl --answerer rule|claude"""
import argparse, json, os, sys

import numpy as np

SYSTEM = (
    "You answer questions about one badminton doubles rally using ONLY the 3D reconstruction given as JSON. "
    "It was produced automatically from a single back-view camera, so it can be noisy or miss strokes; use "
    "physical reasoning about the court and the shuttle's flight. Reply with only a JSON object "
    '{"answer": <value>} where <value> is an integer, a number, or one of the allowed words.'
)


def rule_answer(scene, qid):
    shots = scene["shots"]
    if not shots:
        return None
    if qid == "n_hits":
        return sum(1 for s in shots if s["anchored"]) or len(shots)
    if qid == "last_hitter_side":
        hit = [s for s in shots if s["anchored"]] or shots
        return hit[-1]["hitter_side"]
    if qid == "landing_half":
        return "near" if scene["final_shuttle_xyz_m"][1] < 6.7 else "far"
    if qid == "max_height_m":
        return max(s["apex_m"] for s in shots)
    if qid == "highest_shot":
        anchored = [s for s in shots if s["anchored"]] or shots
        best = max(range(len(shots)), key=lambda i: shots[i]["apex_m"])
        # number strokes by anchored shots up to and including the highest one
        return sum(1 for s in shots[:best + 1] if s["anchored"]) or best + 1
    return None


def claude_answer(client, scene, question):
    import anthropic

    resp = client.beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=2000,
        system=SYSTEM,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": f"Rally reconstruction:\n{json.dumps(scene)}\n\nQuestion: {question}"}],
    )
    if resp.stop_reason == "refusal":
        return None
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1])["answer"]
    except (ValueError, KeyError):
        return None


def correct(q, pred):
    if pred is None:
        return False
    try:
        if q["kind"] == "choice":
            return str(pred).strip().lower() == q["answer"]
        if q["kind"] == "int":
            return int(round(float(pred))) == q["answer"]
        return abs(float(pred) - q["answer"]) <= q["tol"]
    except (TypeError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--answerer", default="rule", choices=["rule", "claude"])
    ap.add_argument("--limit", type=int, default=None, help="max rallies (claude costs money)")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.path)][: a.limit]
    client = None
    if a.answerer == "claude":
        import anthropic

        client = anthropic.Anthropic()
    stats, out = {}, []
    for r in rows:
        for q in r["questions"]:
            pred = rule_answer(r["scene"], q["id"]) if client is None else claude_answer(client, r["scene"], q["question"])
            ok = correct(q, pred)
            stats.setdefault(q["id"], []).append(ok)
            out.append({"match": r["match"], "rally": r["rally"], "id": q["id"], "answer": q["answer"], "pred": pred, "ok": ok})
    print(f"{'question':18s} {'n':>4s} {'acc':>6s}")
    for k, v in stats.items():
        print(f"{k:18s} {len(v):4d} {np.mean(v):6.3f}")
    allv = [x for v in stats.values() for x in v]
    print(f"{'ALL':18s} {len(allv):4d} {np.mean(allv):6.3f}")
    dst = a.path.replace(".jsonl", f".{a.answerer}.results.json")
    json.dump(out, open(dst, "w"), indent=1)


if __name__ == "__main__":
    main()
