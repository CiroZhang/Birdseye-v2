"""Evaluate rally QA answerers on a split (tools/qa2_build.py data):
  rule   deterministic reading of the detected shots (no learning)
  base   Qwen3-4B-Instruct zero-shot, same prompt as training
  lora   Qwen3-4B-Instruct + the fine-tuned LoRA adapter
usage: .venv_qa/Scripts/python tools/qa2_eval.py --answerer rule|base|lora [--split test] [--adapter outputs/qa2_lora]"""
import argparse, json, os, sys
from collections import defaultdict

import numpy as np

try:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True  # fp32 matmul with TF32 off stalls on the RTX 5080
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(__file__))
from qa2_common import MODEL, ROOT, TEST, VAL, correct, load, make_prompt, parse


def rule_answer(row, q):
    shots = row["shots"]
    if not shots:
        return None
    hit = [s for s in shots if s["anchored"]] or shots
    qid = q["id"]
    if qid == "n_hits":
        return len(hit)
    if qid == "last_hitter_side":
        return hit[-1]["hitter_side"]
    if qid == "landing_half":
        return "near" if row["final_xyz"][1] < 6.7 else "far"
    if qid == "max_height_m":
        return max(s["apex_m"] for s in shots)
    if qid == "highest_shot":
        return int(np.argmax([s["apex_m"] for s in hit])) + 1
    if qid == "duration_s":
        return shots[-1]["t_end_s"] - shots[0]["t_start_s"]
    if qid == "near_strokes":
        return sum(s["hitter_side"] == "near" for s in hit)
    k = q.get("k", 0) - 1
    if not (0 <= k < len(hit)):
        return None
    s = hit[k]
    return {"stroke_side": s["hitter_side"], "stroke_apex": s["apex_m"], "stroke_speed": s["speed_kmh"],
            "stroke_down": "yes" if s["end_xyz_m"][2] < s["hit_xyz_m"][2] and s["apex_m"] <= s["hit_xyz_m"][2] + 0.1 else "no"}[qid]


def summarize(results, quiet=False):
    by = defaultdict(list)
    for r in results:
        by[r["id"]].append(r["ok"])
    allv = [r["ok"] for r in results]
    if not quiet:
        print(f"{'question':18s} {'n':>5s} {'acc':>6s}")
        for k in sorted(by):
            print(f"{k:18s} {len(by[k]):5d} {np.mean(by[k]):6.3f}")
        print(f"{'ALL':18s} {len(allv):5d} {np.mean(allv):6.3f}")
        mac = np.mean([np.mean(v) for v in by.values()])
        print(f"{'MACRO (per type)':18s} {'':5s} {mac:6.3f}")
    return float(np.mean(allv))


def eval_llm(model, tok, rows, quiet=False, chunk=24):
    import torch
    results = []
    for row in rows:
        qs = row["questions"]
        for c in range(0, len(qs), chunk):
            part = qs[c:c + chunk]
            prompt = tok.apply_chat_template(make_prompt(row, part), tokenize=False, add_generation_prompt=True)
            ids = torch.tensor(tok(prompt, add_special_tokens=False)["input_ids"])
            n = -(-len(ids) // 256) * 256  # left-pad to few shapes (new shapes are slow on the RTX 5080)
            x = torch.cat([torch.full((n - len(ids),), tok.pad_token_id), ids])[None].to(model.device)
            mask = torch.cat([torch.zeros(n - len(ids), dtype=torch.long), torch.ones(len(ids), dtype=torch.long)])[None].to(model.device)
            with torch.no_grad():
                out = model.generate(input_ids=x, attention_mask=mask, max_new_tokens=8 * len(part) + 16,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            text = tok.decode(out[0, n:], skip_special_tokens=True)
            for q, p in zip(part, parse(text, len(part))):
                results.append({"match": row["match"], "rally": row["rally"], "id": q["id"], "k": q.get("k"),
                                "answer": q["answer"], "pred": p, "ok": correct(q, p)})
    return summarize(results, quiet), results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerer", default="rule", choices=["rule", "base", "lora"])
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--adapter", default=os.path.join(ROOT, "outputs", "qa2_lora"))
    a = ap.parse_args()
    rows = load(TEST if a.split == "test" else VAL)
    print("rallies", len(rows), "questions", sum(len(r["questions"]) for r in rows))
    if a.answerer == "rule":
        results = [{"match": r["match"], "rally": r["rally"], "id": q["id"], "k": q.get("k"), "answer": q["answer"],
                    "pred": (p := rule_answer(r, q)), "ok": correct(q, p)} for r in rows for q in r["questions"]]
        summarize(results)
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
        if a.answerer == "lora":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a.adapter)
        model.eval()
        _, results = eval_llm(model, tok, rows)
    dst = os.path.join(ROOT, "outputs", "qa2", f"results_{a.split}_{a.answerer}.json")
    json.dump(results, open(dst, "w"), indent=0)


if __name__ == "__main__":
    main()
