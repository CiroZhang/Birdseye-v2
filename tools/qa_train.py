"""FBBench QA with one model and one training protocol across input representations.

Model: Qwen2.5-VL-7B-Instruct + LoRA (rank 16, language model only), fine-tuned on FBBench-style questions generated
from FineBadminton-20K (tools/fbqa_build.py), tested on FBBench. The model answers by scoring the option letters at
the first answer position (cross-entropy over the letters in training, arg-max at test), so there is no free-form
generation to parse. COND picks what the model sees besides the question:
  none   the question and options only (what the text alone gives away)
  2d     broadcast-view 2D tracks as text: per side mid-ankle, head, wrists; the shuttle (tools/qa_scene.py d2)
  3d     the Birdseye reconstruction as text: per side pelvis, wrists, racket head; the shuttle; hits (d3)
  2d3d   both texts
  rgb    the video: frames at RGB_FPS (tools/qa_frames.py), at most RGB_MAX, 448 px wide
Scene text is sampled at RATE Hz, lowered for long clips so a scene has at most MAX_LINES lines. A training question
sees only its window (meta.clip_s); an FBBench question the whole clip.

usage: COND=3d python tools/qa_train.py train|eval|zeroshot   (env HF_HOME, OUT, EPOCHS, LR, ...)
out  <OUT>/<COND>/{adapter/, eval_fbbench.jsonl, summary.json}"""
import glob, json, math, os, random, sys, time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COND = os.environ.get("COND", "3d")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
OUT = os.path.join(os.environ.get("OUT", os.path.join(ROOT, "outputs", "qa_runs")), COND + os.environ.get("SUFFIX", ""))
EPOCHS = float(os.environ.get("EPOCHS", "2"))
LR = float(os.environ.get("LR", "1e-4"))
ACCUM = int(os.environ.get("ACCUM", "16"))
RATE = float(os.environ.get("RATE", "5"))
MAX_LINES = int(os.environ.get("MAX_LINES", "300"))
RGB_FPS = float(os.environ.get("RGB_FPS", "2"))
RGB_MAX = int(os.environ.get("RGB_MAX", "64"))
MAX_TOK = int(os.environ.get("MAX_TOK", "16000"))
LIMIT = int(os.environ.get("LIMIT", "0"))  # debug: first N questions of each split
SEED = int(os.environ.get("SEED", "0"))
FBQA = os.path.join(ROOT, "outputs", "fbqa")
FBB = os.path.join(ROOT, "data", "fbbench", "finebadmintonbenchmark")
SCENE = os.path.join(ROOT, "outputs", "qa_scene")
FRAMES = os.path.join(ROOT, "outputs", "qa_frames")
LETTERS = "ABCDEFGH"

SYSTEM = ("You answer multiple-choice questions about a badminton video clip from a broadcast camera. "
          "'top' is the player at the top of the screen (far side), 'bottom' the player at the bottom (near side). "
          "Reply with the letter of the correct option only.")
LEGEND_2D = ("2D TRACKS (image coordinates 0-1000, x to the right, y downwards). One line per time step: "
             "t=seconds | top: f=feet h=head lw/rw=left/right wrist | bottom: same | s=shuttle. '-' = not detected.")
LEGEND_3D = ("3D RECONSTRUCTION (court coordinates in cm: x across the court 0-610, y along it 0-1340 from the bottom "
             "player's baseline, net at y=670, z up). One line per time step: t=seconds | top: p=pelvis x,y lw/rw=left/"
             "right wrist x,y,z r=racket head x,y,z | bottom: same | s=shuttle x,y,z. '-' = player not seen. "
             "HITS lists the detected shuttle contacts (time, hitter).")


# ---------------------------------------------------------------- data
def load_questions():
    tr = [json.loads(l) for l in open(os.path.join(FBQA, "train.jsonl"), encoding="utf8")]
    va = [json.loads(l) for l in open(os.path.join(FBQA, "val.jsonl"), encoding="utf8")]
    for q in tr + va:
        q["set"], q["window"] = "fb20k", q["meta"].get("clip_s")
    te = []
    for p in sorted(glob.glob(os.path.join(FBB, "*.json"))):
        task = os.path.basename(p)[len("benchmark_"):-5]
        for k in ("_updated", "_new", "2", "_questions"):
            task = task.replace(k, "")
        for i, it in enumerate(json.load(open(p, encoding="utf8"))):
            qtext = it["question"]["question_en"] if isinstance(it.get("question"), dict) else it.get("question")
            te.append({"uid": it["video_uid"], "task": task, "question": qtext, "options": it["options_en"],
                       "answer": it["answer"].strip()[:1], "set": "fb", "window": None, "qid": f"{task}/{i}"})
    if LIMIT:
        tr, va, te = tr[:LIMIT], va[:LIMIT // 4], te[:LIMIT // 2]
    return tr, va, te


_scene_cache = {}


def scene(st, uid):
    k = (st, uid)
    if k not in _scene_cache:
        p = os.path.join(SCENE, st, uid + ".json")
        _scene_cache[k] = json.load(open(p)) if os.path.exists(p) else None
        if len(_scene_cache) > 4000:
            _scene_cache.clear()
    return _scene_cache[k]


def sample_idx(t, window):
    t = np.asarray(t, float)
    a, b = (window if window else (t[0] if len(t) else 0, t[-1] if len(t) else 0))
    dur = max(b - a, 1e-3)
    rate = min(RATE, MAX_LINES / dur)
    ts = np.arange(a, b + 1e-6, 1 / rate)
    if not len(t):
        return [], ts
    idx = np.clip(np.searchsorted(t, ts), 0, len(t) - 1)
    return idx, ts


def fmt(v):
    return "-" if v is None else ",".join(str(x) for x in v)


def text_2d(sc, window):
    d = sc.get("d2") if sc else None
    if not d:
        return LEGEND_2D + "\n(no 2D tracks for this clip)"
    idx, ts = sample_idx(sc["t"], window)
    lines = [LEGEND_2D]
    for i, t in zip(idx, ts):
        parts = [f"t={t:.1f}"]
        for side in ("top", "bottom"):
            r = d[side][i]
            parts.append(f"{side}: -" if r is None else
                         f"{side}: f={fmt(r['feet'])} h={fmt(r['head'])} lw={fmt(r['lw'])} rw={fmt(r['rw'])}")
        parts.append(f"s={fmt(d['shuttle'][i])}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def text_3d(sc, window):
    d = sc.get("d3") if sc else None
    if not d:
        return LEGEND_3D + "\n(no reconstruction for this clip)"
    t = np.asarray(sc["t"][:len(d["shuttle"])], float)
    idx, ts = sample_idx(t, window)
    lines = [LEGEND_3D]
    for i, tt in zip(idx, ts):
        parts = [f"t={tt:.1f}"]
        for side in ("top", "bottom"):
            r = d[side][i]
            parts.append(f"{side}: -" if r is None else
                         f"{side}: p={fmt(r['pel'][:2])} lw={fmt(r['lw'])} rw={fmt(r['rw'])} r={fmt(r['racket'])}")
        parts.append(f"s={fmt(d['shuttle'][i])}")
        lines.append(" | ".join(parts))
    a, b = window if window else (t[0] if len(t) else 0, t[-1] if len(t) else 0)
    hits = [f"{t[h]:.1f}s {s}" for h, s in d["hits"] if h < len(t) and a - 1e-6 <= t[h] <= b + 1e-6]
    lines.append("HITS: " + (", ".join(hits) if hits else "none"))
    return "\n".join(lines)


def video_frames(st, uid, window):
    from PIL import Image
    import io
    p = os.path.join(FRAMES, st, uid + ".npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    t, jpg = z["t"], z["jpg"]
    a, b = window if window else (t[0], t[-1])
    keep = np.flatnonzero((t >= a - 1e-6) & (t <= b + 1e-6))
    if not len(keep):
        keep = np.array([int(np.argmin(np.abs(t - a)))])
    step = max(1, int(round((len(keep) / max(t[keep[-1]] - t[keep[0]], 1e-3)) / RGB_FPS))) if len(keep) > 1 else 1
    keep = keep[::step]
    if len(keep) > RGB_MAX:
        keep = keep[np.linspace(0, len(keep) - 1, RGB_MAX).round().astype(int)]
    if len(keep) % 2:  # Qwen2.5-VL merges frames in pairs
        keep = np.r_[keep, keep[-1]]
    return [Image.open(io.BytesIO(jpg[k])).convert("RGB") for k in keep]


def build(q, processor):
    """-> (inputs dict of tensors, letter token ids, index of the correct letter)."""
    st, uid, win = q["set"], q["uid"], q["window"]
    sc = scene(st, uid) if COND in ("2d", "3d", "2d3d") else None
    content = []
    video = None
    if COND == "rgb":
        video = video_frames(st, uid, win)
        if video is not None:
            content.append({"type": "video"})
        else:
            content.append({"type": "text", "text": "(video unavailable)"})
    if COND in ("2d", "2d3d"):
        content.append({"type": "text", "text": text_2d(sc, win)})
    if COND in ("3d", "2d3d"):
        content.append({"type": "text", "text": text_3d(sc, win)})
    opts = "\n".join(q["options"])
    content.append({"type": "text", "text": f"Question: {q['question']}\nOptions:\n{opts}\nAnswer with the letter only."})
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}, {"role": "user", "content": content}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    kw = {"text": [text], "return_tensors": "pt"}
    if video is not None:
        kw["videos"] = [video]
        kw["videos_kwargs"] = {"fps": RGB_FPS, "do_sample_frames": False}
    inp = processor(**kw)
    if inp["input_ids"].shape[1] > MAX_TOK:  # keep the end (question, options) for over-long scenes
        for k in ("input_ids", "attention_mask"):
            inp[k] = inp[k][:, -MAX_TOK:]
    n = len(q["options"])
    ids = [processor.tokenizer.convert_tokens_to_ids(c) for c in LETTERS[:n]]
    return inp, ids, LETTERS.index(q["answer"]) if q["answer"] in LETTERS[:n] else -1


# ---------------------------------------------------------------- model
def load(adapter=None, train=False):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
                                                               attn_implementation=os.environ.get("ATTN", "sdpa"))
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    elif train:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                         target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    if train:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return processor, model


def letter_logits(model, inp, ids):
    inp = {k: v.to("cuda") for k, v in inp.items()}
    out = model(**inp, use_cache=False, logits_to_keep=1)
    return out.logits[0, -1, ids].float()


def evaluate(processor, model, qs, path):
    model.eval()
    res, t0 = [], time.time()
    with torch.no_grad():
        for i, q in enumerate(qs):
            inp, ids, y = build(q, processor)
            ts = time.time()
            lg = letter_logits(model, inp, ids)
            torch.cuda.synchronize()
            pred = int(lg.argmax())
            res.append({"qid": q.get("qid", q.get("uid")), "uid": q["uid"], "task": q["task"], "answer": q["answer"],
                        "pred": LETTERS[pred], "ok": pred == y, "n_tok": int(inp["input_ids"].shape[1]),
                        "sec": round(time.time() - ts, 3)})
            if i % 200 == 0:
                acc = np.mean([r["ok"] for r in res])
                print(f"  eval {i}/{len(qs)} acc {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
    with open(path, "w", encoding="utf8") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    tasks = sorted({r["task"] for r in res})
    summ = {"cond": COND, "n": len(res), "acc": float(np.mean([r["ok"] for r in res])),
            "per_task": {t: float(np.mean([r["ok"] for r in res if r["task"] == t])) for t in tasks},
            "tokens_median": float(np.median([r["n_tok"] for r in res])), "sec_median": float(np.median([r["sec"] for r in res]))}
    model.train()
    return summ


def train():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    tr, va, te = load_questions()
    processor, model = load(train=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    steps = int(math.ceil(len(tr) * EPOCHS / ACCUM))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 20)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    os.makedirs(OUT, exist_ok=True)
    model.train()
    order, k, step, t0, run = [], 0, 0, time.time(), []
    best = -1
    while step < steps:
        if not order:
            order = list(range(len(tr)))
            random.shuffle(order)
        q = tr[order.pop()]
        inp, ids, y = build(q, processor)
        if y < 0:
            continue
        lg = letter_logits(model, inp, ids)
        loss = torch.nn.functional.cross_entropy(lg[None], torch.tensor([y], device="cuda")) / ACCUM
        loss.backward()
        run.append(float(loss) * ACCUM)
        k += 1
        if k % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                print(f"step {step}/{steps} loss {np.mean(run[-ACCUM * 10:]):.3f} ({time.time() - t0:.0f}s)", flush=True)
            if step % max(1, steps // 4) == 0 or step == steps:
                s = evaluate(processor, model, va, os.path.join(OUT, f"val_step{step}.jsonl"))
                print(f"VAL step {step}: acc {s['acc']:.3f} {json.dumps(s['per_task'])}", flush=True)
                if s["acc"] > best:
                    best = s["acc"]
                    model.save_pretrained(os.path.join(OUT, "adapter"))
    json.dump({"best_val": best, "steps": steps, "n_train": len(tr)}, open(os.path.join(OUT, "train_info.json"), "w"))
    del model
    torch.cuda.empty_cache()
    test(os.path.join(OUT, "adapter"))


def test(adapter):
    _, va, te = load_questions()
    processor, model = load(adapter=adapter)
    s = evaluate(processor, model, te, os.path.join(OUT, "eval_fbbench.jsonl"))
    s["adapter"] = adapter
    json.dump(s, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print("TEST", json.dumps(s), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":
        train()
    elif mode == "eval":
        test(os.path.join(OUT, "adapter"))
    elif mode == "zeroshot":
        os.makedirs(OUT, exist_ok=True)
        test(None)
