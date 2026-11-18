"""LoRA fine-tune of Qwen3-4B-Instruct on the rally QA set (tools/qa2_build.py).
One example = one rally scene + a random subset of its questions (re-drawn every epoch); loss on the
answer tokens only. Picks the epoch with the best validation accuracy.
usage: .venv_qa/Scripts/python tools/qa2_train.py --out outputs/qa2_lora"""
import argparse, json, math, os, random, sys, time

import torch

torch.backends.cuda.matmul.allow_tf32 = True  # fp32 matmul with TF32 off stalls for minutes on the RTX 5080 (this torch build)
torch.backends.cudnn.allow_tf32 = False
sys.path.insert(0, os.path.dirname(__file__))
from qa2_common import MODEL, TRAIN, VAL, load, make_prompt, make_target, sample_questions


def encode(tok, row, qs):
    prompt = tok.apply_chat_template(make_prompt(row, qs), tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(make_target(qs) + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return torch.tensor(p_ids + a_ids), len(a_ids)


BUCKET = 256  # every new tensor shape costs seconds of kernel selection on the RTX 5080: pad to few shapes


def pad_left(ids, pad_id, bucket=BUCKET):
    """Left-pad to a multiple of `bucket`, with an attention mask. RoPE is relative, so a constant
    position offset leaves the real tokens' attention unchanged."""
    n = -(-len(ids) // bucket) * bucket
    pad = n - len(ids)
    return (torch.cat([torch.full((pad,), pad_id, dtype=ids.dtype), ids]),
            torch.cat([torch.zeros(pad, dtype=torch.long), torch.ones(len(ids), dtype=torch.long)]))


def answer_loss(model, ids, n_ans, pad_id):
    """Cross-entropy on the answer tokens only; logits are computed only near the end (the vocab is
    150k, full-sequence logits would not fit). The logit count is bucketed too."""
    x, mask = pad_left(ids, pad_id)
    keep = -(-(n_ans + 1) // 64) * 64
    out = model(input_ids=x[None].cuda(), attention_mask=mask[None].cuda(), logits_to_keep=keep)
    logits = out.logits[0, -(n_ans + 1):-1].float()
    return torch.nn.functional.cross_entropy(logits, ids[-n_ans:].cuda())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/qa2_lora")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=12288)
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]),
                           autocast_adapter_dtype=False)  # bf16 adapters: fp32 LoRA matmuls stall on the RTX 5080
    model.print_trainable_parameters()

    train, val = load(TRAIN), load(VAL)
    print("train rallies", len(train), "val rallies", len(val), flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    total = a.epochs * math.ceil(len(train) / a.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s / total, 1))))

    from qa2_eval import eval_llm
    rng = random.Random(0)
    best, step = -1, 0
    for ep in range(a.epochs):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        t0, run = time.time(), []
        for i, k in enumerate(order):
            row = train[k]
            ids, n_ans = encode(tok, row, sample_questions(row, rng))
            if len(ids) > a.max_len:
                continue
            loss = answer_loss(model, ids, n_ans, tok.pad_token_id)
            (loss / a.accum).backward()
            run.append(loss.item())
            if (i + 1) % a.accum == 0 or i == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
            if (i + 1) % 40 == 0:
                print(f"ep {ep} {i + 1}/{len(order)} loss {sum(run[-40:]) / max(len(run[-40:]), 1):.4f} {time.time() - t0:.0f}s", flush=True)
        model.eval()
        acc, _ = eval_llm(model, tok, val, quiet=True)
        print(f"== epoch {ep} val acc {acc:.3f}", flush=True)
        if acc > best:
            best = acc
            model.save_pretrained(a.out)
            json.dump({"epoch": ep, "val_acc": acc, "args": vars(a)}, open(os.path.join(a.out, "train_info.json"), "w"))
    print("best val", best)


if __name__ == "__main__":
    main()
