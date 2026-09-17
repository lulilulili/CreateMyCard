"""P1-4 训练脚本：LoRA SFT（answer-only loss），CPU/GPU 通用。

用法（本机 CPU 冒烟，0.5B）：
  python -m training.train_lora --model Qwen/Qwen2.5-0.5B-Instruct --epochs 1
GPU 机器上换 3B 直接改 --model Qwen/Qwen2.5-3B-Instruct（自动用 bf16+cuda）。

数据：training/datasets/cc_train_v1.jsonl（messages 三元组）。
答案侧才计损失：prompt 部分 labels=-100。超过 --max-len 的样本跳过（不截答案）。
产物：--out 目录下 adapter（peft 格式）+ train_log.json（损失曲线/吞吐/跳过数）。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_samples(path: Path, limit: int, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def _ids(encoded):
    if hasattr(encoded, "ids"):
        return encoded.ids
    if hasattr(encoded, "keys"):  # BatchEncoding
        ids = encoded["input_ids"]
        return list(ids[0]) if ids and isinstance(ids[0], (list, tuple)) else list(ids)
    return list(encoded)


def encode(tokenizer, messages: list[dict], max_len: int):
    """Render chat template; labels only on the assistant answer."""
    prompt_ids = _ids(tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True))
    full_ids = _ids(tokenizer.apply_chat_template(messages, tokenize=True))
    if len(full_ids) > max_len:
        return None
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return full_ids, labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default=str(REPO_ROOT / "training/datasets/cc_train_v1.jsonl"))
    parser.add_argument("--out", default=str(REPO_ROOT / "training/generated/lora_cc"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0=全量，否则抽前 N（洗牌后）")
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    samples = load_samples(Path(args.data), args.limit, args.seed)
    encoded, skipped = [], 0
    for row in samples:
        item = encode(tokenizer, row["messages"], args.max_len)
        if item is None:
            skipped += 1
        else:
            encoded.append(item)
    print(f"样本 {len(encoded)}（跳过超长 {skipped}）")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr)
    steps_per_epoch = math.ceil(len(encoded) / args.grad_accum)
    log: dict = {"args": vars(args), "skipped": skipped, "n": len(encoded),
                 "losses": [], "startedAt": time.strftime("%F %T")}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    started, tokens_seen, running = time.time(), 0, []
    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(encoded)))
        random.Random(args.seed + epoch).shuffle(order)
        optimizer.zero_grad()
        for position, index in enumerate(order, 1):
            input_ids, labels = encoded[index]
            batch = {
                "input_ids": torch.tensor([input_ids], device=device),
                "labels": torch.tensor([labels], device=device)}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            tokens_seen += len(input_ids)
            running.append(loss.item() * args.grad_accum)
            if position % args.grad_accum == 0 or position == len(order):
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step % args.log_every == 0 or position == len(order):
                    avg = sum(running) / len(running)
                    tps = tokens_seen / (time.time() - started)
                    eta_min = (steps_per_epoch * args.epochs - step) * \
                        (time.time() - started) / max(1, step) / 60
                    print(f"epoch {epoch + 1} step {step}/{steps_per_epoch * args.epochs} "
                          f"loss {avg:.4f} | {tps:.0f} tok/s | ETA {eta_min:.0f} min",
                          flush=True)
                    log["losses"].append({"step": step, "loss": round(avg, 4)})
                    running = []
        model.save_pretrained(out_dir / f"epoch{epoch + 1}")
        print(f"已保存 {out_dir / f'epoch{epoch + 1}'}")

    log["finishedAt"] = time.strftime("%F %T")
    log["tokSeen"] = tokens_seen
    (out_dir / "train_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print("完成。日志:", out_dir / "train_log.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
