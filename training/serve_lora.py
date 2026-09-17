"""极简 OpenAI 兼容推理服务，给 eval_harness 评测 transformers/LoRA 模型用。

  python -m training.serve_lora --model Qwen/Qwen2.5-0.5B-Instruct [--adapter <dir>] --port 8899
之后：python -m training.eval_harness --base-url http://127.0.0.1:8899/v1 --model local

只实现 /v1/chat/completions 的最小子集：messages、max_tokens；temperature 恒 0（greedy）；
response_format 忽略（评测提示词本身要求 JSON）。单线程顺序处理，够 42 条用。
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = TOKENIZER = None


def generate(messages: list[dict], max_tokens: int) -> str:
    encoded = TOKENIZER.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    input_ids = input_ids.to(MODEL.device)
    with torch.no_grad():
        output = MODEL.generate(
            input_ids=input_ids, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=TOKENIZER.pad_token_id or TOKENIZER.eos_token_id)
    return TOKENIZER.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            text = generate(body["messages"], int(body.get("max_tokens", 400)))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.send_error(500, str(exc)[:150].encode("ascii", "replace").decode())
            return
        payload = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]},
                             ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> int:
    global MODEL, TOKENIZER
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    TOKENIZER = AutoTokenizer.from_pretrained(args.model)
    MODEL = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    if args.adapter:
        from peft import PeftModel
        MODEL = PeftModel.from_pretrained(MODEL, args.adapter)
        print("adapter:", args.adapter)
    MODEL.eval()
    print(f"serving http://127.0.0.1:{args.port}/v1/chat/completions "
          f"({args.model} on {device})", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
