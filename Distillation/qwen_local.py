#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地调用 Qwen/Qwen3.5-0.8B
优先加载本地缓存，MPS / CPU 自动选择设备。
用法:
    python qwen_local.py                          # 默认对话示例
    python qwen_local.py "你的问题"               # 单次提问
    python qwen_local.py --interactive            # 交互模式，输入 exit 退出
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
CACHE_DIR = os.path.expanduser("~/cache/huggingface")  # 与本地缓存一致


def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    print(f"加载模型 {MODEL_NAME} ... (device={device}, dtype={dtype})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return tokenizer, model, device


def chat(tokenizer, model, device, messages, max_new_tokens=512):
    # 应用 chat template 并直接得到 input_ids
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 去掉输入部分，只保留新生成的内容
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response


def main():
    parser = argparse.ArgumentParser(description="本地调用 Qwen3.5-0.8B")
    parser.add_argument("question", nargs="?", help="单次提问的内容")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    tokenizer, model, device = load_model()

    def ask(user_text):
        messages = [
            {"role": "system", "content": "你是一个有用的AI助手。"},
            {"role": "user", "content": user_text},
        ]
        print(f"\n用户: {user_text}")
        answer = chat(tokenizer, model, device, messages, args.max_new_tokens)
        print(f"\n回答: {answer}\n")

    if args.interactive:
        print("进入交互模式，输入 exit 退出。")
        while True:
            try:
                q = input("\n请输入问题: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("exit", "quit"):
                break
            ask(q)
    elif args.question:
        ask(args.question)
    else:
        ask("解释一下什么是蒸馏学习？")


if __name__ == "__main__":
    main()
