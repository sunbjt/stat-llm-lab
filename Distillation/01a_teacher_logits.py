#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================
阶段 A：教师推理 → 原始 Top-K soft label（与学生词表无关）
========================================================================

对每条文本：clean_text → 教师 tokenize（截断 MAX_TEACHER_TOKENS=1024）
→ forward → Top-K logits → 精确 softmax 概率（分块 logsumexp）
→ 记录每个教师 token 的字符偏移。

落盘"原始教师 soft label"到 --teacher-out-dir（默认 data/processed/teacher_chunks/）：
    text_idx / ct / n_teacher / t_starts / t_ends / teacher_ids / teacher_probs

断点续推：每 chunk 落盘时同步写 chunk_XXX.meta.json（resume_pos/配置/数据指纹），
.arrow / .meta.json 先写临时文件再原子改名；中断后重跑同一命令自动续跑，
已推理文本不重放 forward。--force-restart 清空后从头生成。

运行：
    pip install Cython && pip install youtokentome   # 阶段 B 才需要，本阶段不需要
    python3 Distillation/01a_teacher_logits.py --limit 50    # 小样本试跑
    python3 Distillation/01a_teacher_logits.py --limit 0     # 全量（--limit 0 = 不限）
    # 注意：不带 --limit 时默认只处理前 100000 条（LIMIT_NUM），不是全量。
    # 教师用基座 Qwen3.5-0.8B（Instruct 的 next-token 分布与文档续写系统性错位）；
    # top_k 必须与阶段 B（01b_student_project.py）一致（默认 16）。
    # 全量约 8 小时；改 top_k / 输入文件后务必 --force-restart 全量重建。

下游：01b_student_project.py 读取本目录，投影到学生词表后产出训练 arrow。
========================================================================
"""

import argparse
import bisect
import gc
import hashlib
import json
import os
import platform
import re
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["MKL_NUM_THREADS"] = "4"

# ==================== 1. 环境与路径自动检测 ====================
system_name = platform.system()
is_mac = (system_name == "Darwin")
is_linux = (system_name == "Linux")
cuda_available = torch.cuda.is_available()

if is_mac:
    print("--- 检测到 Mac 环境：切换至【本地测试模式】---")
    WORK_DIR = os.path.expanduser("~/github/stat-llm-lab/")
    CACHE_DIR = os.path.expanduser("~/cache/huggingface")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
elif is_linux and cuda_available:
    print("--- 检测到 Linux + GPU 环境：切换至【服务器 GPU 满载训练模式】---")
    WORK_DIR = "/root/autodl-tmp/stat-llm-lab/"
    CACHE_DIR = "/root/autodl-tmp/cache"
    device = "cuda"
else:
    WORK_DIR = os.path.abspath(".")
    CACHE_DIR = os.path.expanduser("~/cache/huggingface")
    device = "cpu"

os.chdir(WORK_DIR)

INPUT_JSONL = os.path.join(WORK_DIR, "data/raw/pretrain_clean.jsonl")
TEACHER_CHUNK_DIR = os.path.join(WORK_DIR, "data/processed/teacher_chunks/")
MODEL_NAME_OR_PATH = "Qwen/Qwen3.5-0.8B"

# ==================== 2. 模型超参数优化（针对 24G 显存压榨性能） ====================
TOP_K = 16
TEMPERATURE = 1.0
# 提升 Batch Size 充分利用 CUDA 多核吞吐
BATCH_SIZE = 16 if device == "cuda" else 2
CHUNK_SIZE = 2048
LIMIT_NUM = 100000

LSE_CHUNK = 8192
MAX_TEACHER_TOKENS = 1024

_PUNCT_RE = re.compile(r"([,.:;!?\"'(){}[\]，。！？；：—（）《》“”‘’、])")


def clean_text(text: str) -> str:
    out = _PUNCT_RE.sub(r" \1 ", text)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def resolve_offset_unit(teacher_tok, text: str):
    enc = teacher_tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if not ids:
        return "char"
    b2c = []
    for ci, ch in enumerate(text):
        b2c += [ci] * len(ch.encode("utf-8"))

    def ok_rate(use_byte):
        good = 0
        for j in range(len(ids)):
            s, e = offs[j]
            if use_byte:
                cs = b2c[s] if s < len(b2c) else len(text)
                ce = (b2c[e - 1] + 1) if (0 < e <= len(b2c)) else len(text)
            else:
                cs, ce = s, e
            cs = max(0, min(cs, len(text)))
            ce = max(cs, min(ce, len(text)))
            dec = teacher_tok.decode([ids[j]]).replace("Ġ", " ").strip()
            slc = text[cs:ce].strip()
            if dec and (dec == slc or dec in slc or slc in dec):
                good += 1
        return good / max(1, len(ids))

    char_rate, byte_rate = ok_rate(False), ok_rate(True)
    unit = "byte" if byte_rate > char_rate else "char"
    print(f"教师 offset 单位检测: {unit}  (char_rate={char_rate:.2f}, byte_rate={byte_rate:.2f})")
    return unit


def offsets_to_char(offs, unit, text):
    if unit == "char":
        return [(s, e) for s, e in offs]
    b2c = []
    for ci, ch in enumerate(text):
        b2c += [ci] * len(ch.encode("utf-8"))
    out = []
    for s, e in offs:
        cs = b2c[s] if s < len(b2c) else len(text)
        ce = (b2c[e - 1] + 1) if (0 < e <= len(b2c)) else len(text)
        out.append((cs, ce))
    return out


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def cleanup_tmp_files(output_dir):
    if not os.path.isdir(output_dir):
        return
    for fn in os.listdir(output_dir):
        if fn.endswith(".arrow.tmp") or fn.endswith(".meta.json.tmp"):
            p = os.path.join(output_dir, fn)
            if os.path.isfile(p):
                os.remove(p)
                print(f"[cleanup] 移除中断残留 {fn}")


def find_resume_point(output_dir):
    k = 1
    prev_meta = None
    while True:
        arrow = os.path.join(output_dir, f"chunk_{k:03d}.arrow")
        meta = os.path.join(output_dir, f"chunk_{k:03d}.meta.json")
        if not (os.path.isfile(arrow) and os.path.isfile(meta)):
            break
        with open(meta, "r", encoding="utf-8") as f:
            prev_meta = json.load(f)
        k += 1

    if prev_meta is not None:
        return k, int(prev_meta["resume_pos"]), prev_meta, False

    if os.path.isdir(output_dir):
        for fn in os.listdir(output_dir):
            if re.fullmatch(r"chunk_\d{3}\.arrow", fn):
                return 1, 0, None, True
    return 1, 0, None, False


def make_data_fp(indexed_texts):
    h = hashlib.sha256()
    h.update(str(len(indexed_texts)).encode("utf-8"))
    probe = sorted({0, 1, 2, len(indexed_texts) // 2,
                    len(indexed_texts) - 3, len(indexed_texts) - 2, len(indexed_texts) - 1})
    for i in probe:
        if 0 <= i < len(indexed_texts):
            h.update(repr(indexed_texts[i][1][:512]).encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def clean_chunk_files(output_dir):
    for fn in os.listdir(output_dir):
        if re.fullmatch(r"chunk_\d{3}(\.arrow|\.meta\.json)(\.tmp)?", fn):
            p = os.path.join(output_dir, fn)
            try:
                os.remove(p)
                print(f"  删除 {fn}")
            except OSError as e:
                print(f"[warn] 删除 {fn} 失败：{e}")


def run_stage_teacher(args):
    print("=" * 70)
    print("[阶段 A] 教师推理：原始 Top-K soft label（高性能满载版）")
    print(f"[工作路径] {WORK_DIR}")
    print(f"[运行设备] {device}")
    slab_size = args.slab_size if args.slab_size > 0 else args.chunk_size
    print(f"[参数] top_k={TOP_K} chunk_size={args.chunk_size} slab_size={slab_size} "
          f"batch_size={args.batch_size} limit={args.limit}")

    teacher_tok = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    teacher_tok.padding_side = "right"
    if teacher_tok.pad_token is None:
        teacher_tok.pad_token = teacher_tok.eos_token

    model_device_map = "cuda" if device == "cuda" else ("mps" if device == "mps" else "cpu")
    model_dtype = torch.bfloat16 if device in ["cuda", "mps"] else torch.float32

    # 加载模型，不强制指定敏感的 attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        cache_dir=CACHE_DIR,
        dtype=model_dtype,
        device_map=model_device_map,
        trust_remote_code=True,
    )
    model.eval()

    print(f"读取数据源: {args.input_jsonl}")
    indexed_texts = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                item = json.loads(line)
                indexed_texts.append((idx, item["text"]))
    if args.limit and args.limit > 0:
        indexed_texts = indexed_texts[:args.limit]
        print(f"【测试模式】仅处理前 {len(indexed_texts)} 条。")

    probe = next((t for _, t in indexed_texts if any(ord(c) > 127 for c in t)), None)
    offset_unit = resolve_offset_unit(teacher_tok, probe) if probe is not None else "char"

    os.makedirs(args.teacher_out_dir, exist_ok=True)
    cleanup_tmp_files(args.teacher_out_dir)
    data_fp = make_data_fp(indexed_texts)
    chunk_idx, resume_pos, prev_meta, legacy = find_resume_point(args.teacher_out_dir)
    if args.force_restart:
        clean_chunk_files(args.teacher_out_dir)
        chunk_idx, resume_pos, prev_meta, legacy = 1, 0, None, False
        print("[--force-restart] 已清空阶段 A 旧 chunk，从头重新生成。")

    p0 = bisect.bisect_left([i for i, _ in indexed_texts], resume_pos)
    if p0 >= len(indexed_texts):
        print("[完成] 所有文本已完成推理。")
        return
    slab_start = (p0 // slab_size) * slab_size

    chunk_text_idx, chunk_ct, chunk_n = [], [], []
    chunk_tstarts, chunk_tends = [], []
    chunk_ids, chunk_probs = [], []

    def save_a_chunk():
        nonlocal chunk_idx
        if len(chunk_text_idx) == 0:
            return
        table = pa.Table.from_arrays(
            [
                pa.array(chunk_text_idx, type=pa.int64()),
                pa.array(chunk_ct, type=pa.string()),
                pa.array(chunk_n, type=pa.int32()),
                pa.array(chunk_tstarts, type=pa.list_(pa.int32())),
                pa.array(chunk_tends, type=pa.list_(pa.int32())),
                pa.array(chunk_ids, type=pa.list_(pa.int32())),
                pa.array(chunk_probs, type=pa.list_(pa.float32())),
            ],
            names=["text_idx", "ct", "n_teacher", "t_starts", "t_ends",
                   "teacher_ids", "teacher_probs"],
        )
        out_file = os.path.join(args.teacher_out_dir, f"chunk_{chunk_idx:03d}.arrow")
        tmp_file = out_file + ".tmp"
        feather.write_feather(table, tmp_file, compression="zstd")
        os.replace(tmp_file, out_file)

        meta_file = out_file.replace(".arrow", ".meta.json")
        _atomic_write_json(meta_file, {
            "chunk_idx": chunk_idx,
            "resume_pos": resume_pos,
            "data_fp": data_fp,
            "config": {
                "top_k": TOP_K,
                "chunk_size": args.chunk_size,
                "slab_size": slab_size,
                "limit": args.limit,
                "input_jsonl": os.path.basename(args.input_jsonl),
                "model": MODEL_NAME_OR_PATH,
            },
        })
        print(f"\n已存盘 阶段A Chunk {chunk_idx}: {out_file}（{len(chunk_text_idx)} 条文本）")
        chunk_text_idx.clear(); chunk_ct.clear(); chunk_n.clear()
        chunk_tstarts.clear(); chunk_tends.clear()
        chunk_ids.clear(); chunk_probs.clear()
        chunk_idx += 1
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    total = len(indexed_texts)
    n_slabs = (total + slab_size - 1) // slab_size
    for s0 in tqdm(range(slab_start, total, slab_size), desc="Teacher inference"):
        slab = indexed_texts[s0: s0 + slab_size]
        if s0 == slab_start and resume_pos > 0:
            slab = [(i, t) for i, t in slab if i >= resume_pos]
        if not slab:
            continue
        t0 = time.time()

        prepped = []
        for idx, txt in slab:
            ct = clean_text(txt)
            if not ct:
                continue
            enc = teacher_tok(
                ct,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_TEACHER_TOKENS,
            )
            t_ids = enc["input_ids"]
            if len(t_ids) < 2:
                continue
            t_offs = enc["offset_mapping"]
            t_starts = [o[0] for o in offsets_to_char(t_offs, offset_unit, ct)]
            t_ends   = [o[1] for o in offsets_to_char(t_offs, offset_unit, ct)]
            prepped.append((idx, ct, t_ids, t_starts, t_ends))
        if not prepped:
            continue

        store = {}
        # 按序列长度分桶打包，最小化 Padding 开销
        bucketed = sorted(prepped, key=lambda p: len(p[2]))
        for bi in range(0, len(bucketed), args.batch_size):
            group = bucketed[bi: bi + args.batch_size]
            Tmax = max(len(p[2]) for p in group)
            Tmax = max(Tmax, 2)
            
            input_ids_t = torch.zeros(len(group), Tmax, dtype=torch.long, device=device)
            attn_t = torch.zeros(len(group), Tmax, dtype=torch.long, device=device)
            for b, p in enumerate(group):
                tids = p[2]
                input_ids_t[b, :len(tids)] = torch.tensor(tids, dtype=torch.long)
                attn_t[b, :len(tids)] = 1

            with torch.inference_mode():
                # 1. Forward
                logits = model(input_ids=input_ids_t, attention_mask=attn_t).logits
                
                # 2. 截取 Shifted Logits 并缩放
                logits_shifted = logits[:, :-1, :]
                if TEMPERATURE != 1.0:
                    logits_shifted.div_(TEMPERATURE)

                # 3. 提取 Top-K
                tk_logits, tk_ids = torch.topk(logits_shifted, k=TOP_K, dim=-1)
                
                # 4. 块状 Logsumexp (LSE) - 无缝流式运算
                m = logits_shifted.max(dim=-1, keepdim=True).values.float()
                V = logits_shifted.size(-1)
                se_sum = None
                for c0 in range(0, V, LSE_CHUNK):
                    chunk = logits_shifted[..., c0:c0 + LSE_CHUNK].float()
                    chunk.sub_(m)
                    chunk.exp_()
                    se = chunk.sum(dim=-1)
                    se_sum = se if se_sum is None else se_sum + se
                lse = m + se_sum.unsqueeze(-1).log()
                
                # 5. Softmax 概率
                tk_probs = torch.exp(tk_logits.float() - lse).clamp(0.0, 1.0)
                
                tk_ids_np = tk_ids.cpu().numpy()
                tk_probs_np = tk_probs.float().cpu().numpy()

                # 仅在 Python 层面释放句柄，绝不触发流阻塞式的 cuda.empty_cache()
                del logits, logits_shifted, tk_logits, tk_ids, m, se_sum, lse, tk_probs

            for b, p in enumerate(group):
                nt = len(p[2])
                store[p[0]] = (tk_ids_np[b][:nt - 1], tk_probs_np[b][:nt - 1])

        for idx, ct, t_ids, t_starts, t_ends in prepped:
            resume_pos = idx + 1
            tk_ids_np, tk_probs_np = store[idx]
            chunk_text_idx.append(idx)
            chunk_ct.append(ct)
            chunk_n.append(len(t_ids))
            chunk_tstarts.append(t_starts)
            chunk_tends.append(t_ends)
            chunk_ids.append(tk_ids_np.reshape(-1).tolist())
            chunk_probs.append(tk_probs_np.reshape(-1).astype(np.float32).tolist())
            if len(chunk_text_idx) >= args.chunk_size:
                save_a_chunk()

        del store, prepped
        gc.collect()

        slab_i = s0 // slab_size + 1
        print(
            f"\n[slab {slab_i}/{n_slabs}] 完成, 耗时 {time.time() - t0:.1f}s, "
            f"当前 chunk {chunk_idx}",
            flush=True,
        )

    save_a_chunk()
    print("\n阶段 A 全部 teacher chunk 已存盘。")


def main():
    global TOP_K
    ap = argparse.ArgumentParser(description="阶段 A：教师推理出原始 Top-K soft label")
    ap.add_argument("--input-jsonl", default=INPUT_JSONL)
    ap.add_argument("--teacher-out-dir", default=TEACHER_CHUNK_DIR)
    ap.add_argument("--limit", type=int, default=LIMIT_NUM)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--slab-size", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--force-restart", action="store_true")
    args = ap.parse_args()
    TOP_K = args.top_k

    os.makedirs(CACHE_DIR, exist_ok=True)
    run_stage_teacher(args)


if __name__ == "__main__":
    main()