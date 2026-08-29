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

# ==================== 1. 环境与路径自动检测（复刻 config.R 逻辑） ====================
system_name = platform.system()
is_mac = (system_name == "Darwin")
is_linux = (system_name == "Linux")
cuda_available = torch.cuda.is_available()

if is_mac:
    print("--- 检测到 Mac 环境：切换至【本地测试模式】---")
    WORK_DIR = os.path.expanduser("~/github/stat-llm-lab/")
    CACHE_DIR = os.path.expanduser("~/cache/huggingface")
    os.environ["OMP_NUM_THREADS"] = "4"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
elif is_linux and cuda_available:
    print("--- 检测到 Linux + GPU 环境：切换至【服务器 GPU 训练模式】---")
    WORK_DIR = "/root/autodl-tmp/stat-llm-lab/"
    CACHE_DIR = "/root/autodl-tmp/cache"
    os.environ["OMP_NUM_THREADS"] = "4"
    device = "cuda"
else:
    WORK_DIR = os.path.abspath(".")
    CACHE_DIR = os.path.expanduser("~/cache/huggingface")
    device = "cpu"

# 切换工作目录
os.chdir(WORK_DIR)

# 基于 WORK_DIR 动态构建相对路径
INPUT_JSONL = os.path.join(WORK_DIR, "data/raw/pretrain_clean.jsonl")
TEACHER_CHUNK_DIR = os.path.join(WORK_DIR, "data/processed/teacher_chunks/")  # 本阶段输出 / 阶段 B 输入

# 教师模型：用基座（Base）而非 Instruct —— 指令微调模型的 next-token 分布是
# “助手式续写”，与文档真实续写系统性错位（实测真实 token 仅 ~33% 出现在 top-16）。
MODEL_NAME_OR_PATH = "Qwen/Qwen3.5-0.8B"

# ==================== 2. 模型超参数 ====================
TOP_K = 16
TEMPERATURE = 1.0
BATCH_SIZE = 8 if is_mac else 16   # Mac(16GB 内存)MPS 显存紧张：大词表 logits 峰值高，批小点;Linux GPU 保持 16
CHUNK_SIZE = 2048                  # 每个 teacher chunk 的文本条数（按条数分桶）
LIMIT_NUM = 100000

LSE_CHUNK = 4096            # 教师 logits 求 logsumexp 时按词表维分块的块宽（float32 精度、显存峰值≈一块）
LSE_CHUNK_CPU = 16384       # MPS 路径把 LSE 挪到 CPU 后用的块宽（CPU 上 4096 太小、逐块开销大；16384 实测快一个量级）

MAX_TEACHER_TOKENS = 1024           # 教师侧序列安全上限
# ===============================================

_PUNCT_RE = re.compile(r"([,.:;!?\"'(){}[\]，。！？；：—（）《》“”‘’、])")


def clean_text(text: str) -> str:
    """复刻 R clean_text_internal：标点隔离 → 空白折叠 → strip。"""
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


# ==================== 断点续跑公共机制 ====================

def _atomic_write_json(path, obj):
    """先写临时文件再原子改名，保证读方永远看不到半截 JSON。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def cleanup_tmp_files(output_dir):
    """启动时清理上次中断遗留的 .tmp 写文件（.arrow / .meta.json 的临时件）。"""
    if not os.path.isdir(output_dir):
        return
    for fn in os.listdir(output_dir):
        if fn.endswith(".arrow.tmp") or fn.endswith(".meta.json.tmp"):
            p = os.path.join(output_dir, fn)
            if os.path.isfile(p):
                os.remove(p)
                print(f"[cleanup] 移除中断残留 {fn}")


def find_resume_point(output_dir):
    """依据每个 chunk 的断点元数据 chunk_XXX.meta.json 判断续跑点。

    返回 (chunk_idx, resume_pos, prev_meta, legacy)：
      chunk_idx : 下一个要写入的 chunk 序号（chunk_001..(k-1) 的 .arrow+.meta.json 齐全）
      resume_pos: 下一文本在原始 jsonl 中的行号（0 起，未处理文本的起点）
      prev_meta : 最近一个完整 chunk 的断点元数据 dict，或 None
      legacy    : True 表示存在旧版裸 .arrow（无 .meta.json），无法安全续跑
    """
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

    # 没有任何“完整”chunk：检查是否残留旧版裸 .arrow
    if os.path.isdir(output_dir):
        for fn in os.listdir(output_dir):
            if re.fullmatch(r"chunk_\d{3}\.arrow", fn):
                return 1, 0, None, True
    return 1, 0, None, False


def make_data_fp(indexed_texts):
    """轻量数据指纹：行数 + 首/中/尾几条文本前 512 字符的哈希。
    用于续跑时发现“输入 jsonl 被换掉”这类粗粒度变更（不做全量哈希）。"""
    h = hashlib.sha256()
    h.update(str(len(indexed_texts)).encode("utf-8"))
    probe = sorted({0, 1, 2, len(indexed_texts) // 2,
                    len(indexed_texts) - 3, len(indexed_texts) - 2, len(indexed_texts) - 1})
    for i in probe:
        if 0 <= i < len(indexed_texts):
            h.update(repr(indexed_texts[i][1][:512]).encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def warn_config_mismatch(prev_meta, cur_cfg, data_fp):
    """续跑时核对上次运行的配置与数据指纹，提示可能导致断点失效的变更。"""
    prev_cfg = prev_meta.get("config", {}) if prev_meta else {}
    mism = {k: (prev_cfg.get(k), cur_cfg[k]) for k in cur_cfg if prev_cfg.get(k) != cur_cfg[k]}
    if mism:
        print("⚠️  断点元数据中的配置与本次运行不一致，续跑结果可能与旧 chunk 不连续：")
        for k, (old, new) in sorted(mism.items()):
            print(f"      {k}: 旧={old}  新={new}")
        print("      - 仅 limit 变大 / 数据追加时可放心续跑；")
        print("      - 若 chunk_size / top_k / 输入文件已变更，建议 --force-restart 从头重建。")
    if prev_meta and prev_meta.get("data_fp") and prev_meta["data_fp"] != data_fp:
        print("⚠️  数据指纹与断点不一致（输入内容疑似变更），断点位置可能失效！")
        print("      请核对输入，必要时 --force-restart。")


def clean_chunk_files(output_dir):
    """--force-restart：删除输出目录里的旧 chunk（.arrow / .meta.json / 临时件）。"""
    for fn in os.listdir(output_dir):
        if re.fullmatch(r"chunk_\d{3}(\.arrow|\.meta\.json)(\.tmp)?", fn):
            p = os.path.join(output_dir, fn)
            try:
                os.remove(p)
                print(f"  删除 {fn}")
            except OSError as e:
                print(f"[warn] 删除 {fn} 失败：{e}")


# ==================== 阶段 A：教师推理（原始 Top-K soft label） ====================

def run_stage_teacher(args):
    print("=" * 70)
    print("[阶段 A] 教师推理：原始 Top-K soft label（与学生词表无关）")
    print(f"[工作路径] {WORK_DIR}")
    print(f"[运行设备] {device}")
    slab_size = args.slab_size if args.slab_size > 0 else args.chunk_size
    print(f"[参数] top_k={TOP_K} chunk_size={args.chunk_size} slab_size={slab_size} "
          f"batch_size={args.batch_size} limit={args.limit} "
          f"teacher_out_dir={args.teacher_out_dir}")

    teacher_tok = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    teacher_tok.padding_side = "right"
    if teacher_tok.pad_token is None:
        teacher_tok.pad_token = teacher_tok.eos_token

    model_device_map = "cuda" if device == "cuda" else ("mps" if device == "mps" else "cpu")
    model_dtype = torch.bfloat16 if device in ["cuda", "mps"] else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        cache_dir=CACHE_DIR,
        dtype=model_dtype,
        attn_implementation="sdpa" if device == "cuda" else "eager",
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
    # 原始 jsonl 已随机打乱，不做长度重排 —— 长度排序会让 chunk 内样本长度趋同，
    # 导致 chunk_001 全是短文本、loss_mask 有效率骤降（如仅 12.63%）。
    print(f"共读取 {len(indexed_texts)} 条（保持原始随机顺序）。")
    if args.limit and args.limit > 0:
        indexed_texts = indexed_texts[:args.limit]
        print(f"【测试模式】仅处理前 {len(indexed_texts)} 条。")

    # offset_unit 对全语料只解析一次
    probe = next((t for _, t in indexed_texts if any(ord(c) > 127 for c in t)), None)
    if probe is not None:
        offset_unit = resolve_offset_unit(teacher_tok, probe)
    else:
        offset_unit = "char"

    # =====================================================================
    # 断点续跑
    # =====================================================================
    os.makedirs(args.teacher_out_dir, exist_ok=True)
    cleanup_tmp_files(args.teacher_out_dir)
    data_fp = make_data_fp(indexed_texts)
    chunk_idx, resume_pos, prev_meta, legacy = find_resume_point(args.teacher_out_dir)
    if args.force_restart:
        clean_chunk_files(args.teacher_out_dir)
        chunk_idx, resume_pos, prev_meta, legacy = 1, 0, None, False
        print("[--force-restart] 已清空阶段 A 旧 chunk，从头重新生成。")
    if legacy:
        print("[错误] 阶段 A 输出目录存在旧版裸 .arrow（无 .meta.json），无法安全续跑。")
        print("       请 --force-restart 或手动清空 data/processed/teacher_chunks/。")
        sys.exit(1)
    if prev_meta is not None:
        warn_config_mismatch(prev_meta, {
            "top_k": TOP_K,
            "chunk_size": args.chunk_size,
            "slab_size": slab_size,
            "limit": args.limit,
            "input_jsonl": os.path.basename(args.input_jsonl),
            "model": MODEL_NAME_OR_PATH,
        }, data_fp)
        print(f"[断点续跑] 已有 chunk_001..{chunk_idx - 1:03d}，从 resume_pos={resume_pos} 续写。")
    else:
        print("[全新生成] 阶段 A 输出目录无已有 chunk，从头生成。" if chunk_idx == 1
              else f"[断点续跑] 检测到 chunk 序号缺口，从 chunk_{chunk_idx:03d} 续写。")

    # resume_pos 是“原始 jsonl 行号”→ 转成 indexed_texts 下标，并按 slab 下界对齐
    p0 = bisect.bisect_left([i for i, _ in indexed_texts], resume_pos)
    if p0 >= len(indexed_texts):
        print(f"[完成] 断点 resume_pos={resume_pos} 已越过语料末尾"
              f"（共 {len(indexed_texts)} 条），全部文本均已推理，无需续跑。")
        return
    slab_start = (p0 // slab_size) * slab_size

    # 阶段 A chunk 状态：按“文本条数”分桶（不是样本数；每行 = 一条原始文本）
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
        os.replace(tmp_file, out_file)   # 原子改名：中断也不会留下半截 chunk

        # 断点元数据：resume_pos = 下一个待处理文本的原始 jsonl 行号。
        # 先写 .arrow 再写 .meta.json；若在两者之间中断，下次启动时该 chunk 因缺
        # meta 被判定为“不完整”，续跑会覆盖重写，不会产生空洞。
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
        elif device == "mps":
            torch.mps.empty_cache()

    total = len(indexed_texts)
    n_slabs = (total + slab_size - 1) // slab_size
    for s0 in tqdm(range(slab_start, total, slab_size), desc="Teacher inference"):
        slab = indexed_texts[s0: s0 + slab_size]
        if s0 == slab_start and resume_pos > 0:
            # 断点续跑：跳过本段 slab 中 resume_pos 之前的文本（已落入已存 chunk）
            slab = [(i, t) for i, t in slab if i >= resume_pos]
        if not slab:
            continue
        t0 = time.time()

        # 1) 一次性教师 tokenize（无模型计算）：clean 文本 + 教师 token 字符偏移
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

        # 2) 按教师 token 数分桶排序 → 同桶 batch 的 Tmax≈桶内长度，padding 最小化。
        #    结果按原始 idx 存入 store，与 forward 顺序解耦。
        store = {}
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
                logits = model(input_ids=input_ids_t, attention_mask=attn_t).logits
                if device == "mps":
                    # MPS 的 torch.topk 对超大词表维（V=248320）会按输入形状缓存一个 ~26B/元素的
                    # workspace，且 empty_cache 释放不掉：每个不同 Tmax 的分桶都新开一份并一直保留，
                    # 跨桶累积直接把共享池顶爆（实测一次 B=8,T=1024 的 topk 就要 ~50GiB）。
                    # → topk + softmax 整体挪到 CPU 做（forward 仍在 MPS），LSE 用 fp32 大块保证精度。
                    x = logits[:, :-1, :].to("cpu")          # bf16 [B,T-1,V]，拷回 CPU 后释放 MPS 侧
                    x /= TEMPERATURE
                    tk_logits, tk_ids = torch.topk(x, k=TOP_K, dim=-1)   # CPU topk（bf16 直接算，实测 0.5s）
                    m = x.max(dim=-1, keepdim=True).values.float()       # [B,T-1,1] fp32
                    se_sum = None
                    for c0 in range(0, x.size(-1), LSE_CHUNK_CPU):
                        chunk = x[..., c0:c0 + LSE_CHUNK_CPU].float()    # bf16→fp32 大块（CPU 上小块开销大）
                        chunk.sub_(m)
                        chunk.exp_()
                        se = chunk.sum(dim=-1)                           # [B,T-1] fp32
                        se_sum = se if se_sum is None else se_sum + se
                    lse = m + se_sum.unsqueeze(-1).log()
                    tk_probs = torch.exp(tk_logits.float() - lse).clamp(0.0, 1.0)
                    tk_ids_np = tk_ids.numpy()
                    tk_probs_np = tk_probs.numpy()
                    del x
                else:
                    # CUDA：bf16 topk + fp32 分块 LSE（GPU 显存充足、topk 无此问题，保持原逻辑）
                    logits_shifted = logits[:, :-1, :] / TEMPERATURE
                    tk_logits, tk_ids = torch.topk(logits_shifted, k=TOP_K, dim=-1)
                    # lse 仍用 float32 精度算，但按词表维分块，显存峰值 ≈ 一个分块（不再整块 [B,T,V] 转 float32）。
                    # 为什么不能整块 bf16 算：CUDA 上半精度累加会把 lse 算小（实测单点概率可到 1.12），
                    # 而 float32 分块累加恒有 lse >= max(logit)（max 项贡献 exp(0)=1），保证单点概率 <= 1。
                    m = logits_shifted.max(dim=-1, keepdim=True).values.float()   # [B,T-1,1] float32
                    V = logits_shifted.size(-1)
                    se_sum = None
                    for c0 in range(0, V, LSE_CHUNK):
                        chunk = logits_shifted[..., c0:c0 + LSE_CHUNK].float()   # bf16→float32 小块
                        chunk.sub_(m)
                        chunk.exp_()
                        se = chunk.sum(dim=-1)                                    # [B,T-1] float32
                        se_sum = se if se_sum is None else se_sum + se
                    lse = m + se_sum.unsqueeze(-1).log()
                    tk_probs = torch.exp(tk_logits.float() - lse)
                    # 防御性兜底：极端舍入下个别点仍可能略超 1，钳到 [0,1]，避免污染下游训练数据。
                    tk_probs = tk_probs.clamp(0.0, 1.0)
                    tk_ids_np = tk_ids.cpu().numpy()
                    tk_probs_np = tk_probs.float().cpu().numpy()
                del logits
                if device == "mps":
                    # MPS 缓存分配器也会按尺寸无限保留释放的大块（不同 Tmax→不同块大小），
                    # 测到过 16.95GiB 才触发回收；每个 batch 主动归还驱动，给 forward 峰值留足余量。
                    torch.mps.empty_cache()

            for b, p in enumerate(group):
                # 关键：forward 在 [B, Tmax] 的 padding batch 上做，tk_ids/probs 的形状是
                # [Tmax-1, K]，其中 Tmax-1 行包含 padding 位置的"预测"（对 PAD token 的 topk，
                # 纯垃圾）。必须按本条文本真实 token 数 len(p[2])=n_teacher 裁掉 padding 行，
                # 只留预测 token 1..n-1 的 n_teacher-1 行。否则短文本混在长文本的 batch 里，
                # 存盘的 teacher_ids/teacher_probs 会多出 padding 行，阶段 B 的
                # (n_teacher-1)×K 校验直接报错，且越界行本身就是噪声。
                nt = len(p[2])
                store[p[0]] = (tk_ids_np[b][:nt - 1], tk_probs_np[b][:nt - 1])

        # 3) 按原始随机顺序落盘阶段 A chunk（teacher 结果已按文本存好，与 forward 顺序无关）
        for idx, ct, t_ids, t_starts, t_ends in prepped:
            # 本文本已完整推理：无论是否产出样本，都把它记入断点，
            # 避免续跑时重算其 teacher forward。
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
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

        slab_i = s0 // slab_size + 1
        print(
            f"\n[slab {slab_i}/{n_slabs}] 完成,耗时 {time.time() - t0:.1f}s,"
            f"累计文本 {len(chunk_text_idx)},当前 chunk {chunk_idx}",
            flush=True,
        )

    save_a_chunk()
    print("\n阶段 A 全部 teacher chunk 已存盘（原始教师 Top-K soft label，学生无关，可被阶段 B 复用）。")


def main():
    global TOP_K
    ap = argparse.ArgumentParser(description="阶段 A：教师推理出原始 Top-K soft label（01b_student_project.py 的输入）")
    ap.add_argument("--input-jsonl", default=INPUT_JSONL, help="输入 jsonl")
    ap.add_argument("--teacher-out-dir", default=TEACHER_CHUNK_DIR,
                    help="本阶段输出目录（阶段 B 的输入）")
    ap.add_argument("--limit", type=int, default=LIMIT_NUM, help="0=不限（默认只处理前 100000 条）")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="每个 teacher chunk 的文本条数")
    ap.add_argument("--slab-size", type=int, default=0,
                    help="teacher 结果驻留内存的文本批（默认 = chunk_size；调大分桶更细但内存更高）")
    ap.add_argument("--top-k", type=int, default=TOP_K, help="教师 Top-K（默认 16；须与阶段 B 一致）")
    ap.add_argument("--force-restart", action="store_true",
                    help="忽略已有 teacher chunk，清空后从头重新生成")
    args = ap.parse_args()
    TOP_K = args.top_k

    os.makedirs(CACHE_DIR, exist_ok=True)
    run_stage_teacher(args)


if __name__ == "__main__":
    main()
