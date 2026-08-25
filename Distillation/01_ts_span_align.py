#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ts_span_align.py —— 异构 Tokenizer 离线蒸馏数据生成（含学生词表投影）

教师: Qwen2.5-3B-Instruct (Python / GPU 服务器)
学生: 15M RtomicCausalLM (R torch, 16384 字符级 BPE)
产出: 已投影到【学生词表】的 .arrow chunk，可直接被 02_distill_casual_lm.R 训练
      （schema: x/y_hard int32 flat、topk_ids fixed_size_list[K] int32、topk_probs fixed_size_list[K]
       float32、loss_mask bool；zstd 压缩，R 的 arrow 可读）。K 默认 16，可用 --top-k 覆盖。
       【v2 格式】probs 由 float16 升为 float32（仍 fixed_size_list）：R arrow 会把 halffloat 位模式
       按 int16 解读（1.0 → 15360，曾导致 loss~767027），float32 无此问题、读入零 cast。

========================================================================
一、投影算法（与用户确认：取学生子词 + 概率加和 + 归一化）
========================================================================
教师词表(~15.2万字节级 BPE)与学生词表(16384 字符级 BPE)无任何 ID 对应关系，
因此投影必须经过文本桥：
    教师 token id --解码(Qwen)--> 文本 --编码(学生 YTTM BPE)--> 学生子词 id

每学生位置 i 的边界 b = 学生 token i+1 的起始字符。取【包含 b 的教师 token k】：
  * 教师预测 token k 的分布 = logits[k-1]（即上一教师边界处的 logits）
  * 目标字符偏移 off = b - start(k)   （b 恰为 token 起点时 off=0，等价"取第一个子词"）
对 logits[k-1] 的 top-K 每个候选教师 token：
  * 解码 → 文本 → 学生 BPE 编码 → 取【覆盖字符 off 的那个学生子词】id
  * 多个候选投到同一学生 id → 概率【加和】
取 top-K（默认 16，--top-k 可调）并【归一化】，存为该位置的软标签。

注意：b 在第一个教师 token 内部时（k=0，无 logits[-1]）该位置无教师信号 → loss_mask=False。

========================================================================
二、服务器安装与运行
========================================================================
    pip install Cython
    pip install youtokentome           # 与 R 包 tokenizers.bpe 同一 C++ 库
    python ts_span_align.py --self-test   # 先自检学生 BPE 加载/偏移重建
    python ts_span_align.py --limit 50    # 小样本试跑
    python ts_span_align.py               # 全量

若 youtokentome 源码编译失败（备选路线，未实现）：
  本脚本只出教师 top-k ids + 字符偏移 + id→text 词典；
  R 侧用 tokenizers.bpe（同一 C++ 库，已本地验证）完成对齐与投影。
========================================================================
"""

import argparse
import bisect
import gc
import json
import os
import re
import sys

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import youtokentome as yttm
except ImportError:  # 延迟到 main() 再报错，保证 --help 可用
    yttm = None

# ==================== 配置项 ====================
INPUT_JSONL = "/root/autodl-tmp/stat-llm-lab/data/raw/pretrain_clean.jsonl"
OUTPUT_CHUNK_DIR = "/root/autodl-tmp/stat-llm-lab/data/processed/chunks/"
MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-3B-Instruct"
CACHE_DIR = "/root/autodl-tmp/cache"
# 学生 BPE 模型（YTTM 格式，与 R 侧 models/rtomic_bpe.model 相同）
STUDENT_MODEL_PATH = "/root/autodl-tmp/stat-llm-lab/models/rtomic_bpe.model"

MAX_LENGTH = 512
TARGET_LEN = MAX_LENGTH - 1  # Causal LM 输入输出偏移长度 (511)
TOP_K = 16
TEMPERATURE = 1.0
BATCH_SIZE = 16
CHUNK_SIZE = 4096
LIMIT_NUM = 100000
VOCAB_SIZE = 151936  # 教师词表（仅用于日志）

# ---- 学生词表约定（与 R 侧一致）----
STUDENT_VOCAB_SIZE = 16384          # 2^14
PAD_SURFACE = 1                     # R 表层 PAD（YTTM id 0 + 1）
UNK_SURFACE = 2                     # R 表层 UNK（YTTM id 1 + 1）
MAX_STUDENT_LEN = MAX_LENGTH        # 512：学生 token 序列上限（x/y 各 511）
MAX_TEACHER_TOKENS = 1024           # 教师侧序列安全上限（学生 ≤512 时通常用不到）
# ===============================================

# R 侧 clean_text_internal 的标点隔离模式（perl=TRUE 语义一致）
_PUNCT_RE = re.compile(r"([,.:;!?\"'(){}[\]，。！？；：—（）《》“”‘’、])")


def clean_text(text: str) -> str:
    """复刻 R clean_text_internal：标点隔离 → 空白折叠 → strip。"""
    out = _PUNCT_RE.sub(r" \1 ", text)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


# ---------------------------------------------------------------------------
# 学生侧：YTTM 编码 + 字符偏移（偏移规则已用 R 侧同一 C++ 库验证）
# ---------------------------------------------------------------------------
def student_encode(student_model, ct: str):
    """返回 (subs, surface_ids)。surface = min(yttm_id + 1, 16383)。"""
    subs = student_model.encode(ct, output_type=yttm.OutputType.SUBWORD)
    ids0 = student_model.encode(ct, output_type=yttm.OutputType.ID)
    assert len(subs) == len(ids0), (len(subs), len(ids0))
    surface = [min(int(i) + 1, STUDENT_VOCAB_SIZE - 1) for i in ids0]
    return subs, surface


def student_offsets(subs, ct: str):
    """
    子词 → 0-based 字符偏移。
    规则（已 R 验证）：▁→空格；首子词若以空格开头而 ct 不以空格开头，
    去掉该前导空格（YTTM 在文本开头发的是边界标记而非真实空格）。ct 已 strip。
    """
    parts = [s.replace("▁", " ") for s in subs]
    if parts and parts[0].startswith(" ") and not ct.startswith(" "):
        parts[0] = parts[0][1:]
    rebuilt = "".join(parts)
    assert rebuilt == ct, (rebuilt, ct)
    starts, ends = [], []
    pos = 0
    for p in parts:
        starts.append(pos)
        pos += len(p)
        ends.append(pos)
    return starts, ends


def process_text_student(text: str, student_model):
    """清洗 + 学生编码；若 > 512 个学生 token，在边界处截断文本后重编。"""
    ct = clean_text(text)
    if not ct:
        return None
    subs, surface = student_encode(student_model, ct)
    starts, ends = student_offsets(subs, ct)
    L = len(surface)
    if L > MAX_STUDENT_LEN:
        cut = starts[MAX_STUDENT_LEN]  # 第 513 个子词的起始字符
        ct = ct[:cut]
        subs, surface = student_encode(student_model, ct)
        starts, ends = student_offsets(subs, ct)
        L = len(surface)
        assert L <= MAX_STUDENT_LEN
    if L < 2:
        return None  # 无法形成 x/y 对
    return ct, subs, surface, starts, ends


# ---------------------------------------------------------------------------
# 教师侧：offset 单位检测（byte-level BPE 的 offset_mapping 可能是字节偏移）
# ---------------------------------------------------------------------------
def resolve_offset_unit(teacher_tok, text: str):
    """检测教师 offset_mapping 是字符偏移还是字节偏移（对多字节文本才关键）。"""
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
    """把教师 offset_mapping 统一成字符偏移（0-based, 闭开区间）。"""
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


# ---------------------------------------------------------------------------
# 投影：教师 token → 学生子词序列（按 teacher_id 缓存，一次编码处处复用）
# ---------------------------------------------------------------------------
_TEACHER_STUDENT_CACHE = {}


def teacher_token_student_encoding(tid, teacher_tok, student_model):
    """教师 token id → (学生表层 id 序列, 子词 starts, 子词 ends)。"""
    if tid in _TEACHER_STUDENT_CACHE:
        return _TEACHER_STUDENT_CACHE[tid]

    txt = teacher_tok.decode([tid])
    ctxt = clean_text(txt) if txt else ""
    if not ctxt:
        _TEACHER_STUDENT_CACHE[tid] = ([UNK_SURFACE], [0], [1])
        return _TEACHER_STUDENT_CACHE[tid]

    subs = student_model.encode(ctxt, output_type=yttm.OutputType.SUBWORD)
    ids0 = student_model.encode(ctxt, output_type=yttm.OutputType.ID)
    parts = [s.replace("▁", " ") for s in subs]
    if parts and parts[0].startswith(" "):
        parts[0] = parts[0][1:]  # 候选文本已 clean → 无前导空格
    starts, ends = [], []
    pos = 0
    for p in parts:
        starts.append(pos)
        pos += len(p)
        ends.append(pos)
    if pos != len(ctxt) or len(ids0) != len(parts):
        # 防御：重建失败（如含 UNK 子词）→ 退化为首个子词
        if not ids0:
            _TEACHER_STUDENT_CACHE[tid] = ([UNK_SURFACE], [0], [1])
        else:
            _TEACHER_STUDENT_CACHE[tid] = ([min(int(ids0[0]) + 1, STUDENT_VOCAB_SIZE - 1)], [0], [1])
        return _TEACHER_STUDENT_CACHE[tid]
    surface = [min(int(i) + 1, STUDENT_VOCAB_SIZE - 1) for i in ids0]
    _TEACHER_STUDENT_CACHE[tid] = (surface, starts, ends)
    return _TEACHER_STUDENT_CACHE[tid]


def subword_at(surface, starts, ends, off):
    """返回覆盖字符 off 的学生子词 id；off 越界时钳到最后子词。"""
    if not surface:
        return UNK_SURFACE
    idx = bisect.bisect_right(starts, off) - 1
    idx = max(0, min(idx, len(surface) - 1))
    return surface[idx]


def find_teacher_signal(b, t_starts, t_ends):
    """学生边界 b → (教师 logits 位置 j, 候选 token 内偏移 off) 或 None。"""
    k = bisect.bisect_right(t_starts, b) - 1
    if k < 0 or k >= len(t_ends) or b >= t_ends[k]:
        return None
    j = k - 1
    if j < 0:  # b 在第一个教师 token 内部，无 logits[-1] 可用
        return None
    return j, b - t_starts[k]


def project_topk(tids, tprobs, off, teacher_tok, student_model):
    """教师 top-K (id, prob) → 学生 top-K (表层 id, 归一化 prob)。"""
    agg = {}
    for tid, p in zip(tids, tprobs):
        tid, p = int(tid), float(p)
        surface, starts, ends = teacher_token_student_encoding(tid, teacher_tok, student_model)
        sid = subword_at(surface, starts, ends, off)
        agg[sid] = agg.get(sid, 0.0) + p  # 概率加和

    items = sorted(agg.items(), key=lambda kv: -kv[1])
    if not items:
        return [PAD_SURFACE] * TOP_K, [0.0] * TOP_K
    out_ids = [s for s, _ in items[:TOP_K]]
    out_ps = [p for _, p in items[:TOP_K]]
    tot = sum(out_ps)
    if tot > 0:
        out_ps = [p / tot for p in out_ps]  # 归一化
    pad = TOP_K - len(out_ids)
    if pad > 0:
        out_ids += [PAD_SURFACE] * pad
        out_ps += [0.0] * pad
    return out_ids, out_ps


def build_sample(surface, starts, ends, t_ids, t_starts, t_ends,
                 batch_topk_ids, batch_topk_probs, teacher_tok, student_model):
    """组装一条已对齐、已投影的样本（长度 TARGET_LEN，不足补 PAD）。"""
    L = len(surface)
    n_real = L - 1
    xa = np.full(TARGET_LEN, PAD_SURFACE, dtype=np.int32)
    ya = np.full(TARGET_LEN, PAD_SURFACE, dtype=np.int32)
    xa[:n_real] = surface[:-1]
    ya[:n_real] = surface[1:]

    tids_a = np.full((TARGET_LEN, TOP_K), PAD_SURFACE, dtype=np.int32)
    tprobs_a = np.zeros((TARGET_LEN, TOP_K), dtype=np.float16)  # float16 存盘，体积减半
    mask_a = np.zeros(TARGET_LEN, dtype=bool)

    for i in range(n_real):
        b = starts[i + 1]  # 边界 = 学生 token i+1 的起始字符
        sig = find_teacher_signal(b, t_starts, t_ends)
        if sig is None:
            continue  # 无教师信号 → mask=False
        j, off = sig
        if j >= len(t_ids) - 1:  # j 超出教师 logits 范围（预测最后一个 token 之后）
            continue
        tids, tprobs = project_topk(
            batch_topk_ids[j], batch_topk_probs[j], off, teacher_tok, student_model
        )
        tids_a[i] = tids
        tprobs_a[i] = tprobs
        mask_a[i] = True

    return xa, ya, tids_a, tprobs_a, mask_a


def to_fixed_array(np_arr, inner_size):
    flat = np_arr.reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat), inner_size)


# ---------------------------------------------------------------------------
# 自检模式：验证学生 BPE 加载 + 偏移重建（服务器上先跑一次）
# ---------------------------------------------------------------------------
def self_test(student_model):
    print("youtokentome 版本:", getattr(yttm, "__version__", "?"))
    print("学生词表大小:", len(student_model.vocab()))
    assert len(student_model.vocab()) == STUDENT_VOCAB_SIZE, \
        f"学生词表 {len(student_model.vocab())} != {STUDENT_VOCAB_SIZE}"
    tests = [
        "这是刘思喆创造的小型LLM，可以回答人工智能领域的一些问题。",
        "Hello world, this is a test 123!",
        "  前后带空格  ",
    ]
    for txt in tests:
        ct = clean_text(txt)
        subs, surface = student_encode(student_model, ct)
        starts, ends = student_offsets(subs, ct)
        print(f"\n[clean] {ct!r}")
        print("  subs :", subs)
        print("  ids  :", surface)
        print("  off  :", list(zip(starts, ends)))
    print("\n✅ 自检通过：学生 BPE 加载正常、字符偏移重建一致。可进入正式生成。")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    global TOP_K  # 允许命令行覆盖（build_sample / project_topk 读取模块全局）
    ap = argparse.ArgumentParser(description="Qwen→Rtomic 学生词表投影蒸馏数据生成")
    ap.add_argument("--student-model", default=STUDENT_MODEL_PATH)
    ap.add_argument("--input-jsonl", default=INPUT_JSONL)
    ap.add_argument("--output-dir", default=OUTPUT_CHUNK_DIR)
    ap.add_argument("--limit", type=int, default=LIMIT_NUM, help="0=不限")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--top-k", type=int, default=TOP_K, help="教师 Top-K 投影数（默认 16；越小文件越小）")
    args = ap.parse_args()

    # 【版本标记】每次运行必须打印自己的绝对路径与关键参数，杜绝"跑错文件/跑旧拷贝"。
    # 若输出路径与预期不符，说明运行的不是这份脚本。
    print("=" * 70)
    print(f"[运行脚本] {os.path.abspath(__file__)}")
    print(f"[参数] top_k={args.top_k} chunk_size={args.chunk_size} batch_size={args.batch_size} "
          f"limit={args.limit} output_dir={args.output_dir}")
    print("[归一化检查] tk_probs = exp(tk_logits - lse)（必须存在，否则是旧脚本）")
    print("[自检检查] save_chunk 内置 probs>1 报错（必须存在，否则是旧脚本）")
    print("=" * 70)

    TOP_K = args.top_k

    if yttm is None:
        print("缺少依赖 youtokentome（学生 BPE 的 Python 端，与 R 包 tokenizers.bpe 同源）。")
        print("   pip install Cython")
        print("   pip install youtokentome")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    student_model = yttm.BPE(args.student_model)
    print("学生 BPE 加载成功，词表大小:", len(student_model.vocab()))
    assert len(student_model.vocab()) == STUDENT_VOCAB_SIZE, \
        f"学生词表 {len(student_model.vocab())} != {STUDENT_VOCAB_SIZE}"
    if args.self_test:
        self_test(student_model)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在使用设备: {device}")

    teacher_tok = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    teacher_tok.padding_side = "right"
    if teacher_tok.pad_token is None:
        teacher_tok.pad_token = teacher_tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        cache_dir=CACHE_DIR,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda" if device == "cuda" else "cpu",
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
    print(f"共读取 {len(indexed_texts)} 条，按长度排序...")
    indexed_texts.sort(key=lambda x: len(x[1]))
    if args.limit and args.limit > 0:
        indexed_texts = indexed_texts[:args.limit]
        print(f"【测试模式】仅处理前 {len(indexed_texts)} 条。")

    # 教师 offset 单位（首条文本检测一次，全局固定）
    offset_unit = None

    chunk_x, chunk_y = [], []
    chunk_tids, chunk_tprobs, chunk_mask = [], [], []
    chunk_idx, cur_size = 1, 0

    def save_chunk():
        nonlocal chunk_x, chunk_y, chunk_tids, chunk_tprobs, chunk_mask, chunk_idx, cur_size
        if cur_size == 0:
            return
        xb = np.stack(chunk_x)
        yb = np.stack(chunk_y)
        tids_b = np.stack(chunk_tids)
        tprobs_b = np.stack(chunk_tprobs)
        mask_b = np.stack(chunk_mask)
        N = xb.shape[0]

        # ---- 存盘前自检：软标签必须已归一化（防止旧版/错误脚本把未归一化 exp(logits) 写盘）----
        p_max = tprobs_b.max()
        if p_max > 1.0 + 1e-3:
            raise ValueError(
                f"topk_probs 最大值 {p_max} 超出 [0,1]（未归一化，常见于 softmax 漏减 "
                f"logsumexp，或投影后未归一化）。\n请修正生成逻辑后再落盘。"
            )
        row_sums = tprobs_b.sum(axis=-1)
        if row_sums.max() > 1.0 + 1e-3:
            raise ValueError(
                f"topk_probs 行和最大值 {row_sums.max()} 超出 1.0（未归一化）。"
            )

        # v2 格式：仍用 fixed_size_list[K]（Arrow 表要求各列等长），仅把 probs 由 float16 升为
        # float32。halffloat 的位模式会被 R arrow 按 int16 解读（1.0→15360，曾致 loss~767027）；
        # float32 无此问题，R 端 as.vector 直接得到正确数值、免去 cast。
        table = pa.Table.from_arrays(
            [
                pa.array(xb.reshape(-1), type=pa.int32()),
                pa.array(yb.reshape(-1), type=pa.int32()),
                to_fixed_array(tids_b.reshape(N * TARGET_LEN, TOP_K), TOP_K),
                to_fixed_array(tprobs_b.reshape(N * TARGET_LEN, TOP_K).astype(np.float32), TOP_K),
                pa.array(mask_b.reshape(-1), type=pa.bool_()),
            ],
            names=["x", "y_hard", "topk_ids", "topk_probs", "loss_mask"],
        )
        out_file = os.path.join(args.output_dir, f"chunk_{chunk_idx:03d}.arrow")
        feather.write_feather(table, out_file, compression="zstd")  # zstd 压缩，R 的 arrow 可解压
        print(f"\n已存盘 Chunk {chunk_idx}: {out_file}（{cur_size} 条）")
        chunk_x.clear(); chunk_y.clear()
        chunk_tids.clear(); chunk_tprobs.clear(); chunk_mask.clear()
        cur_size = 0
        chunk_idx += 1
        gc.collect()
        torch.cuda.empty_cache()

    total = len(indexed_texts)
    for bi in tqdm(range(0, total, args.batch_size), desc="Projecting soft labels"):
        batch = indexed_texts[bi: bi + args.batch_size]

        # 1) 学生侧：清洗 + 编码 + 偏移
        stud = []
        for _, txt in batch:
            r = process_text_student(txt, student_model)
            if r is not None:
                stud.append(r)
        if not stud:
            continue

        # 2) 教师侧：逐条 tokenize 取 offsets（不 padding）
        t_ids_list, t_offs_list = [], []
        for ct, _subs, _surface, _starts, _ends in stud:
            enc = teacher_tok(
                ct,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_TEACHER_TOKENS,
            )
            t_ids_list.append(enc["input_ids"])
            t_offs_list.append(enc["offset_mapping"])
        if offset_unit is None:
            # 只在含非 ASCII 字符的文本上检测（全 ASCII 时字节偏移 == 字符偏移，无法区分）
            probe = next((s[0] for s in stud if any(ord(c) > 127 for c in s[0])), None)
            if probe is not None:
                offset_unit = resolve_offset_unit(teacher_tok, probe)
            else:
                offset_unit = "char"

        # 3) 前向
        Tmax = max(len(t) for t in t_ids_list)
        Tmax = max(Tmax, 2)
        input_ids_t = torch.zeros(len(stud), Tmax, dtype=torch.long, device=device)
        attn_t = torch.zeros(len(stud), Tmax, dtype=torch.long, device=device)
        for b, tids in enumerate(t_ids_list):
            if len(tids) > Tmax:
                tids = tids[:Tmax]
            input_ids_t[b, :len(tids)] = torch.tensor(tids, dtype=torch.long)
            attn_t[b, :len(tids)] = 1

        with torch.inference_mode():
            logits = model(input_ids=input_ids_t, attention_mask=attn_t).logits
            logits_shifted = logits[:, :-1, :] / TEMPERATURE
            lse = torch.logsumexp(logits_shifted, dim=-1, keepdim=True)
            tk_logits, tk_ids = torch.topk(logits_shifted, k=TOP_K, dim=-1)
            tk_probs = torch.exp(tk_logits - lse)  # [B, Tmax-1, K]
            tk_ids_np = tk_ids.cpu().numpy()
            tk_probs_np = tk_probs.float().cpu().numpy()
            del logits, logits_shifted, tk_logits

        # 4) 对齐 + 投影 + 组装
        for b, (ct, subs, surface, starts, ends) in enumerate(stud):
            t_ids = t_ids_list[b]
            t_starts = [o[0] for o in offsets_to_char(t_offs_list[b], offset_unit, ct)]
            t_ends = [o[1] for o in offsets_to_char(t_offs_list[b], offset_unit, ct)]
            if len(t_ids) < 2 or len(surface) < 2:
                continue
            xa, ya, tids_a, tprobs_a, mask_a = build_sample(
                surface, starts, ends, t_ids, t_starts, t_ends,
                tk_ids_np[b], tk_probs_np[b], teacher_tok, student_model,
            )
            chunk_x.append(xa); chunk_y.append(ya)
            chunk_tids.append(tids_a); chunk_tprobs.append(tprobs_a)
            chunk_mask.append(mask_a)
            cur_size += 1

        torch.cuda.empty_cache()
        if cur_size >= args.chunk_size:
            save_chunk()

    save_chunk()
    print("\n全部 Arrow Chunk 已存盘（已投影到学生词表，可直接训练）。")


if __name__ == "__main__":
    main()
