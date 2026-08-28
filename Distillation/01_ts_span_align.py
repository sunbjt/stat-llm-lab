#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================
服务器安装与运行
========================================================================
    pip install Cython
    pip install youtokentome           # 与 R 包 tokenizers.bpe 同一 C++ 库
    pip install git+https://github.com/LahiLuk/YouTokenToMe # 装不上用这个
    python3  Distillation/01_ts_span_align.py --self-test   # 先自检学生 BPE 加载/偏移重建
    python3 Distillation/01_ts_span_align.py --limit 50    # 小样本试跑
    python3 Distillation/01_ts_span_align.py --limit 0     # 全量（--limit 0 = 不限文本数）
    # 注意：不带 --limit 时默认只处理前 100000 条（LIMIT_NUM），不是全量。
    python3 Distillation/01_ts_span_align.py --no-normalize --force-restart  # 诊断：存原始累计概率，评估 Top-K 覆盖率（仅供分析，勿直接训练）

若 youtokentome 源码编译失败（备选路线，未实现）：
  本脚本只出教师 top-k ids + 字符偏移 + id→text 词典；
  R 侧用 tokenizers.bpe（同一 C++ 库，已本地验证）完成对齐与投影。

断点续跑（中断后重跑同一命令即可自动继续，无需从头开始）：
  1. 每个 chunk 落盘时，同目录同步写一个 chunk_XXX.meta.json（含 resume_pos、
     配置与数据指纹），且 .arrow / .meta.json 都采用"先写临时文件再原子改名"，
     避免中断留下半截文件。
  2. 启动时扫描输出目录：
       - 已有 chunk_001..k（.arrow 与 .meta.json 齐全）→ 从 chunk_{k+1} 续跑，
         已处理过的文本不再重算、也不重放 tokenize；
       - 旧版 chunk（只有 .arrow、没有 .meta.json）→ 无法安全续跑，需
         --force-restart 或手动清掉旧 chunk；
       - --force-restart → 清空旧 chunk，从头重新生成。
  3. resume_pos 记录"下一文本在原始 jsonl 中的行号"，续跑时据此跳过已处理的
     文本；上次中断时内存里未落盘的半截 chunk 会被重新生成（最多重算一个
     chunk 量级的样本），不影响已存盘 chunk 的连续性。
========================================================================
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

try:
    import youtokentome as yttm
except ImportError:
    yttm = None

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
OUTPUT_CHUNK_DIR = os.path.join(WORK_DIR, "data/processed/chunks/")
STUDENT_MODEL_PATH = os.path.join(WORK_DIR, "models/rtomic_bpe.model")

MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-3B-Instruct"

# ==================== 2. 模型超参数（与 R 侧 config.R 对应） ====================
VOCAB_SIZE = 151936               # 教师词表
STUDENT_VOCAB_SIZE = 2**14        # 16384 (2^14)，对应 config.R 中的 VOCAB_SIZE
MAX_LENGTH = 512                  # 对应 config.R 中的 SEQ_LEN
TARGET_LEN = MAX_LENGTH           # Causal LM 输入输出偏移长度 (511)

TOP_K = 16
TEMPERATURE = 1.0
BATCH_SIZE = 16
CHUNK_SIZE = 4096
LIMIT_NUM = 100000

# ---- 学生词表约定 ----
PAD_SURFACE = 1                     # R 表层 PAD
UNK_SURFACE = 2                     # R 表层 UNK
MAX_STUDENT_LEN = MAX_LENGTH + 1    # 512：学生 token 序列上限

# 学生软目标中要剔除的"垃圾表面"：特殊 token (PAD/UNK/BOS/EOS) 与 U+FFFD (词表 id 4346)。
# 教师稀有词在学生 16384 词表里没有对应表项 → yttm 投影成 UNK；含乱码字符的教师 token → 投影成 U+FFFD。
# 若让这些垃圾进入软目标，学生会被训练成"不确定时就吐 UNK/�"——生成时直接塌缩成乱码死循环
# (实测旧数据里 14.8% 的目标质量是垃圾，14.9% 的位置垃圾占比 > 50%)。
JUNK_SURFACES = frozenset({PAD_SURFACE, UNK_SURFACE, 3, 4, 4346})
MAX_TEACHER_TOKENS = 1024           # 教师侧序列安全上限
# ===============================================

_PUNCT_RE = re.compile(r"([,.:;!?\"'(){}[\]，。！？；：—（）《》“”‘’、])")


def clean_text(text: str) -> str:
    """复刻 R clean_text_internal：标点隔离 → 空白折叠 → strip。"""
    out = _PUNCT_RE.sub(r" \1 ", text)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def student_encode(student_model, ct: str):
    subs = student_model.encode(ct, output_type=yttm.OutputType.SUBWORD)
    ids0 = student_model.encode(ct, output_type=yttm.OutputType.ID)
    assert len(subs) == len(ids0), (len(subs), len(ids0))
    surface = [min(int(i) + 1, STUDENT_VOCAB_SIZE - 1) for i in ids0]
    return subs, surface


def student_offsets(subs, ct: str):
    parts = [s.replace("▁", " ") for s in subs]
    if parts and parts[0].startswith(" ") and not ct.startswith(" "):
        parts[0] = parts[0][1:]
    rebuilt = "".join(parts)
    if rebuilt != ct:
        # 报错里带首个差异位置,便于定位(通常是截断点后的结尾空白,见 process_text_student)
        for i, (a, b) in enumerate(zip(rebuilt, ct)):
            if a != b:
                diff = i
                break
        else:
            diff = min(len(rebuilt), len(ct))
        raise AssertionError(
            f"student_offsets: 子词重建与文本不符 @{diff} "
            f"(rebuilt len={len(rebuilt)}, ct len={len(ct)})\n"
            f"  rebuilt: {rebuilt[max(0, diff-20):diff+40]!r}\n"
            f"  ct     : {ct[max(0, diff-20):diff+40]!r}"
        )
    starts, ends = [], []
    pos = 0
    for p in parts:
        starts.append(pos)
        pos += len(p)
        ends.append(pos)
    return starts, ends


def process_text_student(text: str, student_model):
    ct = clean_text(text)
    if not ct:
        return None
    subs, surface = student_encode(student_model, ct)
    starts, ends = student_offsets(subs, ct)
    L = len(surface)
    if L > MAX_STUDENT_LEN:
        cut = starts[MAX_STUDENT_LEN]
        ct = ct[:cut]
        # 截断点可能正好落在一个空格 token 之后（如 " , "），结尾空格在 BPE 里无法重建
        # （yttm 重建会丢尾部空白 → student_offsets 断言失败）。去掉结尾空白即可，
        # 不影响已包含 token 的边界偏移。
        ct = ct.rstrip()
        subs, surface = student_encode(student_model, ct)
        starts, ends = student_offsets(subs, ct)
        L = len(surface)
        assert L <= MAX_STUDENT_LEN
    if L < 2:
        return None
    return ct, subs, surface, starts, ends


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


_TEACHER_STUDENT_CACHE = {}


def teacher_token_student_encoding(tid, teacher_tok, student_model):
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
        parts[0] = parts[0][1:]
    starts, ends = [], []
    pos = 0
    for p in parts:
        starts.append(pos)
        pos += len(p)
        ends.append(pos)
    if pos != len(ctxt) or len(ids0) != len(parts):
        if not ids0:
            _TEACHER_STUDENT_CACHE[tid] = ([UNK_SURFACE], [0], [1])
        else:
            _TEACHER_STUDENT_CACHE[tid] = ([min(int(ids0[0]) + 1, STUDENT_VOCAB_SIZE - 1)], [0], [1])
        return _TEACHER_STUDENT_CACHE[tid]
    surface = [min(int(i) + 1, STUDENT_VOCAB_SIZE - 1) for i in ids0]
    _TEACHER_STUDENT_CACHE[tid] = (surface, starts, ends)
    return _TEACHER_STUDENT_CACHE[tid]


def subword_at(surface, starts, ends, off):
    if not surface:
        return UNK_SURFACE
    idx = bisect.bisect_right(starts, off) - 1
    idx = max(0, min(idx, len(surface) - 1))
    return surface[idx]


def find_teacher_signal(b, t_starts, t_ends):
    k = bisect.bisect_right(t_starts, b) - 1
    if k < 0 or k >= len(t_ends) or b >= t_ends[k]:
        return None
    j = k - 1
    if j < 0:
        return None
    return j, b - t_starts[k]


def project_topk(tids, tprobs, off, teacher_tok, student_model, normalize=True):
    agg = {}
    for tid, p in zip(tids, tprobs):
        tid, p = int(tid), float(p)
        surface, starts, ends = teacher_token_student_encoding(tid, teacher_tok, student_model)
        sid = subword_at(surface, starts, ends, off)
        if sid in JUNK_SURFACES:
            # 该教师候选投影到 UNK/� 等垃圾表面：不计入学生软目标，不参与重归一化。
            continue
        agg[sid] = agg.get(sid, 0.0) + p

    items = sorted(agg.items(), key=lambda kv: -kv[1])
    if not items:
        return [PAD_SURFACE] * TOP_K, [0.0] * TOP_K
    out_ids = [s for s, _ in items[:TOP_K]]
    out_ps = [p for _, p in items[:TOP_K]]
    tot = sum(out_ps)
    if normalize and tot > 0:
        out_ps = [p / tot for p in out_ps]
    pad = TOP_K - len(out_ids)
    if pad > 0:
        out_ids += [PAD_SURFACE] * pad
        out_ps += [0.0] * pad
    return out_ids, out_ps


def build_sample(surface, starts, ends, t_ids, t_starts, t_ends,
                 batch_topk_ids, batch_topk_probs, teacher_tok, student_model,
                 normalize=True):
    L = len(surface)
    n_real = L - 1
    xa = np.full(TARGET_LEN, PAD_SURFACE, dtype=np.int32)
    ya = np.full(TARGET_LEN, PAD_SURFACE, dtype=np.int32)
    xa[:n_real] = surface[:-1]
    ya[:n_real] = surface[1:]

    tids_a = np.full((TARGET_LEN, TOP_K), PAD_SURFACE, dtype=np.int32)
    tprobs_a = np.zeros((TARGET_LEN, TOP_K), dtype=np.float16)
    mask_a = np.zeros(TARGET_LEN, dtype=bool)

    for i in range(n_real):
        b = starts[i + 1]
        sig = find_teacher_signal(b, t_starts, t_ends)
        if sig is None:
            continue
        j, off = sig
        if j >= len(t_ids) - 1:
            continue
        tids, tprobs = project_topk(
            batch_topk_ids[j], batch_topk_probs[j], off, teacher_tok, student_model,
            normalize=normalize,
        )
        tids_a[i] = tids
        tprobs_a[i] = tprobs
        mask_a[i] = True

    return xa, ya, tids_a, tprobs_a, mask_a


def to_fixed_array(np_arr, inner_size):
    flat = np_arr.reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat), inner_size)


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

    判定规则：
      - chunk_001..k 均同时存在 .arrow 与 .meta.json → 从 k+1 续跑；
      - 没有任何"完整"chunk，但存在裸 .arrow → 旧版数据（legacy）；
      - 空目录 → 全新跑。
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


def warn_config_mismatch(prev_meta, args, slab_size, data_fp):
    """续跑时核对上次运行的配置与数据指纹，提示可能导致断点失效的变更。"""
    prev_cfg = prev_meta.get("config", {}) if prev_meta else {}
    cur_cfg = {
        "top_k": TOP_K,
        "normalize": not args.no_normalize,
        "chunk_size": args.chunk_size,
        "slab_size": slab_size,
        "limit": args.limit,
        "input_jsonl": os.path.basename(args.input_jsonl),
        "student_model": os.path.basename(args.student_model),
        "model": MODEL_NAME_OR_PATH,
    }
    mism = {k: (prev_cfg.get(k), cur_cfg[k]) for k in cur_cfg if prev_cfg.get(k) != cur_cfg[k]}
    if mism:
        print("⚠️  断点元数据中的配置与本次运行不一致，续跑结果可能与旧 chunk 不连续：")
        for k, (old, new) in sorted(mism.items()):
            print(f"      {k}: 旧={old}  新={new}")
        print("      - 仅 limit 变大 / 数据追加时可放心续跑；")
        print("      - 若 chunk_size / top_k / 输入文件已变更，建议 --force-restart 从头重建。")
    if prev_meta and prev_meta.get("data_fp") and prev_meta["data_fp"] != data_fp:
        print("⚠️  数据指纹与断点不一致（输入 jsonl 内容疑似变更），断点位置可能失效！")
        print("      请核对输入文件，必要时 --force-restart。")


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


def main():
    global TOP_K
    ap = argparse.ArgumentParser(description="Qwen→Rtomic 学生词表投影蒸馏数据生成")
    ap.add_argument("--student-model", default=STUDENT_MODEL_PATH)
    ap.add_argument("--input-jsonl", default=INPUT_JSONL)
    ap.add_argument("--output-dir", default=OUTPUT_CHUNK_DIR)
    ap.add_argument("--limit", type=int, default=LIMIT_NUM, help="0=不限")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--slab-size", type=int, default=0,
                    help="teacher 结果驻留内存的文本批（默认 = chunk_size；调大分桶更细但内存更高）")
    ap.add_argument("--top-k", type=int, default=TOP_K, help="教师 Top-K 投影数（默认 16；越小文件越小）")
    ap.add_argument("--no-normalize", action="store_true",
                    help="不归一化投影后的 topk 概率（存原始累计概率，便于评估 Top-K 覆盖率；默认归一化）")
    ap.add_argument("--force-restart", action="store_true",
                    help="忽略已有 chunk，清空后从头重新生成")
    args = ap.parse_args()
    normalize = not args.no_normalize

    print("=" * 70)
    print(f"[运行脚本] {os.path.abspath(__file__)}")
    print(f"[工作路径] {WORK_DIR}")
    print(f"[运行设备] {device}")
    slab_size = args.slab_size if args.slab_size > 0 else args.chunk_size
    print(f"[参数] top_k={args.top_k} normalize={normalize} chunk_size={args.chunk_size} "
          f"slab_size={slab_size} batch_size={args.batch_size} limit={args.limit} "
          f"output_dir={args.output_dir}")
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
    cleanup_tmp_files(args.output_dir)   # 清掉上次中断遗留的半截写文件

    student_model = yttm.BPE(args.student_model)
    print("学生 BPE 加载成功，词表大小:", len(student_model.vocab()))
    assert len(student_model.vocab()) == STUDENT_VOCAB_SIZE, \
        f"学生词表 {len(student_model.vocab())} != {STUDENT_VOCAB_SIZE}"
    if args.self_test:
        self_test(student_model)
        return

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

    offset_unit = None

    chunk_x, chunk_y = [], []
    chunk_tids, chunk_tprobs, chunk_mask = [], [], []
    chunk_idx, cur_size, resume_pos = 1, 0, 0

    def save_chunk():
        nonlocal chunk_x, chunk_y, chunk_tids, chunk_tprobs, chunk_mask, chunk_idx, cur_size, resume_pos
        if cur_size == 0:
            return
        xb = np.stack(chunk_x)
        yb = np.stack(chunk_y)
        tids_b = np.stack(chunk_tids)
        tprobs_b = np.stack(chunk_tprobs)
        mask_b = np.stack(chunk_mask)
        N = xb.shape[0]

        p_max = tprobs_b.max()
        if p_max > 1.0 + 1e-3:
            raise ValueError(
                f"topk_probs 最大值 {p_max} 超出 [0,1]（softmax 漏减 logsumexp，"
                f"或投影聚合异常）。\n请修正生成逻辑后再落盘。"
            )
        row_sums = tprobs_b.sum(axis=-1)
        if row_sums.max() > 1.0 + 1e-3:
            raise ValueError(
                f"topk_probs 行和最大值 {row_sums.max()} 超出 1.0（重归一化遗漏，或 softmax 异常）。"
            )

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
                "normalize": normalize,
                "chunk_size": args.chunk_size,
                "slab_size": slab_size,
                "limit": args.limit,
                "input_jsonl": os.path.basename(args.input_jsonl),
                "student_model": os.path.basename(args.student_model),
                "model": MODEL_NAME_OR_PATH,
            },
        })
        print(f"\n已存盘 Chunk {chunk_idx}: {out_file}（{cur_size} 条）")
        chunk_x.clear(); chunk_y.clear()
        chunk_tids.clear(); chunk_tprobs.clear(); chunk_mask.clear()
        cur_size = 0
        chunk_idx += 1
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    total = len(indexed_texts)
    # offset_unit 对全语料只解析一次
    if offset_unit is None:
        probe = next((t for _, t in indexed_texts if any(ord(c) > 127 for c in t)), None)
        if probe is not None:
            offset_unit = resolve_offset_unit(teacher_tok, probe)
        else:
            offset_unit = "char"

    # chunk 组成保持原始随机顺序（每 chunk 长度混合、代表整体语料），
    # 只在 slab 内对 teacher forward 按长度分桶，把每个 batch 的 padding 降到最低。
    # slab 是 teacher 结果驻留内存的上界：越大分桶越细，但内存越高。
    slab_size = args.slab_size if args.slab_size > 0 else args.chunk_size

    # =====================================================================
    # 断点续跑：依据每个 chunk 的 chunk_XXX.meta.json（含 resume_pos）判断进度
    # =====================================================================
    data_fp = make_data_fp(indexed_texts)
    chunk_idx, resume_pos, prev_meta, legacy = find_resume_point(args.output_dir)
    if args.force_restart:
        clean_chunk_files(args.output_dir)
        chunk_idx, resume_pos, prev_meta, legacy = 1, 0, None, False
        print("[--force-restart] 已清空旧 chunk，从头重新生成。")
    if legacy:
        print("[错误] 输出目录存在旧版 chunk（只有 .arrow、没有 .meta.json），无法安全续跑。")
        print("       请任选其一后重试：")
        print("         1) python 01_ts_span_align.py --force-restart  （清空旧 chunk 从头生成）")
        print("         2) 手动删除 data/processed/chunks/ 下所有 chunk_*.arrow 后再运行")
        sys.exit(1)
    if prev_meta is not None:
        warn_config_mismatch(prev_meta, args, slab_size, data_fp)
        print(f"[断点续跑] 已有 chunk_001..{chunk_idx - 1:03d}，从 resume_pos={resume_pos} "
              f"续写 chunk_{chunk_idx:03d}（无需重放 tokenize）。")
    else:
        print("[全新生成] 输出目录无已有 chunk，从头生成。" if chunk_idx == 1
              else f"[断点续跑] 检测到 chunk 序号缺口，从 chunk_{chunk_idx:03d} 续写。")

    # resume_pos 是“原始 jsonl 行号”→ 转成 indexed_texts 下标，并按 slab 下界对齐
    p0 = bisect.bisect_left([i for i, _ in indexed_texts], resume_pos)
    if p0 >= len(indexed_texts):
        print(f"[完成] 断点 resume_pos={resume_pos} 已越过语料末尾"
              f"（共 {len(indexed_texts)} 条），全部文本均已处理，无需续跑。")
        return
    slab_start = (p0 // slab_size) * slab_size

    n_slabs = (total + slab_size - 1) // slab_size
    for s0 in tqdm(range(slab_start, total, slab_size), desc="Projecting soft labels"):
        slab = indexed_texts[s0: s0 + slab_size]
        if s0 == slab_start and resume_pos > 0:
            # 断点续跑：跳过本段 slab 中 resume_pos 之前的文本（样本已落入已存 chunk）
            slab = [(i, t) for i, t in slab if i >= resume_pos]
        if not slab:
            continue
        t0 = time.time()

        # 1) 一次性 token 化（无模型计算）：学生编码 + 教师编码/字符偏移
        prepped = []
        for idx, txt in slab:
            r = process_text_student(txt, student_model)
            if r is None:
                continue
            ct, _subs, surface, starts, ends = r
            enc = teacher_tok(
                ct,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_TEACHER_TOKENS,
            )
            t_ids = enc["input_ids"]
            t_offs = enc["offset_mapping"]
            t_starts = [o[0] for o in offsets_to_char(t_offs, offset_unit, ct)]
            t_ends   = [o[1] for o in offsets_to_char(t_offs, offset_unit, ct)]
            prepped.append((idx, ct, surface, starts, ends, t_ids, t_starts, t_ends))
        if not prepped:
            continue

        # 2) 按教师 token 数分桶排序 → 同桶 batch 的 Tmax≈桶内长度，padding 最小化。
        #    结果按原始 idx 存入 store，与 forward 顺序解耦。
        store = {}
        bucketed = sorted(prepped, key=lambda p: len(p[5]))
        for bi in range(0, len(bucketed), args.batch_size):
            group = bucketed[bi: bi + args.batch_size]
            Tmax = max(len(p[5]) for p in group)
            Tmax = max(Tmax, 2)
            input_ids_t = torch.zeros(len(group), Tmax, dtype=torch.long, device=device)
            attn_t = torch.zeros(len(group), Tmax, dtype=torch.long, device=device)
            for b, p in enumerate(group):
                tids = p[5]
                input_ids_t[b, :len(tids)] = torch.tensor(tids, dtype=torch.long)
                attn_t[b, :len(tids)] = 1

            with torch.inference_mode():
                logits = model(input_ids=input_ids_t, attention_mask=attn_t).logits
                logits_shifted = logits[:, :-1, :] / TEMPERATURE
                # lse 用 float32 算：bf16 下 logsumexp 有 ~5% 误差，单点概率会 >1
                # （被归一化掩盖，但 --no-normalize 存原始累计概率时会失真、且触发超界校验）。
                # float32 下恒有 lse >= max(logit)，保证每个 tk_prob <= 1。
                lse = torch.logsumexp(logits_shifted.float(), dim=-1, keepdim=True)
                tk_logits, tk_ids = torch.topk(logits_shifted, k=TOP_K, dim=-1)
                tk_probs = torch.exp(tk_logits.float() - lse)
                tk_ids_np = tk_ids.cpu().numpy()
                tk_probs_np = tk_probs.float().cpu().numpy()
                del logits, logits_shifted, tk_logits

            for b, p in enumerate(group):
                store[p[0]] = (tk_ids_np[b], tk_probs_np[b])

        # 3) 按原始随机顺序组装 chunk（teacher 结果已按文本存好，与 forward 顺序无关）
        for idx, ct, surface, starts, ends, t_ids, t_starts, t_ends in prepped:
            # 本文本已完整处理（tokenize + teacher forward 已完成）：无论是否产出样本，
            # 都把它记入断点，避免续跑时重算其 teacher forward。
            resume_pos = idx + 1
            if len(t_ids) < 2 or len(surface) < 2:
                continue
            tk_ids_np, tk_probs_np = store[idx]
            xa, ya, tids_a, tprobs_a, mask_a = build_sample(
                surface, starts, ends, t_ids, t_starts, t_ends,
                tk_ids_np, tk_probs_np, teacher_tok, student_model,
                normalize=normalize,
            )
            chunk_x.append(xa); chunk_y.append(ya)
            chunk_tids.append(tids_a); chunk_tprobs.append(tprobs_a)
            chunk_mask.append(mask_a)
            cur_size += 1
            if cur_size >= args.chunk_size:
                save_chunk()

        del store, prepped
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        slab_i = s0 // slab_size + 1
        print(
            f"\n[slab {slab_i}/{n_slabs}] 完成,耗时 {time.time() - t0:.1f}s,"
            f"累计有效样本 {cur_size},当前 chunk {chunk_idx}",
            flush=True,
        )

    save_chunk()
    print("\n全部 Arrow Chunk 已存盘（已投影到学生词表，可直接训练）。")


if __name__ == "__main__":
    main()