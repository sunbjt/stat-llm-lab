#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================
阶段 B：学生词表投影 → 训练 arrow（须在 01a_teacher_logits.py 之后运行）
========================================================================

读阶段 A 落盘的原始教师 Top-K（--teacher-out-dir，默认
data/processed/teacher_chunks/）→ 学生 BPE 重新编码 clean 文本 → 字符偏移对齐
→ 每个教师候选投影到学生词表、按学生表面聚合、丢垃圾（UNK/�）、重归一化
→ 输出训练可直接消费的 arrow（--output-dir，默认 data/processed/chunks/）：
    x / y_hard / topk_ids / topk_probs / loss_mask
（列与旧版单文件一致，02_train.R 直接读取）

只依赖学生 BPE（youtokentome）+ 教师 tokenizer（仅 decode，不加载教师模型）。
分钟级，可反复重跑：换学生词表 / 调 top_k / 换归一化策略只重跑本步。

断点续跑：按 text_idx 记账（resume_pos = 下一文本的原始 jsonl 行号），
同样支持断点续跑；本身分钟级，中断后重跑即可续。

运行：
    pip install Cython
    pip install youtokentome           # 与 R 包 tokenizers.bpe 同一 C++ 库
    python3 Distillation/01b_student_project.py --self-test     # 学生 BPE 自检
    python3 Distillation/01b_student_project.py --force-restart # 重新投影（改配置后务必全量重建）
    python3 Distillation/01b_student_project.py --no-normalize  # 诊断：存原始累计概率（勿直接训练）
    # 注意：--top-k 必须与阶段 A 生成时一致（默认 16）；学生词表默认 16384。
========================================================================
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import bisect
import gc
import json
import platform
import re
import sys

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from transformers import AutoTokenizer

try:
    import youtokentome as yttm
except ImportError:
    yttm = None

# ==================== 1. 环境与路径自动检测 ====================
system_name = platform.system()
if system_name == "Darwin":
    print("--- 检测到 Mac 环境：本地路径 ---")
    WORK_DIR = os.path.expanduser("~/github/stat-llm-lab/")
    CACHE_DIR = os.path.expanduser("~/cache/huggingface")
elif system_name == "Linux":
    WORK_DIR = "/root/autodl-tmp/stat-llm-lab/"
    CACHE_DIR = "/root/autodl-tmp/cache"
else:
    WORK_DIR = os.path.abspath(".")
    CACHE_DIR = os.path.expanduser("~/cache/huggingface")

os.chdir(WORK_DIR)

TEACHER_CHUNK_DIR = os.path.join(WORK_DIR, "data/processed/teacher_chunks/")
OUTPUT_CHUNK_DIR = os.path.join(WORK_DIR, "data/processed/chunks/")
STUDENT_MODEL_PATH = os.path.join(WORK_DIR, "models/rtomic_bpe.model")
MODEL_NAME_OR_PATH = "Qwen/Qwen3.5-0.8B"

# ==================== 2. 学生侧超参数 ====================
STUDENT_VOCAB_SIZE = 2**14        # 16384
MAX_LENGTH = 512                  
TARGET_LEN = MAX_LENGTH           

TOP_K = 16                        
CHUNK_SIZE = 2048                 

PAD_SURFACE = 1                     
UNK_SURFACE = 2                     
MAX_STUDENT_LEN = MAX_LENGTH + 1    

JUNK_SURFACES = frozenset({PAD_SURFACE, UNK_SURFACE, 3, 4, 4346})
# ===============================================

_PUNCT_RE = re.compile(r"([,.:;!?\"'(){}[\]，。！？；：—（）《》“”‘’、])")


def clean_text(text: str) -> str:
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
        ct = ct[:cut].rstrip()
        subs, surface = student_encode(student_model, ct)
        starts, ends = student_offsets(subs, ct)
        L = len(surface)
        assert L <= MAX_STUDENT_LEN
    if L < 2:
        return None
    return ct, subs, surface, starts, ends


# ==================== 教师 top-K → 学生词表投影 ====================

_TEACHER_STUDENT_CACHE = {}


def teacher_token_student_encoding(tid, teacher_tok, student_model):
    """【已修复】不再盲目 clean_text 导致解包空格被误删掉，同时增加上限容错保护。"""
    if tid in _TEACHER_STUDENT_CACHE:
        return _TEACHER_STUDENT_CACHE[tid]

    txt = teacher_tok.decode([tid])
    if not txt:
        _TEACHER_STUDENT_CACHE[tid] = ([UNK_SURFACE], [0], [1])
        return _TEACHER_STUDENT_CACHE[tid]

    # 直接对其编码，不能走 strip() 风格的 clean_text
    subs = student_model.encode(txt, output_type=yttm.OutputType.SUBWORD)
    ids0 = student_model.encode(txt, output_type=yttm.OutputType.ID)
    parts = [s.replace("▁", " ") for s in subs]
    if parts and parts[0].startswith(" "):
        parts[0] = parts[0][1:]
    starts, ends = [], []
    pos = 0
    for p in parts:
        starts.append(pos)
        pos += len(p)
        ends.append(pos)

    if pos != len(txt) or len(ids0) != len(parts):
        if not ids0:
            res = ([UNK_SURFACE], [0], [1])
        else:
            res = ([min(int(ids0[0]) + 1, STUDENT_VOCAB_SIZE - 1)], [0], [max(1, len(txt))])
    else:
        surface = [min(int(i) + 1, STUDENT_VOCAB_SIZE - 1) for i in ids0]
        res = (surface, starts, ends)

    # 限制 Cache 体积防泄露
    if len(_TEACHER_STUDENT_CACHE) < 250000:
        _TEACHER_STUDENT_CACHE[tid] = res
    return res


def subword_at(surface, starts, ends, off):
    """【已修复】加入闭区间严格性判定 starts[idx] <= off < ends[idx]。"""
    if not surface:
        return UNK_SURFACE
    idx = bisect.bisect_right(starts, off) - 1
    if 0 <= idx < len(surface):
        if starts[idx] <= off < ends[idx]:
            return surface[idx]
    return UNK_SURFACE


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


def build_sample(surface, starts, ends, n_teacher, t_starts, t_ends,
                 topk_ids, topk_probs, teacher_tok, student_model,
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
        if j >= n_teacher - 1:
            continue
        tids, tprobs = project_topk(
            topk_ids[j], topk_probs[j], off, teacher_tok, student_model,
            normalize=normalize,
        )
        tids_a[i] = tids
        tprobs_a[i] = tprobs
        mask_a[i] = True

    return xa, ya, tids_a, tprobs_a, mask_a


def build_student_sample(row, teacher_tok, student_model, normalize=True):
    ct = row["ct"]
    n_teacher = int(row["n_teacher"])
    if n_teacher < 2:
        return None
    r = process_text_student(ct, student_model)
    if r is None:
        return None
    _ct, _subs, surface, starts, ends = r
    if len(surface) < 2:
        return None

    K = TOP_K
    ids_flat = row["teacher_ids"]
    probs_flat = row["teacher_probs"]
    expect = (n_teacher - 1) * K
    if len(ids_flat) != expect or len(probs_flat) != expect:
        raise ValueError(
            f"阶段 A 数据与当前 top_k 不一致：n_teacher={n_teacher}, top_k={K}, "
            f"teacher_ids 长度 {len(ids_flat)} != (n-1)×K = {expect}。"
        )

    topk_ids = np.asarray(ids_flat, dtype=np.int32).reshape(n_teacher - 1, K)
    topk_probs = np.asarray(probs_flat, dtype=np.float32).reshape(n_teacher - 1, K)
    return build_sample(
        surface, starts, ends, n_teacher,
        row["t_starts"], row["t_ends"],
        topk_ids, topk_probs, teacher_tok, student_model,
        normalize=normalize,
    )


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
    print("\n✅ 自检通过：学生 BPE 加载正常、字符偏移重建一致。可进入正式投影。")


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


def warn_config_mismatch(prev_meta, cur_cfg, data_fp):
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
    for fn in os.listdir(output_dir):
        if re.fullmatch(r"chunk_\d{3}(\.arrow|\.meta\.json)(\.tmp)?", fn):
            p = os.path.join(output_dir, fn)
            try:
                os.remove(p)
                print(f"  删除 {fn}")
            except OSError as e:
                print(f"[warn] 删除 {fn} 失败：{e}")


def run_stage_student(args):
    print("=" * 70)
    print("[阶段 B] 学生词表投影：原始教师 Top-K → 学生 16384 表面")
    print(f"[工作路径] {WORK_DIR}")

    if yttm is None:
        print("缺少依赖 youtokentome。")
        print("   pip install Cython && pip install youtokentome")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    cleanup_tmp_files(args.output_dir)

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

    a_entries = []
    if os.path.isdir(args.teacher_out_dir):
        for fn in os.listdir(args.teacher_out_dir):
            m = re.fullmatch(r"chunk_(\d{3})\.arrow", fn)
            if m:
                a_entries.append((int(m.group(1)), os.path.join(args.teacher_out_dir, fn)))
    a_entries.sort()
    if not a_entries:
        print(f"[错误] 阶段 A 输出目录为空: {args.teacher_out_dir}")
        sys.exit(1)

    data_fp = None
    last_a_meta = None
    last_a_meta_path = os.path.join(args.teacher_out_dir, f"chunk_{a_entries[-1][0]:03d}.meta.json")
    if os.path.isfile(last_a_meta_path):
        with open(last_a_meta_path, "r", encoding="utf-8") as f:
            last_a_meta = json.load(f)
        data_fp = last_a_meta.get("data_fp")
    a_limit = last_a_meta.get("config", {}).get("limit", 0) if last_a_meta else 0

    normalize = not args.no_normalize
    print(f"[参数] top_k={TOP_K} normalize={normalize} chunk_size={args.chunk_size} ")

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
        if p_max > 1.0 + 0.02:
            raise ValueError(f"topk_probs 最大值 {p_max} 超出 [0,1]")
        row_sums = tprobs_b.sum(axis=-1)
        if row_sums.max() > 1.0 + 0.02:
            raise ValueError(f"topk_probs 行和最大值 {row_sums.max()} 超出 1.0")

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
        os.replace(tmp_file, out_file)

        meta_file = out_file.replace(".arrow", ".meta.json")
        _atomic_write_json(meta_file, {
            "chunk_idx": chunk_idx,
            "resume_pos": resume_pos,
            "data_fp": data_fp,
            "config": {
                "top_k": TOP_K,
                "normalize": normalize,
                "chunk_size": args.chunk_size,
                "limit": a_limit,
                "input_teacher": os.path.basename(args.teacher_out_dir),
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

    chunk_idx, resume_pos, prev_meta, legacy = find_resume_point(args.output_dir)
    if args.force_restart:
        clean_chunk_files(args.output_dir)
        chunk_idx, resume_pos, prev_meta, legacy = 1, 0, None, False
        print("[--force-restart] 已清空阶段 B 旧 chunk，从头重新投影。")
    if legacy:
        print("[错误] 阶段 B 输出目录存在旧版裸 .arrow，无法安全续跑。")
        sys.exit(1)
    if prev_meta is not None:
        warn_config_mismatch(prev_meta, {
            "top_k": TOP_K,
            "normalize": normalize,
            "chunk_size": args.chunk_size,
            "limit": a_limit,
            "input_teacher": os.path.basename(args.teacher_out_dir),
            "student_model": os.path.basename(args.student_model),
            "model": MODEL_NAME_OR_PATH,
        }, data_fp)
        print(f"[断点续跑] 从 resume_pos={resume_pos} 续写。")

    for a_idx, af in a_entries:
        a_meta = None
        a_meta_path = os.path.join(args.teacher_out_dir, f"chunk_{a_idx:03d}.meta.json")
        if os.path.isfile(a_meta_path):
            with open(a_meta_path, "r", encoding="utf-8") as f:
                a_meta = json.load(f)
        if a_meta is not None and int(a_meta.get("resume_pos", 0)) - 1 < resume_pos:
            continue

        table = pa.feather.read_table(af)
        n_rows = table.num_rows
        for row in table.to_pylist():
            text_idx = int(row["text_idx"])
            if text_idx < resume_pos:
                continue
            resume_pos = text_idx + 1
            sample = build_student_sample(
                row, teacher_tok, student_model, normalize=normalize,
            )
            if sample is None:
                continue
            xa, ya, tids_a, tprobs_a, mask_a = sample
            chunk_x.append(xa); chunk_y.append(ya)
            chunk_tids.append(tids_a); chunk_tprobs.append(tprobs_a)
            chunk_mask.append(mask_a)
            cur_size += 1
            if cur_size >= args.chunk_size:
                save_chunk()

        print(f"[阶段 B] 已处理 teacher chunk {a_idx:03d}/{len(a_entries)}"
              f"（{n_rows} 行），累计样本 {cur_size}，chunk {chunk_idx}", flush=True)

    save_chunk()
    print("\n全部 Arrow Chunk 已存盘。")


def main():
    global TOP_K
    ap = argparse.ArgumentParser(description="阶段 B：学生词表投影")
    ap.add_argument("--student-model", default=STUDENT_MODEL_PATH)
    ap.add_argument("--teacher-out-dir", default=TEACHER_CHUNK_DIR)
    ap.add_argument("--output-dir", default=OUTPUT_CHUNK_DIR)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--force-restart", action="store_true")
    args = ap.parse_args()
    TOP_K = args.top_k

    os.makedirs(CACHE_DIR, exist_ok=True)
    run_stage_student(args)


if __name__ == "__main__":
    main()