import gc
import json
import os
import pyarrow as pa
import pyarrow.feather as feather
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==================== 配置项 ====================
INPUT_JSONL = "/root/autodl-tmp/stat-llm-lab/data/raw/pretrain_clean.jsonl"
OUTPUT_CHUNK_DIR = "/root/autodl-tmp/stat-llm-lab/data/processed/chunks/"
MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-3B-Instruct"
CACHE_DIR = "/root/autodl-tmp/cache"

MAX_LENGTH = 512
TARGET_LEN = MAX_LENGTH - 1  # 对应 Causal LM 的输入输出偏移长度 (511)
TOP_K = 32
TEMPERATURE = 1.0
BATCH_SIZE = 16
CHUNK_SIZE = 1000
LIMIT_NUM = 1000
VOCAB_SIZE = 151936
# ===============================================


def to_fixed_array(np_arr, inner_size):
    """将二维数组转为 Arrow FixedSizeListArray"""
    flat = np_arr.reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat), inner_size)


def main():
    os.makedirs(OUTPUT_CHUNK_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在使用设备: {device}")

    # 加载 Tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        cache_dir=CACHE_DIR,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda" if device == "cuda" else "cpu",
        trust_remote_code=True,
    )
    model.eval()

    print(f"读取数据源: {INPUT_JSONL}")
    indexed_texts = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                item = json.loads(line)
                indexed_texts.append((idx, item["text"]))

    print(f"共读取到 {len(indexed_texts)} 条文本，正在按长度排序以优化 Batching Padding...")
    indexed_texts.sort(key=lambda x: len(x[1]))

    if LIMIT_NUM is not None and LIMIT_NUM > 0:
        indexed_texts = indexed_texts[:LIMIT_NUM]
        print(f"【测试模式启用】仅截取前 {len(indexed_texts)} 条文本用于生成评估。")

    # Chunk 缓存容器
    (chunk_x, chunk_y, chunk_topk_ids, chunk_topk_probs, chunk_loss_mask) = ([], [], [], [], [])
    chunk_idx = 1
    current_chunk_size = 0

    def save_chunk():
        nonlocal chunk_x, chunk_y, chunk_topk_ids, chunk_topk_probs, chunk_loss_mask, chunk_idx, current_chunk_size
        if current_chunk_size == 0:
            return

        # ✅ 用整个 chunk 的样本数，而不是单个 batch
        N = sum(x.size(0) for x in chunk_x)

        x_batch = torch.cat(chunk_x, dim=0).cpu().numpy().astype("int32")
        y_batch = torch.cat(chunk_y, dim=0).cpu().numpy().astype("int32")
        mask_batch = torch.cat(chunk_loss_mask, dim=0).cpu().numpy().astype("bool")

        t_ids_batch = torch.cat(chunk_topk_ids, dim=0).cpu().numpy().astype("int32")
        t_probs_batch = torch.cat(chunk_topk_probs, dim=0).to(torch.float32).cpu().numpy()

        arrow_table = pa.Table.from_arrays(
            [
                pa.array(x_batch.reshape(-1), type=pa.int32()),
                pa.array(y_batch.reshape(-1), type=pa.int32()),
                to_fixed_array(t_ids_batch.reshape(N * TARGET_LEN, TOP_K), TOP_K),
                to_fixed_array(t_probs_batch.reshape(N * TARGET_LEN, TOP_K), TOP_K),
                pa.array(mask_batch.reshape(-1), type=pa.bool_()),
            ],
            names=["x", "y_hard", "topk_ids", "topk_probs", "loss_mask"],
        )

        out_file = os.path.join(OUTPUT_CHUNK_DIR, f"chunk_{chunk_idx:03d}.arrow")
        feather.write_feather(arrow_table, out_file, compression="uncompressed")
        print(f"\n已存盘 Chunk {chunk_idx}: {out_file} (包含 {current_chunk_size} 条数据)")

        # 清理
        chunk_x.clear(); chunk_y.clear()
        chunk_topk_ids.clear(); chunk_topk_probs.clear()
        chunk_loss_mask.clear()
        current_chunk_size = 0
        chunk_idx += 1
        gc.collect()
        torch.cuda.empty_cache()

    total_items = len(indexed_texts)
    for i in tqdm(range(0, total_items, BATCH_SIZE), desc="Extracting Soft Labels"):
        batch_data = indexed_texts[i: i + BATCH_SIZE]
        batch_texts = [x[1] for x in batch_data]

        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Causal LM 错位截取
            x = input_ids[:, :-1] + 1
            y = input_ids[:, 1:] + 1
            mask = attention_mask[:, 1:].bool()

            logits_shifted = logits[:, :-1, :] / TEMPERATURE
            log_sum_exp = torch.logsumexp(logits_shifted, dim=-1, keepdim=True)
            topk_logits, topk_ids = torch.topk(logits_shifted, k=TOP_K, dim=-1)
            topk_probs = torch.exp(topk_logits - log_sum_exp)

            topk_ids = topk_ids + 1

            # Padding
            pad_len = TARGET_LEN - x.size(1)
            if pad_len > 0:
                x = F.pad(x, (0, pad_len), value=1)
                y = F.pad(y, (0, pad_len), value=1)
                mask = F.pad(mask, (0, pad_len), value=False)
                topk_ids = F.pad(topk_ids, (0, 0, 0, pad_len), value=1)
                topk_probs = F.pad(topk_probs, (0, 0, 0, pad_len), value=0.0)

            # 缓存
            chunk_x.append(x.cpu())
            chunk_y.append(y.cpu())
            chunk_topk_ids.append(topk_ids.cpu())
            chunk_topk_probs.append(topk_probs.cpu())
            chunk_loss_mask.append(mask.cpu())

            current_chunk_size += x.size(0)

            del outputs, logits
            torch.cuda.empty_cache()

            if current_chunk_size >= CHUNK_SIZE:
                save_chunk()

    save_chunk()
    print("\n所有 Arrow 格式 Chunk 数据已成功独立存盘！")


if __name__ == "__main__":
    main()
