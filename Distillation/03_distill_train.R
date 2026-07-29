# =====================================================================
# 教师 Logits 离线提炼 — 本地文件版 (无需联网)
# 从本地 models/ 目录加载 Qwen3.5-2B 权重，对预训练数据做 Top-K logit 提取
# =====================================================================

library(torch)
library(minhub)  # 仅用于 qwen3() 构造函数和 qwen3_hf_weights_remap (不联网)
library(tok)

# ---- 设备和环境 ----
is_mac <- Sys.info()["sysname"] == "Darwin"
has_cuda <- cuda_is_available()
DEVICE <- if (has_cuda) "cuda" else if (is_mac) "mps" else "cpu"
USE_BF16 <- has_cuda  # 3090 原生支持 bf16, CPU/MPS 不支持
MODEL_DTYPE <- if (USE_BF16) torch_bfloat16() else torch_float32()
cat(sprintf("计算设备: %s | 精度: %s\n", DEVICE, if (USE_BF16) "bfloat16" else "float32"))

# ---- 配置 ----
MODEL_DIR     <- "models"
BIN_INPUT     <- "data/processed/qwen_tokens_aligned.bin"
LOGITS_OUTPUT <- "data/processed/teacher_top16_logits.bin"
SEQ_LEN       <- 512L
BATCH_SIZE    <- 8L
K             <- 16L

# ---- 1. 从本地 config.json 解析模型架构参数 (不需要联网) ----
cat("从本地 config.json 读取模型架构...\n")
config <- jsonlite::fromJSON(file.path(MODEL_DIR, "config.json"))
tc <- config$text_config

# 映射 config.json → qwen3() 构造函数参数
# 注意: layer_types 在 text_config 内部, 不在 config 顶层
teacher <- qwen3(
  vocab_size = tc$vocab_size,
  n_embd     = tc$hidden_size,
  n_inter    = tc$intermediate_size,
  n_head     = tc$num_attention_heads,
  n_kv_head  = tc$num_key_value_heads,
  head_dim   = tc$head_dim,
  n_layer    = tc$num_hidden_layers,
  max_pos    = tc$max_position_embeddings,
  rmsnorm_eps = tc$rms_norm_eps,
  rope_base  = tc$rope_parameters$rope_theta,
  partial_rotary_factor = tc$rope_parameters$partial_rotary_factor,
  layer_types = tc$layer_types,
  n_k_heads   = if (!is.null(tc$linear_num_key_heads)) tc$linear_num_key_heads else 16,
  n_v_heads   = if (!is.null(tc$linear_num_value_heads)) tc$linear_num_value_heads else 16,
  k_head_dim  = if (!is.null(tc$linear_key_head_dim)) tc$linear_key_head_dim else 128,
  v_head_dim  = if (!is.null(tc$linear_value_head_dim)) tc$linear_value_head_dim else 128,
  conv_kernel_size = if (!is.null(tc$linear_conv_kernel_dim)) tc$linear_conv_kernel_dim else 4
)

# ---- 2. 从本地 safetensors 加载权重 (不需要联网) ----
weights_file <- file.path(MODEL_DIR, "model.safetensors-00001-of-00001.safetensors")
cat(sprintf("正在从本地文件加载权重: %s (%.1f GB)...\n",
            weights_file, file.info(weights_file)$size / 1e9))

# safetensors 是独立格式, 不能用 torch::load_state_dict() 直接读
# minhub 内部用 safetensors::safe_load_file()
state_dict <- safetensors::safe_load_file(weights_file, framework = "torch")

# 重映射 HF 权重名 → minhub 内部命名 (minhub 内部函数, 不需要联网)
state_dict <- minhub:::qwen3_hf_weights_remap(state_dict)
teacher$load_state_dict(state_dict, .refer_to_state_dict = TRUE)

cat("权重加载完成。\n")

# ---- 3. 转移到设备并设为 eval 模式 ----
# GPU (3090): 保持 bf16 权重, 更快更省显存
# CPU/Mac: 必须转为 float32, CPU 不支持 bf16 运算
teacher$to(dtype = MODEL_DTYPE, device = DEVICE)
teacher$eval()
cat(sprintf("模型已部署到 %s。\n", DEVICE))

# ---- 4. 读取 Token 数据并构建 DataLoader ----
cat(sprintf("读取 Token 数据: %s\n", BIN_INPUT))
con_in <- file(BIN_INPUT, "rb")
total_tokens <- file.info(BIN_INPUT)$size / 4
tokens <- readBin(con_in, what = "integer", n = total_tokens, size = 4, signed = TRUE)
close(con_in)

# 模型需要 1-based token IDs (与训练时一致)
tokens <- tokens + 1L

num_chunks  <- floor(length(tokens) / SEQ_LEN)
tokens_mat  <- matrix(tokens[1:(num_chunks * SEQ_LEN)], ncol = SEQ_LEN, byrow = TRUE)
tokens_tens <- torch_tensor(tokens_mat, dtype = torch_long())

dataset    <- tensor_dataset(tokens_tens)
dataloader <- dataloader(dataset, batch_size = BATCH_SIZE, shuffle = FALSE)
cat(sprintf("总共 %d 个 chunk, %d 个 batch。\n", num_chunks, length(dataloader)))

# ---- 5. GPU 高速提取 Top-K Logits ----
con_out <- file(LOGITS_OUTPUT, "wb")
cat("开始提取 Top-16 Logits...\n")

with_no_grad({
  batch_idx <- 0
  coro::loop(for (b in dataloader) {
    batch_idx <- batch_idx + 1
    input_ids <- b[[1]]$to(device = DEVICE)

    out <- teacher(input_ids)
    logits <- if (is.list(out)) out$logits else out

    # GPU 端 Top-K
    topk_res <- torch_topk(logits, k = K, dim = -1)

    # 还原为 0-based Qwen ID 并切回 CPU
    topk_indices <- (topk_res[[2]] - 1L)$to(device = "cpu")
    topk_values  <- topk_res[[1]]$to(device = "cpu")

    b_size <- input_ids$size(1)
    for (i in seq_len(b_size)) {
      idx_array <- as.integer(as_array(topk_indices[i]))
      val_array <- as.numeric(as_array(topk_values[i]))

      writeBin(idx_array, con_out, size = 4, endian = "little")
      writeBin(val_array, con_out, size = 4, endian = "little")
    }

    if (batch_idx %% 50 == 0) {
      cat(sprintf("已完成: %d/%d batches (%.1f%%)\n",
                  batch_idx, length(dataloader),
                  (batch_idx / length(dataloader)) * 100))
    }
  })
})

close(con_out)
cat(sprintf("✅ 教师 Top-%d Logits 提炼完成！\n输出文件: %s\n", K, LOGITS_OUTPUT))
