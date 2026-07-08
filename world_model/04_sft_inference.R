# =====================================================================
# JEPA SFT 推理
# =====================================================================

source("config.R")
torch_set_num_threads(1L)

device <- torch_device("cpu")
cat(sprintf("JEPA SFT 推理引擎启动，当前使用加速设备: %s\n", device$type))

source("world_model/jepa_model.R")
source("utils/BPETokenizer.R")
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

# =====================================================================
# 1. 加载 JEPA 基座 + SFT 权重
# =====================================================================
model <- RtomicJEPA_VQ(
  vocab_size = VOCAB_SIZE, dim = DIM, n_layers = N_LAYERS,
  n_heads = N_HEADS, max_seq_len = SEQ_LEN, num_clusters = 4096
)

TARGET_CKPT <- "checkpoints/jepa_sft_epoch_05.pt"
if (file.exists(TARGET_CKPT)) {
  cat(sprintf("正在挂载 JEPA SFT 权重: %s\n", TARGET_CKPT))
  ckpt <- torch_load(TARGET_CKPT)
  state_dict <- if ("model" %in% names(ckpt)) ckpt$model else ckpt
  model$load_state_dict(state_dict, strict = FALSE)
} else {
  stop("找不到指定的 Checkpoint 文件！请检查路径。")
}

model <- model$to(device = device)
model$eval()

# =====================================================================
# 2. 生成流水线 (与 LRP SFT 推理完全一致的接口)
# =====================================================================
generate_response <- function(model, tokenizer, prompt,
                              max_new_tokens = 500,
                              temperature = 0.8,
                              top_k = 10,
                              rep_penalty = 1.1) {

  eos_val <- if (!is.null(tokenizer$eos_idx)) tokenizer$eos_idx else 2L

  p_ids <- tokenizer$encode_raw(prompt)[[1]]
  current_ids <- c(tokenizer$bos_idx, p_ids)

  cat(sprintf("\n[User]: %s\n[Assistant]: ", prompt))

  with_no_grad({
    for (step in 1:max_new_tokens) {
      input_seq <- tail(current_ids, SEQ_LEN)
      x_tensor <- torch_tensor(input_seq, dtype = torch_long(), device = device)$unsqueeze(1)

      output <- model(list(x = x_tensor, y = NULL))

      seq_length <- x_tensor$size(2)
      last_pred <- output$pred[1, seq_length, ]

      # 潜空间 → 词表投影
      cb <- model$tok_emb$weight
      logits <- torch_matmul(last_pred$unsqueeze(1), cb$t())$squeeze(1)
      logits <- logits / temperature

      # 重复惩罚
      unique_past_ids <- unique(current_ids)
      for (past_id in unique_past_ids) {
        logit_val <- as.numeric(logits[past_id])
        if (logit_val < 0) {
          logits[past_id] <- logit_val * rep_penalty
        } else {
          logits[past_id] <- logit_val / rep_penalty
        }
      }

      # Top-K 截断
      if (top_k > 0) {
        topk_res <- torch_topk(logits, k = top_k)
        kth_value <- topk_res[[1]][top_k]
        logits <- torch_where(logits < kth_value, torch_tensor(-Inf, device = device), logits)
      }

      # 概率采样
      probs <- nnf_softmax(logits, dim = -1)
      next_token_id <- as.integer(torch_multinomial(probs, num_samples = 1))

      if (next_token_id == eos_val) break

      new_text <- tokenizer$decode(next_token_id)
      if (new_text == "<EOS>") break

      cat(new_text)
      flush.console()

      current_ids <- c(current_ids, next_token_id)
    }
  })

  cat("\n--------------------------------------------------\n")
  invisible(current_ids)
}

# =====================================================================
# 3. 互动测试
# =====================================================================
torch_manual_seed(42)

test_prompts <- c(
  "解释一下决策树算法",
  "数据科学在企业有哪些直接的应用呢？",
  "简单介绍一下交叉熵的基本原理。",
  "人工智能能做什么？",
  "你是谁？"
)

for (p in test_prompts) {
  generate_response(model, tokenizer, prompt = p,
                    max_new_tokens = 128,
                    temperature = 0.5,
                    top_k = 10,
                    rep_penalty = 1.2)
}
