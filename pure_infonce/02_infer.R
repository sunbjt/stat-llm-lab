# =====================================================================
# Pure InfoNCE 推断与生成脚本
# =====================================================================

Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")

source("config.R")
source("utils/BPETokenizer.R")
source("latent_contrastive/contrastive_model.R")

# 1. 初始化 Tokenizer & 设备
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)
device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
cat(sprintf("推断运行设备: %s\n", device$type))

# 2. 核心文本生成函数
generate_text <- function(model, tokenizer, prompt, max_new_tokens = 100,
                          temperature = 0.7, repetition_penalty = 1.2,
                          top_k = 5, use_contrastive = TRUE) {
  model$eval()

  raw_ids <- tokenizer$encode_raw(prompt)[[1]]
  current_ids <- c(tokenizer$bos_idx, raw_ids)

  cat(sprintf("\n[Input]: %s\n", prompt))
  cat("[Generate]: ")

  with_no_grad({
    for (step in 1:max_new_tokens) {
      input_seq <- tail(current_ids, SEQ_LEN)
      x_tensor <- torch_tensor(input_seq, dtype = torch_long(), device = device)$unsqueeze(1)
      S <- x_tensor$size(2)

      # 编码提取潜表征
      h <- model$encode(x_tensor)

      if (use_contrastive) {
        # Pure InfoNCE 潜空间映射路径
        pred_all <- model$predictor(h)
        pred_last <- pred_all[, S, , drop = FALSE]$squeeze(1)
        
        pred_norm <- nnf_normalize(pred_last, p = 2, dim = -1)
        emb_norm <- nnf_normalize(model$tok_emb$weight, p = 2, dim = -1)
        
        # 计算特征向量与全词表 Embedding 的 Cosine 相似度
        cos_sim <- torch_matmul(pred_norm, emb_norm$transpose(1, 2))$squeeze(1)
        
        # 训练过程中的 Temperature（取模型保存的值或 fallback）
        temp_val <- as.numeric(model$temperature$cpu())
        if (is.null(temp_val) || length(temp_val) == 0) temp_val <- 0.2
        
        logits <- cos_sim / temp_val
      } else {
        # CE 路径（回退）
        logits <- torch_matmul(h[, S, ], model$tok_emb$weight$transpose(1, 2))$squeeze(1)
      }

      # 1. Repetition Penalty 惩罚项
      if (repetition_penalty > 1.0) {
        uniq_ids <- unique(current_ids)
        idx_tensor <- torch_tensor(uniq_ids, dtype = torch_long(), device = device)
        penalized <- logits[idx_tensor]$clone()
        mask_pos <- penalized > 0
        penalized[mask_pos] <- penalized[mask_pos] / repetition_penalty
        penalized[!mask_pos] <- penalized[!mask_pos] * repetition_penalty
        logits[idx_tensor] <- penalized
      }

      # 2. Temperature Scaling 缩放
      if (temperature > 1e-3) {
        logits <- logits / temperature
      }

      # 3. Top-k 采样过滤
      if (top_k > 0 && top_k < logits$size(1)) {
        topk <- torch_topk(logits, k = top_k)
        min_val <- topk[[1]][top_k]$item()
        logits[logits < min_val] <- -Inf
      }

      # 4. 采样 / 贪婪解码
      if (temperature <= 1e-3) {
        next_token_id <- as.integer(torch_argmax(logits, dim = -1))
      } else {
        probs <- nnf_softmax(logits, dim = -1)
        next_token_id <- as.integer(torch_multinomial(probs, num_samples = 1))
      }

      # EOS 解码与终止判断
      if (!is.null(tokenizer$eos_idx) && next_token_id == tokenizer$eos_idx) break

      next_word <- tokenizer$decode(next_token_id, clean = FALSE)
      if (next_word == "<EOS>") break

      cat(next_word)
      flush.console()

      current_ids <- c(current_ids, next_token_id)
    }
  })
  cat("\n\nDone.\n")
  invisible(current_ids)
}

# 3. 加载训练好的权重并开始测试
checkpoints_to_test <- c(
  "checkpoints/infonce_03.pt"
)

test_prompts <- c(
  "我认为 AI 未来对人类",
  "在大学里应该这样学习编程，首先",
  "使用数据科学怎样帮助",
  "agent 技术是",
  "深度学习有"
)

for (ckpt_path in checkpoints_to_test) {
  if (!file.exists(ckpt_path)) {
    cat(sprintf("\n[跳过] 找不到模型文件: %s\n", ckpt_path))
    next
  }

  cat(sprintf("\n=========================================="))
  cat(sprintf("\n=== 加载模型Checkpoint: %s ===", ckpt_path))
  cat(sprintf("\n==========================================\n"))

  checkpoint_data <- torch_load(ckpt_path, device = device)
  cfg <- checkpoint_data$config

  # 动态重建模型（基于 Checkpoint saved config）
  model <- TokenLatentModel(
    vocab_size  = if (!is.null(cfg$vocab_size)) cfg$vocab_size else VOCAB_SIZE,
    dim         = if (!is.null(cfg$dim)) cfg$dim else DIM,
    n_layers    = if (!is.null(cfg$n_layers)) cfg$n_layers else N_LAYERS,
    n_heads     = if (!is.null(cfg$n_heads)) cfg$n_heads else N_HEADS,
    max_seq_len = if (!is.null(cfg$seq_len)) cfg$seq_len else SEQ_LEN,
    temperature = if (!is.null(cfg$temperature)) cfg$temperature else 0.2
  )

  model$load_state_dict(checkpoint_data$model)
  model <- model$to(device = device)
  model$to(dtype = torch_float32())

  for (prompt in test_prompts) {
    generate_text(
      model,
      tokenizer,
      prompt = prompt,
      temperature = 0.3,
      repetition_penalty = 1.2,
      top_k = 5,
      max_new_tokens = 120,
      use_contrastive = TRUE
    )
  }
}

