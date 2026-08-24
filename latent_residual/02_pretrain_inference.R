# =====================================================================
# 推理 (Generate)
# =====================================================================

source("config.R")
source("latent_residual/LRP_model.R")
source("utils/BPETokenizer.R")

tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

# 实例化模型
raw_model <- RtomicLRP(
  vocab_size = VOCAB_SIZE,
  dim = DIM,
  n_layers = N_LAYERS,
  n_heads = N_HEADS,
  max_seq_len = SEQ_LEN
)

device <- torch_device("cpu")
raw_model <- raw_model$to(device = device)

generate_text <- function(model, tokenizer, prompt, max_new_tokens = 50,
                          temperature = 0.8, top_k = 10, rep_penalty = 1.2) {
  device <- model$tok_emb$weight$device
  raw_ids <- tokenizer$encode_raw(prompt)[[1]]
  current_ids <- c(tokenizer$bos_idx, raw_ids)
  
  cat(sprintf("\n[输入 Prompt]: %s\n", prompt))
  cat("[模型生成中]: ")
  
  with_no_grad({
    for (i in 1:max_new_tokens) {
      input_seq <- tail(current_ids, SEQ_LEN)
      x_tensor <- torch_tensor(input_seq, dtype = torch_long(), device = device)$unsqueeze(1)
      seq_len <- x_tensor$size(2)
      
      h <- model$tok_emb(x_tensor)
      
      for (j in 1:length(model$layers)) {
        h <- model$layers[[j]](h)
      }
      
      h <- model$norm_f(h)
      semantic_delta <- model$predictor(h)
      pred <- h + semantic_delta
      
      last_pred <- pred[, seq_len, ]
      logits <- torch_matmul(last_pred, model$tok_emb$weight$t()) / temperature
      logits <- logits$squeeze(1)
      
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
      
      # Top-K
      if (top_k > 0) {
        topk_res <- torch_topk(logits, k = top_k)
        kth_value <- topk_res[[1]][top_k]
        logits <- torch_where(logits < kth_value, torch_tensor(-Inf, device = device), logits)
      }
      
      # 采样
      probs <- nnf_softmax(logits, dim = -1)
      next_token_id <- as.integer(torch_multinomial(probs, num_samples = 1))
      
      if (!is.null(tokenizer$eos_idx) && next_token_id == tokenizer$eos_idx) {
        break
      }
      
      # 1. 逐字打印时，关闭 clean 避免空格被吃掉或触发正则
      next_word_raw <- tokenizer$decode(next_token_id, clean = FALSE)
      if (next_word_raw == "<EOS>") break
      
      cat(next_word_raw)
      flush.console()
      
      current_ids <- c(current_ids, next_token_id)
    }
  })
  
  cat("[EOS]。\n")
  invisible(current_ids)
}

# --- 配置检查点与提示词 ---
checkpoints_to_test <- c(
  "checkpoints/lrp_03.pt"
)

test_prompts <- c(
  # 1. 常识排比测试 (看它能不能接上“英国”或“伦敦”)
  "可以这样理解损失函数，它是",
  
  # 2. 人物传记 (看它能不能接出后续中文)
  "数据科学可以帮助企业",
  
  # 3. 极度死板的硬知识测试 (不给前文，直接测事实记忆)
  "人工智能是一种",
  
  # 4. 数学/逻辑递增测试 (看它能不能接上 5 或者 6)
  "谷歌是",
  "支持向量机是"
)
  

for (ckpt_path in checkpoints_to_test) {
  cat("\n")
  cat(sprintf("正在评估模型版本: %s\n", ckpt_path))
  checkpoint_data <- torch_load(ckpt_path)
  raw_model$load_state_dict(checkpoint_data$model)
  raw_model$to(dtype = torch_float32())
  raw_model$eval()

  for (prompt in test_prompts) {
    generate_text(raw_model, tokenizer, prompt = prompt, max_new_tokens = 100,
                  temperature = 0.3, top_k = 5, rep_penalty = 1.15)
  }
}


