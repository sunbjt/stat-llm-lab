# =====================================================================
# Decode-Only 架构推理 (Generate)
# =====================================================================

source("config.R")
source("utils/BPETokenizer.R")
source("causal_lm/CausalLM_model.R")

tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

# 实例化模型
causal_model <- RtomicCausalLM(
  vocab_size = VOCAB_SIZE,
  dim = DIM,
  n_layers = N_LAYERS,
  n_heads = N_HEADS,
  max_seq_len = SEQ_LEN
)

device <- torch_device("cpu")
causal_model <- causal_model$to(device = device)

# 带有温度、Top-K 和重复惩罚的采样函数
sample_with_penalty <- function(logits,
                                current_ids,
                                top_k = 10,
                                rep_penalty = 1.15) {
  for (id in unique(current_ids)) {
    val <- logits[id]$item()
    logits[id] <- if (val > 0)
      val / rep_penalty
    else
      val * rep_penalty
  }
  
  if (length(logits) > top_k) {
    kth <- torch_topk(logits, k = top_k)[[1]][top_k]$item()
    logits[logits < kth] <- -Inf
  }
  
  probs <- nnf_softmax(logits, dim = -1)
  as.integer(torch_multinomial(probs, num_samples = 1))
}

generate_causal_text <- function(model,
                                 tokenizer,
                                 prompt,
                                 max_new_tokens = 50,
                                 temperature = 0.3,
                                 rep_penalty = 1.15,
                                 top_k = 10) {
  raw_ids <- tokenizer$encode_raw(prompt)[[1]]
  current_ids <- c(tokenizer$bos_idx, raw_ids)
  eos_val <- if (!is.null(tokenizer$eos_idx)) tokenizer$eos_idx else 4L
  
  cat(sprintf("\n[输入 Prompt]: %s\n[模型生成]: ", prompt))
  
  with_no_grad({
    for (i in 1:max_new_tokens) {
      input_seq <- tail(current_ids, SEQ_LEN)
      x_tensor <- torch_tensor(input_seq, dtype = torch_long(), device = device)$unsqueeze(1)
      
      output <- model(list(
        x = x_tensor,
        y = x_tensor,
        loss_mask = NULL
      ))
      
      # 提取序列最后一个 token 的 logits
      seq_length <- output$logits$size(2)
      logits_vec <- output$logits[1, seq_length, ]
      logits_vec <- logits_vec / temperature
      
      next_token_id <- sample_with_penalty(logits_vec, current_ids,
                                           top_k = top_k, rep_penalty = rep_penalty)
      
      if (next_token_id == eos_val) {
        cat(" [EOS]")
        break
      }
      
      next_word <- tokenizer$decode(next_token_id, clean = FALSE)
      if (next_word == "<EOS>")
        break
      
      cat(next_word)
      flush.console()
      
      current_ids <- c(current_ids, next_token_id)
    }
  })
  cat("\n")
  invisible(current_ids)
}

# --- 运行测试 ---
ckpt_path <- "checkpoints/causal_model_02.pt"
checkpoint_data <- torch_load(ckpt_path)
state <- if (!is.null(checkpoint_data$model)) checkpoint_data$model else checkpoint_data
causal_model$load_state_dict(state)
causal_model$eval()
  
test_prompts <- c(
  # 1. 常识排比测试 (看它能不能接上“英国”或“伦敦”)
  "中国的首都是北京，日本的首都是东京，法国的首都是",
  "可以这样理解损失函数，它是",
  
  # 2. 人物传记 (看它能不能接出后续中文)
  "数据科学可以帮助企业",
  
  # 3. 极度死板的硬知识测试 (不给前文，直接测事实记忆)
  "人工智能是一种",
  
  # 4. 数学/逻辑递增测试 (看它能不能接上 5 或者 6)
  "在家庭教育中，父母是",
  "支持向量机是"
)
  
for (prompt in test_prompts) {
  generate_causal_text(
    causal_model,
    tokenizer,
    prompt = prompt,
    max_new_tokens = 150,
    temperature = 0.45,
    rep_penalty = 1.2
  )
}

