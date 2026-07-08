# causal_lm/06_self_verifier.R
# =====================================================================
# 方案 B：外挂验证器 (Verifier) —— 语义交叉质检管道
# =====================================================================

source("config.R")
source("utils/BPETokenizer.R")
source("causal_lm/CausalLM_model.R")

device <- torch_device("cpu") 
TARGET_CKPT <- "checkpoints/causal_sft_epoch_05.pt"
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

model <- RtomicCausalLM(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN)

if (file.exists(TARGET_CKPT)) {
  cat(sprintf("正在挂载 SFT 目标权重: %s\n", TARGET_CKPT))
  model$load_state_dict(torch_load(TARGET_CKPT))
} else {
  stop("找不到指定的微调 Checkpoint 文件，请确认路径！")
}

causal_model <- model$to(device = device)
causal_model$eval()

# =====================================================================
# 基础生成闭包 (支持指定温度)
# =====================================================================
generate_raw_text <- function(model, tokenizer, prompt, max_new_tokens = 40, temperature = 1.0) {
  raw_ids <- tokenizer$encode_raw(prompt)[[1]]
  current_ids <- c(tokenizer$bos_idx, raw_ids)
  eos_val <- if (!is.null(tokenizer$eos_idx)) tokenizer$eos_idx else 4L
  
  with_no_grad({
    for (i in 1:max_new_tokens) {
      input_seq <- tail(current_ids, SEQ_LEN)
      x_tensor <- torch_tensor(input_seq, dtype = torch_long(), device = device)$unsqueeze(1)
      
      # 走标准的默认前向传播，不耗费显存提取隐藏状态
      output <- model(list(x = x_tensor, y = x_tensor, loss_mask = NULL), output_hidden_states = FALSE)
      
      logits_vec <- output$logits[1, output$logits$size(2), ] / temperature
      probs <- nnf_softmax(logits_vec, dim = -1)
      next_token_id <- as.integer(torch_multinomial(probs, num_samples = 1))
      
      if (next_token_id == eos_val) break
      current_ids <- c(current_ids, next_token_id)
    }
  })
  
  generated_ids <- current_ids[(length(raw_ids) + 2):length(current_ids)]
  return(tokenizer$decode(generated_ids))
}

# =====================================================================
# 核心验证器：利用条件概率计算句子之间的“语义互相认可度”
# =====================================================================
compute_verification_score <- function(model, tokenizer, prompt, candidate_text) {
  # 构造验证判据：将 prompt 和候选答案拼接，看模型对其整体的条件对数似然
  full_text <- paste0(prompt, candidate_text)
  raw_ids <- tokenizer$encode_raw(full_text)[[1]]
  input_ids <- c(tokenizer$bos_idx, raw_ids)
  
  x_tensor <- torch_tensor(input_ids, dtype = torch_long(), device = device)$unsqueeze(1)
  
  with_no_grad({
    output <- model(list(x = x_tensor, y = x_tensor, loss_mask = NULL), output_hidden_states = FALSE)
    logits <- output$logits[1, , ]
  })
  
  # 计算生成的答案部分的 log-likelihood (对数似然)
  # 似然值越高，说明模型在潜意识里越认可这句话的逻辑连贯性
  prompt_len <- length(tokenizer$encode_raw(prompt)[[1]]) + 1
  total_len <- length(input_ids)
  
  log_prob_sum <- 0
  count <- 0
  
  for (t in prompt_len:(total_len - 1)) {
    next_token_actual <- input_ids[t + 1]
    token_logits <- logits[t, ]
    token_probs <- nnf_log_softmax(token_logits, dim = -1)
    
    # 累加实际输出 token 的对数概率
    log_prob_sum <- log_prob_sum + token_probs[next_token_actual]$item()
    count <- count + 1
  }
  
  # 返回平均对数似然（消除句子长度的影响）
  return(log_prob_sum / max(count, 1))
}

# =====================================================================
# 外挂验证智能路由管道
# =====================================================================
generate_reliable_text_via_verifier <- function(model, tokenizer, prompt, n_candidates = 3) {
  cat(sprintf("\n[输入 Prompt]: %s\n", prompt))
  cat(sprintf("=> 正在外挂验证器管道中并行生成 %d 个候选样本...\n", n_candidates))
  
  candidates <- character(n_candidates)
  scores <- numeric(n_candidates)
  
  # 1. 蒙特卡洛高随机性采样生成
  for (i in 1:n_candidates) {
    # 保持 T=1.1 轻微热运动，促使模型吐出不同的逻辑链
    candidates[i] <- generate_raw_text(model, tokenizer, prompt, temperature = 1.1)
    cat(sprintf("   候选样本 [%d]: %s\n", i, candidates[i]))
  }
  
  # 2. 验证器进行客观统计审计（对数似然评估）
  cat("=> 验证器开始进行语义合规性与逻辑审查...\n")
  for (i in 1:n_candidates) {
    scores[i] <- compute_verification_score(model, tokenizer, prompt, candidates[i])
    cat(sprintf("   审查得分 (对数似然) [%d]: %.4f\n", i, scores[i]))
  }
  
  # 3. 筛选放行
  best_idx <- which.max(scores)
  best_score <- scores[best_idx]
  best_text <- candidates[best_idx]
  
  # 建立一个统计风控阈值（比如 -3.5，具体数值可以根据你模型的实际输出观察微调）
  # 对数似然是负数，越接近 0 越好。如果即便是最好的样本得分也极低（如 -6.5），说明都在瞎编
  hugging_threshold <- -4.0
  
  cat("\n--- 验证器最终裁决 ---\n")
  if (best_score < hugging_threshold) {
    cat("\033[31m[风控拦截] 所有候选样本均未通过审查（逻辑混乱/幻觉过高），系统执行安全风控拒答。\033[0m\n")
    return("（该问题涉及知识盲区，由于无法确保准确性，系统已拦截输出。）")
  } else {
    cat(sprintf("\033[32m[放行通过] 选中样本 [%d]，其逻辑连贯性最强，事实置信度最高。\033[0m\n", best_idx))
    cat(sprintf("[最终输出]: %s\n", best_text))
    return(best_text)
  }
}

# --- 运行验证流程 ---
test_prompt <- "人工智能应用非常广泛，"
final_res <- generate_reliable_text_via_verifier(causal_model, tokenizer, test_prompt, n_candidates = 3)
