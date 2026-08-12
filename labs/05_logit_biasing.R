# =====================================================================
# SFT 推理 + 统计概率漂移电子签名 (Logit Biasing Watermark)
# =====================================================================

source("config.R")
# 自动降级探测：优先 CUDA，其次 Mac MPS，最后 CPU
device <- torch_device("cpu")
cat(sprintf("SFT 推理引擎启动，当前使用加速设备: %s\n", device$type))

source("latent_residual/LRP_model.R")
source("utils/BPETokenizer.R")
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

# =====================================================================
# 1. 极简模型加载
# =====================================================================
model <- RtomicLRP(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN)

TARGET_CKPT <- "checkpoints/lrp_sft_epoch_05.pt"
if (file.exists(TARGET_CKPT)) {
  cat(sprintf("正在挂载 SFT 权重: %s\n", TARGET_CKPT))
  ckpt <- torch_load(TARGET_CKPT)
  state_dict <- if ("model" %in% names(ckpt)) ckpt$model else ckpt
  model$load_state_dict(state_dict, strict = FALSE)
} else {
  stop("找不到指定的 Checkpoint 文件！请检查路径。")
}

model <- model$to(device = device)
model$eval()

# =====================================================================
# 2. 电子签名核心逻辑：伪随机绿名单划分
# =====================================================================
get_green_list <- function(context_tokens, vocab_size, key = 421337L, gamma = 0.5) {
  # 保护并还原全局随机种子，避免干扰其他采样流程
  old_seed <- get0(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  on.exit({
    if (!is.null(old_seed)) assign(".Random.seed", old_seed, envir = .GlobalEnv)
  })
  
  # 提取上下文哈希值作为伪随机种子
  h <- sum(as.numeric(context_tokens) * seq_along(context_tokens)) + key
  set.seed(abs(as.integer(h %% 2147483647L)))
  
  # 动态确定绿名单 Token 索引集合
  green_size <- floor(vocab_size * gamma)
  green_indices <- sample.int(vocab_size, size = green_size)
  return(green_indices)
}

# =====================================================================
# 1. 生成函数 (先加水印，再缩放 Temperature)
# =====================================================================
generate_response <- function(model, tokenizer, prompt,
                              max_new_tokens = 500,
                              temperature = 0.8,
                              top_k = 10,
                              rep_penalty = 1.1,
                              use_watermark = TRUE,
                              watermark_delta = 2.0,
                              watermark_gamma = 0.5,
                              watermark_key = 421337L,
                              watermark_k = 1L) {
  
  eos_val <- if (!is.null(tokenizer$eos_idx)) tokenizer$eos_idx else 2L
  p_ids <- tokenizer$encode_raw(prompt)[[1]]
  prompt_ids <- c(tokenizer$bos_idx, p_ids)
  current_ids <- prompt_ids
  gen_ids <- integer(0) # 仅记录模型生成的 Token
  
  cat(sprintf("\n[User]: %s\n[Assistant]: ", prompt))
  
  with_no_grad({
    for (step in 1:max_new_tokens) {
      input_seq <- tail(current_ids, SEQ_LEN)
      x_tensor <- torch_tensor(input_seq, dtype = torch_long(), device = device)$unsqueeze(1)
      
      output <- model(list(x = x_tensor, y = NULL))
      seq_length <- x_tensor$size(2)
      last_pred <- output$pred[1, seq_length, ]
      
      cb <- model$tok_emb$weight
      logits <- torch_matmul(last_pred$unsqueeze(1), cb$t())$squeeze(1)
      
      # 注入水印偏置
      if (use_watermark && length(current_ids) >= watermark_k) {
        context_tokens <- tail(current_ids, watermark_k)
        green_indices <- get_green_list(
          context_tokens = context_tokens,
          vocab_size = VOCAB_SIZE,
          key = watermark_key,
          gamma = watermark_gamma
        )
        logits[green_indices] <- logits[green_indices] + watermark_delta
      }
      
      # 再应用 Temperature 缩放
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
      
      probs <- nnf_softmax(logits, dim = -1)
      next_token_id <- as.integer(torch_multinomial(probs, num_samples = 1))
      
      if (next_token_id == eos_val) break
      new_text <- tokenizer$decode(next_token_id)
      if (new_text == "<EOS>") break
      
      cat(new_text)
      flush.console()
      
      current_ids <- c(current_ids, next_token_id)
      gen_ids <- c(gen_ids, next_token_id)
    }
  })
  
  cat("\n--------------------------------------------------\n")
  invisible(list(prompt_ids = prompt_ids, gen_ids = gen_ids))
}

# =====================================================================
# 2. 检测函数 (隔离 Prompt，仅检测模型生成部分)
# =====================================================================
verify_watermark <- function(prompt_ids, gen_ids, vocab_size = VOCAB_SIZE,
                             watermark_key = 421337L, watermark_gamma = 0.5,
                             watermark_k = 1L, z_threshold = 2.0) { # 将短文本阈值调整为 2.0 (97.7% 置信度)
  
  full_ids <- c(prompt_ids, gen_ids)
  prompt_len <- length(prompt_ids)
  n_total <- length(gen_ids)
  
  if (n_total <= 0) {
    cat("生成文本为空，无法验证。\n")
    return(NULL)
  }
  
  green_hits <- 0
  for (i in seq_along(gen_ids)) {
    # 算上下文时把 Prompt 作为前置条件，但只统计 gen_ids 本身
    curr_idx <- prompt_len + i
    context <- full_ids[(curr_idx - watermark_k):(curr_idx - 1)]
    target <- gen_ids[i]
    
    green_indices <- get_green_list(
      context_tokens = context,
      vocab_size = vocab_size,
      key = watermark_key,
      gamma = watermark_gamma
    )
    
    if (target %in% green_indices) {
      green_hits <- green_hits + 1
    }
  }
  
  expected_hits <- n_total * watermark_gamma
  std_dev <- sqrt(n_total * watermark_gamma * (1 - watermark_gamma))
  z_score <- (green_hits - expected_hits) / std_dev
  p_value <- 1 - pnorm(z_score)
  
  cat(sprintf("[签名验证报告]\n"))
  cat(sprintf("仅统计生成 Token 数量: %d\n", n_total))
  cat(sprintf("命中绿名单次数: %d (实际占比: %.2f%%, 期望占比: %.2f%%)\n", 
              green_hits, (green_hits / n_total) * 100, watermark_gamma * 100))
  cat(sprintf("Z-Score 统计量: %.4f\n", z_score))
  cat(sprintf("显著性 p-value: %e\n", p_value))
  cat(sprintf("最终判定: %s\n", ifelse(z_score >= z_threshold, "【包含有效电子签名】", "【未检测到签名/人类撰写】")))
  
  invisible(list(z_score = z_score, p_value = p_value, hit_rate = green_hits / n_total))
}

prompt <- "简单介绍一下数据科学。"

res <- generate_response(model, tokenizer, prompt = prompt,
                         max_new_tokens = 128, 
                         temperature = 0.3, 
                         top_k = 10, 
                         rep_penalty = 1.2,
                         use_watermark = TRUE,
                         watermark_delta = 2.0,
                         watermark_key = 421337L)

# 传入分列的 prompt_ids 与 gen_ids
verify_watermark(res$prompt_ids, res$gen_ids, watermark_key = 421337L)
