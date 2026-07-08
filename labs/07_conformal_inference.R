# causal_lm/07_conformal_inference.R
# =====================================================================
# Decode-Only 架构推理 — Top-p (Nucleus Sampling) 附带不确定性监控
# =====================================================================
# 注意：本实验本质是 Top-p 采样，并非严格意义上的共形预测 (Conformal
# Prediction)。真正的 CP 需要独立的校准集计算非一致性分位数。这里的
# set_size 反映分布平坦度，是启发式不确定性信号，不具备有限样本覆盖保证。

source("config.R")
source("utils/BPETokenizer.R")
source("causal_lm/CausalLM_model.R")

tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

causal_model <- RtomicCausalLM(
  vocab_size = VOCAB_SIZE,
  dim = DIM,
  n_layers = N_LAYERS,
  n_heads = N_HEADS,
  max_seq_len = SEQ_LEN
)

device <- torch_device("cpu")
causal_model <- causal_model$to(device = device)

# =====================================================================
# Top-p (Nucleus Sampling) — 从累积概率达到 1-alpha 的集合中采样
# 同时返回集合大小作为不确定性信号
# =====================================================================
sample_top_p <- function(logits, alpha = 0.1) {
  probs <- nnf_softmax(logits, dim = -1)

  sort_res <- torch_sort(probs, descending = TRUE)
  sorted_probs <- sort_res[[1]]
  sorted_indices <- sort_res[[2]]

  cumsum_probs <- torch_cumsum(sorted_probs, dim = -1)
  mask <- cumsum_probs > (1 - alpha)

  if (mask$any()$item()) {
    cutoff_idx <- mask$nonzero()[1, 1]$item()
  } else {
    cutoff_idx <- sorted_probs$size(1)
  }

  cp_indices <- sorted_indices[1:cutoff_idx]
  cp_probs <- sorted_probs[1:cutoff_idx]
  set_size <- cutoff_idx

  renorm_probs <- cp_probs / cp_probs$sum()
  sample_idx <- torch_multinomial(renorm_probs, num_samples = 1)$item()
  next_token_id <- cp_indices[sample_idx]$item()

  # 注意：这里的占比相对于 vocab_size 而非 top_k，
  # 因为此函数接收的是原始 logits（未经截断）
  vocab_size <- sorted_probs$size(1)
  return(list(
    next_token_id = next_token_id,
    set_size = set_size,
    set_size_pct = set_size / vocab_size * 100
  ))
}

# =====================================================================
# 生成文本 (附带不确定性监控)
# =====================================================================
generate_with_top_p <- function(model,
                                tokenizer,
                                prompt,
                                max_new_tokens = 50,
                                temperature = 1.0,
                                alpha = 0.1,
                                top_k = 0,
                                repetition_penalty = 1.0) {

  raw_ids <- tokenizer$encode_raw(prompt)[[1]]
  current_ids <- c(tokenizer$bos_idx, raw_ids)
  eos_val <- if (!is.null(tokenizer$eos_idx)) tokenizer$eos_idx else 4L

  tokens <- character(0)
  set_sizes <- integer(0)
  set_size_pcts <- numeric(0)

  cat(sprintf("\n[输入 Prompt]: %s\n", prompt))

  with_no_grad({
    for (i in 1:max_new_tokens) {
      input_seq <- tail(current_ids, SEQ_LEN)
      x_tensor <- torch_tensor(input_seq, dtype = torch_long(), device = device)$unsqueeze(1)

      output <- model(list(x = x_tensor, y = NULL, loss_mask = NULL))
      seq_length <- output$logits$size(2)
      logits_raw <- output$logits[1, seq_length, ]$clone()

      # === 步骤 1：基于原始 logits 计算不确定性（不受采样策略污染） ===
      uncertainty <- sample_top_p(logits_raw, alpha = alpha)
      set_size <- uncertainty$set_size
      set_size_pct <- uncertainty$set_size_pct

      # === 步骤 2：对 logits 应用启发式采样策略 ===
      logits_sampling <- logits_raw

      if (repetition_penalty != 1.0) {
        for (idx in unique(current_ids)) {
          s <- logits_sampling[idx]$item()
          if (s > 0) {
            logits_sampling[idx] <- s / repetition_penalty
          } else {
            logits_sampling[idx] <- s * repetition_penalty
          }
        }
      }

      vocab_size <- logits_sampling$size(1)
      if (top_k > 0 && top_k < vocab_size) {
        topk_res <- torch_topk(logits_sampling, k = top_k)
        kth_val <- topk_res[[1]][top_k]
        logits_sampling$masked_fill_(logits_sampling < kth_val, -Inf)
      }

      logits_sampling <- logits_sampling / temperature

      # === 步骤 3：从修改后的分布中实际采样 ===
      probs <- nnf_softmax(logits_sampling, dim = -1)
      next_token_id <- as.integer(torch_multinomial(probs, num_samples = 1))

      if (next_token_id == eos_val) break

      next_word <- tokenizer$decode(next_token_id)
      if (next_word == "<EOS>") break

      tokens <- c(tokens, next_word)
      set_sizes <- c(set_sizes, set_size)
      set_size_pcts <- c(set_size_pcts, set_size_pct)
      current_ids <- c(current_ids, next_token_id)
    }
  })

  cat("--- 生成文本 ---\n")
  cat(paste0(tokens, collapse = ""), "\n")

  if (length(tokens) > 0) {
    cat(sprintf("\n--- Top-p 不确定性监控 (p = %.2f) ---\n", 1 - alpha))
    cat(sprintf("总 token 数: %d\n", length(tokens)))
    cat(sprintf("集合大小 — 最小: %.0f  中位数: %.0f  最大: %.0f\n",
                min(set_sizes), median(set_sizes), max(set_sizes)))
    cat(sprintf("集合占比 — 最小: %.1f%%  中位数: %.1f%%  最大: %.1f%%\n",
                min(set_size_pcts), median(set_size_pcts), max(set_size_pcts)))

    n_high  <- sum(set_size_pcts > 3)
    n_warn  <- sum(set_size_pcts > 1 & set_size_pcts <= 3)
    n_safe  <- sum(set_size_pcts <= 1)
    cat(sprintf("高风险 (占比 > 3%%):       %d (%.0f%%)\n", n_high, 100 * n_high / length(tokens)))
    cat(sprintf("不确定 (1%% < 占比 <= 3%%): %d (%.0f%%)\n", n_warn, 100 * n_warn / length(tokens)))
    cat(sprintf("正常 (占比 <= 1%%):      %d (%.0f%%)\n", n_safe, 100 * n_safe / length(tokens)))

    risky_idx <- which(set_size_pcts > 1)
    if (length(risky_idx) > 0) {
      cat(sprintf("\n--- 风险 token 详情 (%d 个) ---\n", length(risky_idx)))
      for (j in risky_idx) {
        risk_label <- if (set_size_pcts[j] > 3) "HIGH" else "MID "
        cat(sprintf("  [%s] %5.1f%% | size=%4.0f | \"%s\"\n",
                    risk_label, set_size_pcts[j], set_sizes[j], tokens[j]))
      }
    }
  }
  cat("\n")

  invisible(current_ids)
}

# --- 运行测试 ---
ckpt_path <- "checkpoints/causal_model_03.pt"
checkpoint_data <- torch_load(ckpt_path)
state <- if (!is.null(checkpoint_data$model)) checkpoint_data$model else checkpoint_data
causal_model$load_state_dict(state)
causal_model$eval()

test_prompts <- c(
  "数据科学是一项对于企业非常有用的技术，它能够",
  "我认为 AI 未来对人类有非常多的好处，理由如下",
  "在大学里应该这样学习编程，首先"
)

for (prompt in test_prompts) {
  generate_with_top_p(
    causal_model,
    tokenizer,
    prompt = prompt,
    max_new_tokens = 100,
    temperature = 0.5,
    top_k = 50,
    repetition_penalty = 1.15,
    alpha = 0.1
  )
}

# =====================================================================
# 方法总结
# =====================================================================
# Top-p (Nucleus Sampling) 附带不确定性监控：
#
# 1. 核心机制：
#    - 对原始 logits 做 softmax，降序累积概率直到超过 1-alpha (p)，
#      需要的 token 数量 = set_size，反映模型对该位置的"犹豫程度"。
#    - 本质是 Top-p 采样，不是共形预测。缺少校准集和分位数计算，
#      不具备有限样本边际覆盖保证。
#
# 2. 解耦设计：
#    - 不确定性 (set_size) 始终从原始 logits 计算，不受 top-k /
#      temperature / repetition_penalty 污染。
#    - 采样从修改后的 logits 进行，两者互不干扰。
#
# 3. 阈值说明（基于 set_size_pct = set_size / vocab_size * 100）：
#    - 高风险：占比 > 3%（约 500 tokens，分布极平）
#    - 不确定：1% < 占比 <= 3%
#    - 正常：占比 <= 1%（约 164 tokens 以内即达 p 阈值）
#    - 小模型分布天然较平坦，不开启 top-k 时占比偏高属正常现象
