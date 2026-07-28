source("Distillation/atomic_blocks.R")

# =====================================================================
# 分块交叉熵损失 (Chunked Cross-Entropy)
# 沿词表维度分块计算 log-sum-exp，避免显存中物化完整的 [N, V] logits 矩阵
# 对于 Qwen 15 万词表，可将峰值显存从 ~20 GB 降至 ~200 MB
# =====================================================================
chunked_cross_entropy <- function(hidden, embed_weight, targets, chunk_size = 32768L) {
  # hidden:        [N, D] 展平后的隐状态
  # embed_weight:  [V, D] 词嵌入权重矩阵 (与 tok_emb 共享)
  # targets:       [N]    目标 token ID (R torch 1-based 索引)
  # chunk_size:    每次处理的词表片段大小

  V <- embed_weight$size(1)
  N <- hidden$size(1)
  device <- hidden$device

  # ---- 阶段 1: 分块 Log-Sum-Exp (数值稳定版) ----
  # 在 fp32 下累积，防止 float16 下的上溢/下溢
  max_log <- torch_full(c(N, 1), -Inf, dtype = torch_float32(), device = device)
  sum_exp <- torch_zeros(c(N, 1), dtype = torch_float32(), device = device)

  chunk_starts <- seq(1, V, by = chunk_size)
  for (start in chunk_starts) {
    end <- min(start + chunk_size - 1, V)
    chunk_w <- embed_weight[start:end, ]                 # [C, D]
    chunk_logits <- torch_matmul(hidden, chunk_w$t())    # [N, C]
    chunk_logits_fp32 <- chunk_logits$to(dtype = torch_float32())

    chunk_max <- chunk_logits_fp32$max(dim = 2, keepdim = TRUE)[[1]]  # [N, 1]

    # Element-wise maximum: R torch 的 torch_max(a, b) 会将 b 误解析为 dim 参数，
    # 因此改用 cat + max 的组合 (临时 [N, 2] 张量，显存开销极小)
    combined <- torch_cat(list(max_log, chunk_max), dim = 2)
    new_max <- combined$max(dim = 2, keepdim = TRUE)[[1]]  # [N, 1]

    # 重新缩放旧的 sum_exp 并加入当前块的 exp
    sum_exp <- sum_exp * torch_exp(max_log - new_max) +
               torch_exp(chunk_logits_fp32 - new_max)$sum(dim = 2, keepdim = TRUE)
    max_log <- new_max

    # 释放中间张量，控制显存峰值
    rm(chunk_w, chunk_logits, chunk_logits_fp32, chunk_max, combined)
  }

  # ---- 阶段 2: 直接点积获取目标 logit ----
  # embed_weight 使用 1-based R 索引 (与 nn_embedding 对齐)
  target_emb <- embed_weight[targets, drop = FALSE]               # [N, D]
  target_logits <- torch_sum(hidden * target_emb, dim = 2, keepdim = TRUE)  # [N, 1]
  target_logits_fp32 <- target_logits$to(dtype = torch_float32())

  # ---- 阶段 3: 计算最终 Loss ----
  log_sum_exp <- max_log + torch_log(sum_exp + 1e-10)
  loss <- (log_sum_exp - target_logits_fp32)$mean()

  return(loss)
}

# =====================================================================
# 主模型组装 (隐空间残差预测器 LRP)
# =====================================================================
RtomicLRP <- nn_module(
  "RtomicLRP",
  initialize = function(vocab_size, dim, n_layers, n_heads, max_seq_len) {
    self$tok_emb <- nn_embedding(vocab_size, dim)
    nn_init_normal_(self$tok_emb$weight, mean = 0, std = 0.02)
    
    # 将 max_seq_len 传递给子 Block 交由 RoPE 处理
    self$layers <- nn_module_list(lapply(1:n_layers, function(i) RtomicBlock(dim, n_heads, max_seq_len)))
    
    self$norm_f <- RMSNorm(dim)
    self$predictor <- nn_sequential(
      nn_linear(dim, dim),
      nn_silu(),
      nn_linear(dim, dim)
    )
    
    nn_init_zeros_(self$predictor[[3]]$weight)
    if (!is.null(self$predictor[[3]]$bias)) {
      nn_init_zeros_(self$predictor[[3]]$bias)
    }
  },
  
  forward = function(input_data) {
    x <- input_data$x
    y <- input_data$y # 在纯部署/生成阶段，y 可能为空 (NULL)
    device <- x$device
    
    # RoPE 负责位置信息，此处仅需进行 Token Embedding
    h <- self$tok_emb(x)
    
    for (i in 1:length(self$layers)) h <- self$layers[[i]](h)
    
    h <- self$norm_f(h)
    semantic_delta <- self$predictor(h)
    pred <- h + semantic_delta
    
    # 构建基础返回列表
    res <- list(pred = pred)
    
    # 只要提供了目标标签 y (无论是 Training 还是 Validation)，就必须算 Loss
    if (!is.null(y)) {
      # 分块交叉熵：沿词表维度分片计算，避免物化 [B*S, 15万] 的巨型 logits 矩阵
      pred_flat <- pred$view(c(-1, pred$size(3)))  # [B*(S-1), D]
      y_flat <- y$view(c(-1))                        # [B*(S-1)]
      loss <- chunked_cross_entropy(pred_flat, self$tok_emb$weight, y_flat)
      res$loss <- loss
    }
    
    return(res)
  }
)