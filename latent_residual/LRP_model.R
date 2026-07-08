source("utils/atomic_blocks.R")

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
      logits <- torch_matmul(pred, self$tok_emb$weight$t())
      loss <- nnf_cross_entropy(
        logits$view(c(-1, self$tok_emb$weight$size(1))),
        y$view(c(-1))
      )
      res$loss <- loss
    }
    
    return(res)
  }
)