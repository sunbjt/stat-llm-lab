source("utils/atomic_blocks.R")

# =====================================================================
# Pure Token 级潜语义对比模型 (Pure InfoNCE - Fixed R Keyword Error)
# =====================================================================

TokenLatentModel <- nn_module(
  "TokenLatentModel",

  initialize = function(
    vocab_size,
    dim,
    n_layers,
    n_heads,
    max_seq_len,
    temperature = 0.07
  ) {
    self$tok_emb <- nn_embedding(vocab_size, dim)

    self$layers <- nn_module_list(
      lapply(
        1:n_layers,
        function(i) RtomicBlock(dim, n_heads, max_seq_len)
      )
    )

    self$norm_f <- RMSNorm(dim)

    # 非线性 Predictor Projection
    self$predictor <- nn_sequential(
      nn_linear(dim, dim * 2),
      RMSNorm(dim * 2),
      nn_silu(),
      nn_linear(dim * 2, dim),
      RMSNorm(dim),
      nn_silu(),
      nn_linear(dim, dim)
    )

    self$temperature <- nn_buffer(torch_tensor(temperature))
  },

  encode = function(x_tokens) {
    h <- self$tok_emb(x_tokens)

    for (i in 1:length(self$layers)) {
      h <- self$layers[[i]](h)
    }

    self$norm_f(h)
  },

  forward = function(input_data) {
    x_tokens <- input_data$x  # [B, S]
    y_tokens <- input_data$y  # [B, S]

    B <- x_tokens$size(1)
    S <- x_tokens$size(2)
    device <- x_tokens$device

    # 1. Online Encoder 提取特征
    h <- self$encode(x_tokens)            # [B, S, D]
    pred <- self$predictor(h)            # [B, S, D]

    # 2. Target Representation (解冻 Embedding，允许梯度更新)
    target_emb <- self$tok_emb(y_tokens) # [B, S, D]

    # L2 正则化归一化
    pred_norm <- nnf_normalize(pred, p = 2, dim = -1)
    target_norm <- nnf_normalize(target_emb, p = 2, dim = -1)

    # [B, S, D] -> [S, B, D]
    pred_t <- pred_norm$permute(c(2, 1, 3))     # [S, B, D]
    target_t <- target_norm$permute(c(2, 1, 3)) # [S, B, D]

    # [S, B, D] x [S, D, B] -> [S, B, B] 余弦相似度矩阵
    sim <- torch_bmm(pred_t, target_t$transpose(2, 3)) / self$temperature

    # ---------------------------------------------------------------
    # 3. 消除 False Negatives (平滑 Mask 机制)
    # ---------------------------------------------------------------
    y_t <- y_tokens$transpose(1, 2) # [S, B]
    same_token_mask <- (y_t$unsqueeze(3) == y_t$unsqueeze(2)) # [S, B, B]
    
    diag_mask <- torch_eye(B, dtype = torch_bool(), device = device)$unsqueeze(1)$expand(c(S, B, B))
    false_negative_mask <- same_token_mask & (!diag_mask)

    # 填入 -100.0 防止 Softmax 极小值数值溢出与梯度暴胀
    sim$masked_fill_(false_negative_mask, -100.0)

    # ---------------------------------------------------------------
    # 4. 正确对齐 1-based 标签与展平形状
    # ---------------------------------------------------------------
    # 基础标签 [B]: 1, 2, ..., B
    labels_per_seq <- torch_arange(1, B, dtype = torch_long(), device = device)

    # 关键修复：使用 `repeat` 避开 R 保留字语法解析错误
    labels <- labels_per_seq$`repeat`(c(S))

    # 展平相似度矩阵 [S * B, B]
    sim_flat <- sim$reshape(c(S * B, B))

    # 计算全局单节点平均 Cross Entropy Loss
    contrastive_loss <- nnf_cross_entropy(sim_flat, labels)

    list(
      loss = contrastive_loss,
      contrastive = contrastive_loss
    )
  }
)