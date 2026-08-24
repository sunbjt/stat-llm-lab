source("utils/atomic_blocks.R")

# =====================================================================
# Pure Token 级潜语义对比模型 (Pure InfoNCE with Masked Negatives)
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
    x_tokens <- input_data$x
    y_tokens <- input_data$y

    B <- x_tokens$size(1)
    S <- x_tokens$size(2)
    device <- x_tokens$device

    # 1. Online Encoder 提取特征
    h <- self$encode(x_tokens)   # [B, S, D]
    pred <- self$predictor(h)   # [B, S, D]

    # 2. Target Representation (解冻 Embedding，允许双向优化)
    target_emb <- self$tok_emb(y_tokens) # [B, S, D]

    # L2 正则化归一化
    pred_norm <- nnf_normalize(pred, p = 2, dim = -1)
    target_norm <- nnf_normalize(target_emb, p = 2, dim = -1)

    # [B, S, D] -> [S, B, D]
    pred_t <- pred_norm$permute(c(2, 1, 3))
    target_t <- target_norm$permute(c(2, 1, 3))

    # [S, B, D] x [S, D, B] -> [S, B, B] 余弦相似度矩阵
    sim <- torch_bmm(pred_t, target_t$transpose(2, 3)) / self$temperature

    # ---------------------------------------------------------------
    # 核心修复：消除 False Negatives (伪负样本 Mask)
    # ---------------------------------------------------------------
    y_t <- y_tokens$transpose(1, 2)
    
    # 检查同一位置上，哪些 Batch 样本的目标 Token 实际上相同
    same_token_mask <- y_t$unsqueeze(3) == y_t$unsqueeze(2)

    # 排除对角线（正样本 Positive Pair 不遮罩）
    diag_mask <- torch_eye(B, dtype = torch_bool(), device = device)$unsqueeze(1)$expand(c(S, B, B))
    false_negative_mask <- same_token_mask & (!diag_mask)

    # 将伪负样本处的相似度设为 -1e9，使其在 Softmax 中归零
    sim$masked_fill_(false_negative_mask, -1e9)

    # 3. 计算纯粹的 Pure InfoNCE Loss
    labels <- torch_arange(1, B, dtype = torch_long(), device = device)$unsqueeze(1)$expand(c(S, B))

    contrastive_loss <- nnf_cross_entropy(
      sim$reshape(c(S * B, B)),
      labels$reshape(c(S * B))
    )

    list(
      loss = contrastive_loss,
      contrastive = contrastive_loss
    )
  }
)