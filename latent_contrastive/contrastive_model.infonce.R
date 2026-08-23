source("utils/atomic_blocks.R")

# =====================================================================
# Token 级潜语义对比预训练 (Pure Per-Position InfoNCE)
# =====================================================================
#
# 目标：
#   x_t -> Online Encoder -> Predictor -> z_pred
#   y_t -> Target Token Embedding (stop-gradient) -> z_target
#   通过同一 position 的 batch 内样本进行 InfoNCE
#
# Loss：
#   L = L_InfoNCE
#
# 注意：
#   target embedding 使用 detach()，因此 target branch 不接收
#   InfoNCE 的反向梯度；这是当前模型设计的一部分。
# =====================================================================

TokenLatentModel <- nn_module(
  "TokenLatentModel",

  initialize = function(
    vocab_size,
    dim,
    n_layers,
    n_heads,
    max_seq_len,
    temperature = 0.2
  ) {
    self$tok_emb <- nn_embedding(vocab_size, dim)

    self$layers <- nn_module_list(
      lapply(
        1:n_layers,
        function(i) RtomicBlock(dim, n_heads, max_seq_len)
      )
    )

    self$norm_f <- RMSNorm(dim)

    self$predictor <- nn_sequential(
      nn_linear(dim, dim * 2),
      RMSNorm(dim * 2),
      nn_silu(),
      nn_linear(dim * 2, dim),
      RMSNorm(dim),
      nn_silu(),
      nn_linear(dim, dim)
    )

    # Fixed temperature: saved in state_dict, not optimized.
    self$temperature <- nn_buffer(
      torch_tensor(temperature)
    )
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

    # ---------------------------------------------------------------
    # 1. Online encoder
    # ---------------------------------------------------------------
    h <- self$tok_emb(x_tokens)

    for (i in 1:length(self$layers)) {
      h <- self$layers[[i]](h)
    }

    h <- self$norm_f(h)

    # ---------------------------------------------------------------
    # 2. Predictor
    # ---------------------------------------------------------------
    pred <- self$predictor(h)

    # ---------------------------------------------------------------
    # 3. Target representation
    # ---------------------------------------------------------------
    # [B, S] -> [B, S, D]
    # Stop gradient on target branch.
    target_emb <- self$tok_emb(y_tokens)$detach()

    # L2 normalize latent vectors.
    pred_norm <- nnf_normalize(pred, p = 2, dim = -1)
    target_norm <- nnf_normalize(target_emb, p = 2, dim = -1)

    # [B, S, D] -> [S, B, D]
    pred_t <- pred_norm$permute(c(2, 1, 3))
    target_t <- target_norm$permute(c(2, 1, 3))

    # For every sequence position:
    #
    #   [B, D] x [D, B] -> [B, B]
    #
    # Combined over S positions:
    #
    #   [S, B, D] x [S, D, B] -> [S, B, B]
    #
    # Diagonal = positive pairs.
    sim <- torch_bmm(
      pred_t,
      target_t$transpose(2, 3)
    ) / self$temperature

    # Positive target for sample i is target i.
    labels <- torch_arange(
      1,
      B,
      dtype = torch_long(),
      device = device
    )

    labels <- labels$
      unsqueeze(1)$
      expand(c(S, B))

    # [S, B, B] -> [S*B, B]
    contrastive_loss <- nnf_cross_entropy(
      sim$reshape(c(S * B, B)),
      labels$reshape(c(S * B))
    )

    # Pure InfoNCE objective.
    list(
      loss = contrastive_loss,
      contrastive = contrastive_loss
    )
  }
)
