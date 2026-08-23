source("utils/atomic_blocks.R")

# =====================================================================
# Token 级潜语义对比预训练 (Per-Position InfoNCE)
# =====================================================================
TokenLatentModel <- nn_module(
  "TokenLatentModel",
  initialize = function(vocab_size, dim, n_layers, n_heads, max_seq_len) {
    self$tok_emb <- nn_embedding(vocab_size, dim)
    self$layers <- nn_module_list(lapply(1:n_layers, function(i) RtomicBlock(dim, n_heads, max_seq_len)))
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

    self$temperature <- nn_buffer(torch_tensor(0.2))
    self$ce_weight <- nn_buffer(torch_tensor(0.5))
  },

  encode = function(x_tokens) {
    h <- self$tok_emb(x_tokens)
    for (i in 1:length(self$layers)) h <- self$layers[[i]](h)
    self$norm_f(h)
  },

 forward = function(input_data) {
  x_tokens <- input_data$x
  y_tokens <- input_data$y

  B <- x_tokens$size(1)
  S <- x_tokens$size(2)

  # --------------------------------------------------
  # 1. Context encoder
  # --------------------------------------------------
  h <- self$tok_emb(x_tokens)

  for (i in 1:length(self$layers)) {
    h <- self$layers[[i]](h)
  }

  h <- self$norm_f(h)

  # --------------------------------------------------
  # 2. Predictor
  # --------------------------------------------------
  pred <- self$predictor(h)

  # --------------------------------------------------
  # 3. Target embedding
  # --------------------------------------------------
  target_emb <- self$tok_emb(y_tokens)$detach()

  pred_norm <- nnf_normalize(pred, p = 2, dim = -1)
  target_norm <- nnf_normalize(target_emb, p = 2, dim = -1)

  # [B, S, D] -> [S, B, D]
  pred_t <- pred_norm$permute(c(2, 1, 3))
  target_t <- target_norm$permute(c(2, 1, 3))

  # [S, B, D] x [S, D, B] -> [S, B, B]
  sim <- torch_bmm(
    pred_t,
    target_t$transpose(2, 3)
  ) / self$temperature

  # 每个 position 的正样本都是 batch 中同 index
  labels <- torch_arange(
    1, B,
    dtype = torch_long(),
    device = x_tokens$device
  )

  labels <- labels$unsqueeze(1)$expand(c(S, B))

  contrastive_loss <- nnf_cross_entropy(
    sim$reshape(c(S * B, B)),
    labels$reshape(c(S * B))
  )

  # --------------------------------------------------
  # 4. Auxiliary CE
  # --------------------------------------------------
  ce_logits <- torch_matmul(
    h,
    self$tok_emb$weight$transpose(1, 2)
  )

  ce_loss <- nnf_cross_entropy(
    ce_logits$reshape(c(B * S, -1)),
    y_tokens$reshape(c(B * S))
  )

  total_loss <- contrastive_loss +
    self$ce_weight * ce_loss

  list(
    loss = total_loss,
    contrastive = contrastive_loss,
    ce = ce_loss
  )
}
)