# =====================================================================
# 共享 Transformer 原子组件: RMSNorm / RtomicRoPE / RtomicBlock
# =====================================================================

RMSNorm <- nn_module(
  "RMSNorm",
  initialize = function(dim, eps = 1e-6) {
    self$eps <- eps
    self$weight <- nn_parameter(torch_ones(dim))
  },
  forward = function(x) {
    variance <- x$pow(2)$mean(dim = -1, keepdim = TRUE)
    return(self$weight * (x * torch_rsqrt(variance + self$eps)))
  }
)

RtomicRoPE <- nn_module(
  "RtomicRoPE",
  initialize = function(head_dim, max_seq_len = 1024, base = 10000) {
    self$head_dim <- head_dim
    inv_freq <- 1.0 / (base ^ (torch_arange(0, head_dim - 1, 2, dtype = torch_float32()) / head_dim))
    self$register_buffer("inv_freq", inv_freq)

    t <- torch_arange(1, max_seq_len, dtype = torch_float32())
    freqs <- torch_einsum("i,j->ij", list(t, self$inv_freq))
    emb <- torch_cat(list(freqs, freqs), dim = -1)

    self$register_buffer("cos_cached", emb$cos()$unsqueeze(1)$unsqueeze(1))
    self$register_buffer("sin_cached", emb$sin()$unsqueeze(1)$unsqueeze(1))
  },

  forward = function(q, k) {
    seq_len <- q$size(3)
    cos <- self$cos_cached[, , 1:seq_len, ]$to(device = q$device, dtype = q$dtype)
    sin <- self$sin_cached[, , 1:seq_len, ]$to(device = q$device, dtype = q$dtype)

    rotate_half <- function(x) {
      d <- x$size(-1)
      x1 <- x$narrow(dim = -1, start = 1, length = d/2)
      x2 <- x$narrow(dim = -1, start = d/2 + 1, length = d/2)
      torch_cat(list(-x2, x1), dim = -1)
    }

    q_embed <- (q * cos) + (rotate_half(q) * sin)
    k_embed <- (k * cos) + (rotate_half(k) * sin)
    return(list(q = q_embed, k = k_embed))
  }
)

# SwiGLU（Swish Gated Linear Unit）门控线性单元
RtomicBlock <- nn_module(
  "RtomicBlock",
  initialize = function(dim, n_heads, max_seq_len = 1024) {
    self$norm1 <- RMSNorm(dim)
    self$n_heads <- n_heads
    self$head_dim <- dim %/% n_heads

    self$q_proj <- nn_linear(dim, dim, bias = FALSE)
    self$k_proj <- nn_linear(dim, dim, bias = FALSE)
    self$v_proj <- nn_linear(dim, dim, bias = FALSE)
    self$out_proj <- nn_linear(dim, dim, bias = FALSE)

    self$rope <- RtomicRoPE(self$head_dim, max_seq_len)

    self$norm2 <- RMSNorm(dim)
    dim_hidden <- as.integer(dim * 8 / 3)
    self$w1 <- nn_linear(dim, dim_hidden, bias = FALSE)
    self$w2 <- nn_linear(dim, dim_hidden, bias = FALSE)
    self$w3 <- nn_linear(dim_hidden, dim, bias = FALSE)
  },

  forward = function(x) {
    B <- x$size(1); S <- x$size(2); D <- x$size(3)
    h_norm <- self$norm1(x)

    q <- self$q_proj(h_norm)$view(c(B, S, self$n_heads, self$head_dim))$transpose(2, 3)$contiguous()
    k <- self$k_proj(h_norm)$view(c(B, S, self$n_heads, self$head_dim))$transpose(2, 3)$contiguous()
    v <- self$v_proj(h_norm)$view(c(B, S, self$n_heads, self$head_dim))$transpose(2, 3)$contiguous()

    rope_out <- self$rope(q, k)
    q <- rope_out$q; k <- rope_out$k

    orig_dtype <- q$dtype
    if (orig_dtype == torch_float32()) {
      q <- q$to(dtype = torch_float16()); k <- k$to(dtype = torch_float16()); v <- v$to(dtype = torch_float16())
    }

    attn_out <- torch_scaled_dot_product_attention(query = q, key = k, value = v, attn_mask = NULL, is_causal = TRUE)
    attn_out <- attn_out$to(dtype = orig_dtype)$transpose(2, 3)$reshape(c(B, S, D))

    x <- x + self$out_proj(attn_out)
    h_norm2 <- self$norm2(x)
    gate <- nnf_silu(self$w1(h_norm2))
    ffn_out <- self$w3(gate * self$w2(h_norm2))

    return(x + ffn_out)
  }
)
