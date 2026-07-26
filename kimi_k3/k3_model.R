# =====================================================================
# 共享 Transformer 原子组件: RMSNorm / RtomicRoPE / RtomicAttnRes / RtomicBlock
# =====================================================================
library(torch)

# 1. 修正版 RMSNorm：强转 FP32 避免平方溢出
RMSNorm <- nn_module(
  "RMSNorm",
  initialize = function(dim, eps = 1e-6) {
    self$eps <- eps
    self$weight <- nn_parameter(torch_ones(dim))
  },
  forward = function(x) {
    orig_dtype <- x$dtype
    # 强制提升到 FP32 进行平方和求均值，防止 FP16 下超出 65504 导致 Inf
    x_fp32 <- x$to(dtype = torch_float32())
    variance <- x_fp32$pow(2)$mean(dim = -1, keepdim = TRUE)
    
    x_norm <- x_fp32 * torch_rsqrt(variance + self$eps)
    return(self$weight$to(dtype = orig_dtype) * x_norm$to(dtype = orig_dtype))
  }
)

# 2. 旋转位置编码 (保持不变)
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

# 3. 新增: 跨层注意力残差模块 (Attention Residuals)
RtomicAttnRes <- nn_module(
  "RtomicAttnRes",
  initialize = function(dim) {
    # 随机初始化打破对称性，配合残差连接让每层注意力的初始偏好不同
    self$pseudo_query <- nn_parameter(torch_randn(dim) * 0.02)
    self$norm <- RMSNorm(dim)
  },
  
  forward = function(history_outputs) {
    # 始终走完整的 attention residual 路径，确保所有参数都参与计算图，
    # 避免 AMP 梯度 unscaling 时因部分参数无梯度而报 "tensor does not have a device"
    H <- torch_stack(history_outputs, dim = 1) # [L, B, S, D]
    H_norm <- self$norm(H)
    
    # 增加缩放因子 / sqrt(dim)，防止混合精度下内积过大导致 Softmax 溢出
    logits <- torch_einsum("d, l b s d -> l b s", list(self$pseudo_query, H_norm)) / sqrt(self$pseudo_query$size(1))
    weights <- nnf_softmax(logits, dim = 1)
    
    h_l <- torch_einsum("l b s, l b s d -> b s d", list(weights, H))
    return(h_l)
  }
)

# 4. K3 基础块 (融合 AttnRes 与 KDA 线性代理)
RtomicBlock <- nn_module(
  "RtomicBlock",
  initialize = function(dim, n_heads, max_seq_len = 1024, use_kda_proxy = TRUE) {
    self$attn_res <- RtomicAttnRes(dim) 
    self$norm1 <- RMSNorm(dim)
    self$n_heads <- n_heads
    self$head_dim <- dim %/% n_heads
    self$use_kda_proxy <- use_kda_proxy

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

  forward = function(history_outputs) {
    # 1. 跨层检索 + identity 残差：保证梯度高速通道，避免训练初期梯度被 softmax 均匀权重稀释
    x <- self$attn_res(history_outputs) + history_outputs[[length(history_outputs)]]
    
    B <- x$size(1); S <- x$size(2); D <- x$size(3)
    h_norm <- self$norm1(x)

    q <- self$q_proj(h_norm)$view(c(B, S, self$n_heads, self$head_dim))$transpose(2, 3)$contiguous()
    k <- self$k_proj(h_norm)$view(c(B, S, self$n_heads, self$head_dim))$transpose(2, 3)$contiguous()
    v <- self$v_proj(h_norm)$view(c(B, S, self$n_heads, self$head_dim))$transpose(2, 3)$contiguous()

    rope_out <- self$rope(q, k)
    q <- rope_out$q; k <- rope_out$k
    orig_dtype <- q$dtype

    if (self$use_kda_proxy) {
      # =========================================================
      # FP32 安全区: 防止 Cumsum 在 FP16 下累加溢出
      # =========================================================
      q_fp32 <- q$to(dtype = torch_float32())
      k_fp32 <- k$to(dtype = torch_float32())
      v_fp32 <- v$to(dtype = torch_float32())
      
      q_act <- nnf_elu(q_fp32) + 1.0
      k_act <- nnf_elu(k_fp32) + 1.0
      
      kv <- torch_einsum("b h s d, b h s e -> b h s d e", list(k_act, v_fp32))
      k_sum <- k_act
      
      kv_cum <- torch_cumsum(kv, dim = 3)
      k_cum <- torch_cumsum(k_sum, dim = 3)
      
      attn_num <- torch_einsum("b h s d, b h s d e -> b h s e", list(q_act, kv_cum))
      attn_den <- torch_einsum("b h s d, b h s d -> b h s", list(q_act, k_cum))$unsqueeze(-1)
      
      attn_out_fp32 <- attn_num / (attn_den + 1e-6)
      attn_out <- attn_out_fp32$to(dtype = orig_dtype)$transpose(2, 3)$reshape(c(B, S, D))
      
    } else {
      if (orig_dtype == torch_float32()) {
        q <- q$to(dtype = torch_float16()); k <- k$to(dtype = torch_float16()); v <- v$to(dtype = torch_float16())
      }
      attn_out <- torch_scaled_dot_product_attention(query = q, key = k, value = v, attn_mask = NULL, is_causal = TRUE)
      attn_out <- attn_out$to(dtype = orig_dtype)$transpose(2, 3)$reshape(c(B, S, D))
    }

    h_attn <- x + self$out_proj(attn_out)
    
    h_norm2 <- self$norm2(h_attn)
    gate <- nnf_silu(self$w1(h_norm2))
    ffn_out <- self$w3(gate * self$w2(h_norm2))

    return(h_attn + ffn_out)
  }
)

# =====================================================================
# 主模型组装 (隐空间残差预测器 - 适配 K3 架构)
# =====================================================================
RtomicK3 <- nn_module(
  "RtomicK3",
  initialize = function(vocab_size, dim, n_layers, n_heads, max_seq_len, use_kda_proxy = TRUE) {
    self$tok_emb <- nn_embedding(vocab_size, dim)
    nn_init_normal_(self$tok_emb$weight, mean = 0, std = 0.02)
    
    self$layers <- nn_module_list(lapply(1:n_layers, function(i) {
      RtomicBlock(dim, n_heads, max_seq_len, use_kda_proxy = use_kda_proxy)
    }))
    
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
    y <- input_data$y 
    device <- x$device
    
    h <- self$tok_emb(x)
    
    # 初始化张量状态轨迹 (Layer 0)
    history_outputs <- list(h) 
    
    for (i in 1:length(self$layers)) {
      h <- self$layers[[i]](history_outputs)
      history_outputs <- append(history_outputs, list(h))
    }
    
    h <- self$norm_f(h)
    semantic_delta <- self$predictor(h)
    pred <- h + semantic_delta
    
    res <- list(pred = pred)
    
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