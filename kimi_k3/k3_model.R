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

# 3. 跨层注意力残差模块 (Cross-Layer Attention Residuals)
#
# 设计核心：用当前层的表示作为 query，从所有历史层输出中检索信息。
# 这使每个 (batch, position) 拥有独立的跨层注意力权重，而非全局共享。
#
# 初始化策略保证训练起点 ≈ 标准 Transformer：
#   - query_proj = 单位矩阵 → query = 当前层表示自身
#   - 向量自点积天然最大 → softmax 初始集中在最新层
#   - 随训练进行，query_proj 学会提取"需要检索什么"的特征方向
RtomicAttnRes <- nn_module(
  "RtomicAttnRes",
  initialize = function(dim) {
    self$norm_key <- RMSNorm(dim)    # 对历史层输出做归一化
    self$norm_q   <- RMSNorm(dim)    # 对 query（当前层输出）做归一化

    # Query 投影：初始化为单位矩阵，使 query ≈ 当前层归一化表示
    # 训练中学习旋转 query 方向以检索特定层的特征
    self$query_proj <- nn_linear(dim, dim, bias = FALSE)
    with_no_grad({
      self$query_proj$weight$copy_(torch_eye(dim))
    })

    # 可学习温度倒数 (inverse temperature)。
    # 初始化为 log(sqrt(dim)) → tau ≈ sqrt(dim) → softmax 适度尖锐，
    # 由于自点积最大，权重集中在最新层。训练后可降低温度来摊平分布。
    self$inv_temp <- nn_parameter(torch_tensor(log(sqrt(dim))))
  },

  forward = function(history_outputs) {
    n_hist <- length(history_outputs)
    # 将历史层输出 detach 作为静态 key-value store。
    # 梯度只需流过 query（当前层表示），而不过历史层 keys——
    # 这消除了 O(N²) 梯度路径，防止梯度爆炸导致 NaN。
    H <- torch_stack(lapply(history_outputs, function(h) h$detach()), dim = 1) # [L, B, S, D]
    H_norm <- self$norm_key(H)                                                   # [L, B, S, D]

    # Content-based query: 用最新层的表示作为 query（保留梯度）
    current <- history_outputs[[n_hist]]                # [B, S, D]
    query <- self$query_proj(self$norm_q(current))      # [B, S, D]

    # query · key: 自点积天然最大 → softmax 初始集中在最新层
    logits <- torch_einsum("b s d, l b s d -> l b s", list(query, H_norm))
    logits <- logits * torch_exp(self$inv_temp)         # 可学习温度

    weights <- nnf_softmax(logits, dim = 1)             # [L, B, S]

    # 加权融合原始历史输出（非归一化，保留 scale 信息）
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
    # 1. 跨层检索：用当前层表示为 query，从所有历史层中加权提取信息。
    #    初始化时自点积最大 → 集中在最新层 → 行为等价于标准 Transformer
    x <- self$attn_res(history_outputs)
    
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
    
    nn_init_normal_(self$predictor[[3]]$weight, mean = 0, std = 0.02 / sqrt(dim))
    nn_init_zeros_(self$predictor[[3]]$bias)
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