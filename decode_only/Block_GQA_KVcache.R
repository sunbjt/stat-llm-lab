# =====================================================================
# 共享 Transformer 原子组件: RMSNorm / RtomicRoPE / RtomicBlock
# 现代化升级: 完美支持 GQA (Grouped-Query Attention) 与 KV Cache 推理
# =====================================================================

library(torch)

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
  initialize = function(head_dim, max_seq_len = 8192, base = 10000) {
    self$head_dim <- head_dim
    inv_freq <- 1.0 / (base ^ (torch_arange(0, head_dim - 1, 2, dtype = torch_float32()) / head_dim))
    self$register_buffer("inv_freq", inv_freq)
    
    # 包含到 max_seq_len，确保生成足够的缓存空间
    t <- torch_arange(1, max_seq_len, dtype = torch_float32())
    freqs <- torch_einsum("i,j->ij", list(t, self$inv_freq))
    emb <- torch_cat(list(freqs, freqs), dim = -1)
    
    self$register_buffer("cos_cached", emb$cos()$unsqueeze(1)$unsqueeze(1))
    self$register_buffer("sin_cached", emb$sin()$unsqueeze(1)$unsqueeze(1))
  },
  
  # 新增 seq_offset 参数，用于 KV Cache 推理时的位置对齐
  forward = function(q, k, seq_offset = 0) {
    seq_len <- q$size(3)
    # R 语言 1-based 索引，精准切出当前输入 Token 对应的 cos 和 sin
    cos <- self$cos_cached[, , (seq_offset + 1):(seq_offset + seq_len), ]$to(device = q$device, dtype = q$dtype)
    sin <- self$sin_cached[, , (seq_offset + 1):(seq_offset + seq_len), ]$to(device = q$device, dtype = q$dtype)
    
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

RtomicBlock <- nn_module(
  "RtomicBlock",
  # 升级点 1: 增加 n_kv_heads，默认为 NULL (向下兼容 MHA)
  initialize = function(dim, n_heads, max_seq_len = 1024, n_kv_heads = NULL) {
    self$norm1 <- RMSNorm(dim)
    self$n_heads <- n_heads
    
    # 自动识别：如果不传 n_kv_heads，就默认等于 n_heads (即标准 MHA)
    self$n_kv_heads <- if (is.null(n_kv_heads)) n_heads else n_kv_heads
    self$head_dim <- dim %/% n_heads
    
    if (self$n_heads %% self$n_kv_heads != 0) {
      stop("Error: n_heads 必须能被 n_kv_heads 整除！")
    }
    
    # 升级点 2: K 和 V 的投影矩阵变小了，极大地节约了参数和显存
    self$q_proj <- nn_linear(dim, self$n_heads * self$head_dim, bias = FALSE)
    self$k_proj <- nn_linear(dim, self$n_kv_heads * self$head_dim, bias = FALSE)
    self$v_proj <- nn_linear(dim, self$n_kv_heads * self$head_dim, bias = FALSE)
    self$out_proj <- nn_linear(dim, dim, bias = FALSE)
    
    self$rope <- RtomicRoPE(self$head_dim, max_seq_len)
    
    self$norm2 <- RMSNorm(dim)
    
    # 工业标准：SwiGLU 的隐藏层为了与传统 FFN 参数量对齐，缩放为 8/3。
    # 为了确保 GPU Tensor Core 的极致加速和内存对齐，必须将其向上取整为特定倍数。
    multiple_of <- 64  # 根据需求也可设为 128 或 256
    raw_dim_hidden <- dim * 8 / 3
    dim_hidden <- as.integer(multiple_of * ceiling(raw_dim_hidden / multiple_of))
    
    self$w1 <- nn_linear(dim, dim_hidden, bias = FALSE)
    self$w2 <- nn_linear(dim, dim_hidden, bias = FALSE)
    self$w3 <- nn_linear(dim_hidden, dim, bias = FALSE)
  },
  
  # 升级点 3: 支持接收历史缓存 past_kv
  forward = function(x, past_kv = NULL) {
    B <- x$size(1); S <- x$size(2); D <- x$size(3)
    h_norm <- self$norm1(x)
    
    # 映射时，K 和 V 按 n_kv_heads 重塑
    q <- self$q_proj(h_norm)$view(c(B, S, self$n_heads, self$head_dim))$transpose(2, 3)$contiguous()
    k <- self$k_proj(h_norm)$view(c(B, S, self$n_kv_heads, self$head_dim))$transpose(2, 3)$contiguous()
    v <- self$v_proj(h_norm)$view(c(B, S, self$n_kv_heads, self$head_dim))$transpose(2, 3)$contiguous()
    
    # 如果有缓存，说明是推理模式，需要加上偏移量算位置编码
    seq_offset <- if (!is.null(past_kv)) past_kv$k$size(3) else 0
    rope_out <- self$rope(q, k, seq_offset)
    q <- rope_out$q; k <- rope_out$k
    
    # =====================================================================
    # KV Cache 拼接模块
    # =====================================================================
    if (!is.null(past_kv)) {
      # 在序列长度（Sequence Length, dim=3）维度拼接历史缓存
      k <- torch_cat(list(past_kv$k, k), dim = 3)
      v <- torch_cat(list(past_kv$v, v), dim = 3)
    }
    # 返回本层完整的最新 KV 用于下一步
    present_kv <- list(k = k, v = v) 
    
    # =====================================================================
    # GQA 核心模块：头维度展开 (Repeat Interleave)
    # =====================================================================
    num_key_value_groups <- self$n_heads %/% self$n_kv_heads
    if (num_key_value_groups > 1) {
      # 在头维度 (dim=2) 展开 k 和 v，使其与 q 的头数对齐
      k <- k$repeat_interleave(num_key_value_groups, dim = 2)
      v <- v$repeat_interleave(num_key_value_groups, dim = 2)
    }
    
    orig_dtype <- q$dtype
    if (orig_dtype == torch_float32()) {
      q <- q$to(dtype = torch_float16()); k <- k$to(dtype = torch_float16()); v <- v$to(dtype = torch_float16())
    }
    
    # 训练时(长度>1)使用因果掩码，单次推理(长度=1)时关掉掩码以便看到全局Cache
    use_causal <- S > 1
    
    attn_out <- torch_scaled_dot_product_attention(query = q, key = k, value = v, attn_mask = NULL, is_causal = use_causal)
    attn_out <- attn_out$to(dtype = orig_dtype)$transpose(2, 3)$reshape(c(B, S, D))
    
    x <- x + self$out_proj(attn_out)
    h_norm2 <- self$norm2(x)
    gate <- nnf_silu(self$w1(h_norm2))
    ffn_out <- self$w3(gate * self$w2(h_norm2))
    
    # 使用列表返回，方便主模型提取缓存
    return(list(hidden_states = x + ffn_out, present_kv = present_kv))
  }
)

