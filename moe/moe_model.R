# =====================================================================
# 混合专家架构 (Mixture of Experts) 稠密计算优化版
# 核心改动: 干掉所有 torch_nonzero() 与 R 层的条件同步判断
# 改用全量稠密矩阵乘法 + 权重 scatter 还原，释放 GPU 异步流水线吞吐量
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
  initialize = function(head_dim,
                        max_seq_len = 8192,
                        base = 10000) {
    self$head_dim <- head_dim
    inv_freq <- 1.0 / (base^(
      torch_arange(0, head_dim - 1, 2, dtype = torch_float32()) / head_dim
    ))
    self$register_buffer("inv_freq", inv_freq)
    
    t <- torch_arange(1, max_seq_len, dtype = torch_float32())
    freqs <- torch_einsum("i,j->ij", list(t, self$inv_freq))
    emb <- torch_cat(list(freqs, freqs), dim = -1)
    
    self$register_buffer("cos_cached", emb$cos()$unsqueeze(1)$unsqueeze(1))
    self$register_buffer("sin_cached", emb$sin()$unsqueeze(1)$unsqueeze(1))
  },
  
  forward = function(q, k, seq_offset = 0) {
    seq_len <- q$size(3)
    cos <- self$cos_cached[, , (seq_offset + 1):(seq_offset + seq_len), ]$to(device = q$device, dtype = q$dtype)
    sin <- self$sin_cached[, , (seq_offset + 1):(seq_offset + seq_len), ]$to(device = q$device, dtype = q$dtype)
    
    rotate_half <- function(x) {
      d <- x$size(-1)
      x1 <- x$narrow(dim = -1,
                     start = 1,
                     length = d / 2)
      x2 <- x$narrow(dim = -1,
                     start = d / 2 + 1,
                     length = d / 2)
      torch_cat(list(-x2, x1), dim = -1)
    }
    
    q_embed <- (q * cos) + (rotate_half(q) * sin)
    k_embed <- (k * cos) + (rotate_half(k) * sin)
    return(list(q = q_embed, k = k_embed))
  }
)

# =====================================================================
# 1. 独立专家 (Expert): SwiGLU 前馈网络
# =====================================================================
RtomicExpert <- nn_module(
  "RtomicExpert",
  initialize = function(dim, dim_hidden) {
    self$w1 <- nn_linear(dim, dim_hidden, bias = FALSE)
    self$w2 <- nn_linear(dim, dim_hidden, bias = FALSE)
    self$w3 <- nn_linear(dim_hidden, dim, bias = FALSE)
  },
  forward = function(x) {
    gate <- nnf_silu(self$w1(x))        # SiLU 激活函数处理成了“门控”
    return(self$w3(gate * self$w2(x)))  # 线性变化和 gate 相乘，变回原维度 dim
  }
)

# =====================================================================
# 2. 稠密计算版混合专家路由器 (无同步、无稀疏切片)
# =====================================================================
RtomicMoE <- nn_module(
  "RtomicMoE",
  initialize = function(dim,
                        dim_hidden,
                        num_experts = 8,
                        top_k = 2,
                        aux_coef = 0.01,
                        dim_hidden_routed = NULL) {
    self$num_experts <- num_experts
    self$top_k <- top_k
    self$aux_coef <- aux_coef
    
    self$dim_hidden_routed <- if (is.null(dim_hidden_routed)) {
      as.integer(64 * ceiling((dim * 4 / 3) / 64))
    } else {
      dim_hidden_routed
    }
    
    self$router <- nn_linear(dim, num_experts, bias = FALSE)
    
    self$experts <- nn_module_list(lapply(1:num_experts, function(i)
      RtomicExpert(dim, self$dim_hidden_routed)))
    
    self$shared_expert <- RtomicExpert(dim, dim_hidden)
  },
  
  forward = function(x) {
    B <- x$size(1)
    S <- x$size(2)
    D <- x$size(3)
    N <- B * S
    
    x_flat <- x$view(c(-1, D))  # [N, D]
    
    # ---- 1. 路由权重计算 ----
    router_logits <- self$router(x_flat)$to(dtype = torch_float32())
    # 转化为概率分布，此时每个 Token 在所有 Expert 上都有权重
    routing_weights <- nnf_softmax(router_logits, dim = -1)
    
    # 提取 Top-K 索引与权重
    topk_out <- torch_topk(routing_weights$detach(),
                           k = self$top_k,
                           dim = -1)
    routing_indices <- topk_out[[2]]  # 返回第几个的索引
    
    # 把 topK 归一化
    routing_weights_topk <- routing_weights$gather(dim = -1, index = routing_indices)
    routing_weights_topk <- routing_weights_topk / routing_weights_topk$sum(dim = -1, keepdim = TRUE)
    routing_weights_topk <- routing_weights_topk$to(dtype = x$dtype)
    
    # 供外部探查
    self$last_routing_indices <- routing_indices$detach()
    self$last_routing_weights <- routing_weights$detach()
    
    # ---- 2. 负载均衡辅助损失 (一维化张量操作，零循环) ----
    aux_loss <- if (self$training) {
      # 用 one_hot 统计 top_k 里每个专家被选中的次数
      # 对第 1 和 第 2 维度 (即 N 和 top_k) 求和，直接得到每个 expert 的总频次
      expert_counts <- nnf_one_hot(routing_indices,
        num_classes = self$num_experts)$to(dtype = torch_float32())$sum(dim = c(1, 2))
      
      # 计算比例 f：每个专家分到的 Token 占总数的百分比
      f <- expert_counts / N
      # 计算 P：所有 Token 对各个专家的 Softmax 概率的平均值
      P <- routing_weights$mean(dim = 1)
      self$aux_coef * self$num_experts * (f * P)$sum()
    } else {
      torch_tensor(0, requires_grad = FALSE, device = x$device)
    }
    
    # ---- 3. 全量稠密权重矩阵 ----
    # 巨大的 $N \times E$ （总 Token 数 $\times$ 专家总数）的稠密矩阵
    dense_weights <- torch_zeros_like(routing_weights)$scatter(
      dim = 2,  # 在列上 expert 维度上操作
      index = routing_indices,  # 按照坐标填充
      src = routing_weights_topk)$to(dtype = x$dtype)
    
    # ---- 4. 稠密计算循环 (In-place 内存优化) ----
    final_output <- torch_zeros_like(x_flat)
    
    for (i in 1:self$num_experts) {
      weight_i <- dense_weights[, i]$unsqueeze(2)  # 提取当前专家的列权重
      expert_out <- self$experts[[i]](x_flat)      # 全量暴力前向传播
      
      # 使用原地加法 $add_()，拒绝产生中间 Tensor 碎片，大幅降低 R 的垃圾回收压力
      final_output$add_(expert_out * weight_i) # 为 0 的 expert 就被 * 消掉了
    }
    
    # 加上 Shared Expert 输出，同样使用原地加法
    final_output$add_(self$shared_expert(x_flat))
    
    return(list(output = final_output$view(c(B, S, D)), aux_loss = aux_loss))
  }
)

# =====================================================================
# 3. Transformer Block
# =====================================================================
RtomicBlockMoE <- nn_module(
  "RtomicBlockMoE",
  initialize = function(dim,
                        n_heads,
                        max_seq_len = 1024,
                        n_kv_heads = NULL,
                        num_experts = 8,
                        top_k = 2,
                        dim_hidden_routed = NULL) {
    self$norm1 <- RMSNorm(dim)
    self$n_heads <- n_heads
    self$n_kv_heads <- if (is.null(n_kv_heads))
      n_heads
    else
      n_kv_heads
    self$head_dim <- dim %/% n_heads
    
    self$q_proj <- nn_linear(dim, self$n_heads * self$head_dim, bias = FALSE)
    self$k_proj <- nn_linear(dim, self$n_kv_heads * self$head_dim, bias = FALSE)
    self$v_proj <- nn_linear(dim, self$n_kv_heads * self$head_dim, bias = FALSE)
    self$out_proj <- nn_linear(dim, dim, bias = FALSE)
    
    self$rope <- RtomicRoPE(self$head_dim, max_seq_len)
    
    self$norm2 <- RMSNorm(dim)
    
    multiple_of <- 64
    raw_dim_hidden <- dim * 8 / 3
    dim_hidden <- as.integer(multiple_of * ceiling(raw_dim_hidden / multiple_of))
    
    self$moe <- RtomicMoE(
      dim,
      dim_hidden,
      num_experts,
      top_k,
      aux_coef = 0.01,
      dim_hidden_routed = dim_hidden_routed
    )
  },
  
  forward = function(x, past_kv = NULL) {
    B <- x$size(1)
    S <- x$size(2)
    D <- x$size(3)
    h_norm <- self$norm1(x)
    
    q <- self$q_proj(h_norm)$view(c(B, S, self$n_heads, self$head_dim))$transpose(2, 3)$contiguous()
    k <- self$k_proj(h_norm)$view(c(B, S, self$n_kv_heads, self$head_dim))$transpose(2, 3)$contiguous()
    v <- self$v_proj(h_norm)$view(c(B, S, self$n_kv_heads, self$head_dim))$transpose(2, 3)$contiguous()
    
    seq_offset <- if (!is.null(past_kv))
      past_kv$k$size(3)
    else
      0
    rope_out <- self$rope(q, k, seq_offset)
    q <- rope_out$q
    k <- rope_out$k
    
    if (!is.null(past_kv)) {
      k <- torch_cat(list(past_kv$k, k), dim = 3)
      v <- torch_cat(list(past_kv$v, v), dim = 3)
    }
    present_kv <- list(k = k, v = v)
    
    num_key_value_groups <- self$n_heads %/% self$n_kv_heads
    if (num_key_value_groups > 1) {
      k <- k$repeat_interleave(num_key_value_groups, dim = 2)
      v <- v$repeat_interleave(num_key_value_groups, dim = 2)
    }
    
    orig_dtype <- q$dtype
    if (orig_dtype == torch_float32()) {
      q <- q$to(dtype = torch_bfloat16())
      k <- k$to(dtype = torch_bfloat16())
      v <- v$to(dtype = torch_bfloat16())
    }
    
    use_causal <- S > 1
    attn_out <- torch_scaled_dot_product_attention(
      query = q,
      key = k,
      value = v,
      attn_mask = NULL,
      is_causal = use_causal
    )
    attn_out <- attn_out$to(dtype = orig_dtype)$transpose(2, 3)$reshape(c(B, S, D))
    
    x <- x + self$out_proj(attn_out)
    h_norm2 <- self$norm2(x)
    
    moe_out <- self$moe(h_norm2)
    
    return(
      list(
        hidden_states = x + moe_out$output,
        present_kv = present_kv,
        aux_loss = moe_out$aux_loss
      )
    )
  }
)

# =====================================================================
# 4. 主模型总装: RtomicCausalLM
# =====================================================================
RtomicCausalLM <- nn_module(
  "RtomicCausalLM",
  initialize = function(vocab_size,
                        dim,
                        n_layers,
                        n_heads,
                        max_seq_len,
                        n_kv_heads = NULL,
                        num_experts = 8,
                        top_k = 2,
                        dim_hidden_routed = NULL) {
    self$tok_emb <- nn_embedding(vocab_size, dim)
    
    self$layers <- nn_module_list(lapply(1:n_layers, function(i) {
      RtomicBlockMoE(dim,
                     n_heads,
                     max_seq_len,
                     n_kv_heads,
                     num_experts,
                     top_k,
                     dim_hidden_routed)
    }))
    
    self$norm_f <- RMSNorm(dim)
    
    self$apply(function(m) {
      if (inherits(m, c("nn_linear", "nn_embedding"))) {
        nn_init_normal_(m$weight, mean = 0, std = 0.02)
      }
    })
    
    res_std <- 0.02 / sqrt(2 * n_layers)
    for (i in 1:n_layers) {
      nn_init_normal_(self$layers[[i]]$out_proj$weight, std = res_std)
      for (e in 1:num_experts) {
        nn_init_normal_(self$layers[[i]]$moe$experts[[e]]$w3$weight, std = res_std)
      }
      nn_init_normal_(self$layers[[i]]$moe$shared_expert$w3$weight, std = res_std)
    }
  },
  
  forward = function(input_data,
                     output_hidden_states = FALSE,
                     use_cache = FALSE,
                     past_key_values = NULL) {
    x <- input_data$x
    y <- input_data$y
    loss_mask <- input_data$loss_mask
    device <- x$device
    
    h <- self$tok_emb(x)
    presents <- list()
    
    aux_losses <- list()
    for (i in 1:length(self$layers)) {
      layer_past <- if (!is.null(past_key_values))
        past_key_values[[i]]
      else
        NULL
      layer_outputs <- self$layers[[i]](h, past_kv = layer_past)
      h <- layer_outputs$hidden_states
      aux_losses[[i]] <- layer_outputs$aux_loss
      if (use_cache) {
        presents[[i]] <- layer_outputs$present_kv
      }
    }
    
    h <- self$norm_f(h)
    logits <- nnf_linear(h, weight = self$tok_emb$weight)
    
    if (self$training) {
      logits_flat <- logits$view(c(-1, logits$size(-1)))
      y_flat <- y$view(c(-1))
      
      if (!is.null(loss_mask)) {
        mask_flat <- loss_mask$view(c(-1))
        logits_flat <- logits_flat[mask_flat, ]
        y_flat <- y_flat[mask_flat]
      }
      
      if (logits_flat$size(1) == 0) {
        loss <- torch_tensor(0, requires_grad = TRUE, device = device)
      } else {
        loss <- nnf_cross_entropy(logits_flat, y_flat)
      }
      
      total_aux_loss <- Reduce(`+`, aux_losses)
      loss <- loss + total_aux_loss
      
      if (output_hidden_states) {
        return(list(
          logits = logits,
          loss = loss,
          hidden_states = h
        ))
      } else {
        return(list(logits = logits, loss = loss))
      }
      
    } else {
      res <- list(logits = logits)
      if (output_hidden_states)
        res$hidden_states <- h
      if (use_cache)
        res$past_key_values <- presents
      return(res)
    }
  }
)