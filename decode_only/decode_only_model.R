# =====================================================================
# 纯正的 Decode-Only Causal Language Model 架构 (集成 RoPE 与显式初始化)
# =====================================================================

library(torch)
# 增加了 GQA 和 KVcache 逻辑的架构
source("decode_only/Block_GQA_KVcache.R")


RtomicCausalLM <- nn_module(
  "RtomicCausalLM",
  # 升级点 1: 新增 n_kv_heads 参数接收口，默认为 NULL (即兼容 MHA)
  initialize = function(vocab_size, dim, n_layers, n_heads, max_seq_len, n_kv_heads = NULL) {
    self$tok_emb <- nn_embedding(vocab_size, dim)
    
    # 彻底移除旧版绝对位置编码 pos_emb，将 max_seq_len 传递给子 Block
    # 并且把 n_kv_heads 传给每一层的 Block，激活 GQA
    self$layers <- nn_module_list(lapply(1:n_layers, function(i) {
      RtomicBlock(dim, n_heads, max_seq_len, n_kv_heads)
    }))
    
    self$norm_f <- RMSNorm(dim)
    
    # =====================================================================
    # 针对大模型与 SwiGLU 优化的显式参数初始化
    # =====================================================================
    self$apply(function(m) {
      if (inherits(m, c("nn_linear", "nn_embedding"))) {
        nn_init_normal_(m$weight, mean = 0, std = 0.02)
      }
    })
    
    # 对所有残差投射输出层进行深度缩放，阻断深层网络中的方差爆炸
    res_std <- 0.02 / sqrt(2 * n_layers)
    for (i in 1:n_layers) {
      nn_init_normal_(self$layers[[i]]$out_proj$weight, std = res_std)
      nn_init_normal_(self$layers[[i]]$w3$weight, std = res_std)
    }
  },
  
  # 升级点 2: 增加 use_cache 和 past_key_values，支持推理加速
  forward = function(input_data, output_hidden_states = FALSE, 
                     use_cache = FALSE, past_key_values = NULL) {
    x <- input_data$x
    y <- input_data$y
    loss_mask <- input_data$loss_mask
    device <- x$device
    
    # 纯净的 Token Embedding
    h <- self$tok_emb(x)
    presents <- list() # 用于收集各层新产生的 Cache
    
    # =====================================================================
    # 升级点 3: 带有缓存传递的前向传播循环
    # =====================================================================
    for (i in 1:length(self$layers)) {
      # 提取当前层的历史 Cache (如果有)
      layer_past <- if (!is.null(past_key_values)) past_key_values[[i]] else NULL
      
      # 传入隐状态和历史 Cache，返回新的隐状态和更新后的 Cache
      layer_outputs <- self$layers[[i]](h, past_kv = layer_past)
      
      h <- layer_outputs$hidden_states
      if (use_cache) {
        presents[[i]] <- layer_outputs$present_kv
      }
    }
    
    h <- self$norm_f(h)
    logits <- nnf_linear(h, weight = self$tok_emb$weight)
    
    # =====================================================================
    # 训练/推理 分支返回逻辑
    # =====================================================================
    if (self$training) {
      # 训练模式：计算 Loss (此模式下不用管 Cache)
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
      
      if (output_hidden_states) {
        return(list(logits = logits, loss = loss, hidden_states = h))
      } else {
        return(list(logits = logits, loss = loss))
      }
      
    } else {
      # 推理模式：打包输出 logits，如果启用了 cache 则一并返回
      res <- list(logits = logits)
      if (output_hidden_states) res$hidden_states <- h
      if (use_cache) res$past_key_values <- presents
      return(res)
    }
  }
)