source("utils/atomic_blocks.R")

# =====================================================================
# VQ 模块 (点积寻址)
# =====================================================================
RtomicVQ <- nn_module(
  "RtomicVQ",
  initialize = function() {},
  forward = function(pred_states, codebook_weight) {
    orig_shape <- pred_states$size()
    pred_flat <- pred_states$reshape(c(-1, orig_shape[3]))
    
    pred_norm <- nnf_normalize(pred_flat, p = 2, dim = -1)
    cb_norm <- nnf_normalize(codebook_weight, p = 2, dim = -1)
    
    sim <- torch_matmul(pred_norm, cb_norm$t())
    min_indices <- torch_argmax(sim, dim = -1)
    
    quantized_flat <- nnf_embedding(min_indices, codebook_weight)
    quantized_states <- quantized_flat$reshape(orig_shape)
    
    quantized_ste <- pred_states + (quantized_states - pred_states)$detach()
    
    return(list(quantized = quantized_ste, indices = min_indices))
  }
)

# =====================================================================
# 5. 主模型组装 (内置 VICReg 防坍塌引擎)
# =====================================================================
RtomicJEPA_VQ <- nn_module(
  "RtomicJEPA_VQ",
  initialize = function(vocab_size, dim, n_layers, n_heads, max_seq_len, num_clusters = 4096) {
    self$tok_emb <- nn_embedding(vocab_size, dim)
    nn_init_normal_(self$tok_emb$weight, mean = 0, std = 0.02)
    
    self$codebook_weight <- nn_parameter(torch_randn(num_clusters, dim))
    nn_init_normal_(self$codebook_weight, mean = 0, std = 1 / sqrt(dim))
    
    self$layers <- nn_module_list(lapply(1:n_layers, function(i) RtomicBlock(dim, n_heads, max_seq_len)))
    self$norm_f <- RMSNorm(dim)
    
    self$target_layers <- nn_module_list(lapply(1:n_layers, function(i) RtomicBlock(dim, n_heads, max_seq_len)))
    for (p in self$target_layers$parameters) p$requires_grad_(FALSE)
    self$target_norm_f <- RMSNorm(dim)
    for (p in self$target_norm_f$parameters) p$requires_grad_(FALSE)
    
    self$predictor <- nn_sequential(
      nn_linear(dim, dim),
      nn_silu(),
      nn_linear(dim, dim)
    )
    nn_init_zeros_(self$predictor[[3]]$weight)
    if (!is.null(self$predictor[[3]]$bias)) nn_init_zeros_(self$predictor[[3]]$bias)
    
    self$vq_layer <- RtomicVQ()
  },
  
  forward = function(input_data) {
    x <- input_data$x
    y <- input_data$y 
    
    h_x <- self$tok_emb(x)
    for (i in 1:length(self$layers)) h_x <- self$layers[[i]](h_x)
    h_x <- self$norm_f(h_x)
    
    pred_z <- h_x + self$predictor(h_x)
    res <- list(pred = pred_z)
    
    if (!is.null(y)) {
      with_no_grad({
        h_y <- self$tok_emb(y)
        for (i in 1:length(self$target_layers)) h_y <- self$target_layers[[i]](h_y)
        h_y <- self$target_norm_f(h_y)
      })
      
      vq_out <- self$vq_layer(h_y, self$codebook_weight)
      target_z_quantized <- vq_out$quantized
      
      pred_loss <- nnf_mse_loss(pred_z, target_z_quantized$detach())

      quantized_centroids <- self$codebook_weight[vq_out$indices]$view(h_y$size())
      commit_loss <- 0.25 * nnf_mse_loss(quantized_centroids, h_y$detach())
      
      # ==============================================================
      # 终极防坍塌：VICReg 正则化 (强制特征多样性)
      # ==============================================================
      pred_z_flat <- pred_z$reshape(c(-1, pred_z$size(3)))$to(dtype = torch_float32())
      N <- pred_z_flat$size(1) # 这个值高达 65408
      
      # 1. Variance Loss: 强制每个维度的方差 >= 1.0
      std_z <- torch_sqrt(torch_var(pred_z_flat, dim = 1, unbiased = FALSE) + 1e-4)
      var_loss <- torch_mean(nnf_relu(1.0 - std_z))
      
      # 2. Covariance Loss: 强制不同特征维度互相解耦
      z_centered <- pred_z_flat - pred_z_flat$mean(dim = 1, keepdim = TRUE)
      
      # 【核心数学修复：提前缩放，免疫 Float16 溢出】
      # 将除以 N-1 的操作拆解为除以 sqrt(N-1) 并在 matmul 之前执行
      scale_factor <- sqrt(N - 1)
      z_scaled <- z_centered / scale_factor
      
      # 现在送入 Tensor Core 的数字极小，累加绝对不会超过 65504
      cov_z <- torch_matmul(z_scaled$t(), z_scaled) 
      
      cov_z_off_diag <- cov_z - torch_diag(torch_diag(cov_z))
      cov_loss <- torch_sum(cov_z_off_diag$pow(2)) / cov_z$size(1)
      
      # 混合 Loss
      vicreg_loss <- 1.0 * var_loss + 0.04 * cov_loss
      
      # ==============================================================
      # 【核心新增】：真理之锚 (防止 Target 网络忽略当前 Token)
      # ==============================================================
      # 强制要求 pred_z 必须保留能直接映射回原始 Token 的语义信息
      logits_anchor <- torch_matmul(pred_z, self$tok_emb$weight$t())
      ce_loss <- nnf_cross_entropy(logits_anchor$view(c(-1, logits_anchor$size(-1))), y$view(c(-1)))
      
      # 混合 Loss：纯 JEPA 的世界观 (80%) + 强制的符号对齐 (20%)
      res$loss <- pred_loss + commit_loss + vicreg_loss + 0.2 * ce_loss
    }
    
    if (!self$training) {
      vq_out_pred <- self$vq_layer(pred_z, self$codebook_weight)
      res$quantized_pred <- vq_out_pred$quantized
      res$token_ids <- vq_out_pred$indices
    }
    return(res)
  }
)

# =====================================================================
# 6. 轻量级翻译器 (RtomicDecoder - MLP版)
# =====================================================================
RtomicDecoder <- nn_module(
  "RtomicDecoder",
  initialize = function(dim, vocab_size) {
    self$norm <- RMSNorm(dim)
    self$mlp <- nn_sequential(
      nn_linear(dim, dim * 4),
      nn_silu(),
      nn_linear(dim * 4, vocab_size, bias = FALSE)
    )
  },
  forward = function(latent_states) {
    h <- self$norm(latent_states)
    logits <- self$mlp(h)
    return(logits)
  }
)

# =====================================================================
# 7. 生成式包装器 (RtomicGenerator)
# =====================================================================
RtomicGenerator <- nn_module(
  "RtomicGenerator",
  initialize = function(jepa_ckpt_path, vocab_size, dim, jepa_layers, n_heads, max_seq_len, num_clusters = 4096) {
    self$jepa <- RtomicJEPA_VQ(vocab_size, dim, jepa_layers, n_heads, max_seq_len, num_clusters)
    
    ckpt <- torch_load(jepa_ckpt_path, device = "cpu")
    self$jepa$load_state_dict(ckpt$model)
    cat(sprintf("\n[World Model] 成功挂载 JEPA 基座权重: %s\n", jepa_ckpt_path))
    
    for (p in self$jepa$parameters) p$requires_grad_(FALSE)
    self$jepa$eval() 
    
    self$decoder <- RtomicDecoder(dim, vocab_size)
  },
  
  forward = function(input_data) {
    x <- input_data$x; y <- input_data$y
    with_no_grad({
      jepa_out <- self$jepa(list(x = x, y = NULL))
      latent_states <- jepa_out$pred 
    })
    
    logits <- self$decoder(latent_states)
    res <- list(logits = logits)
    
    if (!is.null(y)) {
      res$loss <- nnf_cross_entropy(logits$view(c(-1, logits$size(-1))), y$view(c(-1)))
    }
    return(res)
  }
)