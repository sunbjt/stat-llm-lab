# causal_lm/utils/CausalLM_model.R
# =====================================================================
# 纯正的 Decode-Only Causal Language Model 架构 (集成 RoPE 与显式初始化)
# =====================================================================

library(torch)
source("utils/atomic_blocks.R")

RtomicCausalLM <- nn_module(
  "RtomicCausalLM",
  initialize = function(vocab_size, dim, n_layers, n_heads, max_seq_len) {
    self$tok_emb <- nn_embedding(vocab_size, dim)
    
    # 彻底移除旧版绝对位置编码 pos_emb，将 max_seq_len 传递给子 Block
    self$layers <- nn_module_list(lapply(1:n_layers, function(i) RtomicBlock(dim, n_heads, max_seq_len)))
    self$norm_f <- RMSNorm(dim)
    
    # 针对大模型与 SwiGLU 优化的显式参数初始化
    # 1. 基础映射层：一键全部初始化
    self$apply(function(m) {
      if (inherits(m, c("nn_linear", "nn_embedding"))) {
        nn_init_normal_(m$weight, mean = 0, std = 0.02)
      }
    })
    
    # 2. 残差输出层：单独阻断方差爆炸
    res_std <- 0.02 / sqrt(2 * n_layers)
    for (i in 1:n_layers) {
      nn_init_normal_(self$layers[[i]]$out_proj$weight, std = res_std)
      nn_init_normal_(self$layers[[i]]$w3$weight, std = res_std)
    }
  },
  
forward = function(input_data, output_hidden_states = FALSE, logit_lens = FALSE) {
    x <- input_data$x
    y <- input_data$y
    loss_mask <- input_data$loss_mask
    device <- x$device
    
    # 1. 纯净的 Token Embedding
    h <- self$tok_emb(x)
    
    # 用于存储每一层的 Logits 投影
    layer_logits <- list()
    
    # 【Logit Lens 拦截点 0】：查看刚过 Embedding 还没进任何 Transformer 层时的状态
    if (logit_lens) {
      h_norm <- self$norm_f(h)
      layer_logits[["Layer_0 (Emb)"]] <- nnf_linear(h_norm, weight = self$tok_emb$weight)
    }
    
    # 2. 穿过所有的 Transformer Layers
    for (i in 1:length(self$layers)) {
      h <- self$layers[[i]](h)
      
      # 【Logit Lens 拦截点 i】：拦截第 i 层的输出，用最终的 norm_f 和大头投影
      if (logit_lens) {
        h_norm <- self$norm_f(h)
        layer_logits[[paste0("Layer_", i)]] <- nnf_linear(h_norm, weight = self$tok_emb$weight)
      }
    }
    
    # 标准的 hidden_states 指的是未经过最终 RMSNorm 的残差输出
    hidden_states_raw <- h
    
    # 3. 最终层归一化与解压到 Logits Space
    h <- self$norm_f(h)
    # 嵌入层与输出层权重共享
    logits <- nnf_linear(h, weight = self$tok_emb$weight)
    
    # 4. 统一初始化返回列表，避免冗余的 if-else 结构
    output <- list(logits = logits)
    
    if (output_hidden_states) {
      output$hidden_states <- hidden_states_raw
    }
    
    if (logit_lens) {
      output$layer_logits <- layer_logits
    }
  
    # 依据“是否提供标签 y”来决定是否计算 Loss
    if (!is.null(y)) {
      logits_flat <- logits$view(c(-1, logits$size(-1)))
      y_flat <- y$view(c(-1))
      
      # 预训练没有 mask，SFT 环节 mask 才生效。
      if (!is.null(loss_mask)) {
        mask_flat <- loss_mask$view(c(-1))
        logits_flat <- logits_flat[mask_flat, ]
        y_flat <- y_flat[mask_flat]
      }
      
      # 保证数据为空时，也能维持反向传播的管道不被堵死
      if (logits_flat$size(1) == 0) {
        # 如果被 mask 完了，动态匹配 requires_grad 状态
        output$loss <- torch_tensor(0, requires_grad = self$training, device = device)
      } else {
        output$loss <- nnf_cross_entropy(logits_flat, y_flat)
      }
    }
    
    return(output)
}
)