# server/jepa_generate_stream.R
# =====================================================================
# JEPA 架构专属的异步流式生成流水线
# =====================================================================

generate_lrp_text_async <- function(model, tokenizer, prompt,
                                     max_new_tokens = 150,
                                     temperature = 0.3, # SFT 推荐低温
                                     top_k = 5,
                                     rep_penalty = 1.2,
                                     on_token = NULL,
                                     on_done = NULL) {
  model$eval()
  
  eos_val <- if (!is.null(tokenizer$eos_idx)) tokenizer$eos_idx else 2L
  tokens <- tokenizer$encode_raw(prompt)[[1]]
  tokens <- c(tokenizer$bos_idx, tokens)
  
  device <- model$tok_emb$weight$device
  
  # 将可变状态封装在环境中，避免闭包复制导致内存泄漏和执行缓慢
  state <- new.env(parent = emptyenv())
  state$current_tokens <- tokens
  state$step <- 0
  state$first_visible <- TRUE  # 标记：还未产生第一个可见 token
  state$max_new_tokens <- max_new_tokens
  state$temperature <- temperature
  state$top_k <- top_k
  state$rep_penalty <- rep_penalty
  
  max_ctx <- SEQ_LEN # 预计算，保持闭包轻量
  
  step_fn <- function() {
    if (state$step >= state$max_new_tokens) {
      if (!is.null(on_done)) on_done()
      return()
    }
    
    # 1. 上下文窗口截断
    seq_len <- length(state$current_tokens)
    if (seq_len > max_ctx) {
      context_tokens <- state$current_tokens[(seq_len - max_ctx + 1):seq_len]
    } else {
      context_tokens <- state$current_tokens
    }
    
    x_tensor <- torch_tensor(matrix(context_tokens, nrow = 1),
                             dtype = torch_long(), device = device)
    
    with_no_grad({
      # 2. JEPA 前向传播与特征提取
      output <- model(list(x = x_tensor, y = NULL))
      last_pred <- output$pred[1, length(context_tokens), ]
      
      # 3. 潜空间到词表的点积投影
      cb <- model$tok_emb$weight
      logits <- torch_matmul(last_pred$unsqueeze(1), cb$t())$squeeze(1)
      logits <- logits / state$temperature
      
      # 4. 防弹版重复惩罚逻辑
      if (state$rep_penalty != 1.0) {
        unique_past_ids <- unique(state$current_tokens)
        for (past_id in unique_past_ids) {
          logit_val <- as.numeric(logits[past_id])
          if (logit_val < 0) {
            logits[past_id] <- logit_val * state$rep_penalty
          } else {
            logits[past_id] <- logit_val / state$rep_penalty
          }
        }
      }
      
      # 5. Top-K 截断
      if (state$top_k > 0) {
        topk_res <- torch_topk(logits, k = state$top_k)
        kth_value <- topk_res[[1]][state$top_k]
        logits <- torch_where(logits < kth_value, torch_tensor(-Inf, device = device), logits)
      }
      
      # 6. 概率采样
      probs <- nnf_softmax(logits, dim = -1)
      next_token <- as.integer(torch_multinomial(probs, num_samples = 1)$item())
    })
    
    # 7. 检查 EOS
    if (next_token == eos_val || next_token == 4L) {
      if (!is.null(on_done)) on_done()
      return()
    }
    
    # 8. 执行 WebSocket 回调
    if (!is.null(on_token)) {
      token_text <- tokenizer$decode(list(next_token))
      
      ## 首 Token 幽灵标点拦截（针对第一个可见 token，而非 step==0）
      if (state$first_visible) {
        # 使用 sub 正则替换掉开头的各种空白符以及中英文标点
        # [[:space:]] 匹配空格/换行，[[:punct:]] 匹配英文标点，外加常见的中文标点集合
        token_text <- sub("^[[:space:][:punct:]。？！，、：；]+", "", token_text, perl = TRUE)
      }

      # 只有当清除了前导标点后，Token 仍然有实质性文本时，才推送给前端
      if (nchar(token_text) > 0) {
        on_token(token_text)
        state$first_visible <- FALSE
      }
    }
    
    # 注意：生成的原始 Token 依然要原封不动地加入上下文，保持自回归张量的对齐
    state$current_tokens <- c(state$current_tokens, next_token)
    state$step <- state$step + 1
    
    # 将控制权交还 httpuv 事件循环，保证前端打字机效果平滑
    later::later(step_fn, 0.001)
  }
  
  # 启动异步循环
  later::later(step_fn, 0.001)
}