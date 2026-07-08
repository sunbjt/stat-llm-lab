# server/generate_stream.R

# 同步版本（保留兼容性，供批量推理或非 Web 场景使用）
generate_causal_text_stream <- function(model, tokenizer, prompt,
                                        max_new_tokens = 100,
                                        temperature = 0.15,
                                        rep_penalty = 1.25,
                                        callback = NULL) {
  model$eval()

  # 1. 编码 Prompt
  tokens <- tokenizer$encode_raw(prompt)[[1]]
  tokens <- c(tokenizer$bos_idx, tokens)

  device <- model$tok_emb$weight$device
  current_tokens <- tokens

  max_ctx <- SEQ_LEN  # pos_emb 长度，避免闭包内 torch tensor 方法派发失败

  # 2. 自回归循环开始
  for (step in 1:max_new_tokens) {
    # 截取历史长度，防止超出最大上下文
    seq_len <- length(current_tokens)
    if (seq_len > max_ctx) {
      context_tokens <- current_tokens[(seq_len - max_ctx + 1):seq_len]
    } else {
      context_tokens <- current_tokens
    }

    # 构造 Tensor
    x_tensor <- torch_tensor(matrix(context_tokens, nrow = 1), dtype = torch_long(), device = device)
    input_data <- list(x = x_tensor, y = NULL, loss_mask = NULL)

    # 前向传播
    with_no_grad({
      output <- model(input_data)
      logits <- output$logits[1, length(context_tokens), ] # 提取最后一个 Token 的预测分布
    })

    # 惩罚重复词 (Repetition Penalty)
    if (rep_penalty != 1.0) {
      unique_tokens <- unique(current_tokens)
      logits[unique_tokens] <- logits[unique_tokens] / rep_penalty
    }

    # 采样逻辑
    if (temperature == 0) {
      next_token <- as.integer(torch_argmax(logits)$item())
    } else {
      probs <- nnf_softmax(logits / temperature, dim = -1)
      next_token <- as.integer(torch_multinomial(probs, num_samples = 1)$item())
    }

    # 检查 EOS
    if (next_token == tokenizer$eos_idx || next_token == 4L) {
      break
    }

    # 流式回调
    if (!is.null(callback)) {
      token_text <- tokenizer$decode(list(next_token))
      callback(token_text)
    }

    current_tokens <- c(current_tokens, next_token)
  }

  return(tokenizer$decode(list(current_tokens)))
}

# 异步流式版本：逐 token 生成，每个 token 通过 later::later 调度，
# 让出控制权给 httpuv 事件循环，确保每个 token 被即时推送到浏览器。
generate_causal_text_async <- function(model, tokenizer, prompt,
                                       max_new_tokens = 150,
                                       temperature = 0.15,
                                       rep_penalty = 1.25,
                                       on_token = NULL,
                                       on_done = NULL) {
  model$eval()

  tokens <- tokenizer$encode_raw(prompt)[[1]]
  tokens <- c(tokenizer$bos_idx, tokens)
  device <- model$tok_emb$weight$device

  # 将可变状态封装在环境中，避免闭包复制大量数据
  state <- new.env(parent = emptyenv())
  state$current_tokens <- tokens
  state$step <- 0
  state$first_visible <- TRUE  # 标记：还未产生第一个可见 token
  state$max_new_tokens <- max_new_tokens
  state$temperature <- temperature
  state$rep_penalty <- rep_penalty
  max_ctx <- SEQ_LEN  # 预计算，避免 later 回调内 torch tensor 方法派发失败

  step_fn <- function() {
    if (state$step >= state$max_new_tokens) {
      if (!is.null(on_done)) on_done()
      return()
    }

    # 上下文窗口截断
    seq_len <- length(state$current_tokens)
    if (seq_len > max_ctx) {
      context_tokens <- state$current_tokens[(seq_len - max_ctx + 1):seq_len]
    } else {
      context_tokens <- state$current_tokens
    }

    x_tensor <- torch_tensor(matrix(context_tokens, nrow = 1),
                              dtype = torch_long(), device = device)

    with_no_grad({
      output <- model(list(x = x_tensor, y = NULL, loss_mask = NULL))
      logits <- output$logits[1, length(context_tokens), ]
    })

    # 重复惩罚
    if (state$rep_penalty != 1.0) {
      unique_tokens <- unique(state$current_tokens)
      logits[unique_tokens] <- logits[unique_tokens] / state$rep_penalty
    }

    # 采样
    if (state$temperature == 0) {
      next_token <- as.integer(torch_argmax(logits)$item())
    } else {
      probs <- nnf_softmax(logits / state$temperature, dim = -1)
      next_token <- as.integer(torch_multinomial(probs, num_samples = 1)$item())
    }

    # 检查 EOS
    if (next_token == tokenizer$eos_idx || next_token == 4L) {
      if (!is.null(on_done)) on_done()
      return()
    }

    # 回调：把当前 token 推送给前端
    if (!is.null(on_token)) {
      token_text <- tokenizer$decode(list(next_token))

      ## 首 Token 幽灵标点拦截（针对第一个可见 token）
      if (state$first_visible) {
        token_text <- sub("^[[:space:][:punct:]。？！，、：；]+", "", token_text, perl = TRUE)
        if (nchar(token_text) > 0) state$first_visible <- FALSE
      }

      if (nchar(token_text) > 0) {
        on_token(token_text)
      }
    }

    state$current_tokens <- c(state$current_tokens, next_token)
    state$step <- state$step + 1

    # 让出控制权给 httpuv 事件循环，确保 WebSocket 帧被及时刷新
    later::later(step_fn, 0.001)
  }

  # 启动异步自回归循环
  later::later(step_fn, 0.001)
}