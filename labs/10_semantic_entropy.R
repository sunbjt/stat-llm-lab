# causal_lm/10_semantic_entropy.R
# =====================================================================
# 基于语义熵 (Semantic Entropy) 的不确定性与幻觉量化
# =====================================================================

source("config.R")
source("utils/BPETokenizer.R")
source("causal_lm/CausalLM_model.R")

tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

# 1. 重新构建模型对象
causal_model <- RtomicCausalLM(
  vocab_size = VOCAB_SIZE,
  dim = DIM,
  n_layers = N_LAYERS,
  n_heads = N_HEADS,
  max_seq_len = SEQ_LEN
)

device <- torch_device("cpu")
causal_model <- causal_model$to(device = device)

# 2. 重新加载预训练权重
ckpt_path <- "checkpoints/causal_model_03.pt"
checkpoint_data <- torch_load(ckpt_path)
state <- if (!is.null(checkpoint_data$model)) checkpoint_data$model else checkpoint_data
causal_model$load_state_dict(state)

# 3. 切入评估模式
causal_model$eval()

# 这里封装一个纯文本生成的闭包，屏蔽底层的 token 循环细节
generate_single_response <- function(model, tokenizer, prompt, max_new_tokens = 30, temperature = 1.0) {
  raw_ids <- tokenizer$encode_raw(prompt)[[1]]
  current_ids <- c(tokenizer$bos_idx, raw_ids)
  eos_val <- if (!is.null(tokenizer$eos_idx)) tokenizer$eos_idx else 4L
  
  with_no_grad({
    for (i in 1:max_new_tokens) {
      input_seq <- tail(current_ids, SEQ_LEN)
      x_tensor <- torch_tensor(input_seq, dtype = torch_long(), device = device)$unsqueeze(1)
      output <- model(list(x = x_tensor, y = x_tensor, loss_mask = NULL))
      
      logits_vec <- output$logits[1, output$logits$size(2), ] / temperature
      probs <- nnf_softmax(logits_vec, dim = -1)
      next_token_id <- as.integer(torch_multinomial(probs, num_samples = 1))
      
      if (next_token_id == eos_val) break
      current_ids <- c(current_ids, next_token_id)
    }
  })
  
  # 截取新生成的部分并解码成完整字符串
  generated_ids <- current_ids[(length(raw_ids) + 2):length(current_ids)]
  return(tokenizer$decode(generated_ids))
}

# =====================================================================
# 核心：语义判定与聚类引擎
# =====================================================================
# 在工程落地中，这里通常调用一个轻量级的 NLI（自然语言推理）模型，
# 或者直接用 API (如 Claude) 做 LLM-as-a-judge 判断两句话语义是否等价。
# 为了在本地闭环演示，这里使用一个基于字符串相似度或精确匹配的 Mock 函数。
check_semantic_equivalence <- function(text1, text2) {
  # Mock: 去除空格和标点后，如果高度相似，则认为语义等价
  clean1 <- gsub("[[:punct:][:space:]]", "", text1)
  clean2 <- gsub("[[:punct:][:space:]]", "", text2)
  
  # 现实中这里是： return(NLI_model(text1, text2) == "Entailment")
  return(clean1 == clean2) 
}

# 将 N 个采样结果进行语义聚类
cluster_responses <- function(responses) {
  clusters <- list()
  
  for (res in responses) {
    matched <- FALSE
    if (length(clusters) > 0) {
      for (i in seq_along(clusters)) {
        # 和当前簇的代表（第一个元素）比较语义
        if (check_semantic_equivalence(res, clusters[[i]][1])) {
          clusters[[i]] <- c(clusters[[i]], res)
          matched <- TRUE
          break
        }
      }
    }
    # 如果和已有的簇都不匹配，自立门户
    if (!matched) {
      clusters[[length(clusters) + 1]] <- c(res)
    }
  }
  return(clusters)
}

# =====================================================================
# 计算语义熵 (Semantic Entropy)
# =====================================================================
evaluate_hallucination_with_SE <- function(model, tokenizer, prompt, num_samples = 10) {
  cat(sprintf("\n[输入 Prompt]: %s\n", prompt))
  cat("正在进行蒙特卡洛采样 (N =", num_samples, ")...\n")
  
  responses <- character(num_samples)
  for (i in 1:num_samples) {
    # 必须保持较高的温度以释放模型的随机性，探测其概率空间的边界
    responses[i] <- generate_single_response(model, tokenizer, prompt, temperature = 1.0)
  }
  
  # 语义聚类
  semantic_clusters <- cluster_responses(responses)
  
  # 计算每个簇的概率 P(C) 以及 香农熵
  entropy <- 0
  cat("\n--- 语义聚类结果 ---\n")
  for (i in seq_along(semantic_clusters)) {
    cluster <- semantic_clusters[[i]]
    p_c <- length(cluster) / num_samples
    
    # 香农熵累加
    entropy <- entropy - (p_c * log(p_c))
    
    cat(sprintf("簇 %d (概率 %.0f%%): '%s' (包含 %d 个变体)\n", 
                i, p_c * 100, cluster[1], length(cluster)))
  }
  
  cat(sprintf("\n=> 最终语义熵 (Semantic Entropy): %.4f\n", entropy))
  
  # 阈值判定 (熵越小，越确信；熵越大，幻觉越严重)
  if (entropy < 0.5) {
    cat("\033[32m[结论] 模型非常确信，可安全信任。\033[0m\n")
  } else if (entropy < 1.2) {
    cat("\033[33m[结论] 模型存在一定犹豫，需结合业务阈值使用。\033[0m\n")
  } else {
    cat("\033[31m[结论] 极高幻觉风险！模型处于瞎猜状态，建议触发拒绝回答。\033[0m\n")
  }
}

# 测试
test_prompt <- "机器学习是一项对于企业非常有用的技术，它能够"
evaluate_hallucination_with_SE(causal_model, tokenizer, test_prompt, num_samples = 10)
