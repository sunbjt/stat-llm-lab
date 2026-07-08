# causal_lm/09_representation_engineering.R
# =====================================================================
# 基于内部表征工程 (RepE) 的幻觉检测 (PCA 潜空间投影)
# =====================================================================

source("config.R")
source("utils/BPETokenizer.R")
source("causal_lm/CausalLM_model.R")

# =====================================================================
# 重新实例化模型 (使用带有 hidden_states 开关的最新版类定义)
# =====================================================================
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

# =====================================================================
# 第一步：提取特定 Prompt 的最后一个 Token 的 Hidden State
# =====================================================================
get_last_hidden_state <- function(model, tokenizer, text) {
  raw_ids <- tokenizer$encode_raw(text)[[1]]
  input_ids <- c(tokenizer$bos_idx, raw_ids)
  
  x_tensor <- torch_tensor(input_ids, dtype = torch_long(), device = device)$unsqueeze(1)
  
  with_no_grad({
    output <- model(list(x = x_tensor, y = NULL, loss_mask = NULL),
                    output_hidden_states = TRUE)
  })
  
  # 获取序列最后一个 Token 的隐藏层向量 (Shape: [1, Dim])
  seq_len <- x_tensor$size(2)
  # 假设 hidden_states 的 shape 是 [Batch, Seq, Dim]
  last_hidden <- output$hidden_states[1, seq_len, ] 
  
  # 转换为 R 的 numeric 向量以便后续做 PCA
  return(as.numeric(last_hidden$cpu()))
}

# =====================================================================
# 第二步：构造对比数据，计算“诚实”方向 (PCA PC1)
# =====================================================================
compute_honesty_direction <- function(model, tokenizer) {
  cat("\n[RepE] 正在提取对比数据的潜空间表征...\n")
  
  # 构造具有明显对立事实且领域多样的 Prompt 集合
  true_statements <- c(
    # 基础常识与地理
    "中国的首都是北京",
    "地球是太阳系中的一颗行星",
    "太平洋是地球上最大的洋",
    "人类需要呼吸氧气才能生存",
    "水由氢和氧两种元素组成",
    
    # 数学与物理逻辑
    "三角形的内角和是180度",
    "一加一等于二",
    "偶数是可以被2整除的整数",
    "光在真空中的传播速度极快",
    "水在标准大气压下的沸点是100度",
    
    # 机器学习与数据科学
    "在深度学习中，反向传播用于更新网络权重",
    "随机森林是一种基于决策树的集成算法",
    "R语言广泛应用于统计分析和数据挖掘",
    "主成分分析(PCA)是一种常用的降维技术",
    "过拟合意味着模型在训练集上表现好但在测试集上表现差",
    
    # 历史与科学史
    "艾萨克·牛顿提出了万有引力定律",
    "阿西莫夫提出了著名的机器人三定律",
    "第二次世界大战在1945年结束",
    "爱因斯坦是相对论的提出者",
    "圆周率π是一个无理数"
  )
  
  false_statements <- c(
    # 基础常识与地理 (严格对应虚假)
    "中国的首都是伦敦",
    "太阳是地球系中的一颗行星",
    "太平洋是地球上最小的湖泊",
    "人类需要呼吸二氧化碳才能生存",
    "水由金和银两种元素组成",
    
    # 数学与物理逻辑
    "三角形的内角和是360度",
    "一加一等于五",
    "奇数是可以被2整除的整数",
    "声音在真空中的传播速度极快",
    "水在标准大气压下的沸点是50度",
    
    # 机器学习与数据科学
    "在深度学习中，反向传播用于随机生成训练数据",
    "随机森林是一种基于线性回归的聚类算法",
    "R语言是一种专门用于编写操作系统内核的汇编语言",
    "主成分分析(PCA)是一种用于增加数据维度的技术",
    "过拟合意味着模型在训练集和测试集上都表现极差",
    
    # 历史与科学史
    "艾萨克·牛顿提出了量子纠缠理论",
    "阿西莫夫提出了著名的相对论",
    "第二次世界大战在2015年结束",
    "爱因斯坦是微积分的发明者",
    "圆周率π是一个可以写成有限小数的有理数"
  )
  
  hidden_matrix <- list()
  labels <- c()
  
  # 收集 True 的表征
  for (text in true_statements) {
    hidden_matrix[[length(hidden_matrix) + 1]] <- get_last_hidden_state(model, tokenizer, text)
    labels <- c(labels, 1) # 1 代表真实
  }
  
  # 收集 False 的表征
  for (text in false_statements) {
    hidden_matrix[[length(hidden_matrix) + 1]] <- get_last_hidden_state(model, tokenizer, text)
    labels <- c(labels, -1) # -1 代表虚假/幻觉
  }
  
  # 拼合矩阵 (Row: 样本, Col: 隐藏层维度)
  H <- do.call(rbind, hidden_matrix)
  
  # 对隐向量矩阵进行 PCA 分析
  # 中心化数据
  H_centered <- scale(H, center = TRUE, scale = FALSE)
  pca_res <- prcomp(H_centered, center = FALSE)
  
  # 提取第一主成分 (PC1)，这就是我们在高维潜空间中找到的“真实/幻觉”维度
  honesty_vector <- pca_res$rotation[, 1]
  
  # 校验 PC1 的方向（确保正数代表真实）
  # 将真实样本投影到 PC1 上，如果均值为负，则翻转向量方向
  true_projections <- H_centered[labels == 1, ] %*% honesty_vector
  if (mean(true_projections) < 0) {
    honesty_vector <- -honesty_vector
  }
  
  cat("[RepE] 成功提取 '诚实' 潜空间特征向量 (维度:", length(honesty_vector), ")\n")
  return(honesty_vector)
}

# =====================================================================
# 第三步：带采样策略与 RepE 监控的生成函数
# =====================================================================
generate_with_repe <- function(model, tokenizer, prompt, honesty_vector, max_new_tokens = 50,
                                   temperature = 0.8, top_k = 50, repetition_penalty = 1.1,
                                   verbose = FALSE) {
  raw_ids <- tokenizer$encode_raw(prompt)[[1]]
  current_ids <- c(tokenizer$bos_idx, raw_ids)
  eos_val <- if (!is.null(tokenizer$eos_idx)) tokenizer$eos_idx else 4L

  # 收集生成结果用于干净输出
  tokens <- character(0)
  scores <- numeric(0)

  cat(sprintf("\n[输入 Prompt]: %s\n", prompt))

  with_no_grad({
    for (i in 1:max_new_tokens) {
      input_seq <- tail(current_ids, SEQ_LEN)
      x_tensor <- torch_tensor(input_seq, dtype = torch_long(), device = device)$unsqueeze(1)

      output <- model(list(x = x_tensor, y = NULL, loss_mask = NULL), output_hidden_states = TRUE)
      seq_length <- output$logits$size(2)
      logits_vec <- output$logits[1, seq_length, ]$clone()

      if (repetition_penalty != 1.0) {
        unique_ids <- unique(current_ids)
        for (idx in unique_ids) {
          score <- logits_vec[idx]$item()
          if (score > 0) {
            logits_vec[idx] <- score / repetition_penalty
          } else {
            logits_vec[idx] <- score * repetition_penalty
          }
        }
      }

      if (temperature != 1.0 && temperature > 0) {
        logits_vec <- logits_vec / temperature
      }

      vocab_size <- logits_vec$size(1)
      if (top_k > 0 && top_k < vocab_size) {
        top_k_res <- torch_topk(logits_vec, k = top_k)
        kth_val <- top_k_res[[1]][top_k]
        logits_vec$masked_fill_(logits_vec < kth_val, -Inf)
      }

      probs <- nnf_softmax(logits_vec, dim = -1)
      next_token_id <- as.integer(torch_multinomial(probs, num_samples = 1))

      if (next_token_id == eos_val) break

      next_word <- tokenizer$decode(next_token_id)
      current_hidden <- as.numeric(output$hidden_states[1, seq_length, ]$cpu())
      repe_score <- sum(current_hidden * honesty_vector)

      tokens <- c(tokens, next_word)
      scores <- c(scores, repe_score)
      current_ids <- c(current_ids, next_token_id)
    }
  })

  # --- 输出 ---
  if (verbose) {
    # 逐 token 彩色输出
    cat("[逐 token RepE 监控]: ")
    for (j in seq_along(tokens)) {
      s <- scores[j]
      if (s < -2.0) {
        cat(sprintf("\033[31m%s[%.1f]\033[0m", tokens[j], s))
      } else if (s < 0) {
        cat(sprintf("\033[33m%s[%.1f]\033[0m", tokens[j], s))
      } else {
        cat(tokens[j])
      }
    }
    cat("\n\n")
  } else {
    # 干净模式：先打印完整文本，再打印汇总表
    cat("\n--- 生成文本 ---\n")
    cat(paste0(tokens, collapse = ""), "\n")

    cat("\n--- RepE 监控汇总 ---\n")
    n_high_risk <- sum(scores < -2.0)
    n_low_risk  <- sum(scores < 0 & scores >= -2.0)
    n_safe      <- sum(scores >= 0)
    cat(sprintf("总 token 数: %d\n", length(scores)))
    cat(sprintf("高风险 (score < -2):  %d (%.0f%%)\n", n_high_risk, 100 * n_high_risk / length(scores)))
    cat(sprintf("低风险 (-2 <= score < 0): %d (%.0f%%)\n", n_low_risk,  100 * n_low_risk  / length(scores)))
    cat(sprintf("正常 (score >= 0): %d (%.0f%%)\n", n_safe,      100 * n_safe      / length(scores)))

    if (n_high_risk > 0 || n_low_risk > 0) {
      cat("\n--- 风险 token 详情 ---\n")
      for (j in seq_along(tokens)) {
        if (scores[j] < 0) {
          risk_label <- if (scores[j] < -2.0) "HIGH" else " LOW"
          cat(sprintf("  [%s] score=%+.1f  token=\"%s\"\n", risk_label, scores[j], tokens[j]))
        }
      }
    }
    cat("\n")
  }
}

# --- 运行测试 ---
honesty_vec <- compute_honesty_direction(causal_model, tokenizer)
test_prompt <- "机器学习是一项对于企业非常有用的技术，它能够"
generate_with_repe(causal_model, tokenizer, test_prompt, honesty_vec,
   temperature = 0.15, top_k = 5, repetition_penalty = 1.1, max_new_tokens = 100)

# =====================================================================
# 方法总结
# =====================================================================
# RepE (Representation Engineering) 通过以下步骤检测幻觉：
#
# 1. 构造对比语句对（20 条真实 + 20 条虚假），提取模型最后一层
#    每个语句最后一个 token 的 hidden state（维度 = DIM）。
#
# 2. 对 40×320 的隐向量矩阵做 PCA，取第一主成分 (PC1) 作为
#    "诚实方向" (honesty_vector)。方向已校准：真实语句投影为正。
#
# 3. 生成时，每个新 token 的 hidden state 与 honesty_vector 做点积，
#    得到 RepE 分数。分数 < 0 表示模型内部表征偏向"虚假"方向，
#    < -2.0 为高风险幻觉。
#
# 核心假设：模型对真实/虚假陈述的内部表征在潜空间中线性可分，
# PC1 恰好捕捉了这个分离方向。
