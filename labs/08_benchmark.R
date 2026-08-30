# ==========================================
# 静默版 Log-Likelihood 批量自动化评测脚本
# 专为多模型横向对比设计
# ==========================================
library(jsonlite)
library(dplyr)
library(purrr)
library(tidyr)
library(torch)

source("config.R")
source("utils/BPETokenizer.R")

device <- torch_device("cpu")
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

# ==========================================
# 0. 模型注册表 (确保优先声明)
# ==========================================
MODEL_REGISTRY <- list(
  contrastive = list(
    label  = "Token Latent (Contrastive)",
    source = "latent_contrastive/contrastive_model.R",
    construct = function() {
      TokenLatentModel(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN)
    },
    ckpt   = "checkpoints/contrastive_sft_05.pt",
    load_fn = function(model, raw) {
      sd <- if ("model" %in% names(raw)) raw$model else raw
      model$load_state_dict(sd, strict = FALSE)
    },
    output = "contrastive" # 指定走对比学习路径
  ),

  moe = list(
    label  = "MoE Causal LM",
    source = "moe/moe_model.R",
    construct = function() {
      N_KV_HEADS  <- 2 
      NUM_EXPERTS <- 4
      TOP_K       <- 1
      RtomicCausalLM(
        vocab_size  = VOCAB_SIZE, 
        dim         = DIM, 
        n_layers    = N_LAYERS, 
        n_heads     = N_HEADS, 
        max_seq_len = SEQ_LEN,
        n_kv_heads  = N_KV_HEADS,
        num_experts = NUM_EXPERTS,
        top_k       = TOP_K
      )
    },
    ckpt   = "checkpoints/moe_sft_epoch_05.pt",
    load_fn = function(model, raw) {
      sd <- if ("model" %in% names(raw)) raw$model else raw
      model$load_state_dict(sd, strict = FALSE)
    },
    output = "logits"  # MoE 模型 eval 状态下直接输出 logits，无需 feature 点积
  ),

  # --- 原有模型保持不变 ---
  lrp = list(
    label  = "Latent Residual Predictor",
    source = "latent_residual/LRP_model.R",
    construct = function() RtomicLRP(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN),
    ckpt   = "checkpoints/lrp_sft_epoch_05.pt",
    load_fn = function(model, raw) {
      sd <- if ("model" %in% names(raw)) raw$model else raw
      model$load_state_dict(sd, strict = FALSE)
    },
    output = "pred"
  ),

  jepa = list(
    label  = "JEPA VQ",
    source = "world_model/jepa_model.R",
    construct = function() RtomicJEPA_VQ(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN, num_clusters = 4096),
    ckpt   = "checkpoints/jepa_sft_epoch_05.pt",
    load_fn = function(model, raw) {
      if ("model" %in% names(raw)) {
        clean <- list()
        for (nm in names(raw$model)) {
          clean[[sub("^model\\.", "", nm)]] <- raw$model[[nm]]
        }
        model$load_state_dict(clean, strict = FALSE)
      } else {
        model$load_state_dict(raw, strict = FALSE)
      }
    },
    output = "pred"
  ),

  causal = list(
    label  = "Causal LM",
    source = "causal_lm/CausalLM_model.R",
    construct = function() RtomicCausalLM(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN),
    ckpt   = "checkpoints/causal_sft_epoch_05.pt",
    load_fn = function(model, raw) {
      sd <- if ("model" %in% names(raw)) raw$model else raw
      model$load_state_dict(sd, strict = FALSE)
    },
    output = "logits"
  ),

  distill = list(
    label  = "distill",
    source = "causal_lm/CausalLM_model.R",
    construct = function() RtomicCausalLM(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN),
    ckpt   = "checkpoints/distill_kl_sft_epoch_05.pt",
    load_fn = function(model, raw) {
      sd <- if ("model" %in% names(raw)) raw$model else raw
      model$load_state_dict(sd, strict = FALSE)
    },
    output = "logits"
  )
)

# ==========================================
# 1. 核心 Log-Likelihood 打分函数
# ==========================================
evaluate_single_choice <- function(model, tokenizer, prompt, choice_text, output_type) {
  prompt_tokens <- c(tokenizer$bos_idx, tokenizer$encode_raw(prompt)[[1]])
  choice_tokens <- tokenizer$encode_raw(choice_text)[[1]]
  full_tokens <- c(prompt_tokens, choice_tokens)

  P <- length(prompt_tokens)
  N <- length(full_tokens)
  C <- N - P
  if (C <= 0) return(-Inf)

  with_no_grad({
    # 保持设备一致性
    input_tensor <- torch_tensor(full_tokens, dtype = torch_long(), device = model$tok_emb$weight$device)$unsqueeze(1)
    
    if (output_type == "logits") {
      out <- model(list(x = input_tensor, y = NULL))
      logits_full <- out$logits
    } else if (output_type == "contrastive") {
      # --- 严格对齐 03_inference.R 对比学习路径 ---
      h <- model$encode(input_tensor)
      pred_all <- model$predictor(h)  # [1, N, DIM]
      
      # 压掉 Batch 维度，变成 2D [N, DIM]，防止 C++ 维度判断异常
      pred_2d <- pred_all$squeeze(1) 
      
      # L2 归一化
      pred_norm <- nnf_normalize(pred_2d, p = 2, dim = -1)
      emb_norm <- nnf_normalize(model$tok_emb$weight, p = 2, dim = -1)
      
      # 余弦相似度 [N, VOCAB_SIZE] 并缩放
      cos_sim <- torch_matmul(pred_norm, emb_norm$t())
      logits_full <- cos_sim / model$temperature
      
      # 补充回 [1, N, VOCAB_SIZE] 维度，与后续统一切片逻辑完全兼容
      logits_full <- logits_full$unsqueeze(1)
    } else {
      out <- model(list(x = input_tensor, y = NULL))
      pred_states <- out$pred
      logits_full <- torch_matmul(pred_states, model$tok_emb$weight$t())
    }

    V <- logits_full$size(-1)
    logits_2d <- logits_full$contiguous()$view(c(N, V))
    log_probs_2d <- nnf_log_softmax(logits_2d, dim = -1)

    shift_log_probs <- log_probs_2d$narrow(dim = 1, start = P, length = C)$contiguous()
    shift_log_probs <- shift_log_probs$view(c(C, V))

    shift_targets <- torch_tensor(choice_tokens, dtype = torch_long(), device = logits_full$device)$view(c(C, 1))
    target_log_probs <- torch_gather(shift_log_probs, dim = -1, index = shift_targets)

    total_score <- as.numeric(target_log_probs$sum())
    return(total_score / C)
  })
}

evaluate_choices <- function(model, tokenizer, prompt, choices, output_type) {
  scores <- numeric(length(choices))
  for (i in seq_along(choices)) {
    scores[i] <- evaluate_single_choice(model, tokenizer, prompt, choices[i], output_type)
  }
  return(scores)
}

# ==========================================
# 2. 核心带进度条评测函数
# ==========================================
evaluate_model_silent <- function(model_key, dataset_path) {
  cfg <- MODEL_REGISTRY[[model_key]]
  if (is.null(cfg)) stop(sprintf("未知模型: %s", model_key))
  
  source(cfg$source)
  model <- cfg$construct()
  if (!file.exists(cfg$ckpt)) {
    warning(sprintf("[%s] Checkpoint 不存在: %s，跳过该模型", cfg$label, cfg$ckpt))
    return(NULL)
  }
  
  raw <- torch_load(cfg$ckpt)
  cfg$load_fn(model, raw)
  rm(raw); gc()
  model$eval()

  dataset <- jsonlite::stream_in(file(dataset_path), verbose = FALSE)
  total_questions <- nrow(dataset)
  choice_letters <- c("A", "B", "C", "D")

  cat(sprintf("\n 开始评估模型 [%s] (%d 道题):\n", cfg$label, total_questions))
  pb <- txtProgressBar(min = 0, max = total_questions, style = 3, char = "=")
  
  results_list <- vector("list", total_questions)
  
  for (i in 1:total_questions) {
    q_prompt <- dataset$question[i]
    q_choices <- if (is.matrix(dataset$choices)) as.character(dataset$choices[i, ]) else as.character(unlist(dataset$choices[[i]]))
    
    raw_ans_idx <- dataset$answer[i]
    true_idx <- if (is.numeric(raw_ans_idx)) (raw_ans_idx + 1) else (which(choice_letters == raw_ans_idx))
    
    scores <- evaluate_choices(model, tokenizer, q_prompt, q_choices, cfg$output)
    best_idx <- which.max(scores)
    
    results_list[[i]] <- data.frame(
      model         = cfg$label,
      model_key     = model_key,
      category      = ifelse("category" %in% names(dataset), dataset$category[i], "general"),
      subcategory   = ifelse("subcategory" %in% names(dataset), dataset$subcategory[i], "default"),
      is_correct    = (best_idx == true_idx),
      stringsAsFactors = FALSE
    )
    setTxtProgressBar(pb, i)
  }

  close(pb)
  df_res <- bind_rows(results_list)
  return(df_res)
}

# ==========================================
# 3. 多模型对比主入口
# ==========================================
compare_models <- function(model_keys, dataset_path) {
  all_results <- map_dfr(model_keys, ~evaluate_model_silent(.x, dataset_path))
  
  if (nrow(all_results) == 0) return(NULL)

  overall_summary <- all_results %>%
    group_by(model) %>%
    summarise(
      Total = n(),
      Correct = sum(is_correct),
      Accuracy = sprintf("%.2f%%", (Correct / Total) * 100),
      .groups = "drop"
    )

  subcat_summary <- all_results %>%
    group_by(subcategory, model) %>%
    summarise(acc = mean(is_correct) * 100, .groups = "drop") %>%
    pivot_wider(names_from = model, values_from = acc) %>%
    mutate(across(where(is.numeric), ~ sprintf("%.2f%%", .x)))

  cat("\n==================================================\n")
  cat("             多模型总体准确率对比              \n")
  cat("==================================================\n")
  print(as.data.frame(overall_summary))

  cat("\n==================================================\n")
  cat("             各子维度 (Subcategory) 对比       \n")
  cat("==================================================\n")
  print(as.data.frame(subcat_summary))

  return(list(overall = overall_summary, subcategory = subcat_summary, raw = all_results))
}

# ==========================================
# 4. 执行脚本
# ==========================================
target_models <- c("lrp") # 若要多模型对比传入 c("lrp", "causal", "jepa", "moe")
target_models <- c("causal", "lrp", "contrastive", "jepa", "moe")

benchmark_file <- "labs/files/benchmark_1300_v2.jsonl"
comparison_report <- compare_models(target_models, benchmark_file)

# 输出聚合详情
# 类别映射
category_map <- c(
  "adversarial" = "对抗鲁棒性",
  "instruction" = "指令遵循",
  "knowledge" = "知识记忆",
  "language" = "语言理解",
  "machine_learning" = "机器学习",
  "math" = "数学推理",
  "memory" = "记忆召回",
  "pattern" = "模式识别",
  "reasoning" = "逻辑推理"
)

# 子类别映射（按你原表翻译）
subcategory_map <- c(
  "distractor" = "干扰项识别",
  "instruction_attack" = "指令攻击",
  "format_following" = "格式遵循",
  "selection" = "选择筛选",
  "general_knowledge" = "常识知识",
  "ambiguity" = "歧义消解",
  "coreference" = "指代消解",
  "entailment" = "蕴含判断",
  "reading" = "阅读理解",
  "semantic_relation" = "语义关系",
  "machine_learning" = "机器学习知识",
  "arithmetic" = "算术运算",
  "multi_step" = "多步推理",
  "novel_rule" = "新颖规则",
  "sequence" = "序列预测",
  "word_problem" = "应用题",
  "fact_recall" = "事实回忆",
  "relational_recall" = "关系回忆",
  "reverse_recall" = "反向回忆",
  "analogy" = "类比推理",
  "number_pattern" = "数字规律",
  "symbol_pattern" = "符号规律",
  "conditional" = "条件逻辑",
  "constraints" = "约束满足",
  "ordering" = "排序顺序",
  "syllogism" = "三段论"
)

library(ggplot2)
library(dplyr)
library(forcats)

# 1. 计算一级维度聚合
df_category <- comparison_report$raw %>%
  group_by(model, category) %>%
  summarise(
    total_cnt = n(),
    correct_cnt = sum(is_correct),
    acc = (correct_cnt / total_cnt) * 100,
    .groups = "drop"
  ) %>%
  mutate(category_zh = category_map[category]) %>%
  select(model, category_zh, acc)

# 2. 计算总体维度聚合
df_overall <- comparison_report$raw %>%
  group_by(model) %>%
  summarise(
    acc = sum(is_correct) / n() * 100,
    .groups = "drop"
  ) %>%
  mutate(category_zh = "★ 总体平均")

# 合并数据
df_all <- bind_rows(df_overall, df_category)

# 3. 统计每维度的极值
df_summary <- df_all %>%
  group_by(category_zh) %>%
  summarise(
    min_acc = min(acc),
    max_acc = max(acc),
    mean_acc = mean(acc),
    .groups = "drop"
  )

# 排序逻辑：确保“★ 总体平均”置顶，其余维度按平均分排序
cat_order <- c(
  df_summary %>% filter(category_zh != "★ 总体平均") %>% arrange(mean_acc) %>% pull(category_zh),
  "★ 总体平均"
)

df_plot <- df_all %>%
  left_join(df_summary, by = "category_zh") %>%
  mutate(
    is_max = (acc == max_acc),
    category_zh = factor(category_zh, levels = cat_order)
  )

# 4. 绘图
p <- ggplot(df_plot, aes(y = category_zh)) +
  # 背景线条
  geom_segment(
    data = df_summary %>% mutate(category_zh = factor(category_zh, levels = cat_order)),
    aes(y = category_zh, yend = category_zh, x = min_acc, xend = max_acc),
    color = "#E5E7EB", linewidth = 3, lineend = "round"
  ) +
  # 所有点
  geom_point(
    aes(x = acc, color = model),
    size = 4, alpha = 0.85
  ) +
  # 冠军光圈
  geom_point(
    data = filter(df_plot, is_max),
    aes(x = acc, color = model),
    size = 5.5, shape = 21, stroke = 1.5, fill = "white"
  ) +
  # 冠军数值
  geom_text(
    data = filter(df_plot, is_max),
    aes(x = acc, label = sprintf("%.1f%%", acc)),
    hjust = -0.4, vjust = 0.4, size = 3.5, fontface = "bold", color = "#374151"
  ) +
  scale_x_continuous(
    labels = function(x) paste0(x, "%"),
    limits = c(min(df_plot$acc) - 3, max(df_plot$acc) + 8),
    breaks = seq(0, 100, by = 10)
  ) +
  scale_color_brewer(palette = "Set1") +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(color = "#F3F4F6", linewidth = 0.8),
    axis.text.y = element_text(face = "bold", color = "#1F2937", size = 11),
    axis.title.x = element_text(color = "#4B5563", margin = margin(t = 10)),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    plot.title = element_text(face = "bold", size = 16, color = "#111827", margin = margin(b = 5)),
    plot.subtitle = element_text(color = "#6B7280", size = 11, margin = margin(b = 15)),
    plot.margin = margin(20, 20, 20, 20)
  ) +
  labs(
    title = "大语言模型能力评估对比 (含总体成绩)",
    subtitle = "置顶项为全量题目综合准确率，带光圈的点代表该项最佳模型",
    x = "准确率 (Accuracy %)",
    y = NULL,
    color = "评测模型"
  )
p

ggsave(
  filename = "~/github/stat-llm-lab/img/distill_acc.png",  # 文件名
  plot = p,                  # 你的 ggplot 对象
  width = 8,                 # 图片宽度
  height = 5,                # 图片高度
  units = "in",              # 单位：英寸 (in), 厘米 (cm), 毫米 (mm), 像素 (px)
  dpi = 300                  # 分辨率，300 dpi 是出版物标准
)
