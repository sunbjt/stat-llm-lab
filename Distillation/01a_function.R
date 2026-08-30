library(arrow)
library(tidyverse)

# 1. 载入环境配置（自动设置 WORK_DIR 和加载 RtomicBPETokenizer 依赖）
source("config.R") 
source("utils/BPETokenizer.R")
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

dataset_path <- "~/github/stat-llm-lab/data/processed/teacher_chunks/chunk_001.arrow"
arrow_table <- read_feather(dataset_path)

# 需要的是非 normalize 的结果
# ==================== 1. 提取第 1 条数据的基础元信息 ====================
text_idx <- arrow_table$text_idx[1]
ct <- arrow_table$ct[1]
n_teacher <- arrow_table$n_teacher[1]  # 假设为 200
top_k <- 16                            # 生成脚本中设置的 TOP_K

cat("文本总字符数:", nchar(ct), "\n")
cat("教师模型 Token 数 (n_teacher):", n_teacher, "\n")
cat("预测位置数 (n_teacher - 1):", n_teacher - 1, "\n\n")

# ==================== 2. 还原 Token 字符边界 (starts/ends) ====================
t_starts <- arrow_table$t_starts[[1]]
t_ends <- arrow_table$t_ends[[1]]

# 查看前 5 个 Token 切割的字符位置及文本片段
token_spans <- tibble(
  token_pos = 1:5,
  start_char = t_starts[1:5],
  end_char = t_ends[1:5],
  # R 的 stringr/substring 索引从 1 开始，需 +1 匹配
  sub_text = str_sub(ct, t_starts[1:5] + 1, t_ends[1:5]) 
)
# 前 5 个 Token 对应的原始字符片段：
print(token_spans)

# ==================== 3. 解构还原 teacher_ids 与 teacher_probs ====================
flat_ids <- arrow_table$teacher_ids[[1]]
flat_probs <- arrow_table$teacher_probs[[1]]

# 校验数据长度是否满足：(n_teacher - 1) * top_k
expected_len <- (n_teacher - 1) * top_k
cat("\n平铺数组实际长度:", length(flat_ids), " | 预期长度:", expected_len, "\n")

# 将一维向量转换为 矩阵 [预测位置, Top-K]
# 注意：R 语言默认按列填充 (byrow = FALSE)，必须显式设置 byrow = TRUE 才能正确还原 Python 行优先顺序
ids_matrix <- matrix(flat_ids, nrow = n_teacher - 1, ncol = top_k, byrow = TRUE)
probs_matrix <- matrix(flat_probs, nrow = n_teacher - 1, ncol = top_k, byrow = TRUE)

# ==================== 4. 查看具体预测结果 ====================
# 比如查看第 1 个预测位置（根据第一个 Token 预测下一个 Token）的 Top-5 结果：
pos_1_ids <- ids_matrix[1, 1:5]
pos_1_probs <- probs_matrix[1, 1:5]

pos_1_df <- tibble(
  rank = 1:5,
  teacher_token_id = pos_1_ids,
  probability = pos_1_probs
)

print("\n第 1 个预测位置 (Index 1) 教师模型预测出的 Top-5 候选词及概率：")
print(pos_1_df)

## 分布分析：
library(arrow)
library(tidyverse)

dataset_path <- "~/github/stat-llm-lab/data/processed/teacher_chunks/chunk_001.arrow"
arrow_table <- read_feather(dataset_path)

# ==================== 1. 自动判断概率列名 ====================
# 自动适配 阶段 A (teacher_probs) 或后续映射阶段 (topk_probs)
prob_col_name <- if ("topk_probs" %in% names(arrow_table)) "topk_probs" else "teacher_probs"
cat("当前读取的概率列名为:", prob_col_name, "\n")

top_k <- 16

# ==================== 2. 逐位置提取概率累加和 ====================
prob_summary_list <- map(1:nrow(arrow_table), function(i) {
  n_teacher <- arrow_table$n_teacher[i]
  
  # 动态提取对应列的数据
  flat_probs <- arrow_table[[prob_col_name]][[i]]
  if (is.null(flat_probs)) return(NULL)
  
  # 还原为 [N-1, K] 的概率矩阵
  prob_mat <- matrix(flat_probs, nrow = n_teacher - 1, ncol = top_k, byrow = TRUE)
  
  # 提取当前文本的 loss_mask (如果不存在则默认全选)
  if ("loss_mask" %in% names(arrow_table)) {
    mask <- arrow_table$loss_mask[[i]]
  } else {
    mask <- rep(TRUE, n_teacher - 1)
  }
  
  # 过滤有效预测位置
  valid_mat <- prob_mat[mask, , drop = FALSE]
  if (nrow(valid_mat) == 0) return(NULL)
  
  # 计算每个 Token 位置的 Top-8 和 Top-16 概率和
  tibble(
    sum_top08 = rowSums(valid_mat[, 1:8, drop = FALSE]),
    sum_top16 = rowSums(valid_mat[, 1:16, drop = FALSE])
  )
})

# 合并所有文本的 Token 级别统计数据
df_tokens <- bind_rows(prob_summary_list)

# 转换为长格式
df_box <- df_tokens %>%
  pivot_longer(cols = c(sum_top08, sum_top16), 
               names_to = "Group", 
               values_to = "Sum_Prob") %>%
  mutate(Group = if_else(Group == "sum_top08", "Top-8", "Top-16"))

# ==================== 3. 计算分组统计量 ====================
stats_df <- df_box %>%
  group_by(Group) %>%
  summarise(
    median_val = median(Sum_Prob),
    mean_val = mean(Sum_Prob)
  )

print("--- 概率和统计摘要 ---")
print(stats_df)

# ==================== 4. 绘制密度分布图 ====================
ggplot(df_box, aes(x = Sum_Prob, fill = Group, color = Group)) +
  geom_density(alpha = 0.35, linewidth = 0.8) +
  geom_vline(data = stats_df, aes(xintercept = median_val, color = Group),
             linetype = "dashed", linewidth = 0.8) +
  geom_text(data = stats_df, 
            aes(x = median_val, y = 1.5, 
                label = sprintf("%s Median: %.3f", Group, median_val), 
                color = Group),
            angle = 0, vjust = -0.5, fontface = "bold", inherit.aes = FALSE) +
  scale_x_continuous(limits = c(0, 1.02), breaks = seq(0, 1, by = 0.1)) +
  scale_fill_manual(values = c("Top-8" = "#5B9BD5", "Top-16" = "#ED7D31")) +
  scale_color_manual(values = c("Top-8" = "#2B5B84", "Top-16" = "#B34A00")) +
  coord_flip() +
  labs(
    title = paste0("Top-8 vs Top-16 概率和密度分布 (数据源: ", prob_col_name, ")"),
    subtitle = "展示模型预测分布的集中度与尾部截断损失",
    x = "Sum of Probabilities", y = "Density"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top", plot.title = element_text(face = "bold"))

