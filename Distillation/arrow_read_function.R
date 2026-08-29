library(arrow)
library(tidyverse)

# 1. 载入环境配置（自动设置 WORK_DIR 和加载 RtomicBPETokenizer 依赖）
source("config.R") 
source("utils/BPETokenizer.R")
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

# 需要的是非 normalize 的结果
dataset_path <- "~/github/stat-llm-lab/data/processed/chunks/chunk_001.arrow"
arrow_table <- read_feather(dataset_path)
arrow_table

head(arrow_table$topk_ids[[1]], 5)
head(arrow_table$topk_probs[[1]], 5)

## 分布分析：
library(tidyverse)

# 1. 提取有效数据（注意 tmp08 改为截取 1:8）
valid_probs <- arrow_table[arrow_table$loss_mask == TRUE, ]$topk_probs

tmp16 <- sapply(valid_probs, function(x) sum(x[1:16]))
tmp08 <- sapply(valid_probs, function(x) sum(x[1:8]))

# 2. 构建长格式数据框并绘制双 Boxplot
df_box <- tibble(`Top-8` = tmp08, `Top-16` = tmp16) %>% 
  pivot_longer(cols = everything(), names_to = "Group", values_to = "Sum_Prob")

# 2. 计算各组的中位数与均值表（便于精准控制文字位置）
stats_df <- df_box %>%
  group_by(Group) %>%
  summarise(
    median_val = median(Sum_Prob),
    mean_val = mean(Sum_Prob)
  )

# 密度对比图（展示多峰与偏态）
ggplot(df_box, aes(x = Sum_Prob, fill = Group, color = Group)) +
  geom_density(alpha = 0.35, linewidth = 0.8) +
  # 画出中位数虚线
  geom_vline(data = stats_df, aes(xintercept = median_val, color = Group),
             linetype = "dashed", linewidth = 0.8) +
  # 标注数值
  geom_text(data = stats_df, 
            aes(x = median_val, y = 1.5, label = sprintf("%s Median: %.3f", Group, median_val), color = Group),
            angle = 0, vjust = -0.5, fontface = "bold", inherit.aes = FALSE) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.1)) +
  scale_fill_manual(values = c("Top-8" = "#5B9BD5", "Top-16" = "#ED7D31")) +
  scale_color_manual(values = c("Top-8" = "#2B5B84", "Top-16" = "#B34A00")) +
  coord_flip() +
  labs(
    title = "Top-8 vs Top-16 概率和密度分布 (Density Plot)",
    subtitle = "虚线指示各自的中位数位置",
    x = "Sum of Probabilities", y = "Density"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top", plot.title = element_text(face = "bold"))


## 还原 x / y_hard 序列文本
# 提取第 1 条样本的 x (输入 Token 序列)
# 注意：由于 Arrow 存盘时打平了，需要提取有效序列 (排除 PAD，即 > 1 的部分)
sample_1_x <- arrow_table$x[1:511]  # 对应序列长度 TARGET_LEN (511)
decoded_text <- tokenizer$decode(sample_1_x)
decoded_text

## 还原并查看 Top-K 软标签（Soft Labels）
# 提取第 1 个 Token 位置 (Row 1) 的 Top-K 预测列表
topk_ids_pos1 <- arrow_table$topk_ids[[1]]
topk_probs_pos1 <- arrow_table$topk_probs[[1]]

# 解码 Top-K 中的每个候选词
topk_tokens <- sapply(topk_ids_pos1, function(id) {
  if (id == tokenizer$pad_idx) return("<PAD>")
  tokenizer$decode(id, clean = FALSE) # 不 clean 保持 Token 原始子词形态
})

# 打印对齐结果
tibble(
  Rank = 1:length(topk_tokens),
  Token_ID = topk_ids_pos1,
  Subword = topk_tokens,
  Prob = topk_probs_pos1
)



library(arrow)
library(tidyverse)

darrow_table <- read_feather("~/github/stat-llm-lab/data/processed/chunks/chunk_001.arrow")

# 1. 展开所有的 topk_ids 并统计分布
all_topk_ids <- unlist(arrow_table$topk_ids[arrow_table$loss_mask]) # 只看有效 loss_mask 位置

unk_count <- sum(all_topk_ids == 2L)             # UNK_SURFACE = 2
pad_count <- sum(all_topk_ids == 1L)             # PAD_SURFACE = 1
total_count <- length(all_topk_ids)

cat(sprintf("有效预测点总数: %d\n", total_count))
cat(sprintf("UNK 占比: %.2f%%\n", unk_count / total_count * 100))
cat(sprintf("PAD 占比: %.2f%%\n", pad_count / total_count * 100))

