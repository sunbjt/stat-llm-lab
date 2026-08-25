
library(arrow)
library(dplyr)
library(tidyr)
library(data.table)
library(tidyverse)

dataset_path <- "~/github/stat-llm-lab/data/processed/qwen_soft_labels.parquet"
arrow_table <- read_parquet(dataset_path)

# 1. 查看第 1 条数据的实际 Token ID 序列（前 10 个）
head(arrow_table$input_ids[[1]], 10)

# 2. 查看第 1 个 Token 位置上，Teacher 给出的 Top-K ID (长度应为 32)
arrow_table$teacher_topk_ids[[1]][[1]]

# 3. 查看第 1 个 Token 位置上，Teacher 给出的 Top-K 软概率 (长度应为 32)
arrow_table$teacher_topk_probs[[1]][[1]]
