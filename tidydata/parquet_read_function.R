
library(arrow)
library(dplyr)
library(tidyr)
library(data.table)
library(tidyverse)

dataset_path <- "~/Downloads/001267.parquet"
arrow_table <- read_parquet(dataset_path)

arrow_table |>
  group_by(category) |>
  count()

arrow_table |>
  filter(category == 'educational') |>
  select(text) |>
  head(5) |>
  as.vector()

set.seed(2026) # 设置随机种子，确保结果可重复
sampled_data <- arrow_table |>
  select(category, 'num-tokens', text) |>
  group_by(category) |>               # 按类别分组
  slice_sample(n = 100) |>            # 每组随机抽取 100 行
  ungroup()

output_file <- "data/processed/sample.jsonl"
jsonlite::stream_out(sampled_data, file(output_file), verbose = FALSE)

