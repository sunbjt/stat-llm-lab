# =====================================================================
# Rtomic 黄金语料极速洗练流水线 (工业级全防守版)
# =====================================================================
library(tidyverse)
library(jsonlite)
library(RcppSimdJson)

# === 配置路径 ===
input_path  <- "data/raw/sft_data.jsonl"
output_path <- "data/processed/qa_no_think.jsonl"

cat("开始读取并解析数据...\n")

# 1. 安全读取原始文本
lines <- readLines(input_path, warn = FALSE) |> 
  str_trim() |> 
  keep(~ .x != "")

# 拦截残缺的 JSON 行 (如最后一行的 {"ins )
is_valid <- map_lgl(lines, jsonlite::validate)
lines <- lines[is_valid]

# 使用 map_dfr 安全平替 as_tibble，防止 unnamed list 报错
sft_clean_dt <- map_dfr(lines, ~ jsonlite::fromJSON(.x) |> as_tibble_row())

# =====================================================================
# ==== 2. 深度清洗、标点泛化与多重拦截规则 ====
# =====================================================================
cat("激活多重硬拦截规则，清洗垃圾截断样本...\n")
set.seed(42)

processed_dt <- sft_clean_dt |> 
  # --- 行为 A: 清洗 output 中的 think 标签、Markdown 分割线以及加粗双星号 ---
  mutate(
    # 1. 干掉 think 标签
    output = str_remove_all(output, "(?s)<think>.*?(?:</think>|<\\\\/think>|<\\/think>)(?:\\\\n|\n)*"),
    
    # 2. 干掉 Markdown 的分割线 --- 或 ***
    output = str_remove_all(output, "(?m)^\\s*[-*]{3,}\\s*$"),
    
    # 3. 【新增】精准脱掉成对的 ** 加粗外衣，保留里面的文字内容
    #    比如把 **最核心** 还原为 最核心
    output = str_replace_all(output, "\\*\\*(.*?)\\*\\*", "\\1"),
    
    output = str_trim(output)
  ) |>
  
  # --- 行为 B: 终极贪婪清洗 instruction 的尾部堆叠标点 ---
  mutate(
    instruction = str_remove(instruction, "[\\?\\?。\\.\\!！,，;；\\s\\p{P}]+$")
  ) |> 
  
  # --- 行为 C: 1/3 概率标点随机分流 ---
  mutate(
    rand_val = runif(n()),
    instruction = case_when(
      rand_val < 0.33 ~ str_c(instruction, "？"),
      rand_val < 0.66 ~ str_c(instruction, "。"),
      TRUE            ~ instruction
    ),
    instruction = str_trim(instruction)
  ) |> 
  
  # --- 行为 D: 【核心安全防线】多重刚性过滤条件 ---
  filter(
    # 规则 1: 基础非空约束
    str_length(instruction) > 0,
    str_length(output) > 0,
    
    # 规则 2: 长度必须严格大于 30 字符
    str_length(output) > 30,
    
    # 规则 3: 严格拦截残留 think 的不安全样本
    !str_detect(output, "(?i)think"), 
    
    # 规则 4: 【针对本次 Bad Case 新增】句尾截断检查！
    # 必须以标准结束标点（句号、问号、感叹号、右引号、右括号）结尾，否则视为被硬截断的烂数据
    str_detect(output, "[。？！”’)）\\]]$")
  ) |> 
  
  # 清理临时随机数
  select(-rand_val)

# =====================================================================
# ==== 3. 黄金语料安全落盘 ====
# =====================================================================
cat("正在写出最终微调语料...\n")
con_out <- file(output_path, "w")
jsonlite::stream_out(processed_dt, con_out, verbose = FALSE)
close(con_out)

cat(sprintf("全部处理完毕！已完美过滤截断数据，最终交付 %d 条黄金语料。\n", nrow(processed_dt)))