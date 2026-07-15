library(jsonlite)
library(tidyverse)
library(data.table)

input <- 'data/raw/train_alpaca.jsonl'

library(jsonlite)

# 打开连接，逐行处理
con <- file("data/raw/train_alpaca.jsonl", "r")
df <- stream_in(con, pagesize = 1000)  # 每次读1000行
close(con)

df <- df[ , c('input', 'output')]

# 3. 转化为 data.table
setDT(df)

# 1. 提取并重命名列，直接对齐 instruction 和 output 格式
dt_json_format <- df[, .(
  instruction = input,
  output      = output
)]

dt_json_format <- dt_json_format[!grepl('[A-Za-z]', dt_json_format$instruction), ]
dt_json_format <- dt_json_format[!grepl('抱歉|故事|场景|moss|聊天|app|画|图', dt_json_format$instruction),]
dt_json_format <- dt_json_format[!grepl('您说|抱歉|故事|场景|无法|聊天|是的|好主意|图', dt_json_format$output),]

# ==========================================
# 1. 设定过滤阈值 (根据 10M / 256 词定制)
# ==========================================
MIN_CHAR_LEG <- 45
MAX_CHAR_LEN <- 500   # 字符总长度（问+答）上限，防止 Tokenizer 编码后破 256
MIN_CHINESE_PCT <- 0.99   # 中文字符（含标点）占总字符数的比例门槛（60%）

cat("原始数据总量：", nrow(dt_json_format), "条\n")

# 计算总字符长度
total_chars <- nchar(dt_json_format$instruction) + nchar(dt_json_format$output)

# 利用正则提取中文字符及中文全角标点，并计算各自数量
# [\u4e00-\u9fa5] 是中文汉字，[\u3000-\u303f\uff00-\uffef] 是中文标点符号
chinese_chars <- nchar(gsub("[^\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef]", "", dt_json_format$instruction)) + 
  nchar(gsub("[^\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef]", "", dt_json_format$output))

# 计算中文占比 (防止分母为 0)
cn_ratio <- ifelse(total_chars > 0, chinese_chars / total_chars, 0)

# 过滤条件：太长不要，中文占比太低不要
keep_indices <- which(total_chars >= MIN_CHAR_LEG & total_chars <= MAX_CHAR_LEN & cn_ratio >= MIN_CHINESE_PCT)

dt_clean <- dt_json_format[keep_indices]

# 2. 导出为标准训练用的 .jsonl 文件（带全 UTF-8 中文支持）
output_file <- "data/processed/fineweb_sft.jsonl"
# 用 stream_out 一击必杀，整表流式写入
jsonlite::stream_out(dt_clean, file(output_file), verbose = FALSE)

