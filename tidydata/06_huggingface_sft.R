# https://huggingface.co/datasets/erhwenkuo/moss-003-sft-chinese-zhtw/tree/main/data

library(arrow)
library(dplyr)
library(tidyr)
library(data.table)

dataset_path <- "~/Downloads/train-00013-of-00017-d483152b2ea748b6.parquet"

# 1. 完整读取（不加 col_select，保留完整的 x2 嵌套列）
arrow_table <- read_parquet(dataset_path)

# 2. 转换为 data.frame 并瞬间展平
# 这一步 unnest 会自动把 conversation 里的两列（human 和隐藏列）一起炸开
df_flat <- as.data.frame(arrow_table) %>% 
  unnest(conversation)

# 3. 转化为 data.table
setDT(df_flat)

# 1. 提取并重命名列，直接对齐 instruction 和 output 格式
dt_json_format <- df_flat[, .(
  instruction = human,
  output      = assistant
)]

dt_json_format <- dt_json_format[!grepl('[A-Za-z]', dt_json_format$instruction), ]
dt_json_format <- dt_json_format[!grepl('Joy|AI|MOSS|故事|场景|moss|聊天|app|画|图', dt_json_format$instruction),]
dt_json_format <- dt_json_format[!grepl('MOSS|故事|场景|moss|聊天|app|画|图|1', dt_json_format$output),]

# ==========================================
# 1. 设定过滤阈值 (根据 10M / 256 词定制)
# ==========================================
MIN_CHAR_LEG <- 75
MAX_CHAR_LEN <- 350   # 字符总长度（问+答）上限，防止 Tokenizer 编码后破 256
MIN_CHINESE_PCT <- 0.8   # 中文字符（含标点）占总字符数的比例门槛（60%）

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

cat("过滤后剩余高质量中文单轮数据：", nrow(dt_clean), "条\n")

# ==========================================
# 4. 基于 Mac OpenCC 的极速转换引擎
# ==========================================
library(ropencc)
cc <- converter(T2S)
to_simplified_ropencc <- function(text_vector) {
  return(run_convert(cc, text_vector))
}

dt_clean[, instruction := to_simplified_ropencc(instruction)]
dt_clean[, output      := to_simplified_ropencc(output)]

# 再过滤一遍免得漏网繁体语义

dt_clean <- dt_clean[!grepl('回答|电影|狗|机器人|影片|智慧|这个|人工|情节|故事|场景|歌|聊天|小说|画|图|诗', dt_clean$instruction),]
dt_clean <- dt_clean[!grepl('指的|从上文|作为|请问|抱歉|请提供|请告诉', dt_clean$output),]
dt_clean <- dt_clean[sample(nrow(dt_clean))]

# 2. 导出为标准训练用的 .jsonl 文件（带全 UTF-8 中文支持）
output_file <- "data/processed/rtomic_sft_dataset.jsonl"
# 用 stream_out 一击必杀，整表流式写入
jsonlite::stream_out(dt_clean, file(output_file), verbose = FALSE)

