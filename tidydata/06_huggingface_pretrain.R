# https://huggingface.co/datasets/BAAI/CCI2-Data/tree/main/data

library(arrow)
library(data.table)

dataset_path <- "~/Downloads/cci2-00177-of-00178.parquet"
dt_corpus <- as.data.table(read_parquet(dataset_path, col_select = c("content")))
setnames(dt_corpus, "content", "text")

# 1. 过滤掉 null 文本或绝对空行
dt_corpus <- dt_corpus[!is.na(text) & text != ""]

# 2. 过滤掉字符数少于 150 字的无意义碎片（短句对预训练长上下文注意力没有帮助）
dt_corpus <- dt_corpus[nchar(text) >= 300]
dt_corpus <- dt_corpus[nchar(text) < 3000]
dt_corpus <- dt_corpus[!grep("工作|建议|成立|课|意见|公告日期|首页|您的位置|我想做代理|请寄给我详细资料|加微信|人民法院|javascript", text)]

# 过滤带网址的
url_pattern <- "(https?://[^\\s]+|www\\.[^\\s]+|[a-zA-Z0-9][-a-zA-Z0-9]{0,62}(\\.[a-zA-Z0-9][-a-zA-Z0-9]{0,62})+\\b)"
dt_corpus <- dt_corpus[!grepl(url_pattern, text, perl = TRUE, ignore.case = TRUE)]

# 彻底剔除中文占比低于 40% 的行
get_chinese_ratio <- function(txt_vec) {
  zh_char_count <- nchar(gsub("[^\u4e00-\u9fa5]", "", txt_vec))
  total_char_count <- nchar(txt_vec)
  total_char_count[total_char_count == 0] <- 1
  return(zh_char_count / total_char_count)
}
dt_corpus <- dt_corpus[get_chinese_ratio(text) >= 0.85]

# 过滤时间和新闻尾部
time_pattern <- "\\d{4}-\\d{2}-\\d{2}\\s+\\d{2}:\\d{2}"
dt_corpus[, text := gsub(time_pattern, "", text, perl = TRUE)]
sohu_tail_pattern <- "(?s)(返回搜狐.*|声明：本文由入驻搜狐号.*)"
dt_corpus[, text := gsub(sohu_tail_pattern, "", text, perl = TRUE)]

# 3. 洗掉一些常见的低级网页垃圾（如连续的换行符或空格过多）
dt_corpus[, text := gsub("\n{3,}", "\n\n", text)] # 将3个以上的连续换行替换为双换行
dt_corpus[, text := trimws(text)] # 去除首尾空格
dt_corpus[, text := gsub("\\s+\\*\\s+", "\n", text)]
dt_corpus[, text := gsub("原标题：", "", text, fixed = TRUE)]


library(jsonlite)
output_jsonl_path <- "~/Downloads/train_corpus.jsonl"
con <- file(output_jsonl_path, open = "wb")
stream_out(dt_corpus[, .(text)], con, pagesize = 5000, auto_unbox = TRUE)
close(con)



library(data.table)
library(jsonlite)

# 1. 定义已有 R 中清洗好的数据集：dt_final_train
# 2. 定义外部需要合并的 JSONL 文件路径
external_jsonl_path <- "data/raw/pretrain_clean.jsonl"
self_jsonl_path <- "data/raw/train_corpus.jsonl"
output_combined_path <- "data/raw/comb_train_corpus.jsonl"


# 使用 stream_in 配合不吃内存的 memory-mapping 指针
con_in <- file(external_jsonl_path, open = "rb")
dt_external <- as.data.table(stream_in(con_in, pagesize = 10000))
close(con_in)

# 确保只保留 text 列，剔除可能存在的 id 或其他杂质字段
dt_external <- dt_external[, .(text)]

con_self <- file(self_jsonl_path, open = 'rb')
dt_final_train <- as.data.table(stream_in(con_self), pagesize = 10000)
# 使用 data.table 的 rbindlist 进行 C++ 级别的极速合并
dt_combined <- rbindlist(list(dt_final_train[, .(text)], dt_external), use.names = TRUE)

# 关键动作：去重（防止两个数据集中有重复的网页抓取或重合书籍，稀释模型拟合度）
dt_combined <- unique(dt_combined, by = "text")

# ----------------- 步骤三：打乱顺序（Shuffle） -----------------
set.seed(42) # 锁定随机种子，保证实验可复现
dt_combined <- dt_combined[sample(.N)]


# ----------------- 步骤四：流式导出全新 JSONL 文件 -----------------
con_out <- file(output_combined_path, open = "wb")
stream_out(dt_combined, con_out, pagesize = 5000, auto_unbox = TRUE)
close(con_out)


