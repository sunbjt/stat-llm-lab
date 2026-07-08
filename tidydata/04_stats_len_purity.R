library(RcppSimdJson)
library(data.table)
library(stringr)
library(ggplot2)


# 假定 62 万条，16GB 内存可一次性读入
file_path <- "data/processed/wiki_clean_dedup.jsonl"
lines <- readLines(file_path, warn = FALSE)
df <- data.table::rbindlist(RcppSimdJson::fparse(lines), fill = TRUE)

cat("正在计算统计特征...\n")
df[, char_len := nchar(text)]
df[, han_count := str_count(text, "\\p{Han}")]
df[, alpha_count := str_count(text, "[a-zA-Z]")]
df[, pure_ratio := han_count / (han_count + alpha_count + 1e-5)]

# 打印全局报告
cat(sprintf("总文档数: %d 篇\n", nrow(df)))
cat(sprintf("总字符数估算: ~%.2f 亿字\n", sum(as.numeric(df$char_len)) / 1e8))
cat(sprintf(
  "长度中位数: %d 字 | 95分位数: %d 字\n",
  median(df$char_len),
  quantile(df$char_len, 0.95)
))
cat(sprintf("平均中文纯度: %.2f%%\n", mean(df$pure_ratio) * 100))

# 绘制长度分布直方图 (保存为图片)
ggplot(df, aes(x = char_len)) +
  geom_histogram(fill = "steelblue",
                 bins = 100,
                 color = "white") +
  scale_x_log10(labels = scales::comma) +
  labs(title = "清洗后语料长度分布 (对数坐标)", x = "字符长度 (Log10)", y = "文档数量") +
  theme_minimal()
