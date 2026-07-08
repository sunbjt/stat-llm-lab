library(data.table)
source("config.R")
source("utils/BPETokenizer.R")

tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

# 1. 读取预训练 Bin 文件
# 如果内存不够，可以通过设置 n 来截取前几千万个 Token，通常 1000 万个 Token (20MB) 足够进行统计抽样
bin_file <- "data/processed/zhwiki_tokens_16384.bin"
con <- file(bin_file, "rb")
# 假设读取 5000 万个 Token (占用内存极小，仅百兆级别)
tokens <- readBin(con, what = "integer", n = 50000000, size = 2, signed = FALSE, endian = "little") 
close(con)

# 2. 错位构建 (当前词 -> 下一个词) 的转移矩阵
N_total <- length(tokens)
dt <- data.table(
  x = tokens[-length(tokens)], 
  y = tokens[-1]
)

## 统计频次与边缘概率
# 1. 统计各自的独立频次 (Nx 和 Ny)
freq_x <- dt[, .(Nx = .N), by = x]
freq_y <- dt[, .(Ny = .N), by = y]

# 2. 统计共现频次 (Nxy)
dt_xy <- dt[, .(Nxy = .N), by = .(x, y)]

# 3. 极速合并表
dt_pmi <- dt_xy[freq_x, on = "x"][freq_y, on = "y"]

# 4. 计算 PMI
# 公式推导: pmi = log2( (Nxy / N) / ((Nx / N) * (Ny / N)) ) = log2( (Nxy * N) / (Nx * Ny) )
# 注意：Nx * Ny 极易超出 32位整数上限 (2*10^9)，必须转为 numeric 保证浮点精度
dt_pmi[, pmi := log2( (Nxy * as.numeric(N_total)) / (as.numeric(Nx) * Ny) )]


# 观察分布
library(ggplot2)

# 1. 查看频次和 PMI 的分位数分布
# 比如：取频次分布的 25% ~ 85% 分位作为“低曝光区”，取 PMI 的前 5% 作为“高强关联区”
quantile(dt_pmi$Nxy, probs = c(0.25, 0.5, 0.75, 0.85, 0.95))
quantile(dt_pmi$pmi, probs = c(0.8, 0.9, 0.95, 0.99))

# 2. 画出 Nxy 与 PMI 的联合散点图 (取 Log 缓解长尾畸变)
ggplot(dt_pmi[Nxy > 10], aes(x = log10(Nxy), y = pmi)) +
  geom_hex(bins = 50) + # 使用六边形热力图防止散点重叠成一坨黑
  scale_fill_viridis_c(option = "plasma") +
  labs(
    title = "Nxy 与 PMI 联合分布图", 
    x = "Log10(共现频次 Nxy)", 
    y = "点互信息 PMI"
  ) +
  theme_bw()


# 过滤出我们想要的“强关联、低曝光”专业实体组合
# 基于上面的热力图设定阈值：
target_bigrams <- dt_pmi[Nxy > 15 & Nxy < 300 & pmi > 8.5 & pmi < 13.5][order(-pmi)][1:500]
# 逆向解码
target_bigrams[, word_x := sapply(x, function(id) tokenizer$decode(id))]
target_bigrams[, word_y := sapply(y, function(id) tokenizer$decode(id))]
target_bigrams[, entity := paste0(word_x, word_y)]

# 3. 限制条件
clean_entities <- target_bigrams[
  # 基础汉字与长度限制
  grepl("^\\p{Han}+$", entity, perl = TRUE) & 
    nchar(entity) >= 3 & nchar(entity) <= 6 &
    
    # 不能包含任何标点符号 (全角/半角)
    !grepl("[，。、？！；：“”‘’《》【】（）]", entity) &
    
    # 开头不能是介词/量词，结尾不能是“的、了、地”
    !grepl("^(一|在|从|以|和|与|及)", entity) &
    !grepl("(的|了|地|等)$", entity)
]

cat("\n=== 基于 PMI 挖掘出的高价值领域实体 ===\n")
print(clean_entities[, .(entity, Nxy, pmi)], nrow = 200)

