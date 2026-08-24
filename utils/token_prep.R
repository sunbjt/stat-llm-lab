source("config.R")

library(RcppSimdJson) # 使用底层 SIMD 指令集极速解析 JSON
library(tokenizers.bpe)
library(R6)
set.seed(123)

# 配置路径
jsonl_file <- "data/raw/pretrain_clean.jsonl" 
bpe_sample_file <- "data/processed/bpe_sample.txt"
model_file <- "models/rtomic_bpe.model"

# 流式读取并抽样，不动原文件
extract_bpe_sample_stream <- function(
    jsonl_file, bpe_sample_file,
    sample_rate = 0.1, chunk_size = 50000,
    log_file = "data/processed/parser_errors.log") { # 新增日志路径参数
  
  cat(sprintf("开始流式扫描 JSONL 提取 BPE 样本: %s\n", jsonl_file))
  con_in <- file(jsonl_file, "r")
  con_out_sample <- file(bpe_sample_file, "w")
  
  # 初始化日志文件（如果存在则清空，不存在则创建）
  con_log <- file(log_file, "w")
  writeLines(c("=== JSON 解析错误日志 ===", sprintf("时间: %s\n", Sys.time())), con_log)
  
  lines_sampled <- 0
  total_lines <- 0
  error_lines_count <- 0 
  
  while (length(lines <- readLines(con_in, n = chunk_size, warn = FALSE)) > 0) {
    chunk_length <- length(lines)
    # 计算当前块在整个文件中的起始绝对行号
    chunk_start_line <- total_lines + 1
    total_lines <- total_lines + chunk_length
    
    parsed_list <- NULL
    
    # 尝试硬件级批量解析
    tryCatch({
      parsed_list <- RcppSimdJson::fparse(lines)
    }, error = function(e) {
      cat(sprintf("\n[警告] 当前块内发现脏数据，已启动逐行扫描并记录日志...\n"))
      
      parsed_list_backup <- list()
      for (i in seq_along(lines)) {
        # 计算当前行的绝对行号
        current_absolute_line <- chunk_start_line + i - 1
        
        single_parsed <- tryCatch({
          RcppSimdJson::fparse(lines[i])
        }, error = function(se) {
          error_lines_count <<- error_lines_count + 1
          
          # 将错误信息和写有问题的原始数据写入日志
          log_msg <- sprintf(
            "[%s] [行号: %d] 错误原因: %s\n原始数据: %s\n-----------------------------------",
            Sys.time(), current_absolute_line, conditionMessage(se), lines[i]
          )
          writeLines(log_msg, con_log)
          
          return(NULL) 
        })
        parsed_list_backup[[i]] <- single_parsed
      }
      parsed_list <<- parsed_list_backup
    })
    
    # 在提取 text 后的清洗步骤：
    texts <- vapply(parsed_list, function(x) {
      if (is.list(x) && !is.null(x$text) && !is.na(x$text)) x$text else ""
    }, character(1))

    texts <- texts[texts != ""]

    # 强制标点隔离
    punct_pattern <- "([,.:;!?\"'()\\{\\}\\[\\]，。！？；：—（）《》“”‘’、])"
    texts <- gsub(punct_pattern, " \\1 ", texts, perl = TRUE)
    texts <- gsub("\\s+", " ", texts, perl = TRUE)

    # 抽样与写入
    if (length(texts) > 0) {
      keep_idx <- runif(length(texts)) < sample_rate
      sampled_texts <- texts[keep_idx]
      
      if (length(sampled_texts) > 0) {
        writeLines(sampled_texts, con_out_sample)
        lines_sampled <- lines_sampled + length(sampled_texts)
      }
    }
  }
  
  close(con_in)
  close(con_out_sample)
  close(con_log) # 关闭日志连接
  
  cat(sprintf("\n扫描完毕！共读取 %d 行。\n", total_lines))
  if (error_lines_count > 0) {
    cat(sprintf("[注意] 自动跳过了 %d 行脏数据，详细行号和内容已写入日志: %s\n", error_lines_count, log_file))
  }
  cat(sprintf("Tokenizer 训练样本已生成: %s (共抽样 %d 行)\n", bpe_sample_file, lines_sampled))
}

# 运行抽样 (0.9G 纯文本，抽样 30%)
if (!file.exists(bpe_sample_file)) {
  extract_bpe_sample_stream(
    jsonl_file = jsonl_file, 
    bpe_sample_file = bpe_sample_file,
    sample_rate = 0.3, 
    chunk_size = 50000 
  )
} else {
  cat("检测到 BPE 样本文件已存在，跳过提取步骤...\n")
}

# 实例化 tokenizer (统一对齐为 2^14)
source('utils/BPETokenizer.R')
tokenizer <- RtomicBPETokenizer$new(
  corpus_file = bpe_sample_file, 
  model_file = model_file,
  vocab_size = VOCAB_SIZE
)

# 展示合并的 token
tail(tokenizer$model$vocabulary, 10)

# 测试一下 Tokenizer 是否正常工作a
test_text <- "这是刘思喆创造的小型LLM，可以回答回答人工智能领域的一些问题。"
encoded_ids <- tokenizer$encode(test_text)
decoded_text <- tokenizer$decode(encoded_ids)

cat("原始文本: ", test_text, "\n")
cat("编码 ID: ", paste(encoded_ids, collapse = ", "), "\n")
cat("解码文本: ", decoded_text, "\n")

# ==========================================
# 1. 配置对比实验参数
# ==========================================

tokenizer <- RtomicBPETokenizer$new(model_file = "models/rtomic_bpe.model", vocab_size = VOCAB_SIZE)

# 获取 <UNK> 的 token ID
UNK_ID <- tokenizer$unk_idx
texts <- readLines(bpe_sample_file, warn = FALSE)

# 核心评估函数
evaluate_tokenizer <- function(tokenizer, texts) {

  # 批量编码
  id_list <- lapply(texts, tokenizer$encode)
  
  # 展平为单一长向量
  all_ids <- unlist(id_list, use.names = FALSE)
  total_tokens <- length(all_ids)
  
  # 1. 计算 UNK 比例
  unk_count <- sum(all_ids == UNK_ID)
  unk_ratio <- unk_count / total_tokens
  
  # 2. 计算压缩率 (Token 数量 / 原始字符数)
  total_chars <- sum(nchar(texts, type = "chars"))
  compression_ratio <- total_tokens / total_chars
  
  return(list(
    total_tokens = total_tokens,
    unk_ratio = unk_ratio,
    compression_ratio = compression_ratio
  ))
}

res_sample <- evaluate_tokenizer(tokenizer, texts)
# sample 的总 token 数
format(res_sample$total_tokens, big.mark=",")
# UNK 的占比
sprintf("%.2f%%", res_sample$unk_ratio * 100)
# 编码效率 (每 100 个中文字符消耗的 Token 数)
sprintf("%.2f", res_sample$compression_ratio * 100)

##
## 流式创建二进制预训练数据 bin
##

bin_output_file <- "data/processed/zhwiki_tokens_16384.bin"

# 1. 加载我们训练好的小钢炮 Tokenizer
tokenizer <- RtomicBPETokenizer$new(model_file = model_file, vocab_size = VOCAB_SIZE)

# 2. 准备二进制文件连接 (以 append 模式打开，追加写入)
con_bin <- file(bin_output_file, "ab") 
con_in <- file(jsonl_file, "r")

chunk_size <- 50000
total_tokens_processed <- 0
chunk_idx <- 1

cat("开始将全量文本转化为二进制 Token 流...\n")

# =====================================================================
# 【工业级优化】流式创建二进制预训练数据 bin（全局 Packing + 末端对齐）
# =====================================================================

BLOCK_TARGET <- 512  # 严格对齐你的模型的 SEQ_LEN
PAD_ID       <- tokenizer$pad_idx  # 也就是 1L
EOS_ID       <- tokenizer$eos_idx  # 也就是 4L <--- 完美纠正！

chunk_idx <- 1
total_tokens_processed <- 0

cat("开始将全量文本转化为【全局 Packing 无缝拼接】的二进制 Token 流...\n")

while (length(lines <- readLines(con_in, n = chunk_size, warn = FALSE)) > 0) {
  
  # 高速解析 JSON
  parsed_list <- RcppSimdJson::fparse(lines)
  texts <- vapply(parsed_list, function(x) {
    if (is.list(x) && !is.null(x$text) && !is.na(x$text)) x$text else ""
  }, character(1))
  texts <- texts[texts != ""]
  # 提取 text 后必须同步加上隔离
  punct_pattern <- "([,.:;!?\"'()\\{\\}\\[\\]，。！？；：—（）《》“”‘’、])"
  texts <- gsub(punct_pattern, " \\1 ", texts, perl = TRUE)
  texts <- gsub("\\s+", " ", texts, perl = TRUE)
  
  if (length(texts) == 0) next
  
  # 【核心优化】：批量编码，并立刻在每篇文章尾部焊上一个 EOS
  id_list <- lapply(texts, function(txt) {
    c(tokenizer$encode(txt), EOS_ID)
  })
  
  # 展平为一维数组（像香肠一样无缝拼接，绝对不在这里加 PAD）
  all_ids <- unlist(id_list, use.names = FALSE)
  
  # 直接将这段紧致的数据流写入二进制文件
  writeBin(as.integer(all_ids), con_bin, size = 2, endian = "little")
  
  total_tokens_processed <- total_tokens_processed + length(all_ids)
  
  cat(sprintf("Chunk %02d 完成. 累计写入紧致 Token 数: %11s\n", 
              chunk_idx, format(total_tokens_processed, big.mark=",")))
  chunk_idx <- chunk_idx + 1
  
  rm(lines, parsed_list, texts, id_list, all_ids)
  gc()
}

# =====================================================================
# 循环结束，执行【全局终极对齐】
# =====================================================================
remainder <- total_tokens_processed %% BLOCK_TARGET

if (remainder > 0) {
  pad_len <- BLOCK_TARGET - remainder
  cat(sprintf("\n检测到文件末端存在余数 (%d)。正在补齐 %d 个 PAD Token 以实现全局 %d 整数倍对齐...\n", 
              remainder, pad_len, BLOCK_TARGET))
  
  # 在文件最末尾补上最后的一点点 PAD
  writeBin(as.integer(rep(PAD_ID, pad_len)), con_bin, size = 2, endian = "little")
  total_tokens_processed <- total_tokens_processed + pad_len
}

close(con_in)
close(con_bin)


cat(sprintf("\n 预训练数据集打包完毕！\n总 Token 数: %s\n输出文件: %s\n", 
            format(total_tokens_processed, big.mark=","), bin_output_file))

