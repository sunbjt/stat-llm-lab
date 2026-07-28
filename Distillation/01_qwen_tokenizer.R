library(RcppSimdJson)
library(tok)

jsonl_file <- "data/raw/pretrain_clean.jsonl"
bin_output_file <- "data/processed/qwen_tokens_aligned.bin"
tokenizer_file <- "models/tokenizer.json"

tokenizer <- tok::tokenizer$from_file(tokenizer_file)
EOS_ID <- 151643L  # Qwen 的 <|endoftext|>

BLOCK_TARGET <- 512
chunk_size <- 50000

# 如果旧文件存在，先删掉
if (file.exists(bin_output_file)) file.remove(bin_output_file)

con_bin <- file(bin_output_file, "wb")
con_in <- file(jsonl_file, "r")

total_tokens_processed <- 0
chunk_idx <- 1

cat("开始全量无损编码预训练数据...\n")

while (length(lines <- readLines(con_in, n = chunk_size, warn = FALSE)) > 0) {
  parsed_list <- RcppSimdJson::fparse(lines)
  texts <- vapply(parsed_list, function(x) {
    if (is.list(x) && !is.null(x$text) && !is.na(x$text)) x$text else ""
  }, character(1))
  texts <- texts[texts != ""]
  
  if (length(texts) == 0) next
  
  # 批量编码 (直接使用 Qwen 原始 Token ID，不做任何过滤与重映射)
  encoded_batch <- tokenizer$encode_batch(texts)
  
  # 提取 ids 并在每篇文章尾部焊上原始 EOS
  id_list <- lapply(encoded_batch, function(res) {
    c(res$ids, EOS_ID)
  })
  
  all_ids <- unlist(id_list, use.names = FALSE)
  
  # 写入 32-bit 二进制
  writeBin(as.integer(all_ids), con_bin, size = 4, endian = "little")
  
  total_tokens_processed <- total_tokens_processed + length(all_ids)
  cat(sprintf("Chunk %02d 完成. 累计写入 Token 数: %11s\n", 
              chunk_idx, format(total_tokens_processed, big.mark=",")))
  chunk_idx <- chunk_idx + 1
  
  rm(lines, parsed_list, texts, encoded_batch, id_list, all_ids)
  gc()
}

# 全局对齐
remainder <- total_tokens_processed %% BLOCK_TARGET
if (remainder > 0) {
  pad_len <- BLOCK_TARGET - remainder
  writeBin(as.integer(rep(EOS_ID, pad_len)), con_bin, size = 4, endian = "little")
  total_tokens_processed <- total_tokens_processed + pad_len
}

close(con_in)
close(con_bin)

cat(sprintf("\n 预训练数据集无损打包完成！\n总 Token 数: %s\n", 
            format(total_tokens_processed, big.mark=",")))

