library(RcppSimdJson)
library(tok) # 原生读取 tokenizer.json

# 配置路径
jsonl_file <- "data/raw/pretrain_clean.jsonl" 
bin_output_file <- "data/processed/qwen_tokens_aligned.bin"

# 请手动下载的 Qwen 配置文件
# https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/tokenizer.json
tokenizer_file <- "models/tokenizer.json" 

# ==========================================
# 1. 实例化 Qwen Tokenizer (纯 R/Rust)
# ==========================================
tokenizer <- tok::tokenizer$from_file(tokenizer_file)

# Qwen3.5 的特殊 Token ID 硬编码 (根据官方 config)
EOS_ID <- 151643L  # <|endoftext|>
PAD_ID <- 151643L  # Qwen 默认使用 eos 作为 pad

# 获取完整的词表字典 (Token Text -> Old ID)
tf_json <- jsonlite::fromJSON(tokenizer_file)
vocab_list <- tf_json$model$vocab
tokens <- names(vocab_list)
old_ids <- as.integer(unlist(vocab_list, use.names = FALSE))

# 编写正则表达式：匹配 ASCII (含英文字母、数字、基础标点)、中文 Unicode 范围、以及特殊标记 <|...|>
is_chinese_or_ascii <- function(tok) {
  # 优先保留特殊 Token (如 <|endoftext|>, <|im_start|> 等)
  if (grepl("^<\\|.*\\|>$", tok)) return(TRUE)
  
  # Byte-Level BPE 中 Leading space 被记作 Ġ，换行记作 Ċ，替换后检查
  clean_tok <- gsub("Ġ|Ċ", "", tok)
  if (nchar(clean_tok) == 0) return(TRUE) # 保留纯空格/换行
  
  # 过滤正则：匹配包含 ASCII 字符或 CJK 统一表意文字 (0x4E00 - 0x9FA5) 的 Token
  grepl("^[a-zA-Z0-9[:punct:][:space:]\u4e00-\u9fa5]+$", clean_tok)
}

# 筛选需要保留的原 Token ID 向量
keep_mask <- vapply(tokens, is_chinese_or_ascii, logical(1))

# 确保 EOS / PAD 必须在保留列表中！
keep_mask[old_ids == EOS_ID] <- TRUE
if (!is.na(PAD_ID)) keep_mask[old_ids == PAD_ID] <- TRUE

keep_old_ids <- sort(old_ids[keep_mask])
NEW_VOCAB_SIZE <- length(keep_old_ids)

cat(sprintf("裁剪完成！原始词表: %d -> 精简词表: %d (压缩率: %.1f%%)\n", 
            length(old_ids), NEW_VOCAB_SIZE, (1 - NEW_VOCAB_SIZE/length(old_ids)) * 100))

# 创建旧 ID 到新连续 ID (0-based) 的快速查找向量表
# (Qwen 最大 ID 为 248069，构建一个长度为 248070 的向量用于 O(1) 查找)
max_old_id <- max(old_ids)
id_map_lookup <- integer(max_old_id + 1)
id_map_lookup[] <- -1L # 默认标记为 -1，方便调试未命中项

# 将保留的旧 ID 依次映射到 0, 1, 2, ..., (NEW_VOCAB_SIZE - 1)
# 对应在 R 中的 index 为 keep_old_ids + 1
id_map_lookup[keep_old_ids + 1L] <- 0:(NEW_VOCAB_SIZE - 1L)

# 转换 EOS 和 PAD 对应的新 ID
NEW_EOS_ID <- id_map_lookup[EOS_ID + 1L]
NEW_PAD_ID <- id_map_lookup[PAD_ID + 1L]

# 指定 UNK 的新 ID（直接复用 PAD_ID，或者指定为词表里的某位）
NEW_UNK_ID <- NEW_PAD_ID

# 保存映射配置，供之后 Teacher 蒸馏切片 logits (teacher_logits[,, keep_old_ids + 1]) 使用！
mapping_file <- "models/qwen_vocab_mapping.rds"
saveRDS(list(
  keep_old_ids = keep_old_ids,
  id_map_lookup = id_map_lookup,
  new_vocab_size = NEW_VOCAB_SIZE,
  new_eos_id = NEW_EOS_ID,
  new_pad_id = NEW_PAD_ID,
  new_unk_id = NEW_UNK_ID # 存入 UNK ID
), file = mapping_file)
cat(sprintf("映射配置已保存至: %s\n", mapping_file))

# 测试 Tokenizer
test_text <- "要求用这些关键词写一份概述：环保设施完备，充足的配备，适宜居住，拎包即可入住。"
encoded_res <- tokenizer$encode(test_text)
encoded_ids <- encoded_res$ids

cat("原始文本: ", test_text, "\n")
cat("编码 ID: ", paste(encoded_ids, collapse = ", "), "\n")
cat("还原编码: ", tokenizer$decode(encoded_ids))

# =====================================================================
# 2. 工业级优化：流式创建二进制预训练数据 bin (全局 Packing)[cite: 1]
# =====================================================================

BLOCK_TARGET <- 512  # 对齐 SEQ_LEN[cite: 1]
chunk_size <- 50000  # 维持分块大小[cite: 1]

# 准备二进制文件连接 (以 append 模式打开，追加写入)[cite: 1]
con_bin <- file(bin_output_file, "ab") 
con_in <- file(jsonl_file, "r")

total_tokens_processed <- 0
chunk_idx <- 1

# 全量文本转化为【全局 Packing 无缝拼接】的二进制 Token 流...
while (length(lines <- readLines(con_in, n = chunk_size, warn = FALSE)) > 0) {
  
  # 高速解析 JSON[cite: 1]
  parsed_list <- RcppSimdJson::fparse(lines)
  texts <- vapply(parsed_list, function(x) {
    if (is.list(x) && !is.null(x$text) && !is.na(x$text)) x$text else ""
  }, character(1))
  texts <- texts[texts != ""]
  
  if (length(texts) == 0) next
  
  # 批量编码
  encoded_batch <- tokenizer$encode_batch(texts)
  
  # 提取 ids 并在每篇文章尾部焊上一个 EOS
  id_list <- lapply(encoded_batch, function(res) {
    # 1. O(1) 查表：将原始 ID 映射到新的 0-based 精简 ID
    mapped_ids <- id_map_lookup[res$ids + 1L]
    
    # 2. 🚀 关键修改：将未命中的 -1L 批量替换为 NEW_UNK_ID，不再执行向量过滤！
    # 这样可以 100% 保持原始文本的序列长度和句法结构不被破坏
    mapped_ids[mapped_ids == -1L] <- NEW_UNK_ID
    
    # 3. 焊上新的 EOS Token ID
    c(mapped_ids, NEW_EOS_ID)
  })
  
  # 展平为一维数组，像香肠一样无缝拼接[cite: 1]
  all_ids <- unlist(id_list, use.names = FALSE)
  
  # 由于 Qwen 的词表是 15 万级别，超过了你之前 vocab_size=9000 
  # 时使用的 writeBin size=2 (16-bit，最大支持 32767)[cite: 1]。
  # 此处必须强制改为 size=4 (32-bit)，否则文件会直接因为数据溢出而损坏！
  writeBin(as.integer(all_ids), con_bin, size = 4, endian = "little")
  
  total_tokens_processed <- total_tokens_processed + length(all_ids)
  
  cat(sprintf("Chunk %02d 完成. 累计写入 Token 数: %11s\n", 
              chunk_idx, format(total_tokens_processed, big.mark=",")))
  chunk_idx <- chunk_idx + 1
  
  rm(lines, parsed_list, texts, encoded_batch, id_list, all_ids)
  gc()
}

# 循环结束，执行全局终极对齐
remainder <- total_tokens_processed %% BLOCK_TARGET

if (remainder > 0) {
  pad_len <- BLOCK_TARGET - remainder
  cat(sprintf("\n检测到文件末端存在余数 (%d)。正在补齐 %d 个 PAD Token 以实现全局 %d 整数倍对齐...\n",
              remainder, pad_len, BLOCK_TARGET))
  
  # 在末尾补上 PAD，同样必须保持 size = 4
  writeBin(as.integer(rep(NEW_PAD_ID, pad_len)), con_bin, size = 4, endian = "little")
  total_tokens_processed <- total_tokens_processed + pad_len
}

close(con_in)
close(con_bin)

cat(sprintf("\n 预训练数据集打包完毕！\n总 Token 数: %s\n输出文件: %s\n", 
            format(total_tokens_processed, big.mark=","), bin_output_file))

