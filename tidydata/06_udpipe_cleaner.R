library(RcppSimdJson)
library(data.table)
library(jiebaR)
library(parallel)

# =====================================================================
# 1. 初始化模型与配置
# =====================================================================

input_file <- "data/raw/pretrain_clean.jsonl"
clean_file <- "data/processed/pretrain_udpipe_clean.jsonl"
audit_file <- "data/processed/audit_udpipe.jsonl"

file.create(clean_file)
file.create(audit_file)

# 设置每段文本容忍的最大专有名词数量 (根据语料容忍度调整)
MAX_PROPN_COUNT <- 5 
NUM_CORES <- 2

# =====================================================================
# 2. 定义工作节点函数 (向量化标注)
# =====================================================================
global_tagger <- worker("tag")
process_chunk_smart <- function(lines_batch) {
  parsed <- tryCatch(RcppSimdJson::fparse(lines_batch, always_list = TRUE), error = function(e) NULL)
  if (is.null(parsed)) return(NULL)
  
  texts <- vapply(parsed, function(x) {
    if (is.list(x) && !is.null(x$text) && !is.na(x$text)) x$text else ""
  }, character(1))
  
  valid_idx <- texts != ""
  valid_texts <- texts[valid_idx]
  if (length(valid_texts) == 0) return(NULL)
  
  # 对每段有效长文本执行智能判定
  dirty_flags <- vapply(valid_texts, function(txt) {
    # 1. 极速分词并打上词性标签
    words_with_tags <- segment(txt, global_tagger)
    tags <- names(words_with_tags)
    words <- as.character(words_with_tags)
    
    total_words <- length(words)
    if (total_words < 5) return(FALSE) # 太短的直接放过
    
    # 2. 提取出所有的专有名词 (人名 nr, 地名 ns, 机构 nt, 专名 nz)
    is_propn <- tags %in% c("nr", "ns", "nt", "nz", "nrt")
    propn_words <- words[is_propn]
    
    # 3. 核心逻辑：去重！算独立实体数
    unique_propn_words <- unique(propn_words)
    unique_count <- length(unique_propn_words)
    
    # 4. 计算独立实体密度 (独立实体数 / 总词数)
    density <- unique_count / total_words
    
    # ===========================================================
    # 5. 工业级长短文兼容拦截规则 (双重条件)
    # ===========================================================
    # 规则解析：
    # 条件 A: 独立实体绝对数量太多 (比如超过 8 个不同的人/地名)
    # 条件 B: 并且实体密度过高 (比如整段话超过 10% 的词汇全是不一样的人名)
    # 只有两个条件同时满足，才会被认定为“报菜名”式的脏数据拦截。
    # 这样，就算是一篇 5000 字的长文，只要密度被稀释下来，就不会被误杀。
    
    is_dirty <- (unique_count > 8) && (density > 0.10)
    
    return(is_dirty)
    
  }, logical(1))
  
  return(list(
    raw_lines = lines_batch[valid_idx],
    is_dirty = dirty_flags
  ))
}

# =====================================================================
# 3. 流式主循环
# =====================================================================
con_in <- file(input_file, "r")
con_clean <- file(clean_file, "a")
con_audit <- file(audit_file, "a")

CHUNK_SIZE <- 20000 # udpipe 计算密集度高，适当减小单批次大小
chunk_idx <- 1

cat("启动 udpipe 深度语义清洗管道...\n")

while (length(lines <- readLines(con_in, n = CHUNK_SIZE, warn = FALSE)) > 0) {
  
  # 1. 兼容单核与多核模式的切分
  if (NUM_CORES <= 1) {
    batches <- list(lines)
  } else {
    batch_ids <- cut(seq_along(lines), breaks = NUM_CORES, labels = FALSE)
    batches <- split(lines, batch_ids)
  }
  
  # 2. 并发执行 JiebaR 智能语义分析
  # 【修复点】：改成了 process_chunk_smart，并删除了 model = ud_model 参数
  res_list <- mclapply(batches, process_chunk_smart, mc.cores = NUM_CORES)
  
  # 过滤掉解析失败返回 NULL 的批次
  res_list <- Filter(Negate(is.null), res_list)
  if (length(res_list) == 0) next
  
  # 3. 合并各核结果
  all_lines <- unlist(lapply(res_list, function(x) x$raw_lines))
  all_dirty <- unlist(lapply(res_list, function(x) x$is_dirty))
  
  # 4. 物理分流落盘
  if (any(!all_dirty)) writeLines(all_lines[!all_dirty], con_clean)
  if (any(all_dirty)) writeLines(all_lines[all_dirty], con_audit)
  
  cat(sprintf("Chunk %04d | 放行: %-5d | PROPN 拦截: %-5d\n", 
              chunk_idx, sum(!all_dirty), sum(all_dirty)))
  
  chunk_idx <- chunk_idx + 1
}

close(con_in)
close(con_clean)
close(con_audit)
