library(RcppSimdJson)
library(data.table)
library(jsonlite)
library(textreuse)
library(parallel)

# === 全局配置 ===
# 注意替换为你实际的新数据路径
input_path <- "data/raw/pretrain_clean.jsonl" 
output_path <- "data/processed/dedup.jsonl"
chunk_lines <- 50000
n_hashes <- 100
bands <- 20
rows_per_band <- n_hashes / bands

num_cores <- 1
options(mc.cores = num_cores)

cat(sprintf("[%s] 阶段 1: 分块提取内存指纹...\n", Sys.time()))
minhash <- minhash_generator(n = n_hashes, seed = 42)
con_in <- file(input_path, "r")
chunk_id <- 0
sig_list <- list()

repeat {
  lines <- readLines(con_in, n = chunk_lines, warn = FALSE)
  if (length(lines) == 0) break
  chunk_id <- chunk_id + 1
  
  df_chunk <- data.table::rbindlist(RcppSimdJson::fparse(lines), fill = TRUE)
  
  # 【关键修复 1】: 增加异常文本拦截与安全捕获，防止并行核心崩溃
  chunk_sigs <- mclapply(df_chunk$text, function(txt) {
    if (is.na(txt) || nchar(trimws(txt)) < 5) {
      return(rep(NA_integer_, n_hashes))
    }
    tryCatch({
      tokens <- tokenize_ngrams(txt, n = 5)
      if (length(tokens) == 0) return(rep(NA_integer_, n_hashes))
      minhash(tokens)
    }, error = function(e) {
      return(rep(NA_integer_, n_hashes))
    })
  }, mc.cores = num_cores)
  
  sig_list[[chunk_id]] <- do.call(cbind, chunk_sigs)
  cat(sprintf(" -> 处理分块 %02d 完成 (提取 %d 篇)...\n", chunk_id, length(chunk_sigs)))
  
  rm(lines, df_chunk, chunk_sigs); gc(verbose = FALSE)
}
close(con_in)

# 生成全局变量 full_sigs，随后丢弃过渡列表
full_sigs <- do.call(cbind, sig_list)
total_docs <- ncol(full_sigs)
rm(sig_list); gc()

cat(sprintf("\n[%s] 阶段 2: 内存划带碰撞与冗余精算...\n", Sys.time()))
candidate_pairs <- data.table(doc_A = integer(), doc_B = integer())

for (b in 1:bands) {
  start_row <- (b - 1) * rows_per_band + 1
  end_row <- b * rows_per_band
  
  band_sigs <- full_sigs[start_row:end_row, ]
  band_keys <- apply(band_sigs, 2, paste, collapse = "_")
  
  dt_band <- data.table(doc_id = 1:total_docs, key_str = band_keys)
  
  # 剔除 NA 无效签名，并限制超大哈希桶防止组合爆炸
  dt_band <- dt_band[!grepl("NA", key_str)]
  collisions <- dt_band[, .(docs = list(doc_id), count = .N), by = key_str][count > 1 & count <= 1000]
  
  if (nrow(collisions) > 0) {
    pairs_list <- lapply(collisions$docs, function(idx) {
      combs <- combn(idx, 2)
      data.table(doc_A = combs[1, ], doc_B = combs[2, ])
    })
    candidate_pairs <- rbindlist(list(candidate_pairs, rbindlist(pairs_list)))
  }
}

candidate_pairs <- unique(candidate_pairs)
cat(sprintf(" -> 找到 %d 对嫌疑组合，开始精算...\n", nrow(candidate_pairs)))

# 核验相似度生成黑名单
drop_indices <- integer(0)
if (nrow(candidate_pairs) > 0) {
  match_counts <- sapply(1:nrow(candidate_pairs), function(i) {
    sum(full_sigs[, candidate_pairs$doc_A[i]] == full_sigs[, candidate_pairs$doc_B[i]])
  })
  
  jaccard_scores <- match_counts / n_hashes
  duplicates <- candidate_pairs[jaccard_scores >= 0.8]
  if (nrow(duplicates) > 0) drop_indices <- unique(duplicates$doc_B)
}

# 算完黑名单后，立刻销毁指纹矩阵释放内存
rm(full_sigs, candidate_pairs); gc()
cat(sprintf("确认需剔除 %d 篇冗余文档。\n", length(drop_indices)))

cat(sprintf("\n[%s] 阶段 3: 二次遍历流式写出...\n", Sys.time()))

con_in2 <- file(input_path, "r")
con_out <- file(output_path, "w")

current_idx <- 1
total_written <- 0
chunk_id <- 0

repeat {
  lines <- readLines(con_in2, n = chunk_lines, warn = FALSE)
  if (length(lines) == 0) break
  chunk_id <- chunk_id + 1
  
  df_chunk <- data.table::rbindlist(RcppSimdJson::fparse(lines), fill = TRUE)
  chunk_indices <- current_idx:(current_idx + nrow(df_chunk) - 1)
  
  # 对照全局黑名单进行精准剔除
  local_drop <- which(chunk_indices %in% drop_indices)
  df_final <- if (length(local_drop) > 0) df_chunk[-local_drop, ] else df_chunk
  
  if (nrow(df_final) > 0) {
    # 【关键修复 3】: 新数据没有 title，只保留 text 字段流式写出
    jsonlite::stream_out(df_final[, .(text)], con_out, pagesize = 20000, verbose = FALSE)
    total_written <- total_written + nrow(df_final)
  }
  
  current_idx <- current_idx + nrow(df_chunk)
  cat(sprintf(" -> 分块 %02d 写入完成，累计放行: %d 篇\n", chunk_id, total_written))
}

close(con_in2)
close(con_out)

cat(sprintf("\n初始文档: %d 篇 | 放行文档: %d 篇\n黄金语料已写出至: %s\n",
            total_docs, total_written, output_path))

