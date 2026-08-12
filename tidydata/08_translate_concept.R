# ============================================================
# Wikipedia 概念翻译流水线
# 输入:  爬虫 relations (data.frame 或 checkpoint RData)
# 输出:  concepts_translated.jsonl
# 依赖:  httr, jsonlite, igraph, dplyr
# ============================================================

library(igraph)
library(dplyr)
library(httr)
library(jsonlite)

deepseek_key <- Sys.getenv("deepseek_API_KEY")

# ---- 1. 加载关系数据 ----
# 方式 A: 从 checkpoint 文件加载
load_relations_from_checkpoint <- function(checkpoint_file = "crawl_checkpoint.Rdata") {
  if (!file.exists(checkpoint_file)) {
    stop("Checkpoint file not found: ", checkpoint_file)
  }
  load(checkpoint_file)  # 恢复 all_relations, visited 等
  relations <- do.call(rbind, all_relations)
  rownames(relations) <- NULL
  cat(sprintf("Loaded %d relations from checkpoint (%d pages crawled)\n",
              nrow(relations), length(visited)))
  return(relations)
}

# 方式 B: 从 RData 文件加载
load_relations_from_rdata <- function(rdata_file = "relations.Rdata") {
  if (!file.exists(rdata_file)) {
    stop("RData file not found: ", rdata_file)
  }
  load(rdata_file)
  cat(sprintf("Loaded %d relations from RData\n", nrow(relations)))
  return(relations)
}

# 方式 C: 直接使用当前环境的 relations 变量
# relations <- relations  # 如果在 crawler.r 之后运行，已存在

# 自动检测数据来源
if (!exists("relations") || nrow(relations) == 0) {
  if (file.exists("crawl_checkpoint.Rdata")) {
    relations <- load_relations_from_checkpoint()
  } else if (file.exists("relations.Rdata")) {
    relations <- load_relations_from_rdata()
  } else {
    stop("No relations data found. Run crawler.r first or provide a valid data source.")
  }
}

# ---- 2. 构建图并计算 PageRank ----
cat("\n--- Computing PageRank ---\n")
g <- graph_from_data_frame(relations)
pg <- page_rank(g)$vector

top_n <- 5000
top_concepts <- data.frame(name = names(pg), pagerank = as.numeric(pg)) %>%
  arrange(desc(pagerank)) %>%
  head(top_n)

cat(sprintf("Top %d concepts selected (from %d total nodes)\n",
            nrow(top_concepts), length(pg)))

# ---- 3. DeepSeek API 翻译函数 ----
deepseek_translate <- function(batch_df, api_key,
                               model    = "deepseek-v4-flash",
                               max_tries = 3) {
  # 构造概念列表文本
  concept_text <- paste(
    sprintf("%d. %s (PageRank: %.6f)",
            seq_len(nrow(batch_df)),
            batch_df$name,
            batch_df$pagerank),
    collapse = "\n"
  )

  system_prompt <- paste0(
    "You are a Wikipedia concept translator and encyclopedist. ",
    "For each concept in the input list, you must:\n",
    "1. Translate the English concept name to accurate Chinese.\n",
    "2. Write a detailed Chinese explanation covering:\n",
    "   - What the concept is (core definition)\n",
    "   - Key technical ideas / how it works\n",
    "   - Real-world applications and significance\n",
    "   - Relationship to other concepts in the field if relevant\n",
    "Aim for 8-12 detailed sentences per concept. Do NOT write a one-liner.\n",
    "Return ONLY a valid JSON array. No markdown, no code fences, no extra text.\n",
    'Format: [{"name":"<original>","name_cn":"<Chinese translation>","explanation":"<detailed Chinese explanation>","pagerank":<number>}]'
  )

  body <- list(
    model    = model,
    messages = list(
      list(role = "system", content = system_prompt),
      list(role = "user",   content = paste0("Translate these concepts:\n", concept_text))
    ),
    temperature = 0.2
  )

  for (attempt in seq_len(max_tries)) {
    resp <- tryCatch(
      httr::POST(
        url = "https://api.deepseek.com/chat/completions",
        httr::add_headers(
          Authorization = paste("Bearer", api_key),
          "Content-Type" = "application/json"
        ),
        body   = body,
        encode = "json",
        httr::timeout(120)
      ),
      error = function(e) {
        warning(sprintf("HTTP error (attempt %d/%d): %s", attempt, max_tries, conditionMessage(e)))
        return(NULL)
      }
    )

    if (is.null(resp)) next

    if (httr::status_code(resp) == 200) {
      content <- httr::content(resp, "text", encoding = "UTF-8")
      parsed  <- tryCatch(
        jsonlite::fromJSON(content),
        error = function(e) NULL
      )

      if (!is.null(parsed)) {
        # 处理可能的响应格式
        if (!is.null(parsed$choices$message$content)) {
          result_text <- parsed$choices$message$content
          # 清理可能的 markdown 代码块
          result_text <- gsub("^```json\\s*", "", result_text)
          result_text <- gsub("^```\\s*", "", result_text)
          result_text <- gsub("\\s*```$", "", result_text)
          result_text <- trimws(result_text)

          result <- tryCatch(
            jsonlite::fromJSON(result_text, simplifyDataFrame = TRUE),
            error = function(e) {
              warning(sprintf("JSON parse error (attempt %d/%d): %s",
                              attempt, max_tries, conditionMessage(e)))
              return(NULL)
            }
          )

          if (!is.null(result) && length(result) > 0) {
            return(result)
          }
        }
      }
    } else {
      warning(sprintf("API returned status %d (attempt %d/%d): %s",
                      httr::status_code(resp), attempt, max_tries,
                      substr(httr::content(resp, "text"), 1, 200)))
    }

    if (attempt < max_tries) {
      cat(sprintf("Retrying in %d seconds...\n", attempt * 5))
      Sys.sleep(attempt * 5)
    }
  }

  warning(sprintf("Batch failed after %d attempts, returning empty result.", max_tries))
  return(data.frame())
}

# ---- 4. 批量翻译（即时落盘 + 失败重试） ----
cat("\n--- Starting Translation ---\n")
batch_size <- 20
n_batches  <- ceiling(nrow(top_concepts) / batch_size)

output_file <- "concepts_translated.jsonl"

# 检查是否有断点可以恢复
progress_file <- "translate_progress.Rdata"
start_batch <- 1
failed_batches <- list()
if (file.exists(progress_file)) {
  load(progress_file)  # 恢复 last_completed, failed_batches
  start_batch <- last_completed + 1
  cat(sprintf("Resumed from batch %d (appending to %s)\n", start_batch, output_file))
}

# 打开输出文件（追加模式）
con <- file(output_file, "a", encoding = "UTF-8")
on.exit(close(con), add = TRUE)

# ---- 翻译一个 batch 的辅助函数 ----
process_batch <- function(i, batch_df) {
  cat(sprintf("[%d/%d] Concepts %d-%d... ", i, n_batches,
              (i - 1) * batch_size + 1, min(i * batch_size, nrow(top_concepts))))

  result <- tryCatch(
    deepseek_translate(batch_df, deepseek_key),
    error = function(e) {
      warning(sprintf("Fatal error in batch %d: %s", i, conditionMessage(e)))
      return(data.frame())
    }
  )

  if (!is.null(result) && is.data.frame(result) && nrow(result) > 0) {
    for (j in seq_len(nrow(result))) {
      line <- jsonlite::toJSON(as.list(result[j, ]), auto_unbox = TRUE)
      writeLines(line, con)
    }
    flush(con)
    cat(sprintf("OK (%d concepts) -> %s\n", nrow(result), output_file))
    return(TRUE)
  } else {
    cat("FAILED\n")
    return(FALSE)
  }
}

# ---- 第一轮：按顺序处理所有 batch ----
for (i in seq.int(start_batch, n_batches)) {
  idx_start <- (i - 1) * batch_size + 1
  idx_end   <- min(i * batch_size, nrow(top_concepts))
  batch     <- top_concepts[idx_start:idx_end, ]

  ok <- process_batch(i, batch)
  if (!ok) failed_batches[[as.character(i)]] <- batch

  # 每 10 批保存断点（含失败列表）
  if (i %% 10 == 0) {
    last_completed <- i
    save(last_completed, failed_batches, file = progress_file)
    cat(sprintf("  Checkpoint saved (batch %d/%d, %d failed)\n",
                i, n_batches, length(failed_batches)))
  }

  Sys.sleep(1.5)
}

# ---- 第二轮：重试失败的 batch（最多 2 轮额外重试） ----
max_retry_rounds <- 2
retry_round <- 1
while (length(failed_batches) > 0 && retry_round <= max_retry_rounds) {
  cat(sprintf("\n--- Retry round %d: %d failed batches ---\n",
              retry_round, length(failed_batches)))
  still_failed <- list()

  for (idx in names(failed_batches)) {
    cat(sprintf("Retrying batch %s... ", idx))
    ok <- process_batch(as.integer(idx), failed_batches[[idx]])
    if (!ok) still_failed[[idx]] <- failed_batches[[idx]]
    Sys.sleep(2)  # 失败重试间隔稍长
    save(last_completed, failed_batches = still_failed, file = progress_file)
  }

  failed_batches <- still_failed
  retry_round <- retry_round + 1
}

if (length(failed_batches) > 0) {
  cat(sprintf("\nWARNING: %d batches still failed after all retries:\n",
              length(failed_batches)))
  cat(paste(" ", names(failed_batches)), "\n")
  save(last_completed, failed_batches, file = progress_file)
}

close(con)

# 清理进度文件
if (file.exists(progress_file) && length(failed_batches) == 0) {
  file.remove(progress_file)
}

# 统计最终结果
cat("\n--- Final Stats ---\n")
final_lines <- readLines(output_file, warn = FALSE)
cat(sprintf("%d concepts written to %s\n", length(final_lines), output_file))
if (length(failed_batches) > 0) {
  cat(sprintf("%d batches failed, re-run to retry again\n", length(failed_batches)))
}
