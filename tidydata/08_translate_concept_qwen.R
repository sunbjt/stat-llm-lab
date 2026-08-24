# ============================================================
# Wikipedia 概念翻译流水线 (DashScope / Qwen 异步并发版)
# 输入:  爬虫 relations (data.frame 或 checkpoint RData)
# 输出:  concepts_translated_ali.jsonl
# ============================================================

library(igraph)
library(dplyr)
library(httr2)
library(jsonlite)

# ---- 0. 环境变量检查 ----
glm_key <- Sys.getenv("aliyun_key")
if (glm_key == "") {
  stop("环境变量 'aliyun_key' 未设置，请先配置 API Key。")
}

# ---- 1. 加载关系数据 ----
load_relations_from_checkpoint <- function(checkpoint_file = "China_crawl_checkpoint.Rdata") {
  if (!file.exists(checkpoint_file)) stop("Checkpoint file not found: ", checkpoint_file)
  load(checkpoint_file)
  relations <- do.call(rbind, all_relations)
  rownames(relations) <- NULL
  cat(sprintf("Loaded %d relations from checkpoint (%d pages crawled)\n",
              nrow(relations), length(visited)))
  return(relations)
}

load_relations_from_rdata <- function(rdata_file = "relations.Rdata") {
  if (!file.exists(rdata_file)) stop("RData file not found: ", rdata_file)
  load(rdata_file)
  cat(sprintf("Loaded %d relations from RData\n", nrow(relations)))
  return(relations)
}

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
g  <- graph_from_data_frame(relations)
pg <- page_rank(g)$vector

top_n <- 30000
top_concepts <- data.frame(name = names(pg), pagerank = as.numeric(pg)) %>%
  arrange(desc(pagerank)) %>%
  head(top_n)

cat(sprintf("Top %d concepts selected (from %d total nodes)\n",
            nrow(top_concepts), length(pg)))

# ---- 3. httr2 异步并发蒸馏函数 ----
async_translate <- function(batch_list, api_key,
                            url       = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                            model     = "qwen3.7-flash",
                            #model     = "qwen3.7-flash-2026-07-15",
                            max_tries = 3) {
  
  system_prompt <- paste0(
    "You are a Wikipedia concept translator and encyclopedist. ",
    "For each concept in the input list, you must:\n",
    "1. Translate the English concept name to accurate Chinese.\n",
    "2. Write a detailed Chinese explanation covering:\n",
    "   - What the concept is (core definition)\n",
    "   - Key technical ideas / how it works\n",
    "   - Real-world applications and significance\n",
    "   - Relationship to other concepts in the field if relevant\n",
    "Aim for 10-15 detailed sentences per concept. Do NOT write a one-liner.\n",
    "Return ONLY a valid JSON array. No markdown, no code fences, no extra text.\n",
    'Format: [{"name":"<original>","name_cn":"<Chinese translation>","explanation":"<detailed Chinese explanation>","pagerank":<number>}]'
  )

  # 构建请求列表
  req_list <- lapply(seq_along(batch_list), function(i) {
    batch_df <- batch_list[[i]]
    concept_text <- paste(
      sprintf("%d. %s (PageRank: %.6f)", seq_len(nrow(batch_df)), batch_df$name, batch_df$pagerank),
      collapse = "\n"
    )

    body <- list(
      model           = model,
      messages        = list(
        list(role = "system", content = system_prompt),
        list(role = "user",   content = paste0("Translate these concepts:\n", concept_text))
      ),
      temperature     = 0.2,
      enable_thinking = FALSE
    )

    request(url) %>%
      req_headers(
        Authorization  = paste("Bearer", api_key),
        `Content-Type` = "application/json"
      ) %>%
      req_body_json(body) %>%
      req_timeout(120) %>%
      req_retry(max_tries = max_tries, backoff = ~ 5)
  })

  # 修正点：移除 max_concurrent 参数，直接执行异步并发
  resps <- req_perform_parallel(req_list, on_error = "continue")

  # 解析响应数据
  parsed_results <- lapply(seq_along(resps), function(i) {
    resp <- resps[[i]]
    if (inherits(resp, "error") || is.null(resp) || resp_status(resp) != 200) {
      return(NULL)
    }

    content_str <- resp_body_string(resp, encoding = "UTF-8")
    parsed_json <- tryCatch(jsonlite::fromJSON(content_str), error = function(e) NULL)
    if (is.null(parsed_json)) return(NULL)

    res_content <- NULL
    if (!is.null(parsed_json$choices$message$content)) {
      res_content <- parsed_json$choices$message$content
    } else if (!is.null(parsed_json$choices[[1]]$message$content)) {
      res_content <- parsed_json$choices[[1]]$message$content
    }
    if (is.null(res_content)) return(NULL)

    cleaned_text <- gsub("^```json\\s*", "", res_content)
    cleaned_text <- gsub("^```\\s*", "", cleaned_text)
    cleaned_text <- gsub("\\s*```$", "", cleaned_text)
    cleaned_text <- trimws(cleaned_text)

    tryCatch(
      jsonlite::fromJSON(cleaned_text, simplifyDataFrame = TRUE),
      error = function(e) NULL
    )
  })

  return(parsed_results)
}

# ---- 4. 分块并发翻译（带断点续传与 JSONL 追加落盘） ----
cat("\n--- Starting Parallel Translation ---\n")
batch_size     <- 20   # 每个 API 请求包含的概念数
max_concurrent <- 10   # 同时并发的 API 请求数
output_file    <- "concepts_translated_china.jsonl"
progress_file  <- "translate_progress_china.Rdata"

# 切分全量数据为批次
all_batches <- split(top_concepts, (seq_len(nrow(top_concepts)) - 1) %/% batch_size)
n_batches   <- length(all_batches)

completed_batches <- c()
if (file.exists(progress_file)) {
  load(progress_file) # 恢复 completed_batches
  cat(sprintf("Resumed from progress file (%d / %d batches completed)\n",
              length(completed_batches), n_batches))
}

# 找出未完成的批次
pending_indices <- setdiff(seq_len(n_batches), completed_batches)

if (length(pending_indices) > 0) {
  # 按 max_concurrent 分块推送并发请求
  chunk_size <- max_concurrent
  chunk_groups <- split(pending_indices, (seq_along(pending_indices) - 1) %/% chunk_size)

  con <- file(output_file, "a", encoding = "UTF-8")
  
  for (g_idx in seq_along(chunk_groups)) {
    cur_indices <- chunk_groups[[g_idx]]
    cur_batches <- all_batches[cur_indices]

    cat(sprintf("[%d/%d Groups] Processing batches %d to %d asynchronously...\n",
                g_idx, length(chunk_groups), min(cur_indices), max(cur_indices)))

    # 发起并发请求
    results <- async_translate(
      batch_list = cur_batches,
      api_key    = glm_key
    )

    # 处理写盘与进度更新
    for (b_pos in seq_along(results)) {
      b_idx  <- cur_indices[b_pos]
      res_df <- results[[b_pos]]

      if (!is.null(res_df) && is.data.frame(res_df) && nrow(res_df) > 0) {
        for (r in seq_len(nrow(res_df))) {
          writeLines(jsonlite::toJSON(as.list(res_df[r, ]), auto_unbox = TRUE), con)
        }
        completed_batches <- c(completed_batches, b_idx)
      } else {
        warning(sprintf("Batch %d failed or returned empty result.", b_idx))
      }
    }
    
    flush(con)
    save(completed_batches, file = progress_file)
    cat(sprintf("  Progress saved (%d/%d batches completed)\n", length(completed_batches), n_batches))
  }
  
  close(con)
}

# ---- 5. 统计最终结果 ----
cat("\n--- Final Stats ---\n")
if (file.exists(output_file)) {
  final_lines <- readLines(output_file, warn = FALSE)
  cat(sprintf("%d concepts successfully distilled and written to %s\n",
              length(final_lines), output_file))
}
cat(sprintf("Total completion rate: %.2f%%\n",
            (length(completed_batches) / n_batches) * 100))


if(FALSE){
  # 读取 JSONL 文件（每行一个 JSON 对象）
  data <- stream_in(file("concepts_translated_ali.jsonl"))
  # 修改字段名：将 explanation 改为 text
  names(data)[names(data) == "explanation"] <- "text"
  data <- data[, "text", drop = FALSE]
  stream_out(data, file("output1.jsonl"))
  }
