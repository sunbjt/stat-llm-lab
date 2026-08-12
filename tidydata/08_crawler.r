# 确保你的 API KEY 已经设置
Sys.setenv(https_proxy = "http://127.0.0.1:7897")
Sys.setenv(http_proxy  = "http://127.0.0.1:7897")

library(rvest)
library(stringr)
library(magrittr)
library(utils)
library(igraph)

retry <-
  function(expr,
           isError = function(x)
             "try-error" %in% class(x),
           maxErrors = 5,
           sleep = rnorm(1, 8, 2)) {
    attempts = 0
    retval = try(eval(expr))
    while (isError(retval)) {
      attempts = attempts + 1
      if (attempts >= maxErrors) {
        msg = sprintf("retry: too many retries [[%s]]", capture.output(str(retval)))
        stop(msg)
      } else {
        msg = sprintf(
          "retry: error in attempt %i/%i [[%s]]",
          attempts,
          maxErrors,
          capture.output(str(retval))
        )
        warning(msg)
      }
      if (sleep > 0)
        Sys.sleep(sleep)
      retval = try(eval(expr))
    }
    return(retval)
  }


gettopic <- function(topic = 'Data_science') {
  url <- paste('https://en.wikipedia.org/wiki/', topic, sep = '')
  body <- retry(read_html(url))
  
  href <-
    body %>%
    html_node('#bodyContent') %>%
    html_nodes('a') %>%
    html_attr('href')

  # 筛选 /wiki/ 链接（兼容相对路径 /wiki/... 和绝对路径 https://.../wiki/...）
  # 排除已知的非文章命名空间: File:, Category:, Template:, Wikipedia:, Help: 等
  is_wiki  <- grepl('/wiki/', href, fixed = TRUE) & !is.na(href)
  is_bad   <- grepl('wiki/(File|Category|Template|Wikipedia|Help|Portal|Special|Talk|User|MediaWiki|Draft|TimedText|Module|Gadget|Gadget_definition|Topic):',
                    href, perl = TRUE)
  wiki_hrefs <- href[is_wiki & !is_bad]

  # 从 href 中提取 /wiki/ 之后的文章标题
  topic_right <- str_extract(wiki_hrefs, '(?<=/wiki/).+') %>%
    URLdecode() %>%
    str_replace_all('[#?].*$', '') %>%  # 去掉锚点和查询参数
    unique()

  # 去掉 NA，应用 should_filter 作为二次过滤
  topic_right <- topic_right[!is.na(topic_right)]
  topic_right <- topic_right[topic_right != ""]
  topic_right <- topic_right[!sapply(topic_right, should_filter)]

  # 无有效链接时返回空 data.frame
  if (length(topic_right) == 0) {
    return(data.frame(topic_left = character(0), topic_right = character(0),
                      stringsAsFactors = FALSE))
  }

  data.frame(topic_left = topic, topic_right, stringsAsFactors = FALSE) %>% na.omit()
}

#' 判断话题是否应该被过滤（不爬取、不纳入关系）
#'
#' @param topic 话题名称
#' @param extra_patterns 额外的过滤正则模式
#' @return TRUE 表示应该过滤掉
should_filter <- function(topic, extra_patterns = "Glossary|List|ournal|onference") {
  # NA 或空字符串直接过滤
  if (is.na(topic) || topic == "") return(TRUE)

  # 包含 ":" 的话题（File:, Category:, Template: 等）
  if (grepl(":", topic, fixed = TRUE)) return(TRUE)

  # 匹配额外排除模式
  if (nchar(extra_patterns) > 0 && grepl(extra_patterns, topic, ignore.case = TRUE)) return(TRUE)

  return(FALSE)
}

#' 递归爬取 Wikipedia 概念关系图（BFS，支持断点续传）
#'
#' @param seed 种子话题（字符向量）
#' @param max_depth 最大爬取深度，1 = 只爬种子页面，2 = 种子 + 1层链接，默认 3
#' @param min_sleep 请求间隔最小值（秒），默认 8
#' @param max_sleep 请求间隔最大值（秒），默认 20
#' @param max_pages 最大爬取页数（安全阀），默认 2000
#' @param checkpoint_file 断点续传文件路径，NULL 表示不启用
#' @param checkpoint_every 每爬取多少页保存一次断点，默认 20
#' @return 包含边关系的 data.frame (topic_left, topic_right, depth)
crawl_topic <- function(seed,
                        max_depth        = 3,
                        min_sleep        = 8,
                        max_sleep        = 20,
                        max_pages        = 2000,
                        checkpoint_file  = NULL,
                        checkpoint_every = 20) {
  # ---- 尝试从断点恢复 ----
  if (!is.null(checkpoint_file) && file.exists(checkpoint_file)) {
    load(checkpoint_file)  # 恢复 visited, all_relations, queue, page_count
    message(sprintf("Resumed from checkpoint: %d visited, %d queued",
                    length(visited), length(queue)))
  } else {
    visited       <- character(0)
    all_relations <- list()
    page_count    <- 0
    # 队列元素: list(topic = "xxx", depth = 1)
    queue <- lapply(seed, function(s) list(topic = s, depth = 1))
  }

  # ---- BFS 主循环 ----
  while (length(queue) > 0 && page_count < max_pages) {
    # 出队
    current <- queue[[1]]
    queue   <- queue[-1]
    topic   <- current$topic
    depth   <- current$depth

    # 跳过条件
    if (is.null(topic) || topic == "")           next
    if (topic %in% visited)                      next
    if (depth > max_depth)                       next
    if (should_filter(topic))                    next

    # 标记已访问
    visited    <- c(visited, topic)
    page_count <- page_count + 1

    # 进度输出
    message(sprintf("[Depth %d | %d crawled | %d queued] %s",
                    depth, page_count, length(queue), topic))

    # 爬取（tryCatch 精细错误处理）
    edges <- tryCatch(
      retry(gettopic(topic)),
      error = function(e) {
        warning(sprintf("ERROR crawling '%s': %s", topic, conditionMessage(e)))
        data.frame(topic_left = character(0), topic_right = character(0),
                   stringsAsFactors = FALSE)
      }
    )

    # 存储关系（附深度标记）
    if (!is.null(edges) && nrow(edges) > 0) {
      edges$depth <- depth
      all_relations[[length(all_relations) + 1]] <- edges
    }

    # 入队新话题（未达深度上限时）
    if (depth < max_depth && !is.null(edges) && nrow(edges) > 0) {
      new_topics <- setdiff(unique(edges$topic_right), visited)
      for (t in new_topics) {
        if (!should_filter(t)) {
          queue[[length(queue) + 1]] <- list(topic = t, depth = depth + 1)
        }
      }
    }

    # 断点保存
    if (!is.null(checkpoint_file) && page_count %% checkpoint_every == 0) {
      save(visited, all_relations, queue, page_count, file = checkpoint_file)
      message(sprintf("Checkpoint saved (%d pages)", page_count))
    }

    # 速率限制（均匀分布抖动）
    Sys.sleep(runif(1, min_sleep, max_sleep))
  }

  # ---- 组装最终结果 ----
  if (length(all_relations) == 0) {
    return(data.frame(topic_left = character(0), topic_right = character(0),
                      depth = integer(0), stringsAsFactors = FALSE))
  }

  relations <- do.call(rbind, all_relations)
  rownames(relations) <- NULL

  # 最终断点
  if (!is.null(checkpoint_file)) {
    save(visited, all_relations, queue, page_count, file = checkpoint_file)
  }

  message(sprintf("Crawl complete: %d pages crawled, %d relations found.",
                  page_count, nrow(relations)))
  return(relations)
}

## 递归爬取 3 层，构建概念关系图，控制参数 max_depth
# 0 种子页面本身（Data_mining）
# 1 种子页面上找到的链接
# 2 第 1 层链接页面上找到的链接 
relations <- crawl_topic(
  seed             = "Data_mining",
  max_depth        = 2,
  min_sleep        = 3,
  max_sleep        = 6,
  max_pages        = 1000,
  checkpoint_file  = "crawl_checkpoint.Rdata",
  checkpoint_every = 20
)

## 可选：爬取多个种子后合并
## relations_ml <- crawl_topic("Machine_learning", max_depth = 2)
## relations <- rbind(relations, relations_ml) %>% unique()

# 保存最终结果
# save(relations, file = "relations.Rdata")

## 二次安全过滤（should_filter 在爬取时已过滤大部分，此处兜底）
yes_left <- as.character(relations$topic_left) %>%
  str_detect('Glossary|List|ournal|onference')
yes_right <- as.character(relations$topic_right) %>%
  str_detect('Glossary|List|ournal|onference')
relations <- relations[yes_left + yes_right == 0, ]


## 构造网络关系
g <- graph_from_data_frame(relations)

library(dplyr)
pg <- page_rank(g)$vector
data.frame(name = names(pg), v = pg) %>% 
  arrange(desc(v)) %>% 
  slice(3000:3500)
