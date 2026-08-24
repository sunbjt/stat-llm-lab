library(httr2)
library(jsonlite)
library(purrr)
library(cli)

# ----------------------------------------------------
# 1. API 参数配置
# ----------------------------------------------------
API_KEY  <- "YOUR_API_KEY"                      # 替换为你的 API Key
BASE_URL <- "https://api.deepseek.com/v1"        # 替换为你的 API Endpoint
MODEL    <- "deepseek-chat"

# ----------------------------------------------------
# 2. 类别与数量定义 (总计 1500 条)
# ----------------------------------------------------
categories_config <- list(
  list(category = "language_understanding", code = "lang",  count = 200),
  list(category = "math_reasoning",         code = "math",  count = 200),
  list(category = "logic_reasoning",        code = "logic", count = 200),
  list(category = "pattern_induction",      code = "pat",   count = 150),
  list(category = "working_memory",         code = "mem",   count = 150),
  list(category = "common_sense",           code = "cs",    count = 400),
  list(category = "instruction_following",  code = "inst",  count = 100),
  list(category = "adversarial_robustness", code = "adv",   count = 100)
)

# ----------------------------------------------------
# 3. Prompt 构建函数
# ----------------------------------------------------
build_prompt <- function(category, count) {
  sprintf('
你是一个针对 15M 参数极小语言模型设计 Log Likelihood 四选一基准测试的专家。
请生成 %d 条单选题，类别为：【%s】。

【难度限制】：
题目必须极度简单直观，完全符合 15M 极小模型的物理容量上限（如数学仅限单步加减法，逻辑仅限直觉判断）。

【硬性约束】：
1. 每个题目提供 4 个选项，且 4 个选项的字数长度必须高度一致（防止长度缩放因子干扰对数似然度）。
2. 干扰项必须是客观错误的，确保唯一正确答案。
3. 输出格式必须是纯 JSON 数组，绝不能包含 Markdown 标记或多余说明。

JSON Schema:
[
  {
    "category": "%s",
    "prompt": "完整 Prompt（末尾必须统一以 \'答案：\' 结尾）",
    "options": ["A. xxx", "B. xxx", "C. xxx", "D. xxx"],
    "answer": "A. xxx"
  }
]
', count, category, category)
}

# ----------------------------------------------------
# 4. API 请求与解析函数
# ----------------------------------------------------
fetch_batch <- function(category, code, count, batch_id) {
  sys_prompt <- build_prompt(category, count)
  
  req <- request(BASE_URL) %>%
    req_url_path_append("chat/completions") %>%
    req_headers(
      "Authorization" = paste("Bearer", API_KEY),
      "Content-Type"  = "application/json"
    ) %>%
    req_body_json(list(
      model = MODEL,
      messages = list(list(role = "user", content = sys_prompt)),
      temperature = 0.7,
      response_format = list(type = "json_object")
    )) %>%
    req_retry(max_tries = 3)

  resp <- req_perform(req)
  res_json <- resp_body_json(resp)
  content_text <- res_json$choices[[1]]$message$content
  
  # 解析返回的 JSON 字符串
  parsed_data <- fromJSON(content_text, simplifyDataFrame = FALSE)
  
  # 如果返回外层包裹了 Key，提取列表
  if (!is.null(names(parsed_data))) {
    parsed_data <- parsed_data[[1]]
  }
  
  # 为每条数据赋予递增 ID
  for (i in seq_along(parsed_data)) {
    parsed_data[[i]]$id <- sprintf("%s_%04d", code, (batch_id - 1) * 50 + i)
  }
  
  return(parsed_data)
}

# ----------------------------------------------------
# 5. 主执行循环与 JSONL 导出
# ----------------------------------------------------
generate_benchmark <- function(output_file = "eval_1500_ll_test.jsonl") {
  all_data <- list()
  
  if (file.exists(output_file)) file.remove(output_file)
  con <- file(output_file, open = "a", encoding = "UTF-8")
  on.exit(close(con))

  cli_alert_info("开始生成 1500 条 R 语言评测数据集...")
  
  for (item in categories_config) {
    cat_name <- item$category
    code     <- item$code
    total    <- item$count
    batch_sz <- 50 # 每批生成 50 条
    batches  <- ceiling(total / batch_sz)
    
    cli_h2(sprintf("Processing: %s (Total: %d)", cat_name, total))
    
    for (b in seq_len(batches)) {
      curr_cnt <- min(batch_sz, total - (b - 1) * batch_sz)
      cli_progress_step(sprintf("Batch %d/%d (%d items)", b, batches, curr_cnt))
      
      tryCatch({
        batch_res <- fetch_batch(cat_name, code, curr_cnt, b)
        
        # 逐行写入 jsonl
        for (row in batch_res) {
          writeLines(toJSON(row, auto_unbox = TRUE, ensure_ascii = FALSE), con)
        }
        
        all_data <- c(all_data, batch_res)
        Sys.sleep(0.5) # 避免触发 Rate Limit
      }, error = function(e) {
        cli_alert_danger(sprintf("Batch %d failed: %s", b, e$message))
      })
    }
  }
  
  cli_alert_success(sprintf("成功导出 1500 条数据至 %s", output_file))
}

# 执行生成
generate_benchmark()