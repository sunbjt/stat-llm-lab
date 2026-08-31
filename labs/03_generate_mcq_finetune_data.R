# ============================================================
# 用大模型为 26 个子类别生成"四选一"微调题目
# —— 单请求隔离在子进程(callr) + 逐条落盘 + 基于文件内容的断点续跑
# ============================================================
library(httr2)
library(jsonlite)
library(callr)

# ---- 0. 子类别映射 ----
subcategory_map <- c(
  "distractor" = "干扰项识别",
  "instruction_attack" = "指令攻击",
  "format_following" = "格式遵循",
  "selection" = "选择筛选",
  "general_knowledge" = "常识知识",
  "ambiguity" = "歧义消解",
  "coreference" = "指代消解",
  "entailment" = "蕴含判断",
  "reading" = "阅读理解",
  "semantic_relation" = "语义关系",
  "machine_learning" = "机器学习知识",
  "arithmetic" = "算术运算",
  "multi_step" = "多步推理",
  "novel_rule" = "新颖规则",
  "sequence" = "序列预测",
  "word_problem" = "应用题",
  "fact_recall" = "事实回忆",
  "relational_recall" = "关系回忆",
  "reverse_recall" = "反向回忆",
  "analogy" = "类比推理",
  "number_pattern" = "数字规律",
  "symbol_pattern" = "符号规律",
  "conditional" = "条件逻辑",
  "constraints" = "约束满足",
  "ordering" = "排序顺序",
  "syllogism" = "三段论"
)

# ============================================================
# 1. 单个类别的请求逻辑：整体作为一个自包含函数，供 callr 在独立子进程执行
#    ——子进程里发生的任何崩溃（含 segfault），callr 都能捕捉并转成
#      父进程里一个普通的、可以 tryCatch 的 R 错误，不会带崩整个会话。
# ============================================================
fetch_one_category <- function(item, api_key, model,
                                url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                                max_tries = 3) {
  library(httr2)
  library(jsonlite)

  is_scalar_nonempty_char <- function(x) {
    is.character(x) && length(x) == 1 && !is.na(x) && nzchar(x)
  }

  build_system_prompt <- function(subcategory_cn) {
    paste0(
      "你是一位专业的中文考试命题专家，负责为一个参数量很小(约15M)的语言模型出『训练/评测用』的四选一单选题。\n",
      "本次题型类别：【", subcategory_cn, "】。\n",
      "命题要求：\n",
      "1. 每道题必须有且仅有一个正确答案；四个选项(A/B/C/D)语义清晰、干扰项合理，不能有歧义或多解、不能有语法或事实错误。\n",
      "2. 难度适中偏基础——目标是让小模型学会该题型的『做题模式』和『格式』，避免需要超长推理链条或专业冷僻知识。\n",
      "3. 题干、句式、选项设计要多样化，同一批题目之间不能出现重复或高度相似的模板。\n",
      "4. 正确答案在 A/B/C/D 之间要大致均匀分布，不要总是把正确答案放在同一个位置。\n",
      "5. 每道题需包含：question(题干)、options(键为A/B/C/D的字典，每个值必须是单行纯文本字符串，不能是数组或嵌套结构)、answer(正确选项字母)、explanation(一句话解析)。\n",
      "6. 严格只返回合法 JSON 数组，不要 markdown、不要代码块围栏（```)、不要任何多余文字或注释。\n",
      '格式: [{"question":"<题干>","options":{"A":"...","B":"...","C":"...","D":"..."},"answer":"A","explanation":"<解析>"}]'
    )
  }

  user_prompt <- sprintf(
    "请为题型类别【%s】生成 %d 道全新的四选一单选题，题目之间互不重复，覆盖该题型下不同的具体场景/表述方式。",
    item$subcategory_cn, item$n
  )

  body <- list(
    model           = model,
    messages        = list(
      list(role = "system", content = build_system_prompt(item$subcategory_cn)),
      list(role = "user",   content = user_prompt)
    ),
    temperature     = 0.9,
    enable_thinking = FALSE
  )

  req <- request(url) %>%
    req_headers(
      Authorization  = paste("Bearer", api_key),
      `Content-Type` = "application/json"
    ) %>%
    req_body_json(body) %>%
    req_timeout(180) %>%
    req_retry(max_tries = max_tries, backoff = ~ 5)

  resp <- req_perform(req)  # 出错/崩溃直接让子进程报错退出，父进程用 callr 捕获

  status <- resp_status(resp)
  if (status != 200) {
    stop(sprintf("HTTP状态码=%d", status))
  }

  content <- resp_body_json(resp)$choices[[1]]$message$content
  clean <- gsub("^```(json)?|```$", "", trimws(content))
  parsed <- fromJSON(clean, simplifyVector = FALSE)

  valid <- Filter(function(q) {
    tryCatch({
      is_scalar_nonempty_char(q$question) &&
        is.list(q$options) &&
        all(c("A","B","C","D") %in% names(q$options)) &&
        all(vapply(q$options[c("A","B","C","D")], is_scalar_nonempty_char, logical(1))) &&
        is_scalar_nonempty_char(q$answer) && q$answer %in% c("A","B","C","D") &&
        is_scalar_nonempty_char(q$explanation)
    }, error = function(e) FALSE)
  }, parsed)

  for (j in seq_along(valid)) valid[[j]]$category <- item$subcategory
  valid
}

# ============================================================
# 2. 读取已有输出文件，按类别统计已完成数量 + 收集已存在题干用于跨次去重
# ============================================================
read_existing_state <- function(output_file) {
  if (!file.exists(output_file) || file.size(output_file) == 0) {
    return(list(counts = integer(0), seen = character(0)))
  }
  df <- tryCatch(stream_in(file(output_file), verbose = FALSE), error = function(e) NULL)
  if (is.null(df) || nrow(df) == 0) return(list(counts = integer(0), seen = character(0)))

  counts <- table(df$category)
  seen <- gsub("\\s+", "", df$question)
  list(counts = as.list(counts), seen = seen)
}

# ============================================================
# 3. 主流程：基于"文件里已有多少条"来决定还要不要请求，天然支持断点续跑
#    —— 随时可以 Ctrl+C 中断，或者遇到崩溃后直接重新执行这个函数，
#       已经写盘的题目不会重复请求、也不会重复写入。
# ============================================================
run_generate_all <- function(api_key,
                              output_file       = "data/raw/mcq_finetune.jsonl",
                              n_per_category    = 60,
                              n_per_call        = 10,
                              model             = "qwen3.7-flash",
                              max_rounds_safety = 50) {

  state <- read_existing_state(output_file)
  existing_counts <- state$counts
  seen <- state$seen

  cat("===== 断点续跑状态检查 =====\n")
  for (key in names(subcategory_map)) {
    done <- if (key %in% names(existing_counts)) existing_counts[[key]] else 0
    cat(sprintf("  %-20s 已有 %d / %d 条\n", key, done, n_per_category))
  }

  con_out <- file(output_file, open = "a", encoding = "UTF-8")
  on.exit(close(con_out), add = TRUE)

  round_i <- 0
  repeat {
    round_i <- round_i + 1
    if (round_i > max_rounds_safety) {
      cat("\n[警告] 达到安全轮数上限，停止（可能某些类别持续请求失败，请检查日志）。\n")
      break
    }

    remaining <- sapply(names(subcategory_map), function(k) {
      done <- if (k %in% names(existing_counts)) existing_counts[[k]] else 0
      max(0, n_per_category - done)
    })
    if (all(remaining <= 0)) {
      cat("\n所有类别均已达到目标数量，完成。\n")
      break
    }

    cat(sprintf("\n===== 第 %d 轮 =====\n", round_i))

    for (key in names(subcategory_map)) {
      need <- remaining[[key]]
      if (need <= 0) next
      n_req <- min(n_per_call, need)

      cat(sprintf("  [%s] (%s) 还需 %d 条，本次请求 %d 条 ... ",
                  key, subcategory_map[[key]], need, n_req))

      item <- list(subcategory = key, subcategory_cn = subcategory_map[[key]], n = n_req)

      result <- tryCatch({
        callr::r(
          func    = fetch_one_category,
          args    = list(item = item, api_key = api_key, model = model),
          timeout = 200
        )
      }, error = function(e) {
        cat(sprintf("子进程失败: %s\n", conditionMessage(e)))
        list()
      })

      if (length(result) == 0) {
        cat("本次无有效题目\n")
        next
      }

      # 去重（跨轮次、跨断点续跑，用已加载的 seen 判重）
      fresh <- Filter(function(q) {
        key_q <- gsub("\\s+", "", q$question)
        if (key_q %in% seen) return(FALSE)
        seen[[length(seen) + 1]] <<- key_q
        TRUE
      }, result)

      if (length(fresh) > 0) {
        for (q in fresh) writeLines(toJSON(q, auto_unbox = TRUE), con_out)
        flush(con_out)  # 立刻落盘，崩溃也只丢当前这一条正在跑的请求
        existing_counts[[key]] <- (if (key %in% names(existing_counts)) existing_counts[[key]] else 0) + length(fresh)
      }

      cat(sprintf("成功写入 %d 条（去重后）\n", length(fresh)))
    }
  }

  cat(sprintf("\n输出文件: %s\n", output_file))
}

# ---- 使用示例 ----
run_generate_all(
  api_key         = Sys.getenv("aliyun_key"),
  output_file     = "data/raw/mcq_finetune.jsonl",
  n_per_category  = 60,     # 26 类 * 60 ≈ 1560 条，可按需调整
  n_per_call      = 10,
  model           = "qwen3.7-max-2026-06-08"
)


# causal_lm/05_build_mcq_sft_data.R
# =====================================================================
# 把 generate_mcq_finetune_data.R 生成的 data/raw/mcq_finetune.jsonl
# 转换成 04_sft_train.R 能直接消费的 instruction/output 格式
# =====================================================================
library(jsonlite)
library(dplyr)

set.seed(42)

INPUT_FILE      <- "data/raw/mcq_finetune.jsonl"
OUTPUT_TRAIN    <- "data/raw/mcq_sft_train.jsonl"
OUTPUT_DEV      <- "data/raw/mcq_sft_dev.jsonl"     # 训练过程中自测用，和最终1300题测试集不重叠
DEV_RATIO       <- 0.05                              # 留 5% 做训练期间的验证
INCLUDE_EXPLANATION <- FALSE                         # 见下方说明，先建议 FALSE

# ---- 1. 读入 mcq jsonl（每行一个 JSON 对象） ----
raw <- stream_in(file(INPUT_FILE), verbose = FALSE)
cat(sprintf("读入原始题目: %d 条\n", nrow(raw)))

# ---- 2b. 兼容 stream_in 对嵌套 options 字段的两种可能解析结果：
#      情况一：raw$options 被简化成 4 列(A/B/C/D)的 data.frame（jsonlite 默认行为）
#      情况二：raw$options 是每行一个 named list 组成的 list-column
get_option_col <- function(options_field, letter) {
  if (is.data.frame(options_field)) {
    v <- options_field[[letter]]
    if (is.null(v)) return(rep(NA_character_, nrow(options_field)))
    as.character(v)
  } else if (is.list(options_field)) {
    vapply(options_field, function(o) {
      v <- tryCatch(o[[letter]], error = function(e) NULL)
      if (is.null(v) || length(v) != 1) NA_character_ else as.character(v)
    }, character(1))
  } else {
    stop("未知的 options 字段结构: ", paste(class(options_field), collapse = "/"))
  }
}

opt_A <- get_option_col(raw$options, "A")
opt_B <- get_option_col(raw$options, "B")
opt_C <- get_option_col(raw$options, "C")
opt_D <- get_option_col(raw$options, "D")

build_instruction <- function(question, A, B, C, D) {
  sprintf(
    "以下是一道单选题，请只回答正确选项的字母（A/B/C/D），不要输出其他内容。\n题目：%s\nA. %s\nB. %s\nC. %s\nD. %s\n答案：",
    question, A, B, C, D
  )
}

build_output <- function(answer, explanation) {
  if (INCLUDE_EXPLANATION && !is.null(explanation) && !is.na(explanation) && nzchar(explanation)) {
    sprintf("%s\n解析：%s", answer, explanation)
  } else {
    answer
  }
}

# ---- 3. 构造 instruction/output，并做基本字段校验（防止生成脚本产出的脏数据混进来） ----
valid_rows <- which(
  !is.na(raw$question) & nzchar(raw$question) &
  !is.na(raw$answer)   & raw$answer %in% c("A","B","C","D") &
  !is.na(opt_A) & nzchar(opt_A) &
  !is.na(opt_B) & nzchar(opt_B) &
  !is.na(opt_C) & nzchar(opt_C) &
  !is.na(opt_D) & nzchar(opt_D)
)
cat(sprintf("字段校验通过: %d / %d 条\n", length(valid_rows), nrow(raw)))

raw   <- raw[valid_rows, ]
opt_A <- opt_A[valid_rows]; opt_B <- opt_B[valid_rows]
opt_C <- opt_C[valid_rows]; opt_D <- opt_D[valid_rows]

sft_df <- data.frame(
  category    = as.character(raw$category),
  instruction = mapply(build_instruction, raw$question, opt_A, opt_B, opt_C, opt_D),
  output      = mapply(build_output, raw$answer, raw$explanation),
  stringsAsFactors = FALSE
)

# ---- 4. 按类别分层切分 train/dev，保证每个类别在 dev 里都有覆盖 ----
sft_df <- sft_df[sample(nrow(sft_df)), ]  # 打乱，避免类别成块排列

split_by_category <- lapply(split(sft_df, sft_df$category), function(df) {
  n_dev <- max(1, round(nrow(df) * DEV_RATIO))
  list(dev = df[1:n_dev, ], train = df[(n_dev + 1):nrow(df), ])
})

train_df <- do.call(rbind, lapply(split_by_category, `[[`, "train"))
dev_df   <- do.call(rbind, lapply(split_by_category, `[[`, "dev"))

train_df <- train_df[sample(nrow(train_df)), ]
dev_df   <- dev_df[sample(nrow(dev_df)), ]

# ---- 5. 类别分布检查（确认没有某个类别数据量畸多/畸少） ----
cat("\n各类别训练样本数：\n")
print(table(train_df$category))

# ---- 6. 落盘为 04_sft_train.R 期望的格式（只需要 instruction/output 两列） ----
stream_out(train_df[, c("instruction", "output")], file(OUTPUT_TRAIN), verbose = FALSE)
stream_out(dev_df[,   c("instruction", "output")], file(OUTPUT_DEV),   verbose = FALSE)

cat(sprintf("\n完成。\n训练集: %d 条 -> %s\n验证集: %d 条 -> %s\n",
            nrow(train_df), OUTPUT_TRAIN, nrow(dev_df), OUTPUT_DEV))

# ---- 7. 抽样打印几条，人工确认格式无误 ----
cat("\n===== 样例检查 =====\n")
for (i in 1:min(3, nrow(train_df))) {
  cat(sprintf("--- 样例 %d (category=%s) ---\n", i, train_df$category[i]))
  cat(train_df$instruction[i], "\n")
  cat(">>> output:", train_df$output[i], "\n\n")
}