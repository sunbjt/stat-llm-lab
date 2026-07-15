# Distillation/01_generate_teacher.R
# =====================================================================
# 教师模型批量推理：用 llamaR 加载 Qwen3.5-2B，逐 prompt 生成高质量回答
# 输出 Distillation/teacher_outputs.jsonl
# =====================================================================
library(llamaR)
library(jsonlite)

# ========================== 可调参数 ==========================

MODEL_PATH     <- "models/Qwen3.5-2B.Q4_K_M.gguf"
DATA_PATH      <- "data/raw/qa_no_think.jsonl"
OUTPUT_PATH    <- "Distillation/teacher_outputs.jsonl"
N_SAMPLES      <- 500L          # 本次生成多少条（总共 3771 条）
START_FROM     <- 1L            # 从第几条开始（支持断点续传）
MAX_NEW_TOKENS <- 256L
TEMPERATURE    <- 0.7
TOP_P          <- 0.95
TOP_K          <- 50L
SEED           <- 42L
N_CTX          <- 2048L
N_THREADS      <- 4L

# ========================== 加载模型 ==========================

cat(sprintf("[%s] 正在加载教师模型...\n", Sys.time()))
model <- llama_load_model(MODEL_PATH, n_gpu_layers = 0L)
ctx   <- llama_new_context(model,
  n_ctx      = N_CTX,
  n_threads  = N_THREADS,
  flash_attn = "auto"
)
cat("教师模型就绪。\n")

# ========================== 读取 prompts ==========================

raw_data <- stream_in(file(DATA_PATH), verbose = FALSE)
n_total  <- nrow(raw_data)

if (N_SAMPLES > n_total) N_SAMPLES <- n_total
end_at <- min(START_FROM + N_SAMPLES - 1L, n_total)

cat(sprintf("共 %d 条数据，本次处理第 %d-%d 条\n", n_total, START_FROM, end_at))

# ========================== 后处理函数 ==========================

# Qwen3.5 Base 有时会输出 <think> 标签或额外的对话延续
# 这里做基础清理
clean_teacher_output <- function(text) {
  # 去掉 <think>...</think> 块
  text <- gsub("<think>[\\s\\S]*?</think>", "", text, perl = TRUE)
  text <- gsub("<think>.*", "", text)
  # 去掉多余的换行和空白
  text <- gsub("\n{3,}", "\n\n", text)
  text <- trimws(text)
  text
}

# ========================== 构建 prompt 格式 ==========================

# Qwen3.5 Base (非 Instruct) 对 "问题：...\n回答：" 格式响应最好
build_prompt <- function(instruction) {
  sprintf("问题：%s\n回答：", instruction)
}

# ========================== 批量生成 ==========================

# 检查已有输出，支持断点续传
if (file.exists(OUTPUT_PATH)) {
  existing <- stream_in(file(OUTPUT_PATH), verbose = FALSE)
  cat(sprintf("检测到已有输出 %d 条，支持断点续传。\n", nrow(existing)))
} else {
  # 初始化输出文件
  first <- data.frame(prompt  = character(0),
                      teacher_response = character(0),
                      stringsAsFactors = FALSE)
  stream_out(first, file(OUTPUT_PATH))
}

results <- list()
t_start <- Sys.time()

for (i in START_FROM:end_at) {
  prompt     <- raw_data$instruction[i]
  full_prompt <- build_prompt(prompt)

  cat(sprintf("[%d/%d] ", i, end_at))

  # 逐条生成（llama_generate_batch 在 CPU 上对大模型不太稳定）
  raw_output <- tryCatch(
    llama_generate(
      ctx,
      full_prompt,
      max_new_tokens = MAX_NEW_TOKENS,
      temp       = TEMPERATURE,
      top_k      = TOP_K,
      top_p      = TOP_P,
      seed       = SEED,
      repeat_penalty = 1.1
    ),
    error = function(e) {
      cat(sprintf("ERROR: %s\n", e$message))
      return("")
    }
  )

  cleaned <- clean_teacher_output(raw_output)

  # 如果模型溢出了（继续自言自语），只取第一段有意义的回答
  # 以常见对话分隔符截断
  cut_patterns <- c("\n问题：", "\n回答：", "\nQ：", "\nA：", "\n<|")
  for (pat in cut_patterns) {
    if (grepl(pat, cleaned, fixed = TRUE)) {
      cleaned <- sub(paste0(pat, ".*"), "", cleaned)
    }
  }

  cat(sprintf("%d chars\n", nchar(cleaned)))

  # 追加写入（原子化，保证不丢数据）
  entry <- data.frame(
    prompt          = prompt,
    teacher_response = cleaned,
    stringsAsFactors = FALSE
  )
  cat(paste0(toJSON(entry, auto_unbox = TRUE), "\n"),
      file = OUTPUT_PATH, append = TRUE)

  # 进度预估
  elapsed   <- difftime(Sys.time(), t_start, units = "secs")
  processed <- i - START_FROM + 1L
  eta_sec   <- (elapsed / processed) * (end_at - i)
  cat(sprintf("  [%.0fs elapsed, ETA %.0fs]\n", elapsed, eta_sec))
}

# ========================== 清理 ==========================

total_elapsed <- difftime(Sys.time(), t_start, units = "secs")
cat(sprintf("\n完成！总共生成 %d 条，耗时 %.0f 秒 (%.1f 条/分钟)\n",
            end_at - START_FROM + 1L, total_elapsed,
            (end_at - START_FROM + 1L) / as.numeric(total_elapsed) * 60))

llama_free_context(ctx)
llama_free_model(model)
cat(sprintf("输出文件: %s\n", OUTPUT_PATH))
