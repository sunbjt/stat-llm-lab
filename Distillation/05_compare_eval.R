# Distillation/03_compare_eval.R
# =====================================================================
# 三方对比评估：Teacher (Qwen3.5-2B) vs Baseline SFT vs Distilled Student
# =====================================================================
library(llamaR)
library(torch)
library(jsonlite)

source("config.R")
source("utils/BPETokenizer.R")
source("causal_lm/CausalLM_model.R")

device <- torch_device("cpu")

# ========================== 配置 ==========================

TEACHER_MODEL   <- "models/Qwen3.5-2B.Q4_K_M.gguf"
BASELINE_CKPT   <- "checkpoints/causal_sft_epoch_05.pt"
DISTILL_CKPT    <- "checkpoints/distill_epoch_05.pt"

# 测试 prompts（从训练集外取，这里用有代表性的样本）
TEST_PROMPTS <- c(
  "你是谁？",
  "你可以做什么？",
  "解释一下什么是机器学习。",
  "写一段关于环保的重要性的短文。",
  "为什么天空是蓝色的？",
  "推荐几本好书。",
  "如何提高学习效率？",
  "人工智能会取代人类工作吗？",
  "用简单的语言解释量子力学。",
  "健康饮食有哪些要点？"
)

MAX_NEW_TOKENS <- 200L
TEMPERATURE    <- 0.7

# ========================== 加载模型 ==========================

cat("========== 加载模型 ==========\n")

# --- 教师模型 ---
cat("1/3 教师模型: Qwen3.5-2B...\n")
t_model <- llama_load_model(TEACHER_MODEL, n_gpu_layers = 0L)
t_ctx   <- llama_new_context(t_model, n_ctx = 2048L, n_threads = 4L)

# --- Baseline 学生模型 ---
cat("2/3 Baseline SFT...\n")
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)
b_model <- RtomicCausalLM(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN)
b_model$load_state_dict(torch_load(BASELINE_CKPT), strict = FALSE)
b_model <- b_model$to(device = device)
b_model$eval()

# --- 蒸馏学生模型 ---
cat("3/3 Distilled Student...\n")
d_model <- RtomicCausalLM(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN)
d_model$load_state_dict(torch_load(DISTILL_CKPT), strict = FALSE)
d_model <- d_model$to(device = device)
d_model$eval()

cat("所有模型就绪！\n\n")

# ========================== 学生模型推理 ==========================

student_generate <- function(model, prompt, max_tokens = 200L) {
  tokens <- tokenizer$encode_raw(prompt)[[1]]
  tokens <- c(tokenizer$bos_idx, tokens)

  for (step in 1:max_tokens) {
    seq_len <- length(tokens)
    if (seq_len > SEQ_LEN) {
      ctx_tokens <- tokens[(seq_len - SEQ_LEN + 1):seq_len]
    } else {
      ctx_tokens <- tokens
    }

    x_tensor <- torch_tensor(matrix(ctx_tokens, nrow = 1),
                             dtype = torch_long(), device = device)
    input_data <- list(x = x_tensor, y = NULL)

    with_no_grad({
      out    <- model(input_data)
      logits <- out$logits[1, length(ctx_tokens), ] / TEMPERATURE
      probs  <- nnf_softmax(logits, dim = -1)
      next_token <- as.integer(torch_multinomial(probs, num_samples = 1)$item())
    })

    if (next_token == tokenizer$eos_idx || next_token == 4L) break

    tokens <- c(tokens, next_token)
  }

  # 只返回生成部分（去掉 prompt）
  response_tokens <- tokens[(nchar(prompt) + 1):length(tokens)]
  if (length(response_tokens) == 0) return("")
  tokenizer$decode(list(response_tokens))
}

# ========================== 逐条对比 ==========================

results <- list()

for (i in seq_along(TEST_PROMPTS)) {
  prompt <- TEST_PROMPTS[i]
  cat(sprintf("========== [%d/%d] ==========\n", i, length(TEST_PROMPTS)))
  cat(sprintf("Prompt: %s\n\n", prompt))

  # 教师
  cat("--- Teacher (Qwen3.5-2B) ---\n")
  t_resp <- tryCatch(
    llama_generate(t_ctx, sprintf("问题：%s\n回答：", prompt),
                   max_new_tokens = MAX_NEW_TOKENS, temp = TEMPERATURE,
                   top_p = 0.95),
    error = function(e) paste("ERROR:", e$message)
  )
  cat(trimws(t_resp), "\n\n")

  # Baseline
  cat("--- Baseline SFT ---\n")
  b_resp <- tryCatch(student_generate(b_model, prompt), error = function(e) paste("ERROR:", e$message))
  cat(trimws(b_resp), "\n\n")

  # Distilled
  cat("--- Distilled Student ---\n")
  d_resp <- tryCatch(student_generate(d_model, prompt), error = function(e) paste("ERROR:", e$message))
  cat(trimws(d_resp), "\n\n")

  results[[i]] <- list(
    prompt   = prompt,
    teacher  = trimws(t_resp),
    baseline = trimws(b_resp),
    distill  = trimws(d_resp)
  )
}

# ========================== 保存结果 ==========================

cat("========== 汇总 ==========\n")
for (i in seq_along(results)) {
  r <- results[[i]]
  cat(sprintf("\n[%d] %s\n", i, r$prompt))
  cat(sprintf("  Teacher:   %s\n", substr(r$teacher,  1, 80)))
  cat(sprintf("  Baseline:  %s\n", substr(r$baseline, 1, 80)))
  cat(sprintf("  Distill:   %s\n", substr(r$distill,  1, 80)))
}

# 保存完整对比结果
output_file <- "Distillation/compare_results.json"
write_json(results, output_file, auto_unbox = TRUE, pretty = TRUE)
cat(sprintf("\n完整结果已保存: %s\n", output_file))

# ========================== 清理 ==========================

llama_free_context(t_ctx)
llama_free_model(t_model)
cat("Done!\n")
