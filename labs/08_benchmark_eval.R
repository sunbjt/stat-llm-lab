# ==========================================
# 通用 Log-Likelihood 批量自动化评测脚本
# 支持多模型切换：改 ACTIVE_MODEL 即可
# ==========================================
library(jsonlite)
source("config.R")
source("utils/BPETokenizer.R")
device <- torch_device("cpu")

tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

# ==========================================
# 0. 模型注册表 —— 添加新模型只需在这里加一条
# ==========================================
# 约定：
#   source    : 模型定义文件路径
#   construct : 返回模型实例的函数（闭包捕获 model 变量）
#   ckpt      : checkpoint 路径
#   load_fn   : 接收 (model, raw_ckpt) 执行 load_state_dict
#   output    : "pred" (latent 需投影) 或 "logits" (已投影到词表)
# ==========================================

MODEL_REGISTRY <- list(
  lrp = list(
    label  = "LRP (Latent Residual Predictor)",
    source = "latent_residual/LRP_model.R",
    construct = function() RtomicLRP(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN),
    ckpt   = "checkpoints/lrp_sft_epoch_05.pt",
    load_fn = function(model, raw) {
      sd <- if ("model" %in% names(raw)) raw$model else raw
      model$load_state_dict(sd, strict = FALSE)
    },
    output = "pred"
  ),

  jepa = list(
    label  = "JEPA VQ (World Model)",
    source = "world_model/jepa_model.R",
    construct = function() RtomicJEPA_VQ(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN, num_clusters = 4096),
    ckpt   = "checkpoints/jepa_sft_epoch_05.pt",
    load_fn = function(model, raw) {
      if ("model" %in% names(raw)) {
        clean <- list()
        for (nm in names(raw$model)) {
          clean[[sub("^model\\.", "", nm)]] <- raw$model[[nm]]
        }
        model$load_state_dict(clean, strict = FALSE)
      } else {
        model$load_state_dict(raw, strict = FALSE)
      }
    },
    output = "pred"
  ),

  causal = list(
    label  = "Causal LM",
    source = "causal_lm/CausalLM_model.R",
    construct = function() RtomicCausalLM(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN),
    ckpt   = "checkpoints/causal_sft_epoch_05.pt",
    load_fn = function(model, raw) {
      sd <- if ("model" %in% names(raw)) raw$model else raw
      model$load_state_dict(sd, strict = FALSE)
    },
    output = "logits"
  )
)

# ==========================================
# 1. 选择模型 —— 只改这一行！
# ==========================================
ACTIVE_MODEL <- "causal"

# ==========================================
# 2. 加载模型
# ==========================================
cfg <- MODEL_REGISTRY[[ACTIVE_MODEL]]
if (is.null(cfg)) stop(sprintf("未知模型: %s。可选: %s", ACTIVE_MODEL, paste(names(MODEL_REGISTRY), collapse = ", ")))

cat(sprintf("\n=== 模型: %s ===\n", cfg$label))
source(cfg$source)
model <- cfg$construct()

ckpt_path <- cfg$ckpt
if (!file.exists(ckpt_path)) stop(sprintf("Checkpoint 不存在: %s", ckpt_path))
cat(sprintf("加载权重: %s\n", ckpt_path))
raw <- torch_load(ckpt_path)
cfg$load_fn(model, raw)
rm(raw); gc()

model$eval()

# ==========================================
# 3. 核心评分函数（模型无关）
# ==========================================
evaluate_single_choice <- function(model, tokenizer, prompt, choice_text, output_type) {
  prompt_tokens <- c(tokenizer$bos_idx, tokenizer$encode_raw(prompt)[[1]])
  choice_tokens <- tokenizer$encode_raw(choice_text)[[1]]
  full_tokens <- c(prompt_tokens, choice_tokens)

  P <- length(prompt_tokens)
  N <- length(full_tokens)
  C <- N - P
  if (C <= 0) return(-Inf)

  with_no_grad({
    input_tensor <- torch_tensor(full_tokens, dtype = torch_long())$unsqueeze(1)
    out <- model(list(x = input_tensor, y = NULL))

    # 统一拿到 (N, V) logits
    if (output_type == "logits") {
      logits_full <- out$logits
    } else {
      pred_states <- out$pred
      logits_full <- torch_matmul(pred_states, model$tok_emb$weight$t())
    }

    V <- logits_full$size(-1)
    logits_2d <- logits_full$contiguous()$view(c(N, V))
    log_probs_2d <- nnf_log_softmax(logits_2d, dim = -1)

    shift_log_probs <- log_probs_2d$narrow(dim = 1, start = P, length = C)$contiguous()
    shift_log_probs <- shift_log_probs$view(c(C, V))

    shift_targets <- torch_tensor(choice_tokens, dtype = torch_long())$view(c(C, 1))
    target_log_probs <- torch_gather(shift_log_probs, dim = -1, index = shift_targets)

    total_score <- as.numeric(target_log_probs$sum())
    return(total_score / C)
  })
}

evaluate_choices <- function(model, tokenizer, prompt, choices, output_type) {
  scores <- numeric(length(choices))
  for (i in seq_along(choices)) {
    scores[i] <- evaluate_single_choice(model, tokenizer, prompt, choices[i], output_type)
  }
  return(scores)
}

# ==========================================
# 4. 批量评测
# ==========================================
run_benchmark <- function(model, tokenizer, dataset_path, output_type) {
  cat(sprintf("\n正在加载评测数据集: %s\n", dataset_path))
  dataset <- jsonlite::stream_in(file(dataset_path), verbose = FALSE)

  total_questions <- nrow(dataset)
  correct_count <- 0
  choice_letters <- c("A", "B", "C", "D")

  cat("==========================================\n")
  cat(sprintf("开始批量评测，共 %d 道题...\n", total_questions))
  cat("==========================================\n")

  for (i in 1:total_questions) {
    q_prompt <- dataset$question[i]

    if (is.matrix(dataset$choices)) {
      q_choices <- as.character(dataset$choices[i, ])
    } else {
      q_choices <- as.character(unlist(dataset$choices[[i]]))
    }

    if (length(q_choices) != 4) {
      stop(sprintf("第 %d 题选项数量异常！当前抓取到的选项数为: %d", i, length(q_choices)))
    }

    clean_choices <- gsub("^\\s*[A-D]\\.\\s*", "", q_choices)
    true_ans <- dataset$answer[i]

    scores <- evaluate_choices(model, tokenizer, q_prompt, clean_choices, output_type)
    best_idx <- which.max(scores)
    pred_ans <- choice_letters[best_idx]

    is_correct <- (pred_ans == true_ans)
    if (is_correct) correct_count <- correct_count + 1

    status <- ifelse(is_correct, "✅", "❌")
    cat(sprintf("[题 %02d/%02d] 真实: %s | 预测: %s | %s\n",
                i, total_questions, true_ans, pred_ans, status))
    cat(sprintf("[Debug] -> A: %s, B: %s, C: %s, D: %s\n",
                round(scores[1], 4), round(scores[2], 4), round(scores[3], 4), round(scores[4], 4)))
  }

  accuracy <- correct_count / total_questions
  cat("==========================================\n")
  cat(sprintf("评测完成！最终准确率 (Accuracy): %.2f%%\n", accuracy * 100))
  cat("==========================================\n")

  return(accuracy)
}

# ==========================================
# 5. 执行
# ==========================================
benchmark_file <- "labs/files/machine_learning.jsonl"
# benchmark_file <- "labs/files/chinese_history.jsonl"

final_accuracy <- run_benchmark(model, tokenizer, benchmark_file, cfg$output)
