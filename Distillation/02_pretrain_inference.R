# =====================================================================
# 推理 (Generate) — Qwen Tokenizer 版
# =====================================================================

library(torch)
library(tok)
source("Distillation/LRP_model.R")

# Qwen Tokenizer
tokenizer <- tok::tokenizer$from_file("models/tokenizer.json")
VOCAB_SIZE <- tokenizer$get_vocab_size()
cat(sprintf("Qwen Tokenizer 词表大小: %d\n", VOCAB_SIZE))

# Qwen 特殊 Token
EOS_ID <- 151643L  # <|endoftext|>

# 模型超参 (与训练完全一致)
DIM      <- 320
N_LAYERS <- 8
N_HEADS  <- 8
SEQ_LEN  <- 512

# 实例化模型
raw_model <- RtomicLRP(
  vocab_size = VOCAB_SIZE,
  dim = DIM,
  n_layers = N_LAYERS,
  n_heads = N_HEADS,
  max_seq_len = SEQ_LEN
)

device <- torch_device("cpu")
raw_model <- raw_model$to(device = device)

generate_text <- function(model, tokenizer, prompt, max_new_tokens = 50,
                          temperature = 0.8, top_k = 10, rep_penalty = 1.2) {
  device <- model$tok_emb$weight$device

  # Qwen tokenizer$encode() 返回 0-based token IDs
  # 模型训练时使用 1-based IDs (dataloader 中做了 +1L)，推理时也必须对齐
  encode_result <- tokenizer$encode(prompt)
  raw_ids <- encode_result$ids

  # 1-based: 与训练数据对齐 (Qwen 0-based → R torch 1-based)
  current_ids <- raw_ids + 1L

  cat(sprintf("\n[输入 Prompt]: %s\n", prompt))
  cat("[模型生成中]: ")

  with_no_grad({
    for (i in 1:max_new_tokens) {
      input_seq <- tail(current_ids, SEQ_LEN)
      x_tensor <- torch_tensor(input_seq, dtype = torch_long(), device = device)$unsqueeze(1)
      seq_len <- x_tensor$size(2)

      h <- model$tok_emb(x_tensor)

      # 遍历 Transformer 层
      for (j in 1:length(model$layers)) {
        h <- model$layers[[j]](h)
      }

      h <- model$norm_f(h)

      # 严格对齐前向传播的残差逻辑 (h + predictor)
      semantic_delta <- model$predictor(h)
      pred <- h + semantic_delta

      # 提取最后一个 Token 的特征向量 (B, Dim)
      last_pred <- pred[, seq_len, ]

      # 与最新训练状态对齐，直接计算内积并引入 Temperature
      logits <- torch_matmul(last_pred, model$tok_emb$weight$t()) / temperature
      logits <- logits$squeeze(1)

      # current_ids 已是 1-based，可直接用于索引
      # 重复惩罚 (Repetition Penalty)
      unique_past_ids <- unique(current_ids)
      for (past_id in unique_past_ids) {
        if (past_id < 1 || past_id > VOCAB_SIZE) next
        logit_val <- as.numeric(logits[past_id])
        if (logit_val < 0) {
          logits[past_id] <- logit_val * rep_penalty
        } else {
          logits[past_id] <- logit_val / rep_penalty
        }
      }

      # Top-K 截断过滤长尾噪音
      if (top_k > 0) {
        topk_res <- torch_topk(logits, k = top_k)
        kth_value <- topk_res[[1]][top_k]
        logits <- torch_where(logits < kth_value, torch_tensor(-Inf, device = device), logits)
      }

      # 概率采样 (输出 1-based R torch 索引)
      probs <- nnf_softmax(logits, dim = -1)
      next_token_1based <- as.integer(torch_multinomial(probs, num_samples = 1))

      # 检测停止符 (EOS_ID 是 0-based Qwen ID，+1 转为 1-based)
      if (next_token_1based == (EOS_ID + 1L)) {
        cat(" [EOS]")
        break
      }

      # 解码：转回 0-based Qwen ID
      next_token_0based <- next_token_1based - 1L
      next_word <- tokenizer$decode(c(next_token_0based), skip_special_tokens = TRUE)
      cat(next_word)
      flush.console()

      current_ids <- c(current_ids, next_token_1based)
    }
  })
  cat("\n\n生成完毕。\n")
  invisible(current_ids)
}

# --- 配置检查点与提示词 ---
checkpoints_to_test <- c(
  "checkpoints/qwen_03.pt"
)

test_prompts <- c(
  "可以这样理解损失函数，它是",
  "数据科学可以帮助企业",
  "人工智能是一种",
  "谷歌是",
  "支持向量机是"
)

for (ckpt_path in checkpoints_to_test) {
  cat("\n")
  cat(sprintf("正在评估模型版本: %s\n", ckpt_path))

  if (!file.exists(ckpt_path)) {
    cat(sprintf("[跳过] 检查点不存在: %s\n", ckpt_path))
    next
  }

  checkpoint_data <- torch_load(ckpt_path)

  # 训练时用 torch_save(model$state_dict(), ...) 保存，直接加载 (无 $model 包装)
  raw_model$load_state_dict(checkpoint_data)
  raw_model$to(dtype = torch_float32())
  raw_model$eval()

  for (prompt in test_prompts) {
    generate_text(raw_model, tokenizer, prompt = prompt, max_new_tokens = 100,
                  temperature = 1.5, top_k = 50, rep_penalty = 1.15)
  }
}
