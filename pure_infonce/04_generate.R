Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")

source("config.R")
source("utils/BPETokenizer.R")
source("latent_contrastive/contrastive_model.R")

device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
cat(sprintf("当前推断运行设备: %s\n", device$type))

# ---------------------------------------------------------------------
# 1. 初始化 Tokenizer & 模型
# ---------------------------------------------------------------------
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

model <- TokenLatentModel(
  vocab_size  = VOCAB_SIZE,
  dim         = DIM,
  n_layers    = N_LAYERS,
  n_heads     = N_HEADS,
  max_seq_len = SEQ_LEN
)

SFT_CKPT <- "checkpoints/infonce_sft_05.pt"

if (file.exists(SFT_CKPT)) {
  cat(sprintf("正在加载 SFT 微调权重: %s\n", SFT_CKPT))
  checkpoint_data <- torch_load(SFT_CKPT, device = "cpu")
  state_dict <- if (!is.null(checkpoint_data$model)) checkpoint_data$model else checkpoint_data
  model$load_state_dict(state_dict, strict = FALSE)
  cat("模型权重加载成功！\n")
} else {
  stop(sprintf("【错误】未找到微调权重文件: %s", SFT_CKPT))
}

model <- model$to(device = device, dtype = torch_float32())
model$eval()

# ---------------------------------------------------------------------
# 2. Pure InfoNCE 纯潜语义生成函数
# ---------------------------------------------------------------------
generate_response <- function(
  model, 
  tokenizer, 
  prompt, 
  max_new_tokens = 120,
  temperature = 0.3, 
  repetition_penalty = 1.2,
  top_k = 5
) {
  raw_ids <- tokenizer$encode_raw(prompt)[[1]]
  current_ids <- c(tokenizer$bos_idx, raw_ids)

  cat(sprintf("\n[Input]: %s\n", prompt))
  cat("[Generate]: ")

  with_no_grad({
    for (step in 1:max_new_tokens) {
      # 构造 [1, S] 输入张量
      input_seq <- tail(current_ids, SEQ_LEN)
      x_tensor <- torch_tensor(matrix(input_seq, nrow = 1), dtype = torch_long(), device = device)
      S <- x_tensor$size(2)

      # 1. Encoder 提取特征 + Predictor 非线性投影
      h <- model$encode(x_tensor)            # [1, S, D]
      pred <- model$predictor(h)            # [1, S, D]

      # 取最后一个位置的 latent 向量 [1, D]
      pred_last <- pred[, S, , drop = FALSE]$squeeze(1) # [1, D]

      # 2. 对齐训练时的 Pure InfoNCE 距离计算：L2 归一化后计算全词表相似度
      pred_norm <- nnf_normalize(pred_last, p = 2, dim = -1)
      emb_norm <- nnf_normalize(model$tok_emb$weight, p = 2, dim = -1)

      # [1, D] x [D, VOCAB_SIZE] -> [1, VOCAB_SIZE] -> squeeze 为 1D Tensor
      cos_sim <- torch_matmul(pred_norm, emb_norm$transpose(1, 2))$squeeze(1)
      logits <- cos_sim / model$temperature

      # 3. Repetition Penalty（规避 C++ 原位重叠赋值）
      if (repetition_penalty > 1.0) {
        uniq_ids <- unique(current_ids)
        idx_tensor <- torch_tensor(uniq_ids, dtype = torch_long(), device = device)
        penalized <- logits[idx_tensor]
        mask_pos <- penalized > 0
        penalized[mask_pos] <- penalized[mask_pos] / repetition_penalty
        penalized[!mask_pos] <- penalized[!mask_pos] * repetition_penalty
        logits[idx_tensor] <- penalized
      }

      # 4. Temperature Scaling
      if (temperature > 1e-3) {
        logits <- logits / temperature
      }

      # 5. Top-K 过滤
      if (top_k > 0 && top_k < logits$size(1)) {
        topk <- torch_topk(logits, k = top_k)
        min_val <- topk[[1]][top_k]$item()
        logits[logits < min_val] <- -Inf
      }

      # 6. Sampling / Greedy Decoding
      if (temperature <= 1e-3) {
        next_token_id <- as.integer(torch_argmax(logits, dim = -1))
      } else {
        probs <- nnf_softmax(logits, dim = -1)
        next_token_id <- as.integer(torch_multinomial(probs, num_samples = 1))
      }

      # 7. EOS 判定与实时打印
      if (!is.null(tokenizer$eos_idx) && next_token_id == tokenizer$eos_idx) break

      next_word <- tokenizer$decode(next_token_id, clean = FALSE)
      if (next_word == "<EOS>") break

      cat(next_word)
      flush.console()

      current_ids <- c(current_ids, next_token_id)
    }
  })
  cat("\n\nDone.\n")
  invisible(current_ids)
}

# ---------------------------------------------------------------------
# 3. 执行推断
# ---------------------------------------------------------------------
test_prompts <- c(
  "你认为 AI 未来对人类有哪些好处？",
  "应该怎样学习编程技术？",
  "简述一下深度学习。",
  "介绍一下北京。"
)

for (prompt in test_prompts) {
  generate_response(
    model, tokenizer, prompt = prompt,
    temperature = 0.3, repetition_penalty = 1.2, top_k = 5, max_new_tokens = 120
  )
}
