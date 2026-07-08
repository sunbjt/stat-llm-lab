# causal_lm/05_causal_sft_inference.R
# =====================================================================
# Decode-Only 架构 SFT 推理 (Generate)
# =====================================================================

source("config.R")
torch_set_num_threads(1L)

# 动态选择设备：优先使用 GPU
device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu") 
cat(sprintf("自回归引擎启动，当前推理设备: %s\n", device$type))

TARGET_CKPT <- "checkpoints/causal_sft_epoch_05.pt"

# =====================================================================
# 1. 加载组件与基础模型
# =====================================================================
source("utils/BPETokenizer.R")
source("causal_lm/CausalLM_model.R")

tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

model <- RtomicCausalLM(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN)

if (file.exists(TARGET_CKPT)) {
  cat(sprintf("正在挂载 SFT 目标权重: %s\n", TARGET_CKPT))
  model$load_state_dict(torch_load(TARGET_CKPT))
} else {
  stop("找不到指定的微调 Checkpoint 文件，请确认路径！")
}

model <- model$to(device = device)
model$eval()

# =====================================================================
# 2. 文本采样与自回归生成核心工具
# =====================================================================
sample_with_penalty <- function(logits, current_ids, top_k = 40, rep_penalty = 1.10) {
  # 改进：将惩罚窗口拉长到 256，防止长周期死循环
  window_size <- 256
  recent_ids <- tail(current_ids, window_size)
  
  for (id in unique(recent_ids)) {
    val <- logits[id]$item()
    # 注意：只对非功能性 Token 惩罚，如果你知道 <EOS> 的 ID，可以跳过它
    logits[id] <- if (val > 0) val / rep_penalty else val * rep_penalty
  }
  
  # Top-K 截断 (放宽到 40，避免词穷)
  if (length(logits) > top_k) {
    kth <- torch_topk(logits, k = top_k)[[1]][top_k]$item()
    logits[logits < kth] <- -Inf
  }
  
  probs <- nnf_softmax(logits, dim = -1)
  as.integer(torch_multinomial(probs, num_samples = 1))
}

generate_response <- function(model, tokenizer, prompt,
                              max_new_tokens = 300,
                              temperature = 0.6,
                              top_k = 30,
                              rep_penalty = 1.05) {
  eos_val <- if (!is.null(tokenizer$eos_idx)) tokenizer$eos_idx else 4L

  p_ids <- tokenizer$encode_raw(prompt)[[1]]
  current_ids <- c(tokenizer$bos_idx, p_ids)

  cat(sprintf("\n[User]: %s\n[Assistant]: ", prompt))

  with_no_grad({
    for (step in 1:max_new_tokens) {
      # 维持滑动窗口，防止超出模型的 max_seq_len
      input_seq <- tail(current_ids, SEQ_LEN - 1)
      x_tensor <- torch_tensor(input_seq, dtype = torch_long(), device = device)$unsqueeze(1)

      output <- model(list(x = x_tensor, y = x_tensor, loss_mask = NULL))

      # 纯因果 LM 直接获取最后一层输出的词表预测对数几率 (Logits)
      seq_length <- output$logits$size(2)
      logits_vec <- output$logits[1, seq_length, ]
      
      # 应用温度缩放
      logits_vec <- logits_vec / temperature

      # 执行带局部滑动窗口惩罚的采样
      next_token_id <- sample_with_penalty(logits_vec, current_ids, top_k, rep_penalty)

      if (next_token_id == eos_val) {
        cat(" [检测到 <EOS>]")
        break
      }

      new_text <- tokenizer$decode(next_token_id)
      if (new_text == "<EOS>") break

      # 流式打印到控制台
      cat(new_text)
      flush.console()

      current_ids <- c(current_ids, next_token_id)
    }
  })

  cat("\n--------------------------------------------------\n")
  invisible(current_ids)
}

# =====================================================================
# 3. 运行评测
# =====================================================================
torch_manual_seed(42)

test_prompts <- c(
  "请讲一下深度学习的概念？",
  "数据科学在企业有哪些直接的应用呢？",
  "谷歌为什么能够成为伟大的公司？",
  "你可以做什么？"
)

for (p in test_prompts) {
  # 温度调到 0.6 或 0.7 让语言更自然，Top K 设为 40 避免死局，惩罚固定在 1.10
  generate_response(model, tokenizer, prompt = p,
                    max_new_tokens = 200, 
                    temperature = 0.4, 
                    top_k = 10, 
                    rep_penalty = 1.15)
}
