# =====================================================================
# SFT 推理 (适配最新 RoPE 架构)
# =====================================================================

source("config.R")
torch_set_num_threads(1L)

# 自动降级探测：优先 CUDA，其次 Mac MPS，最后 CPU
device <- torch_device("cpu")
cat(sprintf("SFT 推理引擎启动，当前使用加速设备: %s\n", device$type))


source("latent_residual/LRP_model.R")
source("utils/BPETokenizer.R")
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

# =====================================================================
# 1. 极简模型加载 (直接使用原版架构，无需 Hybrid 包装)
# =====================================================================
model <- RtomicLRP(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN)

TARGET_CKPT <- "checkpoints/lrp_sft_epoch_05.pt"
if (file.exists(TARGET_CKPT)) {
  cat(sprintf("正在挂载 SFT 权重: %s\n", TARGET_CKPT))
  ckpt <- torch_load(TARGET_CKPT)
  state_dict <- if ("model" %in% names(ckpt)) ckpt$model else ckpt
  model$load_state_dict(state_dict, strict = FALSE)
} else {
  stop("找不到指定的 Checkpoint 文件！请检查路径。")
}

model <- model$to(device = device)
model$eval() # 开启推理模式，关闭 Dropout (如果有) 并激活 VQ (虽然这里不用 VQ 算 Loss)

# =====================================================================
# 2. 健壮的采样与生成流水线
# =====================================================================
generate_response <- function(model, tokenizer, prompt,
                              max_new_tokens = 500,
                              temperature = 0.8,
                              top_k = 10,
                              rep_penalty = 1.1) { # SFT 阶段惩罚系数稍微降低，避免破坏指令逻辑
  
  eos_val <- if (!is.null(tokenizer$eos_idx)) tokenizer$eos_idx else 2L
  
  # 构建 Prompt (与 SFT Dataset 对齐: <BOS> + Prompt)
  p_ids <- tokenizer$encode_raw(prompt)[[1]]
  current_ids <- c(tokenizer$bos_idx, p_ids)
  
  cat(sprintf("\n[User]: %s\n[Assistant]: ", prompt))
  
  with_no_grad({
    for (step in 1:max_new_tokens) {
      # 截取最后的 SEQ_LEN 长度防止 OOM
      input_seq <- tail(current_ids, SEQ_LEN)
      x_tensor <- torch_tensor(input_seq, dtype = torch_long(), device = device)$unsqueeze(1)
      
      # 喂入 y=NULL，获取前向传播的潜空间特征
      output <- model(list(x = x_tensor, y = NULL))
      
      seq_length <- x_tensor$size(2)
      # 提取最后一个 Token 的特征向量 pred (B, Dim)
      last_pred <- output$pred[1, seq_length, ]
      
      # 点积投影回词表，并应用 Temperature
      cb <- model$tok_emb$weight
      logits <- torch_matmul(last_pred$unsqueeze(1), cb$t())$squeeze(1)
      logits <- logits / temperature
      
      # 施加重复惩罚 (防弹版赋值法)
      unique_past_ids <- unique(current_ids)
      for (past_id in unique_past_ids) {
        logit_val <- as.numeric(logits[past_id])
        if (logit_val < 0) {
          logits[past_id] <- logit_val * rep_penalty
        } else {
          logits[past_id] <- logit_val / rep_penalty
        }
      }
      
      # Top-K 截断
      if (top_k > 0) {
        topk_res <- torch_topk(logits, k = top_k)
        kth_value <- topk_res[[1]][top_k]
        # 使用 torch_where 防护，比 R 原生索引更安全
        logits <- torch_where(logits < kth_value, torch_tensor(-Inf, device = device), logits)
      }
      
      # 概率采样
      probs <- nnf_softmax(logits, dim = -1)
      next_token_id <- as.integer(torch_multinomial(probs, num_samples = 1))
      
      if (!is.null(tokenizer$eos_idx) && next_token_id == tokenizer$eos_idx) {
        break
      }
      
      # 1. 逐字打印时，关闭 clean 避免空格被吃掉或触发正则
      next_word_raw <- tokenizer$decode(next_token_id, clean = FALSE)
      if (next_word_raw == "<EOS>") break
      
      cat(next_word_raw)
      flush.console()
      
      current_ids <- c(current_ids, next_token_id)
    }
  })
  
  cat("[EOS]。\n")
  invisible(current_ids)
}

# =====================================================================
# 3. 互动测试
# =====================================================================
torch_manual_seed(42) # 锁死随机种子，方便观察参数调整带来的影响

test_prompts <- c(
  "解释一下决策树原理。",
  "数据科学在企业有哪些直接的应用呢？",
  "简单介绍一下transformer的基本原理。",
  "人工智能能做什么？",
  "你是谁。"
)


# SFT 模型的 Temperature 建议调低 (0.2~0.4)，让回答更确定、更符合指令范式
for (p in test_prompts) {
  generate_response(model, tokenizer, prompt = p,
                    max_new_tokens = 128, 
                    temperature = 0.3, 
                    top_k = 5, 
                    rep_penalty = 1.2)
}
