# moe/03_moe_sft_eval.R
# =====================================================================
# Decode-Only MoE 架构 SFT 推理脚本
# =====================================================================

source("config.R")
source("utils/BPETokenizer.R")
source("moe/moe_model.R")

torch_set_num_threads(1L)

# 设备探测
device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
cat(sprintf("MoE SFT 推理引擎启动，当前使用设备: %s\n", device$type))

# =====================================================================
# 1. 加载 Tokenizer 与初始化 MoE 模型
# =====================================================================
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

N_KV_HEADS  <- 2 
NUM_EXPERTS <- 4
TOP_K       <- 1

model <- RtomicCausalLM(
  vocab_size  = VOCAB_SIZE, 
  dim         = DIM, 
  n_layers    = N_LAYERS, 
  n_heads     = N_HEADS, 
  max_seq_len = SEQ_LEN,
  n_kv_heads  = N_KV_HEADS,
  num_experts = NUM_EXPERTS,
  top_k       = TOP_K
)

TARGET_CKPT <- "checkpoints/moe_sft_epoch_05.pt"
if (file.exists(TARGET_CKPT)) {
  cat(sprintf("正在挂载 MoE SFT 权重: %s\n", TARGET_CKPT))
  ckpt <- torch_load(TARGET_CKPT)
  state_dict <- if ("model" %in% names(ckpt)) ckpt$model else ckpt
  model$load_state_dict(state_dict, strict = FALSE)
} else {
  stop("找不到指定的 Checkpoint 文件！请检查路径。")
}

model <- model$to(device = device)
model$eval() # 开启评估模式 (关闭 Router 的 aux_loss 计算)

# =====================================================================
# 2. 采样与生成流水线
# =====================================================================
generate_response <- function(model, tokenizer, prompt,
                              max_new_tokens = 256,
                              temperature = 0.3,
                              top_k = 5,
                              rep_penalty = 1.1) {
  
  eos_val <- if (!is.null(tokenizer$eos_idx)) tokenizer$eos_idx else 4L
  
  # 构建 Prompt (<BOS> + Prompt)
  p_ids <- tokenizer$encode_raw(prompt)[[1]]
  current_ids <- c(tokenizer$bos_idx, p_ids)
  
  cat(sprintf("\n[User]: %s\n[Assistant]: ", prompt))
  
  with_no_grad({
    for (step in 1:max_new_tokens) {
      # 截取 SEQ_LEN 避免序列超出长度
      input_seq <- tail(current_ids, SEQ_LEN)
      x_tensor <- torch_tensor(input_seq, dtype = torch_long(), device = device)$unsqueeze(1) # [1, S]
      
      # 评估模式下，RtomicCausalLM 返回 list(logits = logits)
      output <- model(list(x = x_tensor, y = NULL))
      logits <- output$logits
      
      seq_len <- x_tensor$size(2)
      # 提取最后一个 Token 预测的 Logits并应用 Temperature
      last_logits <- logits[1, seq_len, ] / temperature
      
      # 施加重复惩罚
      unique_past_ids <- unique(current_ids)
      for (past_id in unique_past_ids) {
        logit_val <- as.numeric(last_logits[past_id])
        if (logit_val < 0) {
          last_logits[past_id] <- logit_val * rep_penalty
        } else {
          last_logits[past_id] <- logit_val / rep_penalty
        }
      }
      
      # Top-K 截断
      if (top_k > 0) {
        topk_res <- torch_topk(last_logits, k = top_k)
        kth_value <- topk_res[[1]][top_k]
        last_logits <- torch_where(last_logits < kth_value, torch_tensor(-Inf, device = device), last_logits)
      }
      
      # 概率采样
      probs <- nnf_softmax(last_logits, dim = -1)
      next_token_id <- as.integer(torch_multinomial(probs, num_samples = 1))
      
      if (next_token_id == eos_val) {
        break
      }
      
      # 解码并流式打印
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
# 3. 测试运行
# =====================================================================
torch_manual_seed(42)

test_prompts <- c(
  "解释一下决策树原理。",
  "数据科学在企业有哪些直接的应用呢？",
  "简单介绍一下transformer的基本原理。",
  "人工智能能做什么？",
  "你是谁。"
)

for (p in test_prompts) {
  generate_response(
    model          = model, 
    tokenizer      = tokenizer, 
    prompt         = p,
    max_new_tokens = 128, 
    temperature    = 0.3, 
    top_k          = 5, 
    rep_penalty    = 1.1
  )
}
