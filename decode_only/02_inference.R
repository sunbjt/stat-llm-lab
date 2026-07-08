# =====================================================================
# Decode-Only 架构推理 (Generate) 
# 现代化升级: 完美集成 GQA 与 KV Cache，实现极速单步自回归
# =====================================================================

source("config.R")
source("utils/BPETokenizer.R")
source("decode_only/decode_only_model.R")

tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

# =====================================================================
# 实例化模型 (注意激活 GQA 配置)
# =====================================================================
# 必须与 01_pretrain.R 训练时的配置严格保持一致！
N_KV_HEADS <- 2 
causal_model <- RtomicCausalLM(
  vocab_size = VOCAB_SIZE,
  dim = DIM,
  n_layers = N_LAYERS,
  n_heads = N_HEADS,
  max_seq_len = SEQ_LEN,
  n_kv_heads = N_KV_HEADS
)

device <- torch_device("cpu") # 如果在服务器上推理可改为 "cuda"
causal_model <- causal_model$to(device = device)

# 带有温度、Top-K 和重复惩罚的采样函数
sample_with_penalty <- function(logits,
                                current_ids,
                                top_k = 10,
                                rep_penalty = 1.15) {
  for (id in unique(current_ids)) {
    val <- logits[id]$item()
    logits[id] <- if (val > 0)
      val / rep_penalty
    else
      val * rep_penalty
  }
  
  if (length(logits) > top_k) {
    kth <- torch_topk(logits, k = top_k)[[1]][top_k]$item()
    logits[logits < kth] <- -Inf
  }
  
  probs <- nnf_softmax(logits, dim = -1)
  as.integer(torch_multinomial(probs, num_samples = 1))
}

# =====================================================================
# KV Cache 极速推理循环
# =====================================================================
generate_causal_text <- function(model,
                                 tokenizer,
                                 prompt,
                                 max_new_tokens = 50,
                                 temperature = 0.3,
                                 rep_penalty = 1.15,
                                 top_k = 10) {
  raw_ids <- tokenizer$encode_raw(prompt)[[1]]
  current_ids <- c(tokenizer$bos_idx, raw_ids)
  eos_val <- if (!is.null(tokenizer$eos_idx)) tokenizer$eos_idx else 4L
  
  cat(sprintf("\n[输入 Prompt]: %s\n[模型生成]: ", prompt))
  
  with_no_grad({
    
    # ---------------------------------------------------------
    # 阶段 1: Prefill (预填充) - 一次性消化 Prompt，建立初始缓存
    # ---------------------------------------------------------
    x_tensor <- torch_tensor(current_ids, dtype = torch_long(), device = device)$unsqueeze(1) # shape: [1, seq_len]
    
    # 首次调用，激活 use_cache，无需提供 past_key_values
    output <- model(list(x = x_tensor, y = NULL, loss_mask = NULL), use_cache = TRUE)
    
    # 拿到热乎的初始 Cache！
    past_kv <- output$past_key_values
    
    # 提取序列最后一个 token 的 logits 预测下一个词
    seq_length <- output$logits$size(2)
    logits_vec <- output$logits[1, seq_length, ] / temperature
    
    next_token_id <- sample_with_penalty(logits_vec, current_ids, top_k = top_k, rep_penalty = rep_penalty)
    
    if (next_token_id == eos_val) {
      cat(" [EOS]\n")
      return(invisible(current_ids))
    }
    
    next_word <- tokenizer$decode(next_token_id)
    if (next_word != "<EOS>") cat(next_word)
    flush.console()
    
    current_ids <- c(current_ids, next_token_id)
    
    # ---------------------------------------------------------
    # 阶段 2: Decode (解码) - 每次只输入 1 个 Token，复用 Cache
    # ---------------------------------------------------------
    # 因为我们在阶段 1 已经生成了第 1 个词，所以循环从 2 开始
    for (i in 2:max_new_tokens) {
      
      # 魔法在这里：x_tensor 永远只包含刚刚生成的这 1 个 Token！(shape: [1, 1])
      x_tensor <- torch_tensor(c(next_token_id), dtype = torch_long(), device = device)$unsqueeze(1)
      
      # 传入历史 Cache
      output <- model(list(x = x_tensor, y = NULL, loss_mask = NULL), 
                      use_cache = TRUE, past_key_values = past_kv)
      
      # 更新 Cache 给下一步用
      past_kv <- output$past_key_values
      
      # 因为输入长度只有 1，所以 logits 位于第二维的第 1 个位置
      logits_vec <- output$logits[1, 1, ] / temperature
      
      next_token_id <- sample_with_penalty(logits_vec, current_ids,
                                           top_k = top_k, rep_penalty = rep_penalty)
      
      if (next_token_id == eos_val) {
        cat(" [EOS]")
        break
      }
      
      next_word <- tokenizer$decode(next_token_id)
      if (next_word == "<EOS>") break
      
      cat(next_word)
      flush.console()
      
      current_ids <- c(current_ids, next_token_id)
    }
  })
  cat("\n")
  invisible(current_ids)
}

# --- 运行测试 ---
ckpt_path <- "checkpoints/decode_model_03.pt"
checkpoint_data <- torch_load(ckpt_path)
state <- if (!is.null(checkpoint_data$model)) checkpoint_data$model else checkpoint_data
causal_model$load_state_dict(state)
causal_model$eval()

test_prompts <- c(
  "我非常喜欢读西游记，最喜欢的人物是孙悟空",
  "旷野之息是任天堂的一款游戏",
  "可以这样理解损失函数，它是",
  "数据科学可以帮助企业",
  "人工智能是一种",
  "在家庭教育中，父母是",
  "支持向量机算法的原理可以这样理解"
)

for (prompt in test_prompts) {
  generate_causal_text(
    causal_model,
    tokenizer,
    prompt = prompt,
    max_new_tokens = 150,
    temperature = 0.3,
    rep_penalty = 1.2
  )
}
