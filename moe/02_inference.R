# =====================================================================
# MoE 架构推理 (Generate) — KV Cache 两阶段自回归
# 适配 V3 Dense MoE（也兼容 V1/V2，模型接口一致）
# =====================================================================

source("config.R")
source("utils/BPETokenizer.R")
source("moe/moe_model.R")   # 切换模型版本只需改这一行

tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

# =====================================================================
# 实例化模型（必须与训练配置严格一致）
# =====================================================================
N_KV_HEADS <- 2
NUM_EXPERTS <- 4
TOP_K <- 2

moe_model <- RtomicCausalLM(
  vocab_size = VOCAB_SIZE,
  dim = DIM,
  n_layers = N_LAYERS,
  n_heads = N_HEADS,
  max_seq_len = SEQ_LEN,
  n_kv_heads = N_KV_HEADS,
  num_experts = NUM_EXPERTS,
  top_k = TOP_K
)

device <- torch_device(device)
moe_model <- moe_model$to(device = device)

# =====================================================================
# 采样函数: Top-K 截断 + 符号感知重复惩罚
# 优化: 1次 CPU 搬运 → 后续全 R 向量操作，零同步
# =====================================================================
sample_with_penalty <- function(logits,
                                current_ids,
                                top_k = 10,
                                rep_penalty = 1.15) {
  # 1次同步: GPU → CPU R 向量，之后全在 CPU 上操作
  lv <- as.array(logits$cpu())

  # 重复惩罚（纯 R 操作，零同步）
  for (id in unique(current_ids)) {
    val <- lv[id]
    lv[id] <- if (val > 0) val / rep_penalty else val * rep_penalty
  }

  # Top-K 截断（纯 R，零同步）
  if (length(lv) > top_k) {
    kth <- sort(lv, decreasing = TRUE)[top_k]
    lv[lv < kth] <- -Inf
  }

  # Softmax + 采样（纯 R，零同步）
  lv <- lv - max(lv)
  probs <- exp(lv) / sum(exp(lv))
  sample(seq_along(probs), size = 1, prob = probs)
}

# =====================================================================
# KV Cache 两阶段推理
# 阶段 1: Prefill — 一次性编码 Prompt，建立初始 KV Cache
# 阶段 2: Decode — 每步只输入 1 个 token，复用 Cache
# =====================================================================
generate_moe_text <- function(model,
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

    # --- Prefill: 一次性消化整个 Prompt ---
    x_tensor <- torch_tensor(current_ids, dtype = torch_long(), device = device)$unsqueeze(1)

    output <- model(list(x = x_tensor, y = NULL, loss_mask = NULL), use_cache = TRUE)
    past_kv <- output$past_key_values

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

    # --- Decode: 每步只输入最新 1 个 token，复用 KV Cache ---
    for (i in 2:max_new_tokens) {

      x_tensor <- torch_tensor(c(next_token_id), dtype = torch_long(), device = device)$unsqueeze(1)

      output <- model(list(x = x_tensor, y = NULL, loss_mask = NULL),
                      use_cache = TRUE, past_key_values = past_kv)
      past_kv <- output$past_key_values

      # 输入长度为 1，logits 在位置 1
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

# =====================================================================
# 加载 checkpoint 并运行
# =====================================================================
ckpt_path <- "checkpoints/moe_model_03.pt"
checkpoint_data <- torch_load(ckpt_path, device = device)
state <- if (!is.null(checkpoint_data$model)) checkpoint_data$model else checkpoint_data
moe_model$load_state_dict(state)
moe_model$eval()

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
  generate_moe_text(
    moe_model,
    tokenizer,
    prompt = prompt,
    max_new_tokens = 150,
    temperature = 0.3,
    rep_penalty = 1.2
  )
}
