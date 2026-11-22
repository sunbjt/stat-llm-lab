# causal_lm/04_causal_sft_train.R
# =====================================================================
# Decode-Only 架构 SFT 微调训练 (Native Torch Loop)
# =====================================================================

source("config.R")

# --- 环境专属超参 ---
BATCH_SIZE <- if (is_mac) 2 else 32

device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
cat(sprintf("当前运行设备: %s\n", device$type))

PRETRAIN_CKPT <- "checkpoints/causal_model_02.pt"

# =====================================================================
# 1. 加载 Tokenizer 与模型架构
# =====================================================================
source("utils/BPETokenizer.R")
source("causal_lm/CausalLM_model.R")
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

DEV_FILE <- "data/raw/mcq_sft_dev.jsonl"

# =====================================================================
# 四选一题目的 logit-based 评测：不做自由生成，只比较模型在答案位置上
# 对 A/B/C/D 四个 token 的 logit 谁最大，直接 argmax 判定，避免生成式
# 评测引入的格式/解析噪声。
# =====================================================================

# 拿到 A/B/C/D 四个字母各自对应的 token id（只取第一个子词，正常情况下应为单token）
get_letter_ids <- function(tokenizer) {
  sapply(c("A", "B", "C", "D"), function(ch) {
    ids <- tokenizer$encode_raw(ch)[[1]]
    ids[1]
  })
}

# 对单条 prompt 做一次前向，返回模型在 A/B/C/D 上的 logit 向量
predict_letter_logits <- function(model, tokenizer, prompt, letter_ids, device, max_len) {
  p_ids <- tokenizer$encode_raw(prompt)[[1]]
  p_ids <- c(tokenizer$bos_idx, p_ids)

  if (length(p_ids) > max_len) {
    p_ids <- p_ids[(length(p_ids) - max_len + 1):length(p_ids)]  # 太长就截前面，保留题干末尾
  }

  x <- torch_tensor(matrix(p_ids, nrow = 1), dtype = torch_long())$to(device = device)

  model$eval()
  with_no_grad({
    output <- model(list(x = x, y = NULL, loss_mask = NULL))
  })
  logits <- output$logits  # [1, seq_len, vocab]

  pos <- length(p_ids)  # 预测紧跟在 prompt 最后一个 token 之后的下一个 token（即"答案："后的第一个token）
  vocab_logits <- as.numeric(logits[1, pos, ])

  vocab_logits[letter_ids]  # 长度4，顺序 A/B/C/D
}

# 对一整个 jsonl (instruction/output 格式) 计算整体 acc + 按 category 的 acc
# instruction 里必须已经包含到"答案："为止的完整 prompt（跟训练格式一致）
evaluate_mcq_jsonl <- function(model, tokenizer, jsonl_file, device, max_len,
                                category_col = NULL, max_n = Inf) {
  df <- jsonlite::stream_in(file(jsonl_file), verbose = FALSE)
  if (nrow(df) == 0) return(list(acc = NA, n = 0, by_category = NULL))
  if (is.finite(max_n) && nrow(df) > max_n) {
    df <- df[sample(nrow(df), max_n), ]
  }

  letter_ids <- get_letter_ids(tokenizer)
  letters <- c("A", "B", "C", "D")

  correct <- logical(nrow(df))
  for (i in seq_len(nrow(df))) {
    true_letter <- substr(trimws(df$output[i]), 1, 1)  # 输出可能是"A"或"A\n解析：..."，只取第一个字符
    logit_vec <- predict_letter_logits(model, tokenizer, df$instruction[i], letter_ids, device, max_len)
    pred_letter <- letters[which.max(logit_vec)]
    correct[i] <- (pred_letter == true_letter)
  }

  result <- list(acc = mean(correct), n = nrow(df))

  if (!is.null(category_col) && category_col %in% names(df)) {
    by_cat <- tapply(correct, df[[category_col]], mean)
    n_cat  <- tapply(correct, df[[category_col]], length)
    result$by_category <- data.frame(
      category = names(by_cat),
      acc      = as.numeric(by_cat),
      n        = as.integer(n_cat),
      stringsAsFactors = FALSE
    )
    result$by_category <- result$by_category[order(result$by_category$acc), ]
  }

  result
}


# =====================================================================
# 2. SFT 数据集定义 (集成 Loss Mask 逻辑)
# =====================================================================
GenerativeSFTDataset <- dataset(
  name = "GenerativeSFTDataset",
  initialize = function(prompts, responses, tokenizer, max_len) {
    self$prompts <- prompts
    self$responses <- responses
    self$tokenizer <- tokenizer
    self$max_len <- max_len
  },
  
  .getitem = function(i) {
    p_ids <- self$tokenizer$encode_raw(self$prompts[i])[[1]]
    p_ids <- c(self$tokenizer$bos_idx, p_ids)
    
    r_ids <- self$tokenizer$encode_raw(self$responses[i])[[1]]
    eos_val <- if (!is.null(self$tokenizer$eos_idx)) self$tokenizer$eos_idx else 4L
    r_ids <- c(r_ids, eos_val)
    
    full_seq <- c(p_ids, r_ids)
    seq_len <- length(full_seq)
    
    x_ids <- full_seq[1:(seq_len - 1)]
    y_ids <- full_seq[2:seq_len]
    
    # 核心逻辑：Prompt 部分不计算 Loss (FALSE)，仅对 Response 计算 Loss (TRUE)
    mask <- c(rep(FALSE, length(p_ids) - 1), rep(TRUE, length(r_ids)))
    
    pad_len <- self$max_len - length(x_ids)
    if (pad_len > 0) {
      x_ids <- c(x_ids, rep(1L, pad_len)) # 1L 为 pad_idx
      y_ids <- c(y_ids, rep(1L, pad_len))
      mask <- c(mask, rep(FALSE, pad_len))
    } else {
      x_ids <- x_ids[1:self$max_len]
      y_ids <- y_ids[1:self$max_len]
      mask <- mask[1:self$max_len]
    }
    
    list(
      x = torch_tensor(x_ids, dtype = torch_long()),
      y = torch_tensor(y_ids, dtype = torch_long()),
      loss_mask = torch_tensor(mask, dtype = torch_bool())
    )
  },
  .length = function() length(self$prompts)
)

# =====================================================================
# 3. 实例化模型并热启动权重
# =====================================================================
model <- RtomicCausalLM(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN)

if (file.exists(PRETRAIN_CKPT)) {
  cat(sprintf("正在加载预训练底座权重: %s\n", PRETRAIN_CKPT))
  ckpt <- torch_load(PRETRAIN_CKPT)
  state_dict <- if (!is.null(ckpt$model)) ckpt$model else ckpt
  
  # 动态扩展位置编码长度（从预训练的 256 扩展到 SFT 的 512）
  old_pos_emb <- state_dict[["pos_emb.weight"]]
  if (!is.null(old_pos_emb)) {
    old_len <- old_pos_emb$size(1)
    if (old_len < SEQ_LEN) {
      cat(sprintf("检测到预训练位置编码长度 (%d) < 当前 SFT 长度 (%d)，正在动态扩展...\n", old_len, SEQ_LEN))
      new_pos_emb <- torch_empty(c(SEQ_LEN, DIM))
      nn_init_normal_(new_pos_emb, std = 0.02)
      new_pos_emb[1:old_len, ] <- old_pos_emb
      state_dict[["pos_emb.weight"]] <- new_pos_emb
    }
  }
  model$load_state_dict(state_dict, strict = FALSE)
} else {
  cat("未找到预训练权重，将从头开始随机初始化！\n")
}

model <- model$to(device = device)

# =====================================================================
# 4. 层冻结策略 (可选择性冻结前几层以保护通用表征)
# =====================================================================
for (p in model$parameters) { p$requires_grad_(TRUE) }

FREEZE_LAYERS <- 0 # 建议改为 0 进行全参数 SFT
cat(sprintf("\n[SFT 策略] 冻结前 %d 层 Transformer Blocks...\n", FREEZE_LAYERS))

if (FREEZE_LAYERS > 0) {
  for (i in 1:FREEZE_LAYERS) {
    lapply(model$layers[[i]]$parameters, function(p) p$requires_grad_(FALSE))
  }
}

# =====================================================================
# 5. 准备微调数据集与 Dataloader
# =====================================================================
raw_data <- jsonlite::stream_in(file("data/raw/mcq_sft_train.jsonl"), verbose = FALSE)

sft_ds <- GenerativeSFTDataset(raw_data$instruction, raw_data$output, tokenizer, max_len = SEQ_LEN)
sft_dl <- dataloader(sft_ds, batch_size = BATCH_SIZE, shuffle = TRUE)

# =====================================================================
# 6. 原生 Torch 训练循环
# =====================================================================
trainable_params <- Filter(function(p) p$requires_grad, model$parameters)
optimizer <- optim_adamw(trainable_params, lr = 3e-4, weight_decay = 0.01)

EPOCHS <- 5
scheduler <- lr_cosine_annealing(optimizer, T_max = EPOCHS)

cat("\n启动 Decode-Only 自回归 SFT 训练...\n")

for (epoch in 1:EPOCHS) {
  model$train()
  total_loss <- 0
  batch_idx <- 0
  current_lr <- optimizer$param_groups[[1]]$lr
  
  coro::loop(for (batch in sft_dl) {
    batch_idx <- batch_idx + 1
    optimizer$zero_grad()
    
    input_data <- list(
      x = batch$x$to(device = device),
      y = batch$y$to(device = device),
      loss_mask = batch$loss_mask$to(device = device)
    )
    
    output <- model(input_data)
    loss <- output$loss
    
    loss$backward()
    nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
    optimizer$step()
    
    total_loss <- total_loss + loss$item()
    
    if (batch_idx %% 10 == 0 || batch_idx == 1) {
      cat(sprintf("Epoch [%d/%d], Step [%d], LR: %.6f, Loss: %.4f\n",
                  epoch, EPOCHS, batch_idx, current_lr, loss$item()))
    }
  })
  
  avg_loss <- total_loss / batch_idx
  cat(sprintf("Epoch %d 结束, 平均 交叉熵Loss: %.4f\n", epoch, avg_loss))

  # ---- dev 集 acc 评测（logit-based，不做自由生成） ----
  if (file.exists(DEV_FILE)) {
    dev_res <- evaluate_mcq_jsonl(model, tokenizer, DEV_FILE, device,
                                   max_len = SEQ_LEN, category_col = "category")
    cat(sprintf("Epoch %d Dev Acc: %.2f%% (n=%d)\n", epoch, dev_res$acc * 100, dev_res$n))
    if (!is.null(dev_res$by_category)) {
      cat("  最差的5个类别:\n")
      worst <- head(dev_res$by_category, 5)
      for (r in seq_len(nrow(worst))) {
        cat(sprintf("    %-20s acc=%.2f%% (n=%d)\n", worst$category[r], worst$acc[r] * 100, worst$n[r]))
      }
    }
  } else {
    cat(sprintf("[提示] 未找到 dev 文件 %s，跳过 dev 评测\n", DEV_FILE))
  }

  scheduler$step()

  save_path <- sprintf("checkpoints/mcq_sft_epoch_%02d.pt", epoch)
  torch_save(model$state_dict(), save_path)
}
