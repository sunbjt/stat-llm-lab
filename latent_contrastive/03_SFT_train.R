# =====================================================================
# SFT 微调训练 (TokenLatentModel)
# =====================================================================

source("config.R")

# --- 环境专属超参 ---
BATCH_SIZE <- if (is_mac) 4 else 32

device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
cat(sprintf("当前运行设备: %s\n", device$type))

# =====================================================================
# 1. 加载 Tokenizer
# =====================================================================
source("utils/BPETokenizer.R")
source("latent_contrastive/contrastive_model.R")
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

model <- TokenLatentModel(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN)

# =====================================================================
# 2. SFT 数据集定义
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
    eos_val <- if (!is.null(self$tokenizer$eos_idx)) self$tokenizer$eos_idx else 2L
    r_ids <- c(r_ids, eos_val)

    full_seq <- c(p_ids, r_ids)
    seq_len <- length(full_seq)

    x_ids <- full_seq[1:(seq_len - 1)]
    y_ids <- full_seq[2:seq_len]

    mask <- c(rep(FALSE, length(p_ids) - 1), rep(TRUE, length(r_ids)))

    pad_len <- self$max_len - length(x_ids)
    if (pad_len > 0) {
      x_ids <- c(x_ids, rep(1L, pad_len))
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
# 3. 加载预训练权重
# =====================================================================
PRETRAIN_CKPT <- "checkpoints/contrastive_02.pt"
if (file.exists(PRETRAIN_CKPT)) {
  cat(sprintf("加载预训练底座权重: %s\n", PRETRAIN_CKPT))
  ckpt <- torch_load(PRETRAIN_CKPT)
  state_dict <- if (!is.null(ckpt$model)) ckpt$model else ckpt

  # pos_emb 动态扩展（若 SFT seq_len > 预训练 seq_len）
  old_pos <- state_dict[["pos_emb.weight"]]
  if (!is.null(old_pos)) {
    old_len <- old_pos$size(1)
    if (old_len < SEQ_LEN) {
      cat(sprintf("pos_emb 扩展: %d → %d\n", old_len, SEQ_LEN))
      new_pos <- torch_empty(c(SEQ_LEN, DIM))
      nn_init_normal_(new_pos, std = 0.02)
      new_pos[1:old_len, ] <- old_pos
      state_dict[["pos_emb.weight"]] <- new_pos
    }
  }

  model$load_state_dict(state_dict, strict = FALSE)
} else {
  cat("未找到预训练权重，将从头开始初始化！\n")
}

model <- model$to(device = device)

# =====================================================================
# 4. 层冻结策略 (全参微调)
# =====================================================================
for (p in model$parameters) { p$requires_grad_(TRUE) }
cat("全参微调模式\n")

# =====================================================================
# 5. 准备数据与 Dataloader
# =====================================================================
raw_data <- jsonlite::stream_in(file("data/raw/qa_no_think.jsonl"))
sft_ds <- GenerativeSFTDataset(raw_data$instruction, raw_data$output, tokenizer, max_len = SEQ_LEN)
sft_dl <- dataloader(sft_ds, batch_size = BATCH_SIZE, shuffle = TRUE)

# =====================================================================
# 6. 训练循环
# =====================================================================
trainable_params <- Filter(function(p) p$requires_grad, model$parameters)
optimizer <- optim_adamw(trainable_params, lr = 3e-4, weight_decay = 0.01)

EPOCHS <- 5
scheduler <- lr_cosine_annealing(optimizer, T_max = EPOCHS)

for (epoch in 1:EPOCHS) {
  model$train()
  total_loss <- 0
  batch_idx <- 0
  current_lr <- optimizer$param_groups[[1]]$lr

  coro::loop(for (batch in sft_dl) {
    batch_idx <- batch_idx + 1
    optimizer$zero_grad()

    x_tensor <- batch$x$to(device = device)

    # 1. 编码器输出 h (Transformer + Norm)
    h <- model$encode(x_tensor)

    # 2. 投影到词表: h @ tok_emb.T
    logits <- torch_matmul(h, model$tok_emb$weight$transpose(1, 2))

    # 3. 展平计算带 mask 的 CE Loss
    logits_flat <- logits$reshape(c(-1, VOCAB_SIZE))
    y_flat <- batch$y$to(device = device)$reshape(c(-1))
    mask_flat <- batch$loss_mask$to(device = device)$reshape(c(-1))

    raw_loss <- nnf_cross_entropy(logits_flat, y_flat, reduction = "none")
    valid_loss <- raw_loss * mask_flat
    loss <- valid_loss$sum() / mask_flat$sum()

    loss$backward()
    nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
    optimizer$step()

    total_loss <- total_loss + loss$item()

    if (batch_idx %% 20 == 0 || batch_idx == 1) {
      cat(sprintf("Epoch [%02d/%02d] Step [%03d] LR=%.6f Loss=%.4f\n",
                  epoch, EPOCHS, batch_idx, current_lr, loss$item()))
    }
  })

  avg_loss <- total_loss / batch_idx
  cat(sprintf("=> Epoch %d 结束, 掩码平均 Loss: %.4f\n", epoch, avg_loss))

  scheduler$step()

  save_path <- sprintf("checkpoints/contrastive_sft_%02d.pt", epoch)
  torch_save(list(model = model$state_dict(), epoch = epoch, loss = avg_loss), save_path)
}
