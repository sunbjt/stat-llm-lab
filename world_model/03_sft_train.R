# =====================================================================
# JEPA World Model SFT 微调训练
# =====================================================================

source("config.R")

# --- 环境专属超参 ---
BATCH_SIZE <- if (is_mac) 4 else 64

device <- torch_device(if(cuda_is_available()) "cuda" else "cpu")
cat(sprintf("当前运行设备: %s\n", device$type))

# =====================================================================
# 1. 加载 Tokenizer 与 JEPA 基座
# =====================================================================
source("utils/BPETokenizer.R")
source("world_model/jepa_model.R")
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

model <- RtomicJEPA_VQ(
  vocab_size = VOCAB_SIZE, dim = DIM, n_layers = N_LAYERS,
  n_heads = N_HEADS, max_seq_len = SEQ_LEN, num_clusters = 4096
)

# =====================================================================
# 2. 加载预训练 JEPA 权重 (处理 luz 的 model. 前缀)
# =====================================================================
PRETRAIN_CKPT <- "checkpoints/wm_03.pt"
cat(sprintf("正在加载 JEPA 预训练权重: %s\n", PRETRAIN_CKPT))
ckpt <- torch_load(PRETRAIN_CKPT, device = "cpu")
clean_state_dict <- list()
for (name in names(ckpt$model)) {
  clean_name <- sub("^model\\.", "", name)
  clean_state_dict[[clean_name]] <- ckpt$model[[name]]
}
model$load_state_dict(clean_state_dict, strict = FALSE)
model <- model$to(device = device)

# =====================================================================
# 3. SFT 数据集定义 (与 LRP 版本完全一致)
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
# 4. 层冻结策略
# =====================================================================
# 全部参数解冻
for (p in model$parameters) { p$requires_grad_(TRUE) }

# target_layers 始终冻结 (EMA 网络，推理时不使用)
for (p in model$target_layers$parameters) p$requires_grad_(FALSE)
for (p in model$target_norm_f$parameters) p$requires_grad_(FALSE)

# 可选：冻结部分 context encoder 层
# FREEZE_LAYERS <- 6
# model$tok_emb$weight$requires_grad_(FALSE)
# for (i in 1:FREEZE_LAYERS) {
#   lapply(model$layers[[i]]$parameters, function(p) p$requires_grad_(FALSE))
# }

# =====================================================================
# 5. 准备数据与 Dataloader
# =====================================================================
raw_data <- jsonlite::stream_in(file("data/raw/qa_no_think.jsonl"))
sft_ds <- GenerativeSFTDataset(raw_data$instruction, raw_data$output, tokenizer, max_len = SEQ_LEN)
sft_dl <- dataloader(sft_ds, batch_size = BATCH_SIZE, shuffle = TRUE)

# =====================================================================
# 6. 原生 Torch 训练循环
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

    # JEPA forward: y = NULL 获取潜空间预测 pred
    input_data <- list(
      x = batch$x$to(device = device),
      y = NULL
    )

    output <- model(input_data)
    pred <- output$pred  # (B, S, Dim)

    # 投影到词表
    logits <- torch_matmul(pred, model$tok_emb$weight$t())

    # 展平并计算掩码 Loss
    logits_flat <- logits$view(c(-1, VOCAB_SIZE))
    y_flat <- batch$y$to(device = device)$view(c(-1))
    mask_flat <- batch$loss_mask$to(device = device)$view(c(-1))

    raw_loss <- nnf_cross_entropy(logits_flat, y_flat, reduction = "none")
    valid_loss <- raw_loss * mask_flat
    loss <- valid_loss$sum() / mask_flat$sum()

    loss$backward()
    nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
    optimizer$step()

    total_loss <- total_loss + loss$item()

    if (batch_idx %% 20 == 0 || batch_idx == 1) {
      cat(sprintf("Epoch [%02d/%02d], Step [%03d], LR: %.6f, Loss: %.4f\n",
                  epoch, EPOCHS, batch_idx, current_lr, loss$item()))
    }
  })

  avg_loss <- total_loss / batch_idx
  cat(sprintf("=> Epoch %d 结束, 掩码平均 Loss: %.4f\n", epoch, avg_loss))

  scheduler$step()

  save_path <- sprintf("checkpoints/jepa_sft_epoch_%02d.pt", epoch)
  torch_save(model$state_dict(), save_path)
  cat(sprintf("已保存 Checkpoint: %s\n", save_path))
}
