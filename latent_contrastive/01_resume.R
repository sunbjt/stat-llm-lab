# =====================================================================
# 断点续训 — 从已保存的 checkpoint 继续预训练
# =====================================================================

Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")

source("config.R")
source("utils/BPETokenizer.R")
source("latent_contrastive/contrastive_model.R")

# =====================================================================
# 续训超参 (按需修改)
# =====================================================================
RESUME_CKPT  <- "checkpoints/contrastive_03.pt"
TRAIN_EPOCHS <- 3   # 在已有基础上再训几轮

if (is_mac) {
  BATCH_SIZE  <- 2
  USE_AMP     <- FALSE
} else {
  BATCH_SIZE  <- 64
  USE_AMP     <- cuda_is_available()
}

# =====================================================================
# 1. 数据集 (与 01 保持一致)
# =====================================================================
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

RtomicBinDataset <- dataset(
  name = "RtomicBinDataset",
  initialize = function(bin_file, seq_len = 512, bytes_per_token = 2) {
    self$seq_len <- seq_len
    file_info <- file.info(bin_file)
    total_tokens <- file_info$size / bytes_per_token
    con <- file(bin_file, "rb")
    raw_tokens <- readBin(con, what = "integer", n = total_tokens,
                          size = bytes_per_token, signed = FALSE, endian = "little")
    close(con)
    self$data_tensor <- torch_tensor(raw_tokens, dtype = torch_long())
    self$num_batches <- floor((total_tokens - 1) / self$seq_len)
    cat(sprintf("数据加载完成！%d 个训练块\n", self$num_batches))
  },
  .getitem = function(i) {
    start_idx <- (i - 1) * self$seq_len + 1
    chunk <- self$data_tensor$narrow(dim = 1, start = start_idx, length = self$seq_len + 1)
    list(x = list(x = chunk[1:self$seq_len], y = chunk[2:(self$seq_len + 1)]), y = chunk[2:(self$seq_len + 1)])
  },
  .length = function() self$num_batches
)

BIN_FILE <- "data/processed/zhwiki_tokens_16384.bin"
train_dl <- dataloader(
  RtomicBinDataset(bin_file = BIN_FILE, seq_len = SEQ_LEN),
  batch_size = BATCH_SIZE, shuffle = TRUE, drop_last = TRUE,
  num_workers = 0, pin_memory = !is_mac
)

# =====================================================================
# 2. 加载模型与优化器
# =====================================================================
device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")

model <- TokenLatentModel(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN)$to(device = device)

param_names <- names(model$parameters)
no_decay_pattern <- "norm|bias|tok_emb"
no_decay_names <- grep(no_decay_pattern, param_names, value = TRUE)
decay_names <- setdiff(param_names, no_decay_names)

optimizer <- optim_adamw(
  list(
    list(params = model$parameters[decay_names],    weight_decay = 0.01),
    list(params = model$parameters[no_decay_names], weight_decay = 0.0)
  ),
  lr = 2e-3
)

# =====================================================================
# 3. 加载 checkpoint
# =====================================================================
stopifnot(file.exists(RESUME_CKPT))
cat(sprintf("\n断点续训：加载 %s\n", RESUME_CKPT))
ckpt <- torch_load(RESUME_CKPT)
model$load_state_dict(ckpt$model)
model$to(device = device)   # 加这一行，确保所有 buffer 也在 GPU 上

prev_epoch <- if (!is.null(ckpt$epoch)) ckpt$epoch else 0
cat(sprintf("已恢复 epoch %d，将续训 %d 轮 (epoch %d → %d)\n",
            prev_epoch, TRAIN_EPOCHS, prev_epoch + 1, prev_epoch + TRAIN_EPOCHS))

# =====================================================================
# 4. WSD 调度 (按总轮数计算)
# =====================================================================
batches_per_epoch <- length(train_dl)
steps_per_epoch <- batches_per_epoch
total_epochs <- prev_epoch + TRAIN_EPOCHS
total_steps <- total_epochs * steps_per_epoch

warmup_pct <- 0.02
decay_pct  <- 0.20

wsd_multiplier <- function(step) {
  current <- as.numeric(step); total <- as.numeric(total_steps)
  warmup_steps <- total * warmup_pct
  decay_steps  <- total * decay_pct
  stable_steps <- total - warmup_steps - decay_steps
  if (current <= warmup_steps) { current / warmup_steps }
  else if (current <= warmup_steps + stable_steps) { 1.0 }
  else {
    progress <- (current - warmup_steps - stable_steps) / decay_steps
    max(0.1, 0.5 * (1 + cos(pi * progress)))
  }
}

ce_weight_start <- 0.5
ce_weight_end   <- 0.05
ce_weight_schedule <- function(step) {
  progress <- min(1.0, step / total_steps)
  ce_weight_end + (ce_weight_start - ce_weight_end) * 0.5 * (1 + cos(pi * progress))
}

cat(sprintf("总 epoch: %d, 总步数: %d\n", total_epochs, total_steps))
cat(sprintf("CE weight: %.2f → %.2f\n", ce_weight_start, ce_weight_end))

# =====================================================================
# 5. 续训循环
# =====================================================================
log_file <- paste0("checkpoints/contrastive_resume_", format(Sys.time(), "%d%H%M"), ".csv")
cat("epoch,iter,loss,contrastive,ce,lr,ce_weight\n", file = log_file)

scaler <- if (USE_AMP) cuda_amp_grad_scaler() else NULL

global_step <- prev_epoch * steps_per_epoch
current_lr <- 2e-3 * wsd_multiplier(global_step + 1)
for (pg in optimizer$param_groups) pg$lr <- current_lr

cat("\n开始续训...\n")

start_epoch <- prev_epoch + 1
end_epoch   <- prev_epoch + TRAIN_EPOCHS
w <- nchar(as.character(batches_per_epoch))

for (epoch in start_epoch:end_epoch) {
  model$train()
  epoch_loss <- 0
  batch_idx <- 0

  coro::loop(for (batch in train_dl) {
    batch_idx <- batch_idx + 1

    if (USE_AMP) {
      with_autocast(device_type = "cuda", {
        input_data <- list(x = batch$x$x$to(device = device), y = batch$x$y$to(device = device))
        output <- model(input_data)
        loss <- output$loss
      })
    } else {
      input_data <- list(x = batch$x$x$to(device = device), y = batch$x$y$to(device = device))
      output <- model(input_data)
      loss <- output$loss
    }

    if (USE_AMP) { scaler$scale(loss)$backward() }
    else         { loss$backward() }

    global_step <- global_step + 1

    if (USE_AMP) {
      scaler$unscale_(optimizer)
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      scaler$step(optimizer)
      scaler$update()
    } else {
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      optimizer$step()
    }

    mult <- wsd_multiplier(global_step)
    current_lr <- 2e-3 * mult
    for (pg in optimizer$param_groups) pg$lr <- current_lr
    model$ce_weight$copy_(torch_tensor(ce_weight_schedule(global_step), device = device))
    optimizer$zero_grad()

    total_loss <- loss$item()
    epoch_loss <- epoch_loss + total_loss

    if (batch_idx %% 50 == 0 || batch_idx == 1) {
      cat(sprintf("Epoch [%d/%d] Batch [%*d/%d] step=%*d | loss=%.4f C=%.4f CE=%.4f | lr=%.2e\n",
                  epoch, end_epoch,
                  w, batch_idx, batches_per_epoch,
                  w, global_step,
                  total_loss,
                  as.numeric(output$contrastive),
                  as.numeric(output$ce),
                  current_lr))
    }

    cat(sprintf("%d,%d,%.4f,%.4f,%.4f,%.6e,%.4f\n",
                epoch, batch_idx, total_loss,
                as.numeric(output$contrastive), as.numeric(output$ce),
                current_lr, as.numeric(model$ce_weight)),
        file = log_file, append = TRUE)
  })

  avg_loss <- epoch_loss / batch_idx
  cat(sprintf("=== Epoch %d 结束, 平均 Loss: %.4f ===\n", epoch, avg_loss))

  ckpt_path <- sprintf("checkpoints/contrastive_%02d.pt", epoch)
  torch_save(list(model = model$state_dict(), optimizer = optimizer$state_dict(),
                  epoch = epoch, loss = avg_loss), ckpt_path)
}

cat("\n续训完成！\n")
