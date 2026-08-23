# =====================================================================
# JEPA 预训练 (Train) — Raw Torch + Gradient Accumulation
# =====================================================================

Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")

source("config.R")
source("utils/BPETokenizer.R")
source("latent_contrastive/contrastive_model.R")

# --- 环境专属超参 (覆盖默认值) ---
if (is_mac) {
  ENV_BATCH_SIZE       <- 2
  ENV_USE_AMP          <- FALSE
  ENV_ACCUM_STEPS      <- 1
} else {
  ENV_BATCH_SIZE       <- 64
  ENV_USE_AMP          <- cuda_is_available()
  ENV_ACCUM_STEPS      <- 4   # 如果是 4 则等效于 ENV_BATCH_SIZE * 4
}

# =====================================================================
# 1. 内存友好型数据集构建 (Lazy Tensorization)
# =====================================================================
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

RtomicBinDataset <- dataset(
  name = "RtomicBinDataset",

  initialize = function(bin_file, seq_len = 512, bytes_per_token = 2) {
    self$seq_len <- seq_len
    file_info <- file.info(bin_file)
    total_bytes <- file_info$size
    total_tokens <- total_bytes / bytes_per_token

    cat(sprintf("正在将 %.2f M Tokens 全量载入内存...\n", total_tokens / 1e6))

    con <- file(bin_file, "rb")
    raw_tokens <- readBin(con, what = "integer", n = total_tokens,
                          size = bytes_per_token, signed = FALSE, endian = "little")
    close(con)

    self$data_tensor <- torch_tensor(raw_tokens, dtype = torch_long())
    self$num_batches <- floor((total_tokens - 1) / self$seq_len)

    cat(sprintf("数据加载完成！生成 %d 个训练块 (长度: %d)\n", self$num_batches, seq_len))
  },

  .getitem = function(i) {
    start_idx <- (i - 1) * self$seq_len + 1
    chunk <- self$data_tensor$narrow(dim = 1, start = start_idx, length = self$seq_len + 1)
    x <- chunk[1:self$seq_len]
    y <- chunk[2:(self$seq_len + 1)]
    list(x = list(x = x, y = y), y = y)
  },

  .length = function() {
    self$num_batches
  }
)

BIN_FILE <- "data/processed/zhwiki_tokens_16384.bin"

train_dataset <- RtomicBinDataset(
  bin_file = BIN_FILE,
  seq_len = SEQ_LEN
)

train_dl <- dataloader(
  train_dataset,
  batch_size = ENV_BATCH_SIZE,
  shuffle = TRUE,
  drop_last = TRUE,
  num_workers = 0,
  pin_memory = !is_mac
)

# =====================================================================
# 2. 加载模型
# =====================================================================

device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
cat(sprintf("当前运行设备: %s\n", device$type))

model <- TokenLatentModel(
  vocab_size = VOCAB_SIZE,
  dim = DIM,
  n_layers = N_LAYERS,
  n_heads = N_HEADS,
  max_seq_len = SEQ_LEN
)
model <- model$to(device = device)

# =====================================================================
# 3. 优化器 (分组 Weight Decay)
# =====================================================================

param_names <- names(model$parameters)
no_decay_pattern <- "norm|bias|tok_emb"
no_decay_names <- grep(no_decay_pattern, param_names, value = TRUE)
decay_names <- setdiff(param_names, no_decay_names)

all_params <- model$parameters
decay_params <- all_params[decay_names]
no_decay_params <- all_params[no_decay_names]

cat(sprintf("Weight Decay (0.01) 参数组: %d\n", length(decay_names)))
cat(sprintf("No Weight Decay (0.0) 参数组: %d\n", length(no_decay_names)))

optimizer <- optim_adamw(
  list(
    list(params = decay_params, weight_decay = 0.01),
    list(params = no_decay_params, weight_decay = 0.0)
  ),
  lr = 2e-3
)

# =====================================================================
# 4. WSD 学习率调度
# =====================================================================

TRAIN_EPOCHS <- 2
batches_per_epoch <- length(train_dl)
accum_steps <- ENV_ACCUM_STEPS
global_steps_per_epoch <- ceiling(batches_per_epoch / accum_steps)
total_global_steps <- TRAIN_EPOCHS * global_steps_per_epoch

warmup_pct <- 0.02
decay_pct  <- 0.20

wsd_multiplier <- function(global_step) {
  current <- as.numeric(global_step)
  total   <- as.numeric(total_global_steps)

  warmup_steps <- total * warmup_pct
  decay_steps  <- total * decay_pct
  stable_steps <- total - warmup_steps - decay_steps

  if (current <= warmup_steps) {
    current / warmup_steps
  } else if (current <= warmup_steps + stable_steps) {
    1.0
  } else {
    decay_step <- current - (warmup_steps + stable_steps)
    progress <- decay_step / decay_steps
    max(0.1, 0.5 * (1 + cos(pi * progress)))
  }
}

# 最终预期潜语义占主导地位，所以需要CE 手动退火。
# 初始为 0.5, cosine decay 至 0.05
ce_weight_start <- 0.5
ce_weight_end   <- 0.05

ce_weight_schedule <- function(step) {
  progress <- min(1.0, step / total_global_steps)
  ce_weight_end + (ce_weight_start - ce_weight_end) * 0.5 * (1 + cos(pi * progress))
}

cat(sprintf("\n训练配置:\n"))
cat(sprintf("  Micro-batch: %d\n", ENV_BATCH_SIZE))
cat(sprintf("  Accumulation steps: %d\n", accum_steps))
cat(sprintf("  Effective batch: %d\n", ENV_BATCH_SIZE * accum_steps))
cat(sprintf("  Epochs: %d\n", TRAIN_EPOCHS))
cat(sprintf("  Batches/epoch: %d\n", batches_per_epoch))
cat(sprintf("  Effective steps/epoch: %d\n", global_steps_per_epoch))
cat(sprintf("  Total effective steps: %d\n", total_global_steps))
cat(sprintf("  Base LR: %.1e\n", 2e-3))
cat(sprintf("  CE weight: %.2f → %.2f (cosine)\n", ce_weight_start, ce_weight_end))
cat(sprintf("  AMP: %s\n", ENV_USE_AMP))

# =====================================================================
# 5. 训练循环 (Gradient Accumulation)
# =====================================================================

log_file <- paste0("checkpoints/contrastive_loss_", format(Sys.time(), "%d%H%M"), ".csv")
cat("epoch,iter,loss,contrastive,ce,lr,ce_weight\n", file = log_file)

scaler <- if (ENV_USE_AMP) cuda_amp_grad_scaler() else NULL

cat("\n开始训练...\n")

global_step <- 0
current_lr <- 2e-3 * wsd_multiplier(1)
for (pg in optimizer$param_groups) pg$lr <- current_lr

for (epoch in 1:TRAIN_EPOCHS) {
  model$train()
  epoch_loss <- 0
  batch_idx <- 0

  coro::loop(for (batch in train_dl) {
    batch_idx <- batch_idx + 1

    # --- forward (with optional AMP) ---
    if (ENV_USE_AMP) {
      with_autocast(device_type = "cuda", {
        input_data <- list(
          x = batch$x$x$to(device = device),
          y = batch$x$y$to(device = device)
        )
        output <- model(input_data)
        loss <- output$loss / accum_steps
      })
    } else {
      input_data <- list(
        x = batch$x$x$to(device = device),
        y = batch$x$y$to(device = device)
      )
      output <- model(input_data)
      loss <- output$loss / accum_steps
    }

    # --- backward (with optional scaler) ---
    if (ENV_USE_AMP) {
      scaler$scale(loss)$backward()
    } else {
      loss$backward()
    }

    # --- optimizer step only after accumulation ---
    if (batch_idx %% accum_steps == 0 || batch_idx == batches_per_epoch) {
      global_step <- global_step + 1

      # gradient clipping
      if (ENV_USE_AMP) {
        scaler$unscale_(optimizer)
        nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
        scaler$step(optimizer)
        scaler$update()
      } else {
        nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
        optimizer$step()
      }

      # WSD LR schedule
      mult <- wsd_multiplier(global_step)
      current_lr <- 2e-3 * mult
      for (pg in optimizer$param_groups) pg$lr <- current_lr

      # CE weight annealing (nn_buffer 必须用 copy_ 原地更新)
      model$ce_weight$copy_(torch_tensor(ce_weight_schedule(global_step), device = device))

      optimizer$zero_grad()
    }

    total_loss <- output$loss$item() * accum_steps
    epoch_loss <- epoch_loss + total_loss

    if (batch_idx %% 50 == 0 || batch_idx == 1) {
      w <- nchar(as.character(batches_per_epoch))
      cat(sprintf("Epoch [%d/%d] Batch [%*d/%d] step=%*d | loss=%.4f C=%.4f CE=%.4f | lr=%.2e\n",
                  epoch, TRAIN_EPOCHS,
                  w, batch_idx, batches_per_epoch,
                  w, global_step,
                  total_loss,
                  as.numeric(output$contrastive),
                  as.numeric(output$ce),
                  current_lr))
    }

    # --- log to CSV ---
    cat(sprintf("%d,%d,%.4f,%.4f,%.4f,%.6e,%.4f\n",
                epoch, batch_idx,
                total_loss,
                as.numeric(output$contrastive),
                as.numeric(output$ce),
                current_lr,
                as.numeric(model$ce_weight)),
        file = log_file, append = TRUE)
  })

  avg_loss <- epoch_loss / batch_idx
  cat(sprintf("=== Epoch %d 结束, 平均 Loss: %.4f ===\n", epoch, avg_loss))

  # --- checkpoint ---
  ckpt_path <- sprintf("checkpoints/contrastive_%02d.pt", epoch)
  torch_save(list(
    model = model$state_dict(),
    optimizer = optimizer$state_dict(),
    epoch = epoch,
    loss = avg_loss
  ), ckpt_path)
  cat(sprintf("Checkpoint saved: %s\n", ckpt_path))
}

cat("\n训练完成！\n")
