# =====================================================================
# Pure InfoNCE 预训练 (Train) — Masked Negatives & Corrected LR
# =====================================================================

Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")

source("config.R")
source("latent_contrastive/contrastive_model.infonce.R")

# --- 环境专属超参 ---
if (is_mac) {
  ENV_BATCH_SIZE  <- 2
  ENV_MAX_LINES   <- 1000
  ENV_USE_AMP     <- FALSE
  ENV_ACCUM_STEPS <- 2
} else {
  ENV_BATCH_SIZE  <- 128
  ENV_MAX_LINES   <- -1
  ENV_USE_AMP     <- cuda_is_available()
  ENV_ACCUM_STEPS <- 4
}

# =====================================================================
# 1. Dataset
# =====================================================================

RtomicBinDataset <- dataset(
  name = "RtomicBinDataset",

  initialize = function(
    bin_file,
    seq_len = 512,
    bytes_per_token = 2,
    max_chunks = -1
  ) {
    self$seq_len <- seq_len
    file_info <- file.info(bin_file)

    if (is.na(file_info$size)) {
      stop(sprintf("找不到数据文件: %s", bin_file))
    }

    total_bytes <- file_info$size
    if (total_bytes %% bytes_per_token != 0) {
      stop(sprintf("数据文件大小 (%d bytes) 不是 bytes_per_token=%d 的整数倍。", total_bytes, bytes_per_token))
    }

    total_tokens <- total_bytes / bytes_per_token
    cat(sprintf("正在将 %.2f M Tokens 全量载入内存...\n", total_tokens / 1e6))

    con <- file(bin_file, "rb")
    on.exit(close(con), add = TRUE)

    raw_tokens <- readBin(con, what = "integer", n = total_tokens, size = bytes_per_token, signed = FALSE, endian = "little")
    self$data_tensor <- torch_tensor(raw_tokens, dtype = torch_long())

    num_chunks <- floor((total_tokens - 1) / self$seq_len)
    if (max_chunks > 0) {
      num_chunks <- min(num_chunks, max_chunks)
    }

    self$num_chunks <- num_chunks
    cat(sprintf("数据加载完成！生成 %d 个训练块 (长度: %d)\n", self$num_chunks, seq_len))
  },

  .getitem = function(i) {
    start_idx <- (i - 1) * self$seq_len + 1
    chunk <- self$data_tensor$narrow(dim = 1, start = start_idx, length = self$seq_len + 1)
    x <- chunk[1:self$seq_len]
    y <- chunk[2:(self$seq_len + 1)]
    list(x = x, y = y)
  },

  .length = function() {
    self$num_chunks
  }
)

BIN_FILE <- "data/processed/zhwiki_tokens_16384.bin"
MAX_CHUNKS <- if (ENV_MAX_LINES > 0) ENV_MAX_LINES else -1

train_dataset <- RtomicBinDataset(
  bin_file = BIN_FILE,
  seq_len = SEQ_LEN,
  max_chunks = MAX_CHUNKS
)

train_dl <- dataloader(
  train_dataset,
  batch_size = ENV_BATCH_SIZE,
  shuffle = TRUE,
  drop_last = TRUE,
  num_workers = 0,
  pin_memory = !is_mac
)

batches_per_epoch <- length(train_dl)
if (batches_per_epoch < 1) {
  stop(sprintf("训练 batch 数为 0：dataset chunks=%d, batch_size=%d。", length(train_dataset), ENV_BATCH_SIZE))
}

# =====================================================================
# 2. Model (超参优化关键点)
# =====================================================================

device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
cat(sprintf("当前运行设备: %s\n", device$type))

# 【修改点 1】：温度系数降低至标准的 0.07，配合 Masked Negatives
TEMPERATURE <- 0.07

model <- TokenLatentModel(
  vocab_size  = VOCAB_SIZE,
  dim         = DIM,
  n_layers    = N_LAYERS,
  n_heads     = N_HEADS,
  max_seq_len = SEQ_LEN,
  temperature = TEMPERATURE
)

model <- model$to(device = device)

# =====================================================================
# 3. Optimizer (学习率修正)
# =====================================================================

param_names <- names(model$parameters)
no_decay_pattern <- "norm|bias|tok_emb"
no_decay_names <- grep(no_decay_pattern, param_names, value = TRUE)
decay_names <- setdiff(param_names, no_decay_names)

all_params <- model$parameters
decay_params <- all_params[decay_names]
no_decay_params <- all_params[no_decay_names]

# ：BASE_LR 从 2e-3 降低至 3e-4，防止特征震荡崩塌
BASE_LR <- 3e-4
WEIGHT_DECAY <- 0.01

optimizer <- optim_adamw(
  list(
    list(params = decay_params, weight_decay = WEIGHT_DECAY),
    list(params = no_decay_params, weight_decay = 0.0)
  ),
  lr = BASE_LR
)

# =====================================================================
# 4. WSD Learning Rate Schedule
# =====================================================================

TRAIN_EPOCHS <- 2

accum_steps <- max(1, ENV_ACCUM_STEPS)
global_steps_per_epoch <- ceiling(batches_per_epoch / accum_steps)
total_global_steps <- TRAIN_EPOCHS * global_steps_per_epoch

warmup_pct <- 0.02
decay_pct <- 0.20
LR_MIN_MULTIPLIER <- 0.10

warmup_steps <- max(1, round(total_global_steps * warmup_pct))
decay_steps  <- max(1, round(total_global_steps * decay_pct))

if (warmup_steps + decay_steps >= total_global_steps) {
  stop("WSD schedule requires warmup_steps + decay_steps < total_global_steps.")
}

stable_steps <- total_global_steps - warmup_steps - decay_steps

wsd_multiplier <- function(global_step) {
  current <- as.numeric(global_step)
  if (current <= 0) {
    0.0
  } else if (current <= warmup_steps) {
    current / warmup_steps
  } else if (current <= warmup_steps + stable_steps) {
    1.0
  } else {
    decay_step <- current - (warmup_steps + stable_steps)
    progress <- min(1.0, decay_step / decay_steps)
    LR_MIN_MULTIPLIER + (1 - LR_MIN_MULTIPLIER) * 0.5 * (1 + cos(pi * progress))
  }
}

# =====================================================================
# 5. Training Loop
# =====================================================================

log_file <- paste0("checkpoints/contrastive_loss_", format(Sys.time(), "%d%H%M"), ".csv")
cat("epoch,iter,loss,contrastive,lr,global_step\n", file = log_file)

scaler <- if (ENV_USE_AMP) cuda_amp_grad_scaler() else NULL
optimizer$zero_grad()

cat("\n开始 Pure InfoNCE (Masked Negatives) 训练...\n")
global_step <- 0
current_lr <- BASE_LR * wsd_multiplier(1)

for (pg in optimizer$param_groups) {
  pg$lr <- current_lr
}

for (epoch in 1:TRAIN_EPOCHS) {
  model$train()
  epoch_loss <- 0
  batch_idx <- 0

  coro::loop(for (batch in train_dl) {
    batch_idx <- batch_idx + 1

    remainder <- batches_per_epoch %% accum_steps
    current_accum_steps <- if (remainder != 0 && batch_idx > batches_per_epoch - remainder) {
      remainder
    } else {
      accum_steps
    }

    input_data <- list(
      x = batch$x$to(device = device),
      y = batch$y$to(device = device)
    )

    # Forward
    if (ENV_USE_AMP) {
      with_autocast(device_type = "cuda", {
        output <- model(input_data)
        loss <- output$loss / current_accum_steps
      })
    } else {
      output <- model(input_data)
      loss <- output$loss / current_accum_steps
    }

    # Backward
    if (ENV_USE_AMP) {
      scaler$scale(loss)$backward()
    } else {
      loss$backward()
    }

    # Step
    should_step <- (batch_idx %% accum_steps == 0 || batch_idx == batches_per_epoch)

    if (should_step) {
      global_step <- global_step + 1

      if (ENV_USE_AMP) {
        scaler$unscale_(optimizer)
        nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
        scaler$step(optimizer)
        scaler$update()
      } else {
        nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
        optimizer$step()
      }

      current_lr <- BASE_LR * wsd_multiplier(min(global_step + 1, total_global_steps))
      for (pg in optimizer$param_groups) {
        pg$lr <- current_lr
      }

      optimizer$zero_grad()
    }

    # Logging
    total_loss <- output$loss$item() * current_accum_steps
    epoch_loss <- epoch_loss + total_loss

    if (batch_idx %% 50 == 0 || batch_idx == 1) {
      width <- nchar(as.character(batches_per_epoch))
      cat(sprintf(
        "Epoch [%d/%d] Batch [%*d/%d] step=%*d | InfoNCE=%.4f | lr=%.2e\n",
        epoch, TRAIN_EPOCHS, width, batch_idx, batches_per_epoch, width, global_step, total_loss, current_lr
      ))
    }

    cat(sprintf("%d,%d,%.6f,%.6f,%.6e,%d\n", epoch, batch_idx, total_loss, as.numeric(output$contrastive), current_lr, global_step),
        file = log_file, append = TRUE)
  })

  avg_loss <- epoch_loss / batch_idx
  cat(sprintf("=== Epoch %d 结束, 平均 InfoNCE: %.6f ===\n", epoch, avg_loss))

  # Save Checkpoint
  ckpt_path <- sprintf("checkpoints/infonce_%02d.pt", epoch)
  torch_save(
    list(
      model = model$state_dict(),
      optimizer = optimizer$state_dict(),
      epoch = epoch,
      global_step = global_step,
      loss = avg_loss,
      current_lr = current_lr,
      config = list(
        vocab_size = VOCAB_SIZE,
        dim = DIM,
        n_layers = N_LAYERS,
        n_heads = N_HEADS,
        seq_len = SEQ_LEN,
        batch_size = ENV_BATCH_SIZE,
        accum_steps = accum_steps,
        base_lr = BASE_LR,
        weight_decay = WEIGHT_DECAY,
        temperature = TEMPERATURE,
        objective = "pure_infonce_masked"
      )
    ),
    ckpt_path
  )

  cat(sprintf("Checkpoint saved: %s\n", ckpt_path))
}

cat("\nPure InfoNCE 训练完成！\n")