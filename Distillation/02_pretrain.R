# =====================================================================
# 预训练 (Train) — 优化版: 内存 Dataloader + 梯度累积 + 分块交叉熵
# =====================================================================
library(torch)
library(R6)
library(tok)
library(coro)
torch_manual_seed(42)
Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")

## 环境专属超参 (覆盖默认值)
is_mac       <- Sys.info()["sysname"] == "Darwin"
is_gpu       <- !is_mac && cuda_is_available()
DIM          <- 320
N_LAYERS     <- 8
N_HEADS      <- 8
SEQ_LEN      <- 512

if (is_mac) {
  ENV_BATCH_SIZE  <- 2
  ENV_GRAD_ACCUM  <- 4    # 等效 Batch = 8
  ENV_USE_AMP     <- FALSE
  ENV_WORKERS     <- 0
} else if (is_gpu) {
  ENV_BATCH_SIZE  <- 24    # 微批次大小 (受 15 万词表 logits 显存限制)
  ENV_GRAD_ACCUM  <- 8    # 等效 Batch = 8 × 8 = 64
  ENV_USE_AMP     <- TRUE
  ENV_WORKERS     <- 0    # 数据全在内存，无需多进程
} else {
  ENV_BATCH_SIZE  <- 64
  ENV_GRAD_ACCUM  <- 1
  ENV_USE_AMP     <- FALSE
  ENV_WORKERS     <- 2
}

# 设备检测
device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
cat(sprintf("训练设备: %s | Micro-Batch: %d | Grad-Accum: %d | 等效 Batch: %d\n",
            device$type, ENV_BATCH_SIZE, ENV_GRAD_ACCUM, ENV_BATCH_SIZE * ENV_GRAD_ACCUM))

# =====================================================================
# 1. 纯内存 Dataloader (零磁盘 I/O 阻塞)
#    一次性将整个 .bin 文件载入内存 Tensor，后续通过 narrow() 零拷贝切片
# =====================================================================
source("Distillation/LRP_model.R")

tokenizer_file <- "models/tokenizer.json"
tokenizer <- tok::tokenizer$from_file(tokenizer_file)
VOCAB_SIZE <- tokenizer$get_vocab_size()
cat(sprintf("Qwen Tokenizer 词表大小: %d\n", VOCAB_SIZE))

RtomicBinDataset <- dataset(
  name = "RtomicBinDataset",

  initialize = function(bin_file, seq_len = 256, vocab_size = 151936, bytes_per_token = 4) {
    self$seq_len <- seq_len
    file_info <- file.info(bin_file)
    total_bytes <- file_info$size
    total_tokens <- total_bytes / bytes_per_token

    cat(sprintf("物理总 Token 数: %.2f M\n", total_tokens / 1e6))
    cat(sprintf("正在将全部数据载入内存...\n"))

    # 一次性读取整个文件
    con <- file(bin_file, "rb")
    raw_tokens <- readBin(con, what = "integer", n = total_tokens,
                          size = bytes_per_token, signed = TRUE, endian = "little")
    close(con)

    # Qwen Token ID 是 0-based，R torch 需要 1-based，整体 +1
    raw_tokens <- raw_tokens + 1L
    self$data_tensor <- torch_tensor(raw_tokens, dtype = torch_long())
    self$num_batches <- floor((total_tokens - 1) / self$seq_len)

    cat(sprintf("数据加载完成！生成 %d 个训练块 (序列长度: %d)\n",
                self$num_batches, seq_len))
  },

  .getitem = function(i) {
    # 零拷贝 narrow 切片，几乎无耗时
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

BIN_FILE <- "data/processed/qwen_tokens_aligned.bin"

train_dataset <- RtomicBinDataset(
  bin_file = BIN_FILE,
  seq_len = SEQ_LEN,
  vocab_size = VOCAB_SIZE,
  bytes_per_token = 4      # Qwen Token ID 需要 32-bit
)

train_dl <- dataloader(
  train_dataset,
  batch_size = ENV_BATCH_SIZE,
  shuffle = TRUE,
  drop_last = TRUE,
  num_workers = ENV_WORKERS,
  pin_memory = is_gpu
)

# =====================================================================
# 2. 优化器 & 学习率调度器
# =====================================================================
custom_grouped_optim <- function(params, lr = 1e-3, weight_decay = 0.01, ...) {
  emb_names <- grep("^tok_emb|^pos_emb", names(params), value = TRUE)
  other_names <- setdiff(names(params), emb_names)

  grouped_params <- list(
    list(params = params[emb_names], lr = 1e-4),
    list(params = params[other_names], lr = lr)
  )
  optim_adamw(grouped_params, weight_decay = weight_decay, ...)
}

# WSD (Warmup - Stable - Decay) 学习率乘子计算器
wsd_multiplier <- function(step, total_steps, warmup_pct = 0.1, decay_pct = 0.1) {
  current_step <- as.numeric(step) + 1
  total_steps <- as.numeric(total_steps)

  warmup_steps <- total_steps * warmup_pct
  decay_steps <- total_steps * decay_pct
  stable_steps <- total_steps - warmup_steps - decay_steps

  if (current_step <= warmup_steps) {
    return(current_step / warmup_steps)
  } else if (current_step <= warmup_steps + stable_steps) {
    return(1.0)
  } else {
    decay_step <- current_step - (warmup_steps + stable_steps)
    progress <- decay_step / decay_steps
    cosine_decay <- 0.5 * (1 + cos(pi * progress))
    return(max(0.1, cosine_decay))
  }
}

# =====================================================================
# 3. 模型初始化
# =====================================================================
model <- RtomicLRP(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN)
model$to(device = device)
cat(sprintf("模型已加载至: %s\n", device$type))

# =====================================================================
# 4. 训练配置
# =====================================================================
TRAIN_EPOCHS <- 1
# 实际优化步数 = 总 batch 数 / 梯度累积步数
total_steps <- (TRAIN_EPOCHS * length(train_dl)) / ENV_GRAD_ACCUM
cat(sprintf("总训练步数 (优化器更新次数): %.0f\n", total_steps))

# 学习率缩放法则: LR_new = LR_base × sqrt(K)
lr_rate <- sqrt(ENV_GRAD_ACCUM)
optimizer <- custom_grouped_optim(model$parameters, lr = lr_rate * 2e-3, weight_decay = 0.01)
scheduler <- lr_lambda(
  optimizer,
  lr_lambda = function(step) wsd_multiplier(step, total_steps, warmup_pct = 0.1, decay_pct = 0.2)
)

# AMP 混合精度
use_scaler <- ENV_USE_AMP && cuda_is_available()
if (use_scaler) {
  scaler <- cuda_amp_grad_scaler()
  cat("已启用 AMP 混合精度训练 (bfloat16)\n")
}

# Loss 记录文件
loss_log_file <- paste0("checkpoints/qwen_loss_", format(Sys.time(), "%d%H%M"), ".csv")
cat(sprintf("Loss 日志: %s\n", loss_log_file))

# =====================================================================
# 5. 手动训练循环 (支持梯度累积)
# =====================================================================
global_step <- 0
model$train()

for (epoch in 1:TRAIN_EPOCHS) {
  iter_idx <- 0
  epoch_loss <- 0

  coro::loop(for (b in train_dl) {
    iter_idx <- iter_idx + 1

    # 异步 CPU → GPU 传输
    batch_x <- b$x$x$to(device = device, non_blocking = TRUE)
    batch_y <- b$y$to(device = device, non_blocking = TRUE)

    # 前向传播 (AMP 自动混合精度)
    if (use_scaler) {
      with_autocast(device_type = "cuda", dtype = torch_bfloat16(), enabled = TRUE, {
        output <- model(list(x = batch_x, y = batch_y))
        loss <- output$loss / ENV_GRAD_ACCUM
      })
    } else {
      output <- model(list(x = batch_x, y = batch_y))
      loss <- output$loss / ENV_GRAD_ACCUM
    }

    # 反向传播
    if (use_scaler) {
      scaler$scale(loss)$backward()
    } else {
      loss$backward()
    }

    # 达到累积阈值，执行权重更新
    if (iter_idx %% ENV_GRAD_ACCUM == 0) {
      if (use_scaler) {
        scaler$unscale_(optimizer)
        torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
        scaler$step(optimizer)
        scaler$update()
      } else {
        torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
        optimizer$step()
      }

      optimizer$zero_grad()
      scheduler$step()
      global_step <- global_step + 1

      current_loss <- as.numeric(output$loss)  # 原始 Loss (未缩放)
      epoch_loss <- epoch_loss + current_loss

      # 记录到 CSV
      cat(sprintf("%d,%d,%.6f\n", epoch, global_step, current_loss),
          file = loss_log_file, append = TRUE)

      if (global_step %% 20 == 0 || global_step == 1) {
        cat(sprintf("Epoch: %d | Step: %d | Loss: %.4f | LR: %.2e\n",
                    epoch, global_step, current_loss, optimizer$param_groups[[2]]$lr))
      }
    }
  })

  avg_loss <- epoch_loss / (iter_idx / ENV_GRAD_ACCUM)
  cat(sprintf("Epoch %d 完成, Avg Loss: %.4f\n", epoch, avg_loss))

  # 保存 Checkpoint
  checkpoint_path <- sprintf("checkpoints/qwen_%02d.pt", epoch)
  torch_save(model$state_dict(), checkpoint_path)
  cat(sprintf("已保存 Checkpoint: %s\n", checkpoint_path))
}

cat("\n预训练完成！\n")
