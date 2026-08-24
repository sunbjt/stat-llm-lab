# =====================================================================
# SFT 微调训练脚本 (TokenLatentModel - High-Performance Pure InfoNCE SFT)
# =====================================================================

Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")

source("config.R")
source("utils/BPETokenizer.R")
source("latent_contrastive/contrastive_model.infonce.R")

# --- 环境专属超参 ---
if (is_mac) {
  ENV_BATCH_SIZE  <- 4
  ENV_ACCUM_STEPS <- 2
  ENV_USE_AMP     <- FALSE
  ENV_NUM_WORKERS <- 0
} else {
  ENV_BATCH_SIZE  <- 32   # 24G 显存可根据文本长度适当调大到 32~64
  ENV_ACCUM_STEPS <- 2
  ENV_USE_AMP     <- cuda_is_available()
  ENV_NUM_WORKERS <- 4    # 开启多Worker加速
}

device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
cat(sprintf("当前运行设备: %s\n", device$type))

# =====================================================================
# 1. 加载 Tokenizer
# =====================================================================
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

# =====================================================================
# 2. 高性能 SFT 数据集定义 (纯原生 Integer 切片，彻底消除 Worker 警告)
# =====================================================================
GenerativeSFTDataset <- dataset(
  name = "GenerativeSFTDataset",

  initialize = function(prompts, responses, tokenizer, max_len = 512) {
    self$max_len <- max_len
    cat(sprintf("正在预处理并 Tokenize %d 条 SFT 数据...\n", length(prompts)))

    bos_id <- tokenizer$bos_idx
    eos_id <- if (!is.null(tokenizer$eos_idx)) tokenizer$eos_idx else 2L
    pad_id <- 1L  # 假设 Padding Token ID 为 1

    # 预先在内存中对所有文本切词并转为 Integer List，提速 CPU 子进程
    encoded_samples <- vector("list", length(prompts))

    for (i in seq_along(prompts)) {
      p_ids <- c(bos_id, tokenizer$encode_raw(prompts[i])[[1]])
      r_ids <- c(tokenizer$encode_raw(responses[i])[[1]], eos_id)

      full_seq <- c(p_ids, r_ids)
      seq_len  <- length(full_seq)

      # 构建 autoregressive 移位对
      x_ids <- full_seq[1:(seq_len - 1)]
      y_ids <- full_seq[2:seq_len]

      # Mask: 仅 Responses 对应的下一个 Token 位置计算 Loss (TRUE)
      mask  <- c(rep(FALSE, length(p_ids) - 1), rep(TRUE, length(r_ids)))

      # Padding 处理
      pad_len <- max_len - length(x_ids)
      if (pad_len > 0) {
        x_ids <- c(x_ids, rep(pad_id, pad_len))
        y_ids <- c(y_ids, rep(pad_id, pad_len))
        mask  <- c(mask, rep(FALSE, pad_len))
      } else {
        x_ids <- x_ids[1:max_len]
        y_ids <- y_ids[1:max_len]
        mask  <- mask[1:max_len]
      }

      encoded_samples[[i]] <- list(
        x = as.integer(x_ids),
        y = as.integer(y_ids),
        mask = as.logical(mask)
      )
    }

    self$samples <- encoded_samples
    cat("SFT 数据集预处理完成！\n")
  },

  .getitem = function(i) {
    item <- self$samples[[i]]
    list(
      x = torch_tensor(item$x, dtype = torch_long()),
      y = torch_tensor(item$y, dtype = torch_long()),
      loss_mask = torch_tensor(item$mask, dtype = torch_bool())
    )
  },

  .length = function() {
    length(self$samples)
  }
)

# =====================================================================
# 3. 数据加载与 Dataloader
# =====================================================================
SFT_JSONL_FILE <- "data/raw/qa_no_think.jsonl"
if (!file.exists(SFT_JSONL_FILE)) {
  stop(sprintf("找不到 SFT 数据文件: %s", SFT_JSONL_FILE))
}

raw_data <- jsonlite::stream_in(file(SFT_JSONL_FILE), verbose = FALSE)

sft_ds <- GenerativeSFTDataset(
  prompts   = raw_data$instruction,
  responses = raw_data$output,
  tokenizer = tokenizer,
  max_len   = SEQ_LEN
)

sft_dl <- dataloader(
  sft_ds,
  batch_size  = ENV_BATCH_SIZE,
  shuffle     = TRUE,
  drop_last   = TRUE,
  num_workers = ENV_NUM_WORKERS,
  pin_memory  = !is_mac
)

batches_per_epoch <- length(sft_dl)

# =====================================================================
# 4. 初始化模型与加载预训练底座权重
# =====================================================================
TEMPERATURE <- 0.07

model <- TokenLatentModel(
  vocab_size  = VOCAB_SIZE,
  dim         = DIM,
  n_layers    = N_LAYERS,
  n_heads     = N_HEADS,
  max_seq_len = SEQ_LEN,
  temperature = TEMPERATURE
)

PRETRAIN_CKPT <- "checkpoints/infonce_02.pt"  # 自动读取最新的 InfoNCE Checkpoint
if (file.exists(PRETRAIN_CKPT)) {
  cat(sprintf("正在加载 InfoNCE 预训练底座权重: %s\n", PRETRAIN_CKPT))
  ckpt <- torch_load(PRETRAIN_CKPT, device = "cpu")
  state_dict <- if (!is.null(ckpt$model)) ckpt$model else ckpt

  model$load_state_dict(state_dict, strict = FALSE)
  cat("底座权重加载成功！\n")
} else {
  cat("【警告】未找到预训练 Checkpoint，将使用随机初始化权重开始 SFT！\n")
}

model <- model$to(device = device)

# =====================================================================
# 5. 优化器与 Learning Rate 调度 (SFT 推荐较小 LR)
# =====================================================================
SFT_LR <- 1e-4  # SFT 通常使用比预训练更小的 LR (1e-4 ~ 5e-5)
WEIGHT_DECAY <- 0.01
EPOCHS <- 5

accum_steps <- max(1, ENV_ACCUM_STEPS)
global_steps_per_epoch <- ceiling(batches_per_epoch / accum_steps)
total_global_steps <- EPOCHS * global_steps_per_epoch

param_names <- names(model$parameters)
no_decay_pattern <- "norm|bias|tok_emb"
no_decay_names <- grep(no_decay_pattern, param_names, value = TRUE)
decay_names <- setdiff(param_names, no_decay_names)

all_params <- model$parameters
decay_params <- all_params[decay_names]
no_decay_params <- all_params[no_decay_names]

optimizer <- optim_adamw(
  list(
    list(params = decay_params, weight_decay = WEIGHT_DECAY),
    list(params = no_decay_params, weight_decay = 0.0)
  ),
  lr = SFT_LR
)

scaler <- if (ENV_USE_AMP) cuda_amp_grad_scaler() else NULL

# Cosine 学习率衰减
get_sft_lr <- function(step, total_steps, base_lr) {
  progress <- min(1.0, max(0.0, step / total_steps))
  base_lr * 0.5 * (1 + cos(pi * progress))
}

# =====================================================================
# 6. SFT 训练循环
# =====================================================================
cat("\n========================================\n")
cat(" 开始 TokenLatentModel SFT 掩码微调 ...\n")
cat("========================================\n")

global_step <- 0
optimizer$zero_grad()

for (epoch in 1:EPOCHS) {
  model$train()
  epoch_loss <- 0
  batch_idx  <- 0

  coro::loop(for (batch in sft_dl) {
    batch_idx <- batch_idx + 1

    remainder <- batches_per_epoch %% accum_steps
    current_accum_steps <- if (remainder != 0 && batch_idx > batches_per_epoch - remainder) {
      remainder
    } else {
      accum_steps
    }

    # 异步推送数据至 GPU
    x_tensor <- batch$x$to(device = device, non_blocking = TRUE)
    y_tensor <- batch$y$to(device = device, non_blocking = TRUE)
    mask_tensor <- batch$loss_mask$to(device = device, non_blocking = TRUE)

    # ---------------------------------------------------------------
    # Autoregressive Logits 推演机制：
    # 将模型编码层输出 h 通过与 Token Embedding 的转置求点积，生成 Logits
    # ---------------------------------------------------------------
    forward_sft_loss <- function() {
      h <- model$encode(x_tensor) # [B, S, D]
      
      # 经过 Predictor 变换映射回 Latent Space
      pred <- model$predictor(h)  # [B, S, D]

      # Logits = pred @ tok_emb.T -> [B, S, VOCAB_SIZE]
      logits <- torch_matmul(pred, model$tok_emb$weight$transpose(1, 2))

      # 展平计算带 Mask 的 Cross Entropy Loss
      logits_flat <- logits$reshape(c(-1, VOCAB_SIZE))
      y_flat      <- y_tensor$reshape(c(-1))
      mask_flat   <- mask_tensor$reshape(c(-1))

      # 仅计算 Mask 为 TRUE (Response 部分) 的 Token Loss
      raw_loss   <- nnf_cross_entropy(logits_flat, y_flat, reduction = "none")
      valid_loss <- raw_loss * mask_flat
      
      # 避免除 0
      denom <- mask_flat$sum()$clamp(min = 1.0)
      loss  <- valid_loss$sum() / denom
      
      loss
    }

    # Forward
    if (ENV_USE_AMP) {
      with_autocast(device_type = "cuda", {
        loss <- forward_sft_loss()
        loss_to_backprop <- loss / current_accum_steps
      })
    } else {
      loss <- forward_sft_loss()
      loss_to_backprop <- loss / current_accum_steps
    }

    # Backward
    if (ENV_USE_AMP) {
      scaler$scale(loss_to_backprop)$backward()
    } else {
      loss_to_backprop$backward()
    }

    # Step
    should_step <- (batch_idx %% accum_steps == 0 || batch_idx == batches_per_epoch)

    if (should_step) {
      global_step <- global_step + 1
      current_lr <- get_sft_lr(global_step, total_global_steps, SFT_LR)

      if (ENV_USE_AMP) {
        scaler$unscale_(optimizer)
        nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
        scaler$step(optimizer)
        scaler$update()
      } else {
        nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
        optimizer$step()
      }

      for (pg in optimizer$param_groups) {
        pg$lr <- current_lr
      }

      optimizer$zero_grad()
    }

    raw_loss_val <- loss$item()
    epoch_loss   <- epoch_loss + raw_loss_val

    if (batch_idx %% 20 == 0 || batch_idx == 1) {
      width <- nchar(as.character(batches_per_epoch))
      cat(sprintf(
        "Epoch [%d/%d] Batch [%*d/%d] step=%*d | SFT Loss=%.4f | lr=%.2e\n",
        epoch, EPOCHS, width, batch_idx, batches_per_epoch, width, global_step, raw_loss_val, optimizer$param_groups[[1]]$lr
      ))
    }
  })

  avg_loss <- epoch_loss / batch_idx
  cat(sprintf("=== Epoch %d 结束, 掩码平均 SFT Loss: %.4f ===\n", epoch, avg_loss))

  # 保存 SFT Checkpoint
  save_path <- sprintf("checkpoints/infonce_sft_%02d.pt", epoch)
  torch_save(
    list(
      model = model$state_dict(),
      optimizer = optimizer$state_dict(),
      epoch = epoch,
      global_step = global_step,
      loss = avg_loss,
      config = list(
        vocab_size = VOCAB_SIZE,
        dim = DIM,
        n_layers = N_LAYERS,
        n_heads = N_HEADS,
        seq_len = SEQ_LEN,
        batch_size = ENV_BATCH_SIZE
      )
    ),
    save_path
  )

  cat(sprintf("SFT Checkpoint saved: %s\n", save_path))
}

cat("\nSFT 微调成功完成！\n")