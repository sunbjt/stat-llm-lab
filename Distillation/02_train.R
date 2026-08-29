Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")
library(torch)
library(coro)
library(future)
library(arrow)

# --- 兼容从 Distillation/ 子目录或仓库根目录运行 ---
if (basename(normalizePath(getwd())) == "Distillation") setwd("..")

source("config.R")
source("causal_lm/CausalLM_model.R")

# ==============================================================================
# 1. 基础配置与准备工作
# ==============================================================================
plan(multisession, workers = 2)

if (is_mac) {
  BATCH_SIZE  <- 2
  ENV_USE_AMP <- FALSE
  ACCUM_STEPS <- 1
} else {
  BATCH_SIZE  <- 64 
  ENV_USE_AMP <- TRUE
  ACCUM_STEPS <- 2
}

LR     <- 5e-4
DEVICE <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")

DATA_DIR    <- "data/processed/chunks" 
arrow_files <- list.files(DATA_DIR, pattern = "\\.(arrow|feather)$", full.names = TRUE)
arrow_files <- normalizePath(arrow_files)

if (length(arrow_files) == 0) {
  stop(sprintf("在目录 '%s' 下没有找到任何 Arrow 文件！请检查路径。", DATA_DIR))
}

# ==============================================================================
# 2. 数据读取与异步生成器定义
# ==============================================================================
load_raw_data_from_arrow <- function(file_path) {
  tb <- arrow::read_feather(file_path, as_data_frame = FALSE)
  n_rows <- tb$num_rows
  if (n_rows %% 512 != 0) {
    stop(sprintf("%s 行数 %d 不是 512 的整数倍,数据布局不符", file_path, n_rows))
  }
  n_samples <- as.integer(n_rows / 512)

  x_vec    <- as.integer(as.vector(tb$x))
  y_vec    <- as.integer(as.vector(tb$y_hard))
  mask_vec <- as.logical(as.vector(tb$loss_mask))

  ids_vec <- do.call(c, lapply(tb$topk_ids$chunks, function(c) unlist(as.vector(c), use.names = FALSE)))
  pr_vec  <- do.call(c, lapply(tb$topk_probs$chunks, function(c) unlist(as.vector(c), use.names = FALSE)))

  list(
    x          = matrix(x_vec, nrow = n_samples, ncol = 512, byrow = TRUE),
    y_hard     = matrix(y_vec, nrow = n_samples, ncol = 512, byrow = TRUE),
    loss_mask  = matrix(mask_vec, nrow = n_samples, ncol = 512, byrow = TRUE),
    topk_ids   = ids_vec,
    topk_probs = pr_vec,
    n_samples  = n_samples
  )
}

async_arrow_generator <- function(arrow_files, batch_size, seq_len = 512, top_k = 16) {
  generator(function() {
    num_files <- length(arrow_files)
    if (num_files == 0) return(NULL)
    
    next_future <- future({ load_raw_data_from_arrow(arrow_files[1]) }, seed = TRUE)
    
    for (i in seq_len(num_files)) {
      raw_data <- value(next_future)
      
      if (i < num_files) {
        next_file <- arrow_files[i + 1]
        next_future <- future({ load_raw_data_from_arrow(next_file) }, seed = TRUE)
      }
      
      n_samples    <- raw_data$n_samples
      x_t          <- torch_tensor(raw_data$x, dtype = torch_long())
      y_hard_t     <- torch_tensor(raw_data$y_hard, dtype = torch_long())
      mask_t       <- torch_tensor(raw_data$loss_mask, dtype = torch_bool())
      topk_ids_t   <- torch_tensor(raw_data$topk_ids, dtype = torch_long())$view(c(n_samples, seq_len, top_k))
      topk_probs_t <- torch_tensor(raw_data$topk_probs, dtype = torch_float32())$view(c(n_samples, seq_len, top_k))
      
      idx_seq <- seq(1, n_samples, by = batch_size)
      for (start_idx in idx_seq) {
        end_idx <- min(start_idx + batch_size - 1, n_samples)
        sl      <- start_idx:end_idx
        
        yield(list(
          x          = x_t[sl, ],
          y_hard     = y_hard_t[sl, ],
          loss_mask  = mask_t[sl, ],
          topk_ids   = topk_ids_t[sl, ..],
          topk_probs = topk_probs_t[sl, ..]
        ))
      }
    }
  })
}

# ==============================================================================
# 3. 蒸馏 Loss 函数定义 (修正维度 BUG、边界 protection 与 0.5% 全 0 掩码)
# ==============================================================================
distill_loss_fn <- function(alpha = 0.4) {
  function(student_logits, target) {
    dev <- DEVICE 
    # R torch 中 3 维张量 dim(3) 即为 Vocab Dim
    vocab_size <- student_logits$size(3) 
    
    y_hard       <- target$y_hard$to(device = dev)
    t_topk_ids   <- target$topk_ids$to(device = dev)
    t_topk_probs <- target$topk_probs$to(device = dev)
    loss_mask    <- target$loss_mask$to(device = dev)
    
    # 1. 越界保护 (Torch 索引按 1 开始，需保护范围 [1, vocab_size])
    invalid_mask    <- (t_topk_ids <= 0L) | (t_topk_ids > vocab_size)
    safe_topk_ids   <- torch_where(invalid_mask, torch_tensor(1L, device = dev), t_topk_ids)
    safe_topk_probs <- torch_where(invalid_mask, torch_tensor(0.0, device = dev), t_topk_probs)
    
    # 2. Log Softmax & Gather 提取学生 logits
    student_log_probs      <- nnf_log_softmax(student_logits, dim = 3)
    student_topk_log_probs <- torch_gather(student_log_probs, dim = 3, index = safe_topk_ids)

    # 3. KL Divergence (屏蔽 loss_mask=FALSE 以及 0.5% 概率全 0 的位置)
    loss_mask_float     <- loss_mask$to(dtype = torch_float32())
    prob_sums           <- safe_topk_probs$sum(dim = 3) # [batch, seq_len]
    valid_kl_mask       <- (prob_sums > 0.0)$to(dtype = torch_float32()) * loss_mask_float

    t_topk_probs_masked <- safe_topk_probs * valid_kl_mask$unsqueeze(3)
    token_kl            <- - torch_sum(t_topk_probs_masked * student_topk_log_probs, dim = 3)
    
    num_valid_kl_tokens <- valid_kl_mask$sum()$clamp(min = 1.0)
    kl_loss             <- (token_kl * valid_kl_mask)$sum() / num_valid_kl_tokens
    
    # 4. Hard Label Cross Entropy (更安全的向量化展开)
    logits_flat <- student_logits$view(c(-1, vocab_size))
    y_flat      <- y_hard$view(c(-1))
    mask_flat   <- loss_mask$view(c(-1))
    
    # 将被 mask 掉的位置用 -100 替代，nnf_cross_entropy 默认忽略 -100
    y_flat_masked <- torch_where(mask_flat, y_flat, torch_tensor(-100L, device = dev))
    ce_loss       <- nnf_cross_entropy(logits_flat, y_flat_masked, ignore_index = -100L)
    
    total <- alpha * kl_loss + (1 - alpha) * ce_loss
    list(total = total, kl = kl_loss, ce = ce_loss)
  }
}

# ==============================================================================
# 4. 训练主程序入口
# ==============================================================================
model <- RtomicCausalLM(
  vocab_size  = VOCAB_SIZE,
  dim         = DIM,
  n_layers    = N_LAYERS,
  n_heads     = N_HEADS,
  max_seq_len = SEQ_LEN
)
model <- model$to(device = DEVICE)

message(">>> 训练开始，设备: ", DEVICE$type)

IS_CUDA <- cuda_is_available()
USE_AMP <- IS_CUDA && ENV_USE_AMP
scaler  <- if (USE_AMP) { cuda_amp_grad_scaler() } else { NULL }
optimizer    <- optim_adamw(model$parameters, lr = LR)
compute_loss <- distill_loss_fn(alpha = 0.4)

EPOCHS <- 2
TOP_K  <- 16

for (epoch in seq_len(EPOCHS)) {

  message(sprintf("\n[Epoch %d/%d]", epoch, EPOCHS))
  model$train()

  data_stream <- async_arrow_generator(
    arrow_files,
    batch_size = BATCH_SIZE,
    seq_len = SEQ_LEN,
    top_k = TOP_K
  )

  batch_idx <- 0L
  optimizer$zero_grad()

  coro::loop(for (batch in data_stream()) {
    batch_idx <- batch_idx + 1L
    x_input <- batch$x$to(device = DEVICE)

    if (USE_AMP) {
      with_autocast(device_type = "cuda", {
        model_out <- model(list(x = x_input, y = NULL, loss_mask = NULL))
        logits <- model_out$logits
        loss_list <- compute_loss(logits, batch)
        loss <- loss_list$total / ACCUM_STEPS
      })
      scaler$scale(loss)$backward()
    } else {
      model_out <- model(list(x = x_input, y = NULL, loss_mask = NULL))
      logits <- model_out$logits
      loss_list <- compute_loss(logits, batch)
      loss <- loss_list$total / ACCUM_STEPS
      loss$backward()
    }

    # Gradient accumulation
    if (batch_idx %% ACCUM_STEPS == 0L) {
      if (USE_AMP) {
        scaler$unscale_(optimizer)
        nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
        scaler$step(optimizer)
        scaler$update()
      } else {
        nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
        optimizer$step()
      }

      optimizer$zero_grad()

      if ((batch_idx / ACCUM_STEPS) %% 10 == 0) {
        cat(sprintf(
          "Batch %d | Total Loss: %.4f (KL: %.4f, CE: %.4f)\n",
          batch_idx,
          loss_list$total$item(),
          loss_list$kl$item(),
          loss_list$ce$item()
        ))
      }
    }
  })

  # Epoch 结束时清空未满 ACCUM_STEPS 的残余梯度
  optimizer$zero_grad()

  message(sprintf(">>> Epoch %d/%d finished. Batches: %d", epoch, EPOCHS, batch_idx))

  if (!dir.exists("checkpoints")) dir.create("checkpoints", showWarnings = FALSE)
  ckpt_path <- sprintf("checkpoints/distill_model_%02d.pt", epoch)
  torch_save(
    list(
      model     = model$state_dict(),
      optimizer = optimizer$state_dict(),
      epoch     = epoch,
      batch_idx = batch_idx,
      config    = list(
        vocab_size  = VOCAB_SIZE,
        dim         = DIM,
        n_layers    = N_LAYERS,
        n_heads     = N_HEADS,
        max_seq_len = SEQ_LEN
      )
    ),
    ckpt_path
  )
  message(sprintf(">>> 已保存检查点: %s", ckpt_path))
}

message("\n>>> 训练完成！")