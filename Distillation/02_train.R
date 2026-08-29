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
# 分配 1 个主线程 + 1 个后台预取 Worker 进程
plan(multisession, workers = 2)

# 设置超参数
## 1. 环境专属超参配置
if (is_mac) {
  BATCH_SIZE  <- 2
  ENV_USE_AMP <- FALSE
  ACCUM_STEPS <- 1
} else {
  BATCH_SIZE  <- 64  # 15M 小模型显存开销极小，RTX 3090 可轻松开启 32~64
  ENV_USE_AMP <- TRUE
  ACCUM_STEPS <- 2
}

LR          <- 5e-4
DEVICE      <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")

# 获取 Arrow 文件路径列表 (请修改为你的实际目录)
DATA_DIR    <- "data/processed/chunks" 
arrow_files <- list.files(DATA_DIR, pattern = "\\.(arrow|feather)$", full.names = TRUE)
arrow_files <- normalizePath(arrow_files) # 必须转为绝对路径，防止子进程找不到文件

if (length(arrow_files) == 0) {
  stop(sprintf("在目录 '%s' 下没有找到任何 Arrow 文件！请检查路径。", DATA_DIR))
}

# ==============================================================================
# 2. 数据读取与异步生成器定义
# ==============================================================================
load_raw_data_from_arrow <- function(file_path) {
  # 【性能关键】as_data_frame=FALSE:只 mmap 不物化,避免把千万行 list 列转成 R data.frame
  # (as_data_frame=TRUE 对 1600 万行的 chunk 要几分钟,这里亚秒级)
  tb <- arrow::read_feather(file_path, as_data_frame = FALSE)
  n_rows <- tb$num_rows
  if (n_rows %% 512 != 0) {
    stop(sprintf("%s 行数 %d 不是 512 的整数倍,数据布局不符", file_path, n_rows))
  }
  n_samples <- as.integer(n_rows / 512)   # 512 = 序列长度

  x_vec    <- as.integer(as.vector(tb$x))
  y_vec    <- as.integer(as.vector(tb$y_hard))
  mask_vec <- as.logical(as.vector(tb$loss_mask))

  # fixed_size_list 列:逐 chunk 展开比整体 unlist 快 ~2x
  # (arrow 25.x 的 FixedSizeListArray$values 绑定缺失不可用,否则可再快 ~50 倍)
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

# 异步双缓冲区 Generator (补充 x 的切片转换)
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
# 3. 蒸馏 Loss 函数定义
# ==============================================================================
distill_loss_fn <- function(alpha = 0.4) {
  function(student_logits, target) {
    # Use global DEVICE instead of target$y_hard$device to avoid closure evaluation errors
    dev <- DEVICE 
    vocab_size <- student_logits$size(3)
    
    # Send target tensors to target device
    y_hard       <- target$y_hard$to(device = dev)
    t_topk_ids   <- target$topk_ids$to(device = dev)
    t_topk_probs <- target$topk_probs$to(device = dev)
    loss_mask    <- target$loss_mask$to(device = dev)
    
    # 1. Index boundary protection
    invalid_mask    <- (t_topk_ids <= 0L) | (t_topk_ids > vocab_size)
    safe_topk_ids   <- torch_where(invalid_mask, torch_tensor(1L, device = dev), t_topk_ids)
    safe_topk_probs <- torch_where(invalid_mask, torch_tensor(0.0, device = dev), t_topk_probs)
    
    # 2. Log Softmax & Gather（T=1，与教师端落盘温度一致；不再除温度）
    student_log_probs      <- nnf_log_softmax(student_logits, dim = 3)
    student_topk_log_probs <- torch_gather(student_log_probs, dim = 3, index = safe_topk_ids)

    # 3. KL Divergence（截断 Top-K；教师概率已 normalize，直接使用，不再乘 T^2）
    loss_mask_float     <- loss_mask$to(dtype = torch_float32())
    t_topk_probs_masked <- safe_topk_probs * loss_mask_float$unsqueeze(3)

    token_kl <- - torch_sum(t_topk_probs_masked * student_topk_log_probs, dim = 3)
    num_valid_tokens <- loss_mask_float$sum()$clamp(min = 1.0)
    kl_loss <- (token_kl * loss_mask_float)$sum() / num_valid_tokens
    
    # 4. Hard Label Cross Entropy
    logits_flat   <- student_logits$view(c(-1, vocab_size))
    y_flat        <- y_hard$view(c(-1))
    valid_indices <- loss_mask$view(c(-1))$nonzero()$squeeze(2)
    ce_loss       <- nnf_cross_entropy(logits_flat[valid_indices, ], y_flat[valid_indices])
    
    total <- alpha * kl_loss + (1 - alpha) * ce_loss
    list(total = total, kl = kl_loss, ce = ce_loss)
  }
}

# ==============================================================================
# 4. 训练主程序入口 (直接执行)
# ==============================================================================

model <- RtomicCausalLM(
  vocab_size  = VOCAB_SIZE,
  dim         = DIM,
  n_layers    = N_LAYERS,
  n_heads     = N_HEADS,
  max_seq_len = SEQ_LEN
)
# 【修正】：改为大写的 DEVICE
model <- model$to(device = DEVICE)

message(">>> 训练开始，设备: ", DEVICE$type)

IS_CUDA <- cuda_is_available()
USE_AMP <- IS_CUDA && ENV_USE_AMP
scaler <- if (USE_AMP) { cuda_amp_grad_scaler() } else { NULL }
optimizer    <- optim_adamw(model$parameters, lr = LR)
# alpha 从 0.7 降到 0.4：教师软标签与真实续写错位时，CE 需要更强地锚定真实 token
compute_loss <- distill_loss_fn(alpha = 0.4)


EPOCHS = 2
# 与 01a_teacher_logits.py / 01b_student_project.py 的 --top-k 保持一致
# （默认 16；A/B 两阶段生成时必须相同，01b 投影后的 topk 列数即此值）
TOP_K = 16

for (epoch in seq_len(EPOCHS)) {

  message(sprintf("\n[Epoch %d/%d]", epoch, EPOCHS))
  model$train()

  # Create data stream
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
        # forward 期望 input_data 为 list(x, y, loss_mask)，返回 list(logits=...)
        model_out <- model(list(x = x_input, y = NULL, loss_mask = NULL))
        logits <- model_out$logits
        loss_list <- compute_loss(
          logits,
          batch
        )
        loss <- loss_list$total / ACCUM_STEPS
      })
      scaler$scale(loss)$backward()
    } else {
      model_out <- model(list(x = x_input, y = NULL, loss_mask = NULL))
      logits <- model_out$logits
      loss_list <- compute_loss(
        logits,
        batch
      )
      loss <- loss_list$total / ACCUM_STEPS
      loss$backward()
    }

    # Gradient accumulation
    if (batch_idx %% ACCUM_STEPS == 0L) {
      # CUDA + AMP optimizer step
      if (USE_AMP) {
        # Unscale before gradient clipping
        scaler$unscale_(optimizer)
        nn_utils_clip_grad_norm_(
          model$parameters,
          max_norm = 1.0
        )
        scaler$step(optimizer)
        scaler$update()
      } else {
        nn_utils_clip_grad_norm_(
          model$parameters,
          max_norm = 1.0
        )
        optimizer$step()
      }

      optimizer$zero_grad()
      # Logging
      optimizer_step_idx <- batch_idx / ACCUM_STEPS

      if (optimizer_step_idx %% 10 == 0) {

        total_loss_value <- loss_list$total$item()
        kl_loss_value    <- loss_list$kl$item()
        ce_loss_value    <- loss_list$ce$item()

        cat(sprintf(
          "Batch %d | Total Loss: %.4f (KL: %.4f, CE: %.4f)\n",
          batch_idx,
          total_loss_value,
          kl_loss_value,
          ce_loss_value
        ))
      }
    }
  })
  message(sprintf(
    ">>> Epoch %d/%d finished. Batches: %d",
    epoch, EPOCHS, batch_idx
  ))

  # 保存本 epoch 检查点（与 03_inference.R 的加载约定一致）
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