# causal_lm/02_distill.R
Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")

library(torch)
library(luz)
library(arrow)

source("config.R")
source("causal_lm/CausalLM_model.R")

## 1. 环境专属超参配置
if (is_mac) {
  ENV_BATCH_SIZE  <- 2
  ENV_USE_AMP     <- FALSE
} else {
  ENV_BATCH_SIZE  <- 32  # 15M 小模型显存开销极小，RTX 3090 可轻松开启 32~64
  ENV_USE_AMP     <- TRUE
}

# =====================================================================
# 2. 零 CPU 耗时内存 Dataset
# =====================================================================
RtomicLazyChunksDataset <- dataset(
  name = "RtomicLazyChunksDataset",
  
  initialize = function(chunk_dir) {
    self$chunk_dir <- chunk_dir
    # 改为检索 Python 生成的 .pt 文件
    self$chunk_files <- sort(list.files(chunk_dir, pattern = "^chunk_.*\\.pt$", full.names = TRUE))
    
    message("正在扫描多 Chunk 二进制 Tensor 索引...")
    self$total_rows <- 0
    self$lens <- c()
    
    for (f in self$chunk_files) {
      # 快速加载 .pt 文件获取行数 (Tensor加载非常快)
      tmp <- torch_load(f)
      c_len <- tmp$x$size(1) 
      self$lens <- c(self$lens, c_len)
      self$total_rows <- self$total_rows + c_len
    }
    
    self$current_chunk_idx <- -1
    self$current_chunk_data <- NULL
    
    message(sprintf("二进制索引加载完成！共管理 %d 个 Chunk，总样本数: %d", length(self$chunk_files), self$total_rows))
  },
  
  .getitem = function(i) {
    accum <- 0
    target_chunk <- -1
    local_idx <- -1
    
    for (c_idx in seq_along(self$lens)) {
      if (i <= accum + self$lens[c_idx]) {
        target_chunk <- c_idx
        local_idx <- i - accum
        break
      }
      accum <- accum + self$lens[c_idx]
    }
    
    # 如果跨 Chunk，加载新的 .pt 块，旧块会被 R 的垃圾回收自动清理
    if (self$current_chunk_idx != target_chunk) {
      self$current_chunk_idx <- target_chunk
      self$current_chunk_data <- torch_load(self$chunk_files[target_chunk])
    }
    
    cd <- self$current_chunk_data
    
    # 【核心优化】由于已经是 Torch Tensor，直接行切片即可。
    # R torch 底层会自动创建零拷贝视图，速度瞬间拉满，且不会引发内存膨胀。
    list(
      x = list(x = cd$x[local_idx, ]),
      y = list(
        y_hard     = cd$y_hard[local_idx, ],
        topk_ids   = cd$topk_ids[local_idx, , ],
        topk_probs = cd$topk_probs[local_idx, , ],
        loss_mask  = cd$loss_mask[local_idx, ]
      )
    )
  },
  
  .length = function() {
    self$total_rows
  }
)

# 训练脚本中的 DataLoader 组装
train_dataset <- RtomicLazyChunksDataset(chunk_dir = "data/processed/chunks/", chunk_size = 2000)

train_dl <- dataloader(
  train_dataset,
  batch_size  = ENV_BATCH_SIZE,
  shuffle     = TRUE,
  drop_last   = TRUE,
  num_workers = 0,
  pin_memory  = !is_mac
)

# =====================================================================
# 3. 带 Loss Mask 的 Top-K 蒸馏混合 Loss 函数 (原生 R torch 兼容版)
# =====================================================================
distill_loss_fn <- function(temperature = 1.0, alpha = 0.8) {
  function(output, target) {
    student_logits <- output$logits             # Shape: [B, S, Vocab]
    y_hard         <- target$y_hard             # Shape: [B, S]
    t_topk_ids     <- target$topk_ids           # Shape: [B, S, K]
    t_topk_probs   <- target$topk_probs         # Shape: [B, S, K]
    loss_mask      <- target$loss_mask          # Shape: [B, S] (torch_bool)
    
    # R 1-based 索引：第 3 维为 Vocab
    vocab_size <- student_logits$size(3)
    dev        <- student_logits$device
    
    # -------------------------------------------------------------------
    # 1. 索引边界安全钳位与 int64 类型对齐
    # -------------------------------------------------------------------
    t_topk_ids_long <- t_topk_ids$to(device = dev, dtype = torch_long())
    
    # 检查索引是否超出范围 [0, vocab_size - 1]
    invalid_mask <- (t_topk_ids_long < 0L) | (t_topk_ids_long >= vocab_size)
    
    # 替换非法索引为 0L（必须显式声明 dtype = torch_long()）
    safe_topk_ids <- torch_where(
      invalid_mask,
      torch_tensor(0L, device = dev, dtype = torch_long()),
      t_topk_ids_long
    )
    
    # 非法位置的教师概率置 0
    safe_topk_probs <- torch_where(
      invalid_mask,
      torch_tensor(0.0, device = dev, dtype = torch_float32()),
      t_topk_probs$to(device = dev, dtype = torch_float32())
    )

    # -------------------------------------------------------------------
    # 2. 掩码计算与 Log Softmax (显式使用 R 1-based dim = 3)
    # -------------------------------------------------------------------
    loss_mask_float <- loss_mask$to(device = dev, dtype = torch_float32())
    loss_mask_3d    <- loss_mask_float$unsqueeze(3) # 扩展至 [B, S, 1]
    t_topk_probs_masked <- safe_topk_probs * loss_mask_3d

    # 对 Student 词表维 (dim = 3) 做 Log Softmax
    student_log_probs <- nnf_log_softmax(student_logits / temperature, dim = 3)
    student_log_probs <- torch_clamp(student_log_probs, min = -100.0, max = 0.0)

    # -------------------------------------------------------------------
    # 3. 提取 Top-K 位置 log_probs (明确指定 dim = 3)
    # -------------------------------------------------------------------
    student_topk_log_probs <- torch_gather(
      student_log_probs,
      dim = 3,  # 必须使用 3 指向 Vocab 维，防止 R 转换 -1 错位
      index = safe_topk_ids
    )
    student_topk_log_probs <- torch_clamp(student_topk_log_probs, min = -100.0, max = 0.0)

    # -------------------------------------------------------------------
    # 4. KL 散度与损失融合
    # -------------------------------------------------------------------
    elem_kl <- t_topk_probs_masked * student_topk_log_probs
    
    # 熔断异常值
    bad_mask <- torch_isnan(elem_kl) | torch_isinf(elem_kl)
    if (bad_mask$any()$item()) {
      elem_kl <- torch_where(
        bad_mask, 
        torch_tensor(0.0, device = dev, dtype = elem_kl$dtype), 
        elem_kl
      )
    }

    # 对 Top-K 维 (dim = 3) 求和，得到 [B, S]
    token_kl <- - torch_sum(elem_kl, dim = 3)

    num_valid_tokens <- loss_mask_float$sum()$clamp(min = 1.0)
    kl_loss <- (token_kl * loss_mask_float)$sum() / num_valid_tokens
    kl_loss <- kl_loss * (temperature ^ 2)

    # 硬标签 Cross Entropy Loss
    logits_flat <- student_logits$view(c(-1, vocab_size))
    y_flat      <- y_hard$view(c(-1))
    mask_flat   <- loss_mask$view(c(-1))

    ce_loss     <- nnf_cross_entropy(logits_flat[mask_flat, ], y_flat[mask_flat])

    alpha * kl_loss + (1 - alpha) * ce_loss
  }
}

# =====================================================================
# 4. 优化器与 WSD 学习率调度器
# =====================================================================
custom_grouped_optim <- function(params, lr = 5e-4, weight_decay = 0.05, ...) {
  emb_names <- grep("^tok_emb|^pos_emb|^lm_head", names(params), value = TRUE)
  other_names <- setdiff(names(params), emb_names)
  grouped_params <- list(
    list(params = params[emb_names], lr = lr * 0.5),
    list(params = params[other_names], lr = lr)
  )
  optim_adamw(grouped_params, weight_decay = weight_decay, ...)
}

wsd_multiplier <- function(step, total_steps, warmup_pct = 0.05, decay_pct = 0.15) {
  current_step <- as.numeric(step) + 1 
  total_steps  <- as.numeric(total_steps)
  
  warmup_steps <- total_steps * warmup_pct
  decay_steps  <- total_steps * decay_pct
  stable_steps <- total_steps - warmup_steps - decay_steps
  
  if (current_step <= warmup_steps) {
    return(current_step / warmup_steps)
  } else if (current_step <= warmup_steps + stable_steps) {
    return(1.0)
  } else {
    decay_step   <- current_step - (warmup_steps + stable_steps)
    progress     <- decay_step / decay_steps
    cosine_decay <- 0.5 * (1 + cos(pi * progress))
    return(max(0.1, cosine_decay))
  }
}

DISTILL_EPOCHS <- 3
total_steps    <- DISTILL_EPOCHS * length(train_dl)

luz_callback_clip_grad <- luz_callback(
  "clip_grad",
  initialize = function(max_norm = 1.0) {
    self$max_norm <- max_norm
  },
  on_backward_end = function() {
    torch::nn_utils_clip_grad_norm_(ctx$model$parameters, max_norm = self$max_norm)
  }
)

base_callbacks <- list(
  luz_callback_lr_scheduler(
    torch::lr_lambda,
    lr_lambda = function(step) {
      wsd_multiplier(step, total_steps = total_steps, warmup_pct = 0.05, decay_pct = 0.15)
    },
    call_on = "on_train_batch_end"
  ),
  luz_callback_clip_grad(1.0),
  luz_callback_model_checkpoint(
    path = "checkpoints/distill_model_{epoch:02d}.pt", 
    save_best_only = FALSE, 
    monitor = "train_loss"
  )
)

if (ENV_USE_AMP && cuda_is_available()) {
  base_callbacks <- append(base_callbacks, list(luz_callback_mixed_precision()))
}

cat(sprintf("\n启动 15M 学生模型从头蒸馏训练 (Total Steps: %d)...\n", total_steps))

fitted_distill_lm <- RtomicCausalLM |>
  setup(
    loss = distill_loss_fn(temperature = 1.0, alpha = 0.8),
    optimizer = custom_grouped_optim
  ) |>
  set_hparams(
    vocab_size = VOCAB_SIZE,
    dim = DIM,
    n_layers = N_LAYERS,
    n_heads = N_HEADS,
    max_seq_len = SEQ_LEN
  ) |>
  set_opt_hparams(lr = 3e-4, weight_decay = 0.05) |>
  fit(
    data = train_dl,
    epochs = DISTILL_EPOCHS,
    accelerator = accelerator(),
    callbacks = base_callbacks,
    verbose = TRUE
  )
