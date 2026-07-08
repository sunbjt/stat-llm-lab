# =====================================================================
# JEPA 预训练 (Train)
# =====================================================================
Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")
source("config.R")

## 环境专属超参 (覆盖默认值)
if (is_mac) {
  ENV_BATCH_SIZE  <- 2
  ENV_MAX_LINES   <- 1000
  ENV_USE_AMP     <- FALSE
  ENV_WORKERS     <- 0
} else {
  ENV_BATCH_SIZE  <- 128
  ENV_MAX_LINES   <- -1
  ENV_USE_AMP     <- TRUE
  ENV_WORKERS     <- 2
}

# =====================================================================
# 1. 内存友好型数据集构建
# =====================================================================
source("utils/BPETokenizer.R")
source("world_model/jepa_model.R")

tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

RtomicBinDataset <- dataset(
  name = "RtomicBinDataset",
  
  initialize = function(bin_file, seq_len = 256, vocab_size = 16384, bytes_per_token = 2) {
    self$bin_file <- bin_file
    self$seq_len <- seq_len
    self$vocab_size <- vocab_size
    self$bytes_per_token <- bytes_per_token
    
    file_info <- file.info(bin_file)
    total_bytes <- file_info$size
    total_tokens <- total_bytes / bytes_per_token
    
    self$chunk_size <- seq_len 
    self$num_batches <- floor(total_tokens / self$chunk_size)
    
    cat(sprintf("物理总 Token 数: %.2f M\n", total_tokens / 1e6))
    cat(sprintf("按 SEQ_LEN = %d 极速读取，总计生成 %d 个纯净训练块\n", 
                seq_len, self$num_batches))
  },
  
  .getitem = function(i) {
    start_token <- (i - 1) * self$chunk_size
    offset_bytes <- start_token * self$bytes_per_token
    
    con <- file(self$bin_file, "rb")
    seek(con, where = offset_bytes, origin = "start")
    chunk <- readBin(con, what = "integer", n = self$chunk_size, 
                     size = self$bytes_per_token, signed = FALSE, endian = "little")
    close(con)
    
    x_ids <- chunk[1:(self$seq_len - 1)]
    y_ids <- chunk[2:self$seq_len]
    
    x <- torch_tensor(x_ids, dtype = torch_long())
    y <- torch_tensor(y_ids, dtype = torch_long())
    
    list(x = list(x = x, y = y), y = y)
  },
  
  .length = function() {
    self$num_batches
  }
)

BIN_FILE <- "data/processed/zhwiki_tokens_16384.bin"

train_dataset <- RtomicBinDataset(
  bin_file = BIN_FILE,
  seq_len = SEQ_LEN,
  vocab_size = VOCAB_SIZE
)

train_dl <- dataloader(
  train_dataset,
  batch_size = ENV_BATCH_SIZE,
  shuffle = TRUE,
  drop_last = TRUE,
  num_workers = ENV_WORKERS,
  pin_memory = !is_mac
)

# =====================================================================
# 3. 训练循环配置与自定义优化器
# =====================================================================
custom_grouped_optim <- function(params, lr = 1e-3, weight_decay = 0.01, ...) {
  
  # 拦截并过滤掉 EMA 目标网络中那些冻结的参数
  active_idx <- sapply(params, function(p) p$requires_grad)
  active_params <- params[active_idx]

  # 后续分组逻辑只针对 active_params
  emb_names <- grep("^tok_emb|^codebook_weight", names(active_params), value = TRUE)
  other_names <- setdiff(names(active_params), emb_names)

  grouped_params <- list(
    list(params = active_params[emb_names], lr = 1e-4),
    list(params = active_params[other_names], lr = lr)
  )
  optim_adamw(grouped_params, weight_decay = weight_decay, ...)
}

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

luz_callback_simple_batch <- luz_callback(
  "simple_batch",
  initialize = function(filename = paste0("checkpoints/wm_loss_", format(Sys.time(), "%d%H%M"), ".csv")) {
    self$file <- filename
  },
  on_train_batch_end = function() {
    cat(ctx$epoch, ",", ctx$iter, ",", as.numeric(ctx$loss[[1]]), "\n", 
        file = self$file, append = TRUE)
  }
)

luz_callback_clip_grad <- luz_callback(
  "clip_grad",
  initialize = function(max_norm = 1.0) { self$max_norm <- max_norm },
  on_backward_end = function() {
    torch::nn_utils_clip_grad_norm_(ctx$model$parameters, max_norm = self$max_norm)
  }
)

# 【核心新增】：Target 网络指数滑动平均更新回调
luz_callback_ema_update <- luz_callback(
  "ema_update",
  initialize = function(tau = 0.99) { self$tau <- tau },
  
  on_fit_begin = function() {
    # 训练开始前，将 Target 网络权重完全对齐 Context 网络
    with_no_grad({
      ctx$model$target_layers$parameters %>% 
        purrr::walk2(ctx$model$layers$parameters, function(target_p, online_p) {
          target_p$copy_(online_p)
        })
      ctx$model$target_norm_f$weight$copy_(ctx$model$norm_f$weight)
    })
    cat("Target EMA 模型权重已完成初始同步\n")
  },
  
  on_train_batch_end = function() {
    # 每一个 batch 结束后，平滑更新 Target 权重
    with_no_grad({
      ctx$model$target_layers$parameters %>% 
        purrr::walk2(ctx$model$layers$parameters, function(target_p, online_p) {
          target_p$mul_(self$tau)$add_(online_p * (1 - self$tau))
        })
      
      ctx$model$target_norm_f$weight$mul_(self$tau)$add_(ctx$model$norm_f$weight * (1 - self$tau))
    })
  }
)

TRAIN_EPOCHS <- 3
total_steps <- TRAIN_EPOCHS * length(train_dl)

base_callbacks <- list(
  luz_callback_lr_scheduler(
    torch::lr_lambda,
    lr_lambda = function(step) {
      wsd_multiplier(step, total_steps = total_steps, warmup_pct = 0.1, decay_pct = 0.2)
    },
    call_on = "on_train_batch_end"
  ),
  luz_callback_simple_batch(), 
  luz_callback_clip_grad(1.0),
  luz_callback_ema_update(tau = 0.99), # 挂载 EMA 更新器
  luz_callback_model_checkpoint(
    path = "checkpoints/wm_{epoch:02d}.pt", save_best_only = FALSE, monitor = "train_loss"
  )
)

if (ENV_USE_AMP && cuda_is_available()) {
  base_callbacks <- append(base_callbacks, list(luz_callback_mixed_precision()))
  cat("已挂载混合精度训练 (AMP) 回调\n")
}
options(luz.force_progress_bar = TRUE) # 在 positron 下也能够显示进度条

fitted_jepa <- RtomicJEPA_VQ |>
  setup(
    loss = function(output, target) output$loss,
    optimizer = custom_grouped_optim
  ) |>
  set_hparams(
    vocab_size = VOCAB_SIZE, dim = DIM, n_layers = N_LAYERS, 
    n_heads = N_HEADS, max_seq_len = SEQ_LEN, num_clusters = 4096
  ) |>
  set_opt_hparams(lr = 2e-3, weight_decay = 0.01) |>
  fit(
    data = train_dl,
    epochs = TRAIN_EPOCHS,
    accelerator = accelerator(),
    callbacks = base_callbacks,
    verbose = TRUE
  )
