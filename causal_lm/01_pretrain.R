# causal_lm/01_pretrain.R
Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")

source("config.R")
source("utils/BPETokenizer.R")
source("causal_lm/CausalLM_model.R")

## 环境专属超参
if (is_mac) {
  ENV_BATCH_SIZE  <- 2
  ENV_MAX_LINES   <- 1000
  ENV_USE_AMP     <- FALSE
} else {
  ENV_BATCH_SIZE  <- 64
  ENV_MAX_LINES   <- -1
  ENV_USE_AMP     <- TRUE
}

tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

# =====================================================================
# 完美对齐版 Dataloader (替换原有的 RtomicBinDataset)
# =====================================================================
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
    
    # 【核心改变 1】：精准切块。因为数据已完美对齐，不需要再强行多读 1 个 Token
    self$chunk_size <- seq_len 
    self$num_batches <- floor(total_tokens / self$chunk_size)
    
    cat(sprintf("物理总 Token 数: %.2f M\n", total_tokens / 1e6))
    cat(sprintf("按 SEQ_LEN = %d 极速读取，总计生成 %d 个纯净训练块\n", 
                seq_len, self$num_batches))
  },
  
  .getitem = function(i) {
    # 严格计算底层字节偏移量
    start_token <- (i - 1) * self$chunk_size
    offset_bytes <- start_token * self$bytes_per_token
    
    con <- file(self$bin_file, "rb")
    seek(con, where = offset_bytes, origin = "start")
    chunk <- readBin(con, what = "integer", n = self$chunk_size, 
                     size = self$bytes_per_token, signed = FALSE, endian = "little")
    close(con)
    
    # 【核心改变 2】：删除了 chunk[chunk > self$vocab_size] <- 2L 的坏桩过滤
    # 因为对齐后的文件，底层字节绝不可能再被切断！
    
    # 【核心改变 3】：在内存中优雅错位 (Shift)，生成自回归因果标签
    # x 取第 1 到倒数第 2 个 Token
    x_ids <- chunk[1:(self$seq_len - 1)]
    # y 取第 2 到最后一个 Token
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
  num_workers = 0,
  pin_memory = !is_mac
)

custom_grouped_optim <- function(params,
                                 lr = 1e-3,
                                 weight_decay = 0.05,
                                 ...) {
  emb_names <- grep("^tok_emb|^pos_emb|^lm_head", names(params), value = TRUE)
  other_names <- setdiff(names(params), emb_names)
  grouped_params <- list(list(params = params[emb_names], lr = 1e-4),
                         list(params = params[other_names], lr = lr))
  optim_adamw(grouped_params, weight_decay = weight_decay, ...)
}

## WSD (Warmup - Stable - Decay) 学习率乘子计算器
wsd_multiplier <- function(step, total_steps, warmup_pct = 0.1, decay_pct = 0.1) {
  # 强制类型转换，防止除法产生浮点数误差
  current_step <- as.numeric(step) + 1 
  total_steps <- as.numeric(total_steps)
  
  warmup_steps <- total_steps * warmup_pct
  decay_steps <- total_steps * decay_pct
  stable_steps <- total_steps - warmup_steps - decay_steps
  
  if (current_step <= warmup_steps) {
    # 阶段 1: 线性 Warmup (从 0 爬升到 100% 火力)
    return(current_step / warmup_steps)
    
  } else if (current_step <= warmup_steps + stable_steps) {
    # 阶段 2: Stable (全程保持 100% 满血火力)
    return(1.0)
    
  } else {
    # 阶段 3: 余弦 Decay (从 100% 平滑衰减到 10% 保底)
    decay_step <- current_step - (warmup_steps + stable_steps)
    progress <- decay_step / decay_steps
    
    # 余弦退火公式：平滑且优雅的着陆
    cosine_decay <- 0.5 * (1 + cos(pi * progress))
    
    # 设定保底乘子 0.1，防止学习率彻底变成 0 导致后期死寂
    return(max(0.1, cosine_decay))
  }
}

TRAIN_EPOCHS <- 3
total_steps <- TRAIN_EPOCHS * length(train_dl)

## 极简版 Batch Loss 记录器
luz_callback_simple_batch <- luz_callback(
  "simple_batch",
  initialize = function(
    filename = paste0("checkpoints/causal_loss_", format(Sys.time(), "%d%H%M"), ".csv")
    ) {
    self$file <- filename
  },
  on_train_batch_end = function() {
    cat(ctx$epoch, ",", ctx$iter, ",", as.numeric(ctx$loss[[1]]), "\n", 
        file = self$file, append = TRUE)
  }
)

## 梯度裁剪 (Gradient Clipping) 回调器
luz_callback_clip_grad <- luz_callback(
  "clip_grad",
  initialize = function(max_norm = 1.0) {
    self$max_norm <- max_norm
  },
  on_backward_end = function() {
    # ctx$model$parameters 会自动获取当前网络内所有需要求导的权重
    # 强制在 backward 之后、optimizer$step 之前将异常梯度剪碎
    torch::nn_utils_clip_grad_norm_(ctx$model$parameters, max_norm = self$max_norm)
  }
)

base_callbacks <- list(
  luz_callback_lr_scheduler(
    torch::lr_lambda,
    lr_lambda = function(step) {
      wsd_multiplier(step, total_steps = total_steps, warmup_pct = 0.1, decay_pct = 0.2)
    },
    call_on = "on_train_batch_end" # 核心：强制精确到每一步更新
  ),
  luz_callback_simple_batch(), # 挂载极简 Batch 记录器
  luz_callback_clip_grad(1.0),
  luz_callback_model_checkpoint(
    path = "checkpoints/causal_model_{epoch:02d}.pt", save_best_only = FALSE, monitor = "train_loss"
    )
)

if (ENV_USE_AMP && cuda_is_available()) {
  base_callbacks <- append(base_callbacks, list(luz_callback_mixed_precision()))
}

## 开始训练纯 Causal LM 模型
Sys.time()  # 显示开始时间，也可以看 loss 文件名
fitted_causal_lm <- RtomicCausalLM |>
  setup(
    loss = function(output, target)
      output$loss,
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
    # 直接 TRAIN_EPOCHS 轮训练得到最好的结果，如果不行可以接着下个环节断点续训
    epochs = TRAIN_EPOCHS,
    accelerator = accelerator(),
    callbacks = base_callbacks,
    verbose = TRUE
  )

# =====================================================================
# 如果你发现你的训练还没有达到合理的 loss，但 epoch 已经结束可以加载权重接着训练
# 第二阶段：平滑退火预训练 (Stage 2)
# =====================================================================

# 1. 挂载已有的最优权重
STAGE1_CKPT <- "checkpoints/causal_model_03.pt"

luz_callback_warm_start <- luz_callback(
  name = "warm_start",
  initialize = function(checkpoint_path) {
    self$checkpoint_path <- checkpoint_path
  },
  on_fit_begin = function() {
    ckpt <- torch_load(self$checkpoint_path)
    # 兼容 luz 的状态字典提取
    state <- if (!is.null(ckpt$model)) ckpt$model else ckpt
    ctx$model$load_state_dict(state)
    cat(sprintf("\n [热启动] 已成功注入第一阶段参数：%s\n", self$checkpoint_path))
  }
)

# 2. 重新配置更温和的调度器 (Cosine Annealing)
stage2_epochs <- 3
stage2_total_steps <- stage2_epochs * length(train_dl)

stage2_callbacks <- list(
  luz_callback_warm_start(STAGE1_CKPT),
  
  # 使用余弦退火，最高学习率压低到 1e-4，并缓慢衰减到底部 1e-5
  luz_callback_lr_scheduler(
    lr_cosine_annealing, 
    T_max = stage2_total_steps, 
    eta_min = 1e-5
  ),
  
  luz_callback_model_checkpoint(
    path = "checkpoints/causal_model_stage2_{epoch:02d}.pt",
    save_best_only = FALSE, 
    monitor = "train_loss"
  ),
  
  luz_callback_memory_cleaner() # 沿用之前的 GC 清理器
)

if (ENV_USE_AMP && cuda_is_available()) {
  stage2_callbacks <- append(stage2_callbacks, list(luz_callback_mixed_precision()))
}

cat("\n启动 Stage 2 预训练退火...\n")

fitted_stage2 <- RtomicCausalLM |>
  setup(
    loss = function(output, target) output$loss,
    optimizer = custom_grouped_optim
  ) |>
  set_hparams(
    vocab_size = VOCAB_SIZE, dim = DIM, n_layers = N_LAYERS, n_heads = N_HEADS, max_seq_len = SEQ_LEN
  ) |>
  # 初始基础 LR 也相应压低
  set_opt_hparams(lr = 1e-4, weight_decay = 0.05) |> 
  fit(
    data = train_dl,
    epochs = stage2_epochs,
    accelerator = accelerator(),
    callbacks = stage2_callbacks,
    verbose = TRUE
  )
