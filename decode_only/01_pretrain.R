# causal_lm/01_pretrain.R
# =====================================================================
# 预训练执行层 (优化 Dataloader 与梯度累加)
# =====================================================================
Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")

source("config.R")
source("utils/BPETokenizer.R")
source("decode_only/decode_only_model.R")

# 梯度累加有两个好处：
# 1. 可以突破显卡物理内存 (VRAM) 限制，时间换空间。
# 2. 梯度方向更准，整体收敛更稳

if (is_mac) {
  ENV_BATCH_SIZE  <- 2
  ENV_GRAD_ACCUM  <- 4   # 相当于模拟 Batch = 8
  ENV_USE_AMP     <- FALSE
  
} else {
  ENV_BATCH_SIZE  <- 64
  ENV_GRAD_ACCUM  <- 4   # 如果 = 4，Batch Size = 64，则相当于 256
  ENV_USE_AMP     <- TRUE
}

tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

# =====================================================================
# 纯内存切片 Dataloader (零磁盘 I/O 阻塞)
# =====================================================================
RtomicBinDataset <- dataset(
  name = "RtomicBinDataset",
  
  initialize = function(bin_file, seq_len = 256, vocab_size = 16384, bytes_per_token = 2) {
    self$seq_len <- seq_len
    file_info <- file.info(bin_file)
    total_bytes <- file_info$size
    total_tokens <- total_bytes / bytes_per_token
    
    cat(sprintf("正在将 %.2f M Tokens 全量载入内存...\n", total_tokens / 1e6))
    
    # 极速读取整个文件入内存
    con <- file(bin_file, "rb")
    raw_tokens <- readBin(con, what = "integer", n = total_tokens, 
                          size = bytes_per_token, signed = FALSE, endian = "little")
    close(con)
    
    # 构建全局 1D Tensor，后续全部依靠内存视图 (View/Narrow) 零拷贝切分
    self$data_tensor <- torch_tensor(raw_tokens, dtype = torch_long())
    self$num_batches <- floor((total_tokens - 1) / self$seq_len)
    
    cat(sprintf("数据加载完成！生成 %d 个训练块 (序列长度: %d)\n", self$num_batches, seq_len))
  },
  
  .getitem = function(i) {
    # 精确定位起点，提取 seq_len + 1 个 token 用于自回归错位
    start_idx <- (i - 1) * self$seq_len + 1
    
    # narrow 操作在内存中是连续的，几乎无耗时
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
  seq_len = SEQ_LEN,
  vocab_size = VOCAB_SIZE
)

train_dl <- dataloader(
  train_dataset,
  batch_size = ENV_BATCH_SIZE,
  shuffle = TRUE,
  drop_last = TRUE,
  num_workers = 0, # 数据全在内存，无需多线程打乱上下文
  pin_memory = !is_mac
)

# 按 Weight Decay 分组，原则是：
# 所有的二维矩阵（如 QKV 投影、FFN 线性层）应用 Weight Decay。
# 所有的 1D 向量（如 RMSNorm 的权重）以及 Token Embedding，不要应用 Weight Decay
custom_grouped_optim <- function(params, lr = 1e-3, weight_decay = 0.05, ...) {
  
  # 筛选出不需要 weight_decay 的参数：
  # 通常包含 RMSNorm 权重 (norm)、所有偏置项 (bias，如果有的话)、以及词向量 (tok_emb)
  no_decay_names <- grep("norm|bias|tok_emb", names(params), value = TRUE)
  decay_names <- setdiff(names(params), no_decay_names)
  
  # 打印一下分组信息，确保没有遗漏
  cat(sprintf("开启 Weight Decay (%.2f) 的参数组数量: %d\n", weight_decay, length(decay_names)))
  cat(sprintf("关闭 Weight Decay (0.0) 的参数组数量: %d\n", length(no_decay_names)))
  
  # 核心改变：所有层共享传入的 lr，只在 weight_decay 上做区分
  grouped_params <- list(
    list(params = params[decay_names], weight_decay = weight_decay, lr = lr),
    list(params = params[no_decay_names], weight_decay = 0.0, lr = lr)
  )
  
  optim_adamw(grouped_params, ...)
}

wsd_multiplier <- function(step, total_steps, warmup_pct = 0.1, decay_pct = 0.1) {
  current_step <- as.numeric(step) + 1 
  total_steps <- as.numeric(total_steps)
  warmup_steps <- total_steps * warmup_pct
  decay_steps <- total_steps * decay_pct
  stable_steps <- total_steps - warmup_steps - decay_steps
  
  if (current_step <= warmup_steps) return(current_step / warmup_steps)
  if (current_step <= warmup_steps + stable_steps) return(1.0)
  
  decay_step <- current_step - (warmup_steps + stable_steps)
  progress <- decay_step / decay_steps
  cosine_decay <- 0.5 * (1 + cos(pi * progress))
  return(max(0.1, cosine_decay))
}

TRAIN_EPOCHS <- 3
# 实际优化步数需要除以梯度累加的批次数
total_steps <- (TRAIN_EPOCHS * length(train_dl)) / ENV_GRAD_ACCUM
cat("\n总训练步数：", total_steps, "\n")

# =====================================================================
# 初始化模型 (激活 GQA)
# =====================================================================
# 假设 N_HEADS = 8，设置 N_KV_HEADS = 2 (每 4 个 Query 头共享 1 个 KV 头)
# 如果你需要 MHA，将参数设为等于 N_HEADS 即可。
N_KV_HEADS <- 2 
model <- RtomicCausalLM(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN, n_kv_heads = N_KV_HEADS)
cat('设备为：', device, '\n')
model$to(device = device)

# 学习率缩放法则
# 分布式训练或使用梯度累加扩大全局 Batch Size 时，必须同步放大初始学习率
# $LR_{new} = LR_{base} \times \sqrt{K}$
lr_rate <- sqrt(ENV_GRAD_ACCUM)
optimizer <- custom_grouped_optim(model$parameters, lr = lr_rate * 3e-4)
scheduler <- lr_lambda(optimizer, lr_lambda = function(step) wsd_multiplier(step, total_steps, 0.1, 0.2))

# 仅在使用 AMP 且为 CUDA 时启用 Scaler
use_scaler <- ENV_USE_AMP && cuda_is_available()
if (use_scaler) scaler <- cuda_amp_grad_scaler()

# 记录 loss 原始值
loss_log_file <- paste0("checkpoints/decode_loss_", format(Sys.time(), "%d%H%M"), ".csv")

global_step <- 0
model$train()
for (epoch in 1:TRAIN_EPOCHS) {
  
  iter_idx <- 0
  coro::loop(for (b in train_dl) {
    iter_idx <- iter_idx + 1
    
    # 开启 non_blocking = TRUE 异步传输
    batch_x <- b$x$x$to(device = device, non_blocking = TRUE)
    batch_y <- b$y$to(device = device, non_blocking = TRUE)
    
    # Mac (MPS) 下 use_scaler 为 FALSE，直接走原生前向传播，避开 with_autocast 的底层检查
    if (use_scaler) {
      with_autocast(device_type = "cuda", dtype = torch_bfloat16(), enabled = TRUE, {
        output <- model(list(x = batch_x, y = batch_y, loss_mask = NULL))
        loss <- output$loss / ENV_GRAD_ACCUM 
      })
    } else {
      # Mac/CPU 走这条分支
      output <- model(list(x = batch_x, y = batch_y, loss_mask = NULL))
      loss <- output$loss / ENV_GRAD_ACCUM 
    }
    
    # 2. 反向传播
    if (use_scaler) {
      scaler$scale(loss)$backward()
    } else {
      loss$backward()
    }
    
    # 3. 达到累加阈值，执行权重更新
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
      
      current_loss <- as.numeric(output$loss)
      cat(sprintf("%d,%d,%.6f\n", epoch, global_step, current_loss), 
          file = loss_log_file, append = TRUE)
      
      if (global_step %% 20 == 0) {
        cat(sprintf("Epoch: %d | Step: %d | Loss: %.4f\n", epoch, global_step, as.numeric(output$loss)))
      }
    }
  })
  
  torch_save(model$state_dict(), sprintf("checkpoints/decode_model_%02d.pt", epoch))
}
