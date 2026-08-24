# causal_lm/01_pretrain.R
# =====================================================================
# 预训练执行层 (优化 Dataloader 与梯度累加)
# =====================================================================
Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")

source("config.R")
source("utils/BPETokenizer.R")
source("moe/moe_model.R")

# 梯度累加有两个好处：
# 1. 可以突破显卡物理内存 (VRAM) 限制，时间换空间。
# 2. 梯度方向更准，整体收敛更稳

if (is_mac) {
  ENV_BATCH_SIZE  <- 2
  ENV_GRAD_ACCUM  <- 4   # 相当于模拟 Batch = 8
  ENV_USE_AMP     <- FALSE
  
} else {
  ENV_BATCH_SIZE  <- 32
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
    
    # 构建全局 1D Tensor
    self$data_tensor <- torch_tensor(raw_tokens, dtype = torch_long())
    
    # 【核心防御 2】：极速销毁 R 侧的 400MB 原始向量，防止 C++ 转换后产生内存滞留
    rm(raw_tokens)
    gc(verbose = FALSE)
    
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
  num_workers = 0, 
  pin_memory = !is_mac
)

# 按 Weight Decay 分组
custom_grouped_optim <- function(params, lr = 1e-3, weight_decay = 0.05, ...) {
  no_decay_names <- grep("norm|bias|tok_emb", names(params), value = TRUE)
  decay_names <- setdiff(names(params), no_decay_names)
  
  cat(sprintf("开启 Weight Decay (%.2f) 的参数组数量: %d\n", weight_decay, length(decay_names)))
  cat(sprintf("关闭 Weight Decay (0.0) 的参数组数量: %d\n", length(no_decay_names)))
  
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

TRAIN_EPOCHS <- 2
total_steps <- (TRAIN_EPOCHS * length(train_dl)) / ENV_GRAD_ACCUM
cat("\n总训练步数：", total_steps, "\n")

# =====================================================================
# 初始化模型 (激活 GQA + MoE)
# =====================================================================
N_KV_HEADS <- 2 
NUM_EXPERTS <- 4
TOP_K <- 1

# 调用主模型 RtomicCausalLM，参数名一定要对齐！
model <- RtomicCausalLM(
  VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN, 
  n_kv_heads = N_KV_HEADS,
  num_experts = NUM_EXPERTS,
  top_k = TOP_K
)

# 打印一下真实的参数量，证明它并没有 18GB 那么夸张
total_params <- sum(sapply(model$parameters, function(p) p$numel()))
cat(sprintf("MoE 模型真实总参数量: %.2f M (浮点大小约 %.1f MB)\n", total_params / 1e6, (total_params * 4) / 1024 / 1024))

model$to(device = device)

lr_rate <- sqrt(ENV_GRAD_ACCUM)
optimizer <- custom_grouped_optim(model$parameters, lr = lr_rate * 3e-4)
scheduler <- lr_lambda(optimizer, lr_lambda = function(step) wsd_multiplier(step, total_steps, 0.1, 0.2))

use_scaler <- ENV_USE_AMP && cuda_is_available()
if (use_scaler) scaler <- cuda_amp_grad_scaler()

loss_log_file <- paste0("checkpoints/moe_loss_", format(Sys.time(), "%d%H%M"), ".csv")

global_step <- 0
model$train()
for (epoch in 1:TRAIN_EPOCHS) {
  
  iter_idx <- 0
  coro::loop(for (b in train_dl) {
    iter_idx <- iter_idx + 1
    
    batch_x <- b$x$x$to(device = device, non_blocking = TRUE)
    batch_y <- b$y$to(device = device, non_blocking = TRUE)
    
    if (use_scaler) {
      with_autocast(device_type = "cuda", dtype = torch_bfloat16(), enabled = TRUE, {
        output <- model(list(x = batch_x, y = batch_y, loss_mask = NULL))
        loss <- output$loss / ENV_GRAD_ACCUM 
      })
    } else {
      output <- model(list(x = batch_x, y = batch_y, loss_mask = NULL))
      loss <- output$loss / ENV_GRAD_ACCUM 
    }
    
    if (use_scaler) {
      scaler$scale(loss)$backward()
    } else {
      loss$backward()
    }
    
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
    
    # 训练循环内的动态清理
    rm(batch_x, batch_y, output, loss)
    if (iter_idx %% 2 == 0) {
      gc(verbose = FALSE)
    }
  })
  
  torch_save(model$state_dict(), sprintf("checkpoints/moe_model_%02d.pt", epoch))
}

