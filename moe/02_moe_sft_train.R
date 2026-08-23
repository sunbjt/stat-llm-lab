# moe/02_moe_sft_train.R
# =====================================================================
# Decode-Only MoE 架构 SFT 微调训练脚本
# =====================================================================

Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")

source("config.R")
source("utils/BPETokenizer.R")
source("moe/moe_model.R")

# --- 1. 环境与超参设置 ---
if (is_mac) {
  ENV_BATCH_SIZE <- 2
  ENV_GRAD_ACCUM <- 4
  ENV_USE_AMP    <- FALSE
} else {
  ENV_BATCH_SIZE <- 8   # SFT 序列较长，可根据显存情况微调 batch size
  ENV_GRAD_ACCUM <- 4
  ENV_USE_AMP    <- TRUE
}

device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
cat(sprintf("当前运行设备: %s\n", device$type))

PRETRAIN_CKPT <- "checkpoints/moe_model_02.pt"

# --- 2. 加载 Tokenizer 与数据集定义 ---
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

GenerativeSFTDataset <- dataset(
  name = "GenerativeSFTDataset",
  initialize = function(prompts, responses, tokenizer, max_len) {
    self$prompts <- prompts
    self$responses <- responses
    self$tokenizer <- tokenizer
    self$max_len <- max_len
  },
  
  .getitem = function(i) {
    p_ids <- self$tokenizer$encode_raw(self$prompts[i])[[1]]
    p_ids <- c(self$tokenizer$bos_idx, p_ids)
    
    r_ids <- self$tokenizer$encode_raw(self$responses[i])[[1]]
    eos_val <- if (!is.null(self$tokenizer$eos_idx)) self$tokenizer$eos_idx else 4L
    r_ids <- c(r_ids, eos_val)
    
    full_seq <- c(p_ids, r_ids)
    seq_len <- length(full_seq)
    
    x_ids <- full_seq[1:(seq_len - 1)]
    y_ids <- full_seq[2:seq_len]
    
    # Prompt 部分不计算 Loss (FALSE)，仅对 Response 计算 Loss (TRUE)
    mask <- c(rep(FALSE, length(p_ids) - 1), rep(TRUE, length(r_ids)))
    
    pad_len <- self$max_len - length(x_ids)
    if (pad_len > 0) {
      x_ids <- c(x_ids, rep(1L, pad_len)) # 1L 为 pad_idx
      y_ids <- c(y_ids, rep(1L, pad_len))
      mask <- c(mask, rep(FALSE, pad_len))
    } else {
      x_ids <- x_ids[1:self$max_len]
      y_ids <- y_ids[1:self$max_len]
      mask <- mask[1:self$max_len]
    }
    
    list(
      x = torch_tensor(x_ids, dtype = torch_long()),
      y = torch_tensor(y_ids, dtype = torch_long()),
      loss_mask = torch_tensor(mask, dtype = torch_bool())
    )
  },
  .length = function() length(self$prompts)
)

# --- 3. 实例化 MoE 模型与加载预训练权重 ---
N_KV_HEADS  <- 2 
NUM_EXPERTS <- 4
TOP_K       <- 1

model <- RtomicCausalLM(
  vocab_size  = VOCAB_SIZE, 
  dim         = DIM, 
  n_layers    = N_LAYERS, 
  n_heads     = N_HEADS, 
  max_seq_len = SEQ_LEN,
  n_kv_heads  = N_KV_HEADS,
  num_experts = NUM_EXPERTS,
  top_k       = TOP_K
)

if (file.exists(PRETRAIN_CKPT)) {
  cat(sprintf("正在加载预训练 MoE 权重: %s\n", PRETRAIN_CKPT))
  ckpt <- torch_load(PRETRAIN_CKPT)
  state_dict <- if (!is.null(ckpt$model)) ckpt$model else ckpt
  model$load_state_dict(state_dict, strict = FALSE)
} else {
  cat("未找到预训练权重，将从头开始随机初始化！\n")
}

model <- model$to(device = device)

# --- 4. 优化器与冻结策略 ---
FREEZE_LAYERS <- 0 # 设置为大于 0 的整数可冻结底座 Layer
if (FREEZE_LAYERS > 0) {
  cat(sprintf("\n[SFT 策略] 冻结前 %d 层 Transformer Blocks...\n", FREEZE_LAYERS))
  for (i in 1:FREEZE_LAYERS) {
    lapply(model$layers[[i]]$parameters, function(p) p$requires_grad_(FALSE))
  }
}

trainable_params <- Filter(function(p) p$requires_grad, model$parameters)

# 考虑梯度累加进行 LR 缩放
base_lr <- 1.5e-4
scaled_lr <- sqrt(ENV_GRAD_ACCUM) * base_lr
optimizer <- optim_adamw(trainable_params, lr = scaled_lr, weight_decay = 0.01)

# --- 5. 数据加载 ---
raw_data <- jsonlite::stream_in(file("data/raw/qa_no_think.jsonl"), verbose = FALSE)

sft_ds <- GenerativeSFTDataset(raw_data$instruction, raw_data$output, tokenizer, max_len = SEQ_LEN)
sft_dl <- dataloader(
  sft_ds, 
  batch_size = ENV_BATCH_SIZE, 
  shuffle = TRUE,
  drop_last = TRUE,
  pin_memory = !is_mac
)

EPOCHS <- 5
total_steps <- (EPOCHS * length(sft_dl)) / ENV_GRAD_ACCUM
scheduler <- lr_cosine_annealing(optimizer, T_max = EPOCHS)

use_scaler <- ENV_USE_AMP && cuda_is_available()
if (use_scaler) scaler <- cuda_amp_grad_scaler()

# --- 6. 训练循环 ---
cat("\n启动 MoE 自回归 SFT 微调训练...\n")

global_step <- 0
for (epoch in 1:EPOCHS) {
  model$train()
  total_loss <- 0
  iter_idx <- 0
  current_lr <- optimizer$param_groups[[1]]$lr
  
  coro::loop(for (batch in sft_dl) {
    iter_idx <- iter_idx + 1
    
    input_data <- list(
      x = batch$x$to(device = device, non_blocking = TRUE),
      y = batch$y$to(device = device, non_blocking = TRUE),
      loss_mask = batch$loss_mask$to(device = device, non_blocking = TRUE)
    )
    
    # 混合精度前向传播
    if (use_scaler) {
      with_autocast(device_type = "cuda", dtype = torch_bfloat16(), enabled = TRUE, {
        output <- model(input_data)
        loss <- output$loss / ENV_GRAD_ACCUM
      })
      scaler$scale(loss)$backward()
    } else {
      output <- model(input_data)
      loss <- output$loss / ENV_GRAD_ACCUM
      loss$backward()
    }
    
    # 真实 Loss 还原用于统计打印
    loss_val <- as.numeric(output$loss$detach())
    total_loss <- total_loss + loss_val
    
    # 梯度累加更新步
    if (iter_idx %% ENV_GRAD_ACCUM == 0) {
      if (use_scaler) {
        scaler$unscale_(optimizer)
        nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
        scaler$step(optimizer)
        scaler$update()
      } else {
        nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
        optimizer$step()
      }
      
      optimizer$zero_grad()
      global_step <- global_step + 1
      
      if (global_step %% 10 == 0 || global_step == 1) {
        cat(sprintf("Epoch [%d/%d] | Micro Step [%d] | Global Step [%d] | LR: %.6f | Loss: %.4f\n",
                    epoch, EPOCHS, iter_idx, global_step, current_lr, loss_val))
      }
    }
    
    # 内存管理
    rm(input_data, output, loss)
    if (iter_idx %% 5 == 0) gc(verbose = FALSE)
  })
  
  avg_loss <- total_loss / iter_idx
  cat(sprintf("\n===> Epoch %d 完成 | 平均 Loss (含 Aux Loss): %.4f <===\n\n", epoch, avg_loss))
  
  scheduler$step()
  
  save_path <- sprintf("checkpoints/moe_sft_epoch_%02d.pt", epoch)
  torch_save(model$state_dict(), save_path)
}