# causal_lm/04_causal_sft_train.R
# =====================================================================
# Decode-Only 架构 SFT 微调训练 (Native Torch Loop)
# =====================================================================

source("config.R")

# --- 环境专属超参 ---
BATCH_SIZE <- if (is_mac) 2 else 32

device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
cat(sprintf("当前运行设备: %s\n", device$type))

PRETRAIN_CKPT <- "checkpoints/distill_model_02.pt"

# =====================================================================
# 1. 加载 Tokenizer 与模型架构
# =====================================================================
source("utils/BPETokenizer.R")
source("causal_lm/CausalLM_model.R")

tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

# =====================================================================
# 2. SFT 数据集定义 (集成 Loss Mask 逻辑)
# =====================================================================
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
    
    # 核心逻辑：Prompt 部分不计算 Loss (FALSE)，仅对 Response 计算 Loss (TRUE)
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

# =====================================================================
# 3. 实例化模型并热启动权重
# =====================================================================
model <- RtomicCausalLM(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN)

if (file.exists(PRETRAIN_CKPT)) {
  cat(sprintf("正在加载预训练底座权重: %s\n", PRETRAIN_CKPT))
  ckpt <- torch_load(PRETRAIN_CKPT)
  state_dict <- if (!is.null(ckpt$model)) ckpt$model else ckpt
  
  # 动态扩展位置编码长度（从预训练的 256 扩展到 SFT 的 512）
  old_pos_emb <- state_dict[["pos_emb.weight"]]
  if (!is.null(old_pos_emb)) {
    old_len <- old_pos_emb$size(1)
    if (old_len < SEQ_LEN) {
      cat(sprintf("检测到预训练位置编码长度 (%d) < 当前 SFT 长度 (%d)，正在动态扩展...\n", old_len, SEQ_LEN))
      new_pos_emb <- torch_empty(c(SEQ_LEN, DIM))
      nn_init_normal_(new_pos_emb, std = 0.02)
      new_pos_emb[1:old_len, ] <- old_pos_emb
      state_dict[["pos_emb.weight"]] <- new_pos_emb
    }
  }
  model$load_state_dict(state_dict, strict = FALSE)
} else {
  cat("未找到预训练权重，将从头开始随机初始化！\n")
}

model <- model$to(device = device)

# =====================================================================
# 4. 层冻结策略 (可选择性冻结前几层以保护通用表征)
# =====================================================================
for (p in model$parameters) { p$requires_grad_(TRUE) }

FREEZE_LAYERS <- 0 # 建议改为 0 进行全参数 SFT
cat(sprintf("\n[SFT 策略] 冻结前 %d 层 Transformer Blocks...\n", FREEZE_LAYERS))

if (FREEZE_LAYERS > 0) {
  for (i in 1:FREEZE_LAYERS) {
    lapply(model$layers[[i]]$parameters, function(p) p$requires_grad_(FALSE))
  }
}

# =====================================================================
# 5. 准备微调数据集与 Dataloader
# =====================================================================
raw_data <- jsonlite::stream_in(file("data/raw/qa_no_think.jsonl"), verbose = FALSE)

sft_ds <- GenerativeSFTDataset(raw_data$instruction, raw_data$output, tokenizer, max_len = SEQ_LEN)
sft_dl <- dataloader(sft_ds, batch_size = BATCH_SIZE, shuffle = TRUE)

# =====================================================================
# 6. 原生 Torch 训练循环
# =====================================================================
trainable_params <- Filter(function(p) p$requires_grad, model$parameters)
optimizer <- optim_adamw(trainable_params, lr = 3e-4, weight_decay = 0.01)

EPOCHS <- 5
scheduler <- lr_cosine_annealing(optimizer, T_max = EPOCHS)

cat("\n启动 Decode-Only 自回归 SFT 训练...\n")

for (epoch in 1:EPOCHS) {
  model$train()
  total_loss <- 0
  batch_idx <- 0
  current_lr <- optimizer$param_groups[[1]]$lr
  
  coro::loop(for (batch in sft_dl) {
    batch_idx <- batch_idx + 1
    optimizer$zero_grad()
    
    input_data <- list(
      x = batch$x$to(device = device),
      y = batch$y$to(device = device),
      loss_mask = batch$loss_mask$to(device = device)
    )
    
    output <- model(input_data)
    loss <- output$loss
    
    loss$backward()
    nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
    optimizer$step()
    
    total_loss <- total_loss + loss$item()
    
    if (batch_idx %% 10 == 0 || batch_idx == 1) {
      cat(sprintf("Epoch [%d/%d], Step [%d], LR: %.6f, Loss: %.4f\n",
                  epoch, EPOCHS, batch_idx, current_lr, loss$item()))
    }
  })
  
  avg_loss <- total_loss / batch_idx
  cat(sprintf("Epoch %d 结束, 平均 交叉熵Loss: %.4f\n", epoch, avg_loss))
  
  scheduler$step()
  
  save_path <- sprintf("checkpoints/distill_sft_epoch_%02d.pt", epoch)
  torch_save(model$state_dict(), save_path)
}
