# =====================================================================
# SFT 微调训练 (已适配 RoPE 架构与精确 Loss Mask)
# =====================================================================

source("config.R")

# --- 环境专属超参 ---
BATCH_SIZE <- if (is_mac) 4 else 32

device <- torch_device(if(cuda_is_available()) "cuda" else "cpu")
cat(sprintf("当前运行设备: %s\n", device$type))

# =====================================================================
# 1. 加载 Tokenizer
# =====================================================================
source("utils/BPETokenizer.R")
source("latent_residual/LRP_model.R")
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

model <- RtomicLRP(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN)

# =====================================================================
# 2. SFT 数据集定义
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
    eos_val <- if (!is.null(self$tokenizer$eos_idx)) self$tokenizer$eos_idx else 2L
    r_ids <- c(r_ids, eos_val)
    
    full_seq <- c(p_ids, r_ids)
    seq_len <- length(full_seq)
    
    x_ids <- full_seq[1:(seq_len - 1)]
    y_ids <- full_seq[2:seq_len]
    
    # 制作 Mask: 提示词部分不计算 Loss (FALSE)，只对回复部分算 (TRUE)
    mask <- c(rep(FALSE, length(p_ids) - 1), rep(TRUE, length(r_ids)))
    
    pad_len <- self$max_len - length(x_ids)
    if (pad_len > 0) {
      x_ids <- c(x_ids, rep(1L, pad_len))
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
# 3. 加载模型架构 (适配无 pos_emb 版本)
# =====================================================================
PRETRAIN_CKPT <- "checkpoints/lrp_03.pt"
ckpt <- torch_load(PRETRAIN_CKPT)
state_dict <- if (!is.null(ckpt$model)) ckpt$model else ckpt
model$load_state_dict(state_dict, strict = FALSE)
model <- model$to(device = device)

# =====================================================================
# 4. 层冻结策略
# =====================================================================
for (p in model$parameters) { p$requires_grad_(TRUE) }

#FREEZE_LAYERS <- 7
#model$tok_emb$weight$requires_grad_(FALSE)

#for (i in 1:FREEZE_LAYERS) {
#  lapply(model$layers[[i]]$parameters, function(p) p$requires_grad_(FALSE))
#}

# =====================================================================
# 5. 准备数据与 Dataloader
# =====================================================================
raw_data <- jsonlite::stream_in(file("data/raw/qa_no_think.jsonl"))
sft_ds <- GenerativeSFTDataset(raw_data$instruction, raw_data$output, tokenizer, max_len = SEQ_LEN)
sft_dl <- dataloader(sft_ds, batch_size = BATCH_SIZE, shuffle = TRUE)

# =====================================================================
# 6. 原生 Torch 训练循环 (接管 Loss 计算)
# =====================================================================
trainable_params <- Filter(function(p) p$requires_grad, model$parameters)
optimizer <- optim_adamw(trainable_params, lr = 3e-4, weight_decay = 0.01)

EPOCHS <- 5
scheduler <- lr_cosine_annealing(optimizer, T_max = EPOCHS)

## SFT 全参微调 (屏蔽 Prompt Loss)
for (epoch in 1:EPOCHS) {
  model$train()
  total_loss <- 0
  batch_idx <- 0
  current_lr <- optimizer$param_groups[[1]]$lr
  
  coro::loop(for (batch in sft_dl) {
    batch_idx <- batch_idx + 1
    optimizer$zero_grad()
    
    # 强制让 y = NULL，阻止模型内部计算无效的全序列 Loss
    input_data <- list(
      x = batch$x$to(device = device),
      y = NULL 
    )
    
    # 1. 拿到特征表示 (B, S, Dim)
    output <- model(input_data)
    pred <- output$pred 
    
    # 2. 手动投影到词表计算 Logits
    logits <- torch_matmul(pred, model$tok_emb$weight$t())
    
    # 3. 展平并提取数据
    logits_flat <- logits$view(c(-1, VOCAB_SIZE))
    y_flat <- batch$y$to(device = device)$view(c(-1))
    mask_flat <- batch$loss_mask$to(device = device)$view(c(-1))
    
    # 4. 计算逐元素的原始交叉熵 (reduction = "none")
    raw_loss <- nnf_cross_entropy(logits_flat, y_flat, reduction = "none")
    
    # 5. 精确的掩码 Loss：只保留 mask_flat 为 TRUE 的部分的 Loss，求均值
    valid_loss <- raw_loss * mask_flat
    loss <- valid_loss$sum() / mask_flat$sum()
    
    loss$backward()
    nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
    optimizer$step()
    
    total_loss <- total_loss + loss$item()
    
    if (batch_idx %% 20 == 0 || batch_idx == 1) {
      cat(sprintf("Epoch [%02d/%02d], Step [%03d], LR: %.6f, Loss: %.4f\n",
                  epoch, EPOCHS, batch_idx, current_lr, loss$item()))
    }
  })
  
  avg_loss <- total_loss / batch_idx
  cat(sprintf("=> Epoch %d 结束, 掩码平均 Loss: %.4f\n", epoch, avg_loss))
  
  scheduler$step()
  
  save_path <- sprintf("checkpoints/lrp_sft_epoch_%02d.pt", epoch)
  torch_save(model$state_dict(), save_path)
}
