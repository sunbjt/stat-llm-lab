library(data.table)
source("config.R")
source("latent_residual/LRP_model.R")
source("utils/BPETokenizer.R")

tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)
device <- torch_device(if(cuda_is_available()) "cuda" else "cpu")

# 1. 挂载已训练的模型
model <- RtomicLRP(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN)
ckpt <- torch_load("checkpoints/lrp_03.pt", device = "cpu")
model$load_state_dict(ckpt$model, strict = FALSE)
model <- model$to(device = device)
model$eval() # 极其重要：关闭 Dropout，稳定输出

# 2. 准备探测数据集
# 这里复用你的 RtomicBinDataset 逻辑，但不要 Shuffle
VAL_BIN <- "data/processed/zhwiki_tokens_16384.bin"

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

val_ds <- RtomicBinDataset(VAL_BIN, seq_len = SEQ_LEN)
val_dl <- dataloader(val_ds, batch_size = if(is_mac) 4 else 32, shuffle = TRUE)

# 3. 开启探针收集循环
loss_records <- list()
batch_idx <- 0
max_batches <- 2000 # 你想要的随机截断点，比如随机测 1000 个 batch
total_batches <- max_batches # 用于日志显示


with_no_grad({
  coro::loop(for (batch in val_dl) {
    batch_idx <- batch_idx + 1
    
    x_input <- batch$x$x$to(device = device)
    y_target <- batch$y$to(device = device)
    
    # 前向传播 (你的模型会输出 pred 潜变量)
    output <- model(list(x = x_input, y = NULL))
    
    # 映射回词表
    logits <- torch_matmul(output$pred, model$tok_emb$weight$t())
    logits_flat <- logits$view(c(-1, VOCAB_SIZE))
    y_flat <- y_target$view(c(-1))
    
    # 核心：关闭均值计算，保留每一个字的 Loss
    raw_loss <- nnf_cross_entropy(logits_flat, y_flat, reduction = "none")
    
    # 将 Tensor 抽回 R 环境
    loss_records[[length(loss_records) + 1]] <- data.table(
      token_id = as.integer(y_flat$cpu()),
      loss_val = as.numeric(raw_loss$cpu())
    )
    if (batch_idx %% 50 == 0) {
      cat(sprintf("\r已处理 Batch: %d / %d", batch_idx, total_batches), '\n')
      flush.console() # 强制立刻刷新输出缓冲区，防止 RStudio 控制台卡顿
    }
    if (batch_idx >= max_batches) {
      cat("\n 达到指定的随机样本量，提前结束收集。\n")
        break
    }
  })
})

# 4. 聚合分析：找出模型的“噩梦词汇”
all_losses_dt <- rbindlist(loss_records)

# 计算每个 Token 的平均 Loss 和方差
token_loss_profile <- all_losses_dt[, .(
  avg_loss = mean(loss_val),
  loss_sd = sd(loss_val),
  freq = .N
), by = token_id]

# 筛选出高频失误的靶点 (出现次数不少，但 Loss 极高，常年飙在 6.0 甚至 8.0 以上)
bad_tokens <- token_loss_profile[freq > 15][order(-avg_loss)][1:50]
bad_tokens[, word := sapply(token_id, function(id) tokenizer$decode(id))]

cat("\n=== 找到的模型平均预测误差最高的概念 (急需补充相关前置解释语料) ===\n")
print(bad_tokens[, .(word, avg_loss, freq)])
