# =====================================================================
# MoE Expert 路由诊断工具
# 检查各层 expert 的 token 分配是否均匀，判断是否出现 expert 坍塌
# =====================================================================
source("config.R")
source("utils/BPETokenizer.R")
source("moe/moe_model.R")

# 临时参数
ENV_BATCH_SIZE  <- 64
N_KV_HEADS <- 2 
NUM_EXPERTS <- 4
TOP_K <- 1

tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

model <- RtomicCausalLM(
  VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN, 
  n_kv_heads = N_KV_HEADS,
  num_experts = NUM_EXPERTS,
  top_k = TOP_K
)

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

diagnose_expert_balance <- function(model, dataloader, device, num_batches = 3) {
  model$eval()

  cat("\n===== MoE Expert 路由诊断 =====\n")
  cat(sprintf("采样 %d 个 batch 的路由分布\n\n", num_batches))

  with_no_grad({
    # 使用 coro 包将 dataloader 转换为标准迭代器
    iter <- coro::as_iterator(dataloader)
    
    for (b_idx in 1:num_batches) {
      b <- iter()
      # R torch 迭代结束时返回的是 coro::exhausted() 而不是 NULL
      if (coro::is_exhausted(b)) break

      batch_x <- b$x$x$to(device = device)
      batch_y <- b$y$to(device = device)

      # 跑一次 forward，各层 MoE 会自动存储 last_routing_indices
      model(list(x = batch_x, y = batch_y, loss_mask = NULL))

      cat(sprintf("--- Batch %d (B=%d, S=%d) ---\n", b_idx, batch_x$size(1), batch_x$size(2)))

      for (i in 1:length(model$layers)) {
        moe <- model$layers[[i]]$moe
        ri <- as.array(moe$last_routing_indices$cpu())  # [N, top_k]
        E <- moe$num_experts

        # 统计每个 expert 被选中的次数（包括 top_k 中的每个 slot）
        counts <- tabulate(as.integer(ri), nbins = E)
        total <- sum(counts)
        pcts <- round(counts / total * 100, 1)

        # 理想均匀: 每个 expert 应占 100/E %
        ideal <- round(100 / E, 1)

        bar <- function(pct, ideal_pct) {
          n <- round(pct / 2)
          paste(rep("#", n), collapse = "")
        }

        cat(sprintf("  Layer %d: ", i))
        for (e in 1:E) {
          cat(sprintf("E%d=%5.1f%%%s ", e, pcts[e],
                      if (pcts[e] < ideal * 0.5) "!" else " "))
        }

        # 最大偏差
        max_dev <- max(abs(pcts - ideal))
        status <- if (max_dev > 30) "COLLAPSE"
                  else if (max_dev > 15) "SKEWED"
                  else "OK"
        cat(sprintf(" [%s]\n", status))
      }
      cat("\n")
    }
  })

  model$train()
  cat("===== 诊断完成 =====\n")
  cat("理想值: 每个 Expert 各占 25% (4 experts) 或 12.5% (8 experts)\n")
  cat("状态说明: OK=均匀 | SKEWED=偏斜 | COLLAPSE=坍塌\n\n")
}

diagnose_expert_balance(model, train_dl, num_batches = 1,
                        device = torch_device('cpu'))

