# 相关文件地址
# https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

library(torch)
library(safetensors)
library(tok)
setwd('/Users/liusizhe/github/Rtomic_4me/')

# 加载分词器
tokenizer <- tok::tokenizer$from_file("src/Pre-train/sentence-transformers/tokenizer.json")
# 加载权重
weights <- safetensors::safe_load_file("src/Pre-train/sentence-transformers/model.safetensors", framework = "torch")
# print(names(weights)) 

# 加载 config 文件
config <- jsonlite::fromJSON("src/Pre-train/sentence-transformers/config.json")
emb_layer <- nn_embedding(config$vocab_size, config$hidden_size)

# 将预训练的真实知识注入 Embedding 层
emb_w <- weights[["embeddings.word_embeddings.weight"]]
with_no_grad({
  emb_layer$weight$set_(emb_w)
})

# 定义函数
encode_semantic <- function(text_batch) {
  with_no_grad({
    encs <- tokenizer$encode_batch(text_batch)
    ids_list <- lapply(encs, function(x) x$ids)
    lens <- lengths(ids_list)
    max_len <- max(lens)
    batch_n <- length(text_batch)
    padded_ids <- matrix(0L, nrow = batch_n, ncol = max_len)
    for (i in seq_len(batch_n)) {
      padded_ids[i, seq_len(lens[i])] <- ids_list[[i]]
    }
    
    # 构造 Tensor 并执行【+1】操作 (适配 R 索引)
    ids <- torch_tensor(padded_ids, dtype = torch_long()) + 1L
    
    # 映射到向量空间并 Mean Pooling
    vectors <- emb_layer(ids)
    return(vectors$mean(dim = 2))
  })
}

# 自检
test_vec <- encode_semantic("这是一条测试")
test_vec$size()

# 第二部分：带有动态进度条的过滤引擎
target_prompts <- c("人工智能与深度学习", "统计学与概率论", "神经网络算法", "大数据", "数据分析和数据科学")
target_vecs <- encode_semantic(target_prompts)

wiki_input <- "~/Downloads/2023-14_zh_middle_0004.jsonl"
wiki_output <- "~/github/stat-llm-lab/data/processed/pre8g.jsonl"
# wiki_input <- "data/coig_neo_sft.jsonl"
# wiki_output <- "data/coig_neo_sft_clean.jsonl"
con_in <- file(wiki_input, "r")
con_out <- file(wiki_output, "w")

batch_size <- 256  # 优化后，Batch Size 可以放心调大到 128 甚至 256
threshold <- 0.73  # 注入真实权重后，0.65 是一个非常严格的高质量阈值

# 进度统计变量
batch_count <- 0
total_lines_processed <- 0
total_kept <- 0
start_time <- Sys.time()

while (TRUE) {
  lines <- readLines(con_in, n = batch_size, warn = FALSE)
  if (length(lines) == 0) break
  
  batch_len <- length(lines)
  total_lines_processed <- total_lines_processed + batch_len
  
  # 过滤极短文本和纯符号行，不让它们浪费算力
  valid_idx <- nchar(lines) > 5
  valid_lines <- lines[valid_idx]
  
  if (length(valid_lines) > 0) {
    # 只对有效文本进行张量运算
    batch_vecs <- encode_semantic(valid_lines)
    
    sims <- torch_matmul(
      nnf_normalize(batch_vecs, p = 2, dim = 2),
      nnf_normalize(target_vecs, p = 2, dim = 2)$t()
    )
    
    max_sim <- as.array(sims$max(dim = 2)[[1]])
    
    # 筛选并保存
    keep_lines <- valid_lines[max_sim > threshold]
    if (length(keep_lines) > 0) {
      writeLines(keep_lines, con_out)
      total_kept <- total_kept + length(keep_lines)
    }
  }
  
  batch_count <- batch_count + 1
  
  # 【进度条渲染】每 10 个 Batch 更新一次控制台
  if (batch_count %% 10 == 0) {
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    speed <- round(total_lines_processed / elapsed, 0)
    
    # 使用 \r 实现原地刷新，不刷屏
    cat(sprintf("\r 处理进度: %d 行 | 命中精华: %d 行 | 速度: %d 行/秒 | 耗时: %.1f 秒", 
                total_lines_processed, total_kept, speed, elapsed))
  }
  
  # 【防爆显存】每 100 个 Batch 手动触发一次 R 的垃圾回收
  if (batch_count %% 100 == 0) gc(verbose = FALSE)
}

close(con_in)
close(con_out)

# 最终总结
total_time <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
cat(sprintf("\n\n 过滤圆满完成！\n总处理: %d 行\n总保留: %d 行 (提纯率 %.2f%%)\n总耗时: %.2f 分钟\n", 
            total_lines_processed, total_kept, (total_kept/total_lines_processed)*100, total_time))
