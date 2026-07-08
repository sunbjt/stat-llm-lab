library(dplyr)
library(purrr)
library(forcats)
library(ggplot2)
library(tidytext)
library(jsonlite)

source("config.R")
source("utils/BPETokenizer.R")
source("moe/moe_model.R")

tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

# =====================================================================
# 实例化模型（必须与训练配置严格一致）
# =====================================================================
N_KV_HEADS <- 2
NUM_EXPERTS <- 4
TOP_K <- 1

moe_model <- RtomicCausalLM(
  vocab_size = VOCAB_SIZE,
  dim = DIM,
  n_layers = N_LAYERS,
  n_heads = N_HEADS,
  max_seq_len = SEQ_LEN,
  n_kv_heads = N_KV_HEADS,
  num_experts = NUM_EXPERTS,
  top_k = TOP_K
)

device <- torch_device(device)
moe_model <- moe_model$to(device = device)
ckpt_path <- "checkpoints/moe_model_03.pt"
checkpoint_data <- torch_load(ckpt_path, device = device)
state <- if (!is.null(checkpoint_data$model)) checkpoint_data$model else checkpoint_data
moe_model$load_state_dict(state)
moe_model$eval()

# =====================================================================
# 工具函数
# =====================================================================

extract_routing_data <- function(model) {
  n_layers <- length(model$layers)
  layer_indices <- map(1:n_layers, function(i) {
    as_array(model$layers[[i]]$moe$last_routing_indices)
  })
  return(layer_indices)
}

analyze_load_balance <- function(routing_data, num_experts = 4) {
  n_layers <- length(routing_data)
  map_df(1:n_layers, function(layer_idx) {
    indices <- routing_data[[layer_idx]]
    counts <- table(factor(as.vector(indices), levels = 1:num_experts))
    data.frame(
      Layer = layer_idx,
      Expert = as.integer(names(counts)),
      Token_Count = as.numeric(counts)
    )
  })
}

probe_expert_specialty <- function(indices, input_ids, tokenizer, target_expert) {
  # indices: [seq_len, top_k] 的矩阵
  activated_positions <- which(apply(indices == target_expert, 1, any))
  if (length(activated_positions) == 0) return(NULL)
  activated_token_ids <- input_ids[activated_positions]
  activated_words <- sapply(activated_token_ids, tokenizer$decode)
  sort(table(activated_words), decreasing = TRUE)
}

# =====================================================================
# 从 pretrain_clean.jsonl 随机采样多条文本
# =====================================================================
set.seed(42)
N_SAMPLES <- 1000

cat(sprintf("正在读取数据文件...\n"))
all_lines <- readLines("data/raw/pretrain_clean.jsonl")
sampled_lines <- sample(all_lines, min(N_SAMPLES, length(all_lines)))
cat(sprintf("从 %d 行中随机采样了 %d 条文本\n", length(all_lines), length(sampled_lines)))

# =====================================================================
# 批量处理：对每条文本跑 forward，聚合路由统计
# =====================================================================

# 累积负载统计 (Layer × Expert 的 Token_Count 总和)
total_load <- expand.grid(Layer = 1:N_LAYERS, Expert = 1:NUM_EXPERTS)
total_load$Token_Count <- 0

# 累积每个 Expert 的词频表 (list of table)
expert_word_freq <- vector("list", NUM_EXPERTS)
names(expert_word_freq) <- sprintf("Expert_%d", 1:NUM_EXPERTS)

total_tokens <- 0
processed <- 0

for (i in seq_along(sampled_lines)) {
  text <- fromJSON(sampled_lines[i])$text
  if (is.null(text) || nchar(text) < 10) next

  input_ids <- tokenizer$encode(text)
  input_ids <- head(input_ids, SEQ_LEN)

  x_tensor <- torch_tensor(input_ids, dtype = torch_long(), device = device)$unsqueeze(1)
  input_data <- list(x = x_tensor, y = NULL, loss_mask = NULL)

  with_no_grad({ output <- moe_model(input_data) })

  routing_data <- extract_routing_data(moe_model)
  load_stats <- analyze_load_balance(routing_data, num_experts = NUM_EXPERTS)

  # 累加负载统计
  total_load <- total_load %>%
    left_join(load_stats, by = c("Layer", "Expert"), suffix = c("", "_new")) %>%
    mutate(Token_Count = Token_Count + ifelse(is.na(Token_Count_new), 0, Token_Count_new)) %>%
    select(-Token_Count_new)

  # 累加各 Expert 词频 (仅分析最后一层以节省内存)
  target_layer <- N_LAYERS
  layer_indices <- routing_data[[target_layer]]
  for (expert_id in 1:NUM_EXPERTS) {
    freq <- probe_expert_specialty(layer_indices, input_ids, tokenizer, expert_id)
    if (!is.null(freq)) {
      existing <- expert_word_freq[[expert_id]]
      for (word in names(freq)) {
        if (word %in% names(existing)) {
          existing[word] <- existing[word] + freq[word]
        } else {
          existing[word] <- freq[word]
        }
      }
      expert_word_freq[[expert_id]] <- existing
    }
  }

  total_tokens <- total_tokens + length(input_ids)
  processed <- processed + 1
  if (processed %% 20 == 0) {
    cat(sprintf("已处理 %d/%d 条文本...\n", processed, length(sampled_lines)))
  }
}

cat(sprintf("完成！共处理 %d 条文本，累计 %d 个 token\n", processed, total_tokens))

# =====================================================================
# 绘图 1: 聚合后的负载均衡热力图
# =====================================================================
p <- ggplot(total_load, aes(x = factor(Expert), y = Layer, fill = Token_Count)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "#f0f9e8", high = "#0868ac") +
  scale_y_reverse(breaks = 1:N_LAYERS) +
  theme_minimal(base_family = "sans") +
  labs(title = "MoE Router Load Balance Heatmap (Aggregated)",
       subtitle = sprintf("%d texts / %d tokens across %d Experts", processed, total_tokens, NUM_EXPERTS),
       x = "Expert ID",
       y = "Layer ID",
       fill = "Token Count")

print(p)

# =====================================================================
# 绘图 2: 聚合后的 Expert 探针 (最后一层)
# =====================================================================
plot_data_probes <- map_dfr(1:NUM_EXPERTS, function(expert_id) {
  freq <- expert_word_freq[[expert_id]]
  if (is.null(freq) || length(freq) == 0) return(data.frame())
  data.frame(
    Token = names(freq),
    Freq = as.numeric(freq),
    Expert = sprintf("Expert %d", expert_id),
    stringsAsFactors = FALSE
  ) %>%
    slice_max(order_by = Freq, n = 15, with_ties = FALSE)
})

plot_family <- if (Sys.info()[["sysname"]] == "Darwin") "STHeiti" else "SimHei"

p1 <- ggplot(plot_data_probes, aes(
    x = reorder_within(Token, Freq, Expert),
    y = Freq,
    fill = Expert
  )) +
  geom_col(show.legend = FALSE, width = 0.7) +
  facet_wrap(~Expert, scales = "free", ncol = 4) +
  coord_flip() +
  scale_x_reordered() +
  theme_minimal(base_family = plot_family) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = sprintf("MoE 专家特异性探针 (Layer %d, %d 条文本聚合)", target_layer, processed),
    x = "捕获的 Token (BPE 解码)",
    y = "累计激活频次"
  )

print(p1)
ggsave("img/expert_act.png", plot = p1, width = 8, height = 5, dpi = 150)
