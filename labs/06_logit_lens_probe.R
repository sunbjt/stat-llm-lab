library(torch)
library(data.table)

# 假定你已经在环境中加载了你的模型和分词器
source("config.R")
source("causal_lm/CausalLM_model.R")
source("utils/BPETokenizer.R")

tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

# 实例化模型
causal_model <- RtomicCausalLM(
  vocab_size = VOCAB_SIZE,
  dim = DIM,
  n_layers = N_LAYERS,
  n_heads = N_HEADS,
  max_seq_len = SEQ_LEN
)

device <- torch_device("cpu")
causal_model <- causal_model$to(device = device)
ckpt_path <- "checkpoints/causal_model_03.pt"
checkpoint_data <- torch_load(ckpt_path)
state <- if (!is.null(checkpoint_data$model)) checkpoint_data$model else checkpoint_data
causal_model$load_state_dict(state)

# 封装一个透镜可视化函数
run_logit_lens <- function(model, tokenizer, prompt_text, device = "cpu") {
  model$eval()
  
  # 1. Token 编码
  tokens <- tokenizer$encode(prompt_text)
  x_input <- torch_tensor(matrix(tokens, nrow = 1), dtype = torch_long(), device = device)
  
  # 2. 前向传播，开启 logit_lens
  with_no_grad({
    output <- model(list(x = x_input, y = NULL), logit_lens = TRUE)
  })
  
  # 3. 解析每一层对每个位置预测的 Top-1 Token
  seq_len <- length(tokens)
  layer_names <- names(output$layer_logits)
  
  # 构建一个矩阵展示结果
  result_mat <- matrix("", nrow = length(layer_names), ncol = seq_len)
  rownames(result_mat) <- layer_names
  
  # 填充输入文本作为表头
  input_words <- sapply(tokens, function(tok) tokenizer$decode(tok))
  colnames(result_mat) <- paste0("Pos_", 1:seq_len, "(", input_words, ")")
  
  for (i in seq_along(layer_names)) {
    layer_name <- layer_names[i]
    # 取出该层的 logits 矩阵 [seq_len, vocab_size]
    logits_s <- output$layer_logits[[layer_name]][1, , ] 
    
    # 算出每个位置概率最大的 Token ID
    top_ids <- as.integer(as.array(torch_argmax(logits_s, dim = -1)))
    
    # 解码为具体字符
    for (pos in 1:seq_len) {
      pred_word <- tokenizer$decode(top_ids[pos])
      # 替换换行或空格符号方便打印
      pred_word <- gsub("\n", "\\n", pred_word, fixed = TRUE)
      result_mat[i, pos] <- pred_word
    }
  }
  
  
  # 显式通过 data.frame 组合，并关闭 check.names 防止括号变点号
  res_df <- data.frame(
    层级 = rownames(result_mat), 
    result_mat, 
    check.names = FALSE, 
    stringsAsFactors = FALSE
  )
  
  print(as.data.table(res_df))
}

run_logit_lens(model = causal_model, tokenizer, prompt_text = "机器学习是一项应用很广")

library(knitr)
run_logit_lens(model = causal_model, tokenizer, prompt_text = "机器学习是一项应用很广") |> 
  kable()


