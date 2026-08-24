# =====================================================================
# 纯血 JEPA 端到端推理 (带真理之锚)
# =====================================================================
source("config.R")
source("world_model/jepa_model.R") # 确保路径正确
source("utils/BPETokenizer.R")

tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)
device <- torch_device("cpu")

# 1. 彻底抛弃 Generator/Decoder，直接加载裸的 JEPA 基座
JEPA_CKPT <- "checkpoints/wm_02.pt" # 如果你跑完了 3 个 Epoch，请改成 wm_03.pt

model <- RtomicJEPA_VQ(
  vocab_size = VOCAB_SIZE, dim = DIM, n_layers = N_LAYERS, 
  n_heads = N_HEADS, max_seq_len = SEQ_LEN, num_clusters = 4096
)

# 2. 挂载权重 (处理 Luz 保存时的 model. 前缀)
ckpt_data <- torch_load(JEPA_CKPT, device = "cpu")
clean_state_dict <- list()
for (name in names(ckpt_data$model)) {
  clean_name <- sub("^model\\.", "", name)
  clean_state_dict[[clean_name]] <- ckpt_data$model[[name]]
}
model$load_state_dict(clean_state_dict, strict = FALSE)

model <- model$to(device = device, dtype = torch_float32())
model$eval()

# 3. 极简的前向传播生成逻辑
generate_text <- function(model, tokenizer, prompt, max_new_tokens = 50, temperature = 0.8, top_k = 10, rep_penalty = 1.2) {
  raw_ids <- tokenizer$encode_raw(prompt)[[1]]
  current_ids <- c(tokenizer$bos_idx, raw_ids)
  
  cat(sprintf("\n[输入 Prompt]: %s\n", prompt))
  cat("[模型生成中]: ")
  
  with_no_grad({
    for (i in 1:max_new_tokens) {
      input_seq <- tail(current_ids, SEQ_LEN)
      x_tensor <- torch_tensor(input_seq, dtype = torch_long(), device = device)$unsqueeze(1)
      seq_len <- x_tensor$size(2)
      
      # A. 让 JEPA 提取高维隐逻辑 (返回 pred_z)
      out <- model(list(x = x_tensor, y = NULL))
      pred_z <- out$pred  # 维度: [Batch, Seq_Len, Dim]
      
      # B. 取出最后一步的“未来预测向量”
      last_pred <- pred_z[, seq_len, ]
      
      # C. 【核心魔法】：直接用自带的 Token Embedding 矩阵做内积，完成“认知”到“表达”的降维打击
      logits <- torch_matmul(last_pred, model$tok_emb$weight$t()) / temperature
      logits <- logits$squeeze(1)
      
      # D. 重复惩罚
      unique_past_ids <- unique(current_ids)
      for (past_id in unique_past_ids) {
        logit_val <- as.numeric(logits[past_id])
        if (logit_val < 0) logits[past_id] <- logit_val * rep_penalty
        else logits[past_id] <- logit_val / rep_penalty
      }
      
      # E. Top-K 采样
      if (top_k > 0) {
        topk_res <- torch_topk(logits, k = top_k)
        kth_value <- topk_res[[1]][top_k]
        logits <- torch_where(logits < kth_value, torch_tensor(-Inf, device = device), logits)
      }
      
      probs <- nnf_softmax(logits, dim = -1)
      next_token_id <- as.integer(torch_multinomial(probs, num_samples = 1))
      
      if (!is.null(tokenizer$eos_idx) && next_token_id == tokenizer$eos_idx) {
        cat(" [EOS]")
        break
      }
      
      next_word <- tokenizer$decode(next_token_id, clean = FALSE)
      cat(next_word)
      flush.console()
      
      current_ids <- c(current_ids, next_token_id)
    }
  })
  cat("\n\n生成完毕。\n")
}

# --- 开始测试 ---
test_prompts <- c("可以这样理解损失函数，它是",
 "数据科学可以帮助企业","西红柿是一种蔬菜，对")
for (prompt in test_prompts) {
  # 对于刚跑完 1 个 Epoch 的模型，适当调高温度、放宽 top_k 来增加输出的多样性
  generate_text(model, tokenizer, prompt = prompt, 
                max_new_tokens = 150, 
                temperature = 0.9,   
                top_k = 20,          
                rep_penalty = 1.5)   
}
