Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")

source("config.R")
source("utils/BPETokenizer.R")
source("latent_contrastive/contrastive_model.R")

tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)

raw_model <- TokenLatentModel(
  vocab_size = VOCAB_SIZE,
  dim = DIM,
  n_layers = N_LAYERS,
  n_heads = N_HEADS,
  max_seq_len = SEQ_LEN
)

device <- torch_device("cpu")
raw_model <- raw_model$to(device = device)

generate_text <- function(model, tokenizer, prompt, max_new_tokens = 100,
                          temperature = 0.7, repetition_penalty = 1.2,
                          top_k = 5, use_contrastive = TRUE) {
  device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
  model <- model$to(device = device)
  model$eval()

  raw_ids <- tokenizer$encode_raw(prompt)[[1]]
  current_ids <- c(tokenizer$bos_idx, raw_ids)

  cat(sprintf("\n[Input]: %s\n", prompt))
  cat("[Generate]: ")

  with_no_grad({
    for (step in 1:max_new_tokens) {
      input_seq <- tail(current_ids, SEQ_LEN)
      x_tensor <- torch_tensor(input_seq, dtype = torch_long(), device = device)$unsqueeze(1)
      S <- x_tensor$size(2)

      h <- model$encode(x_tensor)

      if (use_contrastive) {
        # 对比学习 / 潜空间解构路径
        pred_all <- model$predictor(h)
        pred_last <- pred_all[, S, , drop = FALSE]$squeeze(1)
        pred_norm <- nnf_normalize(pred_last, p = 2, dim = -1)
        emb_norm <- nnf_normalize(model$tok_emb$weight, p = 2, dim = -1)
        cos_sim <- torch_matmul(pred_norm, emb_norm$transpose(1, 2))$squeeze(1)
        logits <- cos_sim / model$temperature
      } else {
        # 标准交叉熵 (CE) 路径
        logits <- torch_matmul(h[, S, ], model$tok_emb$weight$transpose(1, 2))$squeeze(1)
      }

      # 1. Repetition Penalty
      if (repetition_penalty > 1.0) {
        uniq_ids <- unique(current_ids)
        idx_tensor <- torch_tensor(uniq_ids, dtype = torch_long(), device = device)
        penalized <- logits[idx_tensor]$clone()
        mask_pos <- penalized > 0
        penalized[mask_pos] <- penalized[mask_pos] / repetition_penalty
        penalized[!mask_pos] <- penalized[!mask_pos] * repetition_penalty
        logits[idx_tensor] <- penalized
      }

      # 2. Temperature Scaling
      if (temperature > 1e-3) {
        logits <- logits / temperature
      }

      # 3. Top-k Filtering (默认为 5)
      if (top_k > 0 && top_k < logits$size(1)) {
        topk <- torch_topk(logits, k = top_k)
        min_val <- topk[[1]][top_k]$item()
        logits[logits < min_val] <- -Inf
      }

      # 4. Sampling / Greedy Decoding
      if (temperature <= 1e-3) {
        next_token_id <- as.integer(torch_argmax(logits, dim = -1))
      } else {
        probs <- nnf_softmax(logits, dim = -1)
        next_token_id <- as.integer(torch_multinomial(probs, num_samples = 1))
      }

      # EOS 判断与解码输出
      if (!is.null(tokenizer$eos_idx) && next_token_id == tokenizer$eos_idx) break

      next_word <- tokenizer$decode(next_token_id, clean = FALSE)
      if (next_word == "<EOS>") break

      cat(next_word)
      flush.console()

      current_ids <- c(current_ids, next_token_id)
    }
  })
  cat("\n\nDone.\n")
  invisible(current_ids)
}

checkpoints_to_test <- c(
  "checkpoints/contrastive_02.pt",
  "checkpoints/contrastive_sft_05.pt"
)

test_prompts <- c(
  "我认为 AI 未来对人类有哪些好处？",
  "在大学里应该这样学习编程？",
  "使用数据科学怎样帮助企业提高生产效率？",
  "简述一下 agent 技术。",
  "深度学习有哪些常用方法？"
)

for (ckpt_path in checkpoints_to_test) {
  cat(sprintf("\n=== Model: %s ===\n", ckpt_path))

  checkpoint_data <- torch_load(ckpt_path)
  raw_model$load_state_dict(checkpoint_data$model)
  raw_model$to(dtype = torch_float32())
  raw_model$eval()

  for (prompt in test_prompts) {
    generate_text(
      raw_model,
      tokenizer,
      prompt = prompt,
      temperature = 0.3,
      repetition_penalty = 1.2,
      top_k = 5,
      max_new_tokens = 120,
      use_contrastive = TRUE
    )
  }
}

# --- CE 路径推理 ---
cat("\n\n=== CE Path Inference ===\n")

generate_text_ce <- function(model, tokenizer, prompt, max_new_tokens = 80,
                              temperature = 0.7, repetition_penalty = 1.2, top_p = 0.9) {
  device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
  model <- model$to(device = device)
  model$eval()

  raw_ids <- tokenizer$encode_raw(prompt)[[1]]
  current_ids <- c(tokenizer$bos_idx, raw_ids)

  cat(sprintf("\n[Input]: %s\n", prompt))
  cat("[Generate]: ")

  with_no_grad({
    for (step in 1:max_new_tokens) {
      input_seq <- tail(current_ids, SEQ_LEN)
      x_tensor <- torch_tensor(input_seq, dtype = torch_long(), device = device)$unsqueeze(1)
      S <- x_tensor$size(2)

      h <- model$encode(x_tensor)

      logits <- torch_matmul(h[, S, ], model$tok_emb$weight$transpose(1, 2))$squeeze(1)

      if (repetition_penalty > 1.0) {
        uniq_ids <- unique(current_ids)
        idx_tensor <- torch_tensor(uniq_ids, dtype = torch_long(), device = device)
        penalized <- logits[idx_tensor]
        mask_pos <- penalized > 0
        penalized[mask_pos] <- penalized[mask_pos] / repetition_penalty
        penalized[!mask_pos] <- penalized[!mask_pos] * repetition_penalty
        logits[idx_tensor] <- penalized
      }

      if (temperature <= 1e-3) {
        next_token_id <- as.integer(torch_argmax(logits, dim = -1))
      } else {
        logits <- logits / temperature
        if (top_p < 1.0) {
          sorted <- torch_sort(logits, descending = TRUE)
          sorted_vals <- sorted[[1]]
          cumsum <- torch_cumsum(nnf_softmax(sorted_vals, dim = -1), dim = -1)
          cutoff <- sorted_vals$size(1) - as.integer(torch_sum(cumsum > top_p)) + 1
          if (cutoff > 1) logits[logits < sorted_vals[cutoff]] <- -Inf
        }
        probs <- nnf_softmax(logits, dim = -1)
        next_token_id <- as.integer(torch_multinomial(probs, num_samples = 1))
      }

      if (!is.null(tokenizer$eos_idx) && next_token_id == tokenizer$eos_idx) break
      next_word <- tokenizer$decode(next_token_id, clean = FALSE)
      if (next_word == "<EOS>") break

      cat(next_word)
      flush.console()
      current_ids <- c(current_ids, next_token_id)
    }
  })
  cat("\n\nDone.\n")
  invisible(current_ids)
}

for (ckpt_path in checkpoints_to_test) {
  checkpoint_data <- torch_load(ckpt_path)
  raw_model$load_state_dict(checkpoint_data$model)
  raw_model$to(dtype = torch_float32())
  raw_model$eval()

  for (prompt in test_prompts) {
    generate_text_ce(
      raw_model, tokenizer, prompt = prompt,
      temperature = 0.3, repetition_penalty = 1.2, top_p = 0.4, max_new_tokens = 120
    )
  }
}

