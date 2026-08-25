# causal_lm/02_distill.R
Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")

# --- 兼容从 Distillation/ 子目录或仓库根目录运行 ---
if (basename(normalizePath(getwd())) == "Distillation") setwd("..")

library(torch)
library(luz)
library(arrow)

source("config.R")
source("causal_lm/CausalLM_model.R")

## 1. 环境专属超参配置
if (is_mac) {
  ENV_BATCH_SIZE  <- 2
  ENV_USE_AMP     <- FALSE
} else {
  ENV_BATCH_SIZE  <- 128  # 15M 小模型显存开销极小，RTX 3090 可轻松开启 32~64
  ENV_USE_AMP     <- TRUE
}

# =====================================================================
# 2. 零 CPU 耗时内存 Dataset
# =====================================================================
# 数据序列长度 = 教师端 MAX_LENGTH - 1（Python 侧 TARGET_LEN = 511）
TARGET_LEN <- SEQ_LEN - 1L

RtomicLazyChunksDataset <- dataset(
  name = "RtomicLazyChunksDataset",

  initialize = function(chunk_dir) {
    self$chunk_dir <- chunk_dir
    # 数据格式：.arrow（feather / Arrow IPC，ts_span_align.py 生成，v2 平面 float32/int32）。
    files <- sort(list.files(
      chunk_dir,
      pattern = "^chunk_.*\\.arrow$",
      full.names = TRUE
    ))

    if (length(files) == 0) {
      stop(sprintf(
        "未在目录 %s 找到任何 .arrow chunk 数据文件。\n请先运行 ts_span_align.py 生成已投影到学生词表的 chunk。",
        chunk_dir
      ))
    }

    # =====================================================================
    # 惰性加载：只扫每个文件的【元数据】（行数），不加载数据。
    # 全量跑（几万个 chunk / 几十万条样本）也不再吃内存、启动秒级。
    # 训练期按“块级洗牌”逐个读盘：每个 epoch 打乱 chunk 顺序 + chunk 内打乱样本，
    # 内存只常驻【一个】chunk（约 60~70MB），杜绝“全局 shuffle + 单 chunk 缓存 →
    # 相邻随机索引跨 chunk → 逐 batch 重读盘”的卡顿（比预载全部更省内存、启动更快）。
    # =====================================================================
    self$lens <- c()
    for (f in files) {
      n_rows <- arrow::read_feather(f, as_data_frame = FALSE)$num_rows
      if (n_rows %% TARGET_LEN != 0) {
        warning(sprintf(
          "%s 的行数 %d 不是序列长度 %d 的整数倍，将丢弃余数行",
          f, n_rows, TARGET_LEN
        ))
      }
      n <- n_rows %/% TARGET_LEN
      if (n <= 0) {
        warning(sprintf("跳过空 chunk %s", f))
        next
      }
      self$lens <- c(self$lens, n)
    }
    if (length(self$lens) == 0) {
      stop(sprintf(
        "目录 %s 下没有可读取的 .arrow chunk 文件（全部为空或无法解析）。",
        chunk_dir
      ))
    }

    self$chunk_files <- files
    self$n_chunks    <- length(self$lens)
    self$total_rows  <- sum(self$lens)

    # 块级洗牌计划 [total_rows, 2]：第 1 列 chunk 索引、第 2 列 chunk 内局部样本号。
    # 每个 epoch 重建（.getitem(1) 时触发），保证跨 epoch 顺序不同。
    self$perm <- NULL
    self$current_chunk_idx  <- -1L   # 当前常驻 chunk
    self$current_chunk_data <- NULL
    self$epoch_perm_count   <- 0L    # 已重建块级洗牌计划的次数（= 已开始的 epoch 数）

    message(sprintf(
      "发现 %d 个 chunk、共 %d 条样本（TARGET_LEN=%d）。采用【块级洗牌 + 惰性加载】：\n  启动零预载（不再吃内存），每个 epoch 按打乱的 chunk 顺序逐块读盘一次，单 chunk 常驻内存。",
      self$n_chunks, self$total_rows, TARGET_LEN
    ))
  },

  # 每个 epoch 重建块级洗牌计划：打乱 chunk 顺序 + chunk 内打乱样本
  new_epoch_perm = function() {
    co <- sample.int(self$n_chunks)   # 打乱的 chunk 顺序
    mats <- lapply(co, function(c) cbind(c, sample.int(self$lens[c])))
    self$perm <- do.call(rbind, mats)  # [total_rows, 2]，行主序即洗牌后的全局顺序
    self$epoch_perm_count <- self$epoch_perm_count + 1L
  },

  # 将 .arrow / feather 文件加载为张量结构（单 chunk）
  #   x / y_hard    : [N, T]    int64
  #   topk_ids      : [N, T, K] int32（存盘即 int32；loss 内会 to(long) 后 gather）
  #   topk_probs    : [N, T, K] float32（v2 平面 float32；loss 内直接使用）
  #   loss_mask     : [N, T]    bool
  load_arrow_chunk = function(f) {
    tb <- arrow::read_feather(f, as_data_frame = FALSE)
    n_rows <- tb$num_rows
    if (n_rows %% TARGET_LEN != 0) {
      warning(sprintf(
        "%s 的行数 %d 不是序列长度 %d 的整数倍，将丢弃余数行",
        f, n_rows, TARGET_LEN
      ))
    }
    N <- n_rows %/% TARGET_LEN

    # fixed_size_list 每元素恰好 K 个 → unlist 即行主序扁平向量，直接 view 成 [N,T,K]。
    # 【关键】topk_probs：v2 已是 fixed_size_list<float32>（ts_span_align.py 已升级），cast 是廉价
    # no-op；旧 v1 文件是 halffloat，R arrow 会把位模式按 int16 解读（f16 的 1.0=0x3C00→15360，
    # 曾致 loss~767027），必须先 cast 成 fixed_size_list<float32> 走正确的 f16→f32 转换。
    # 统一保留该 cast：对 v2 零开销、对 v1 修复读取。
    ids_list <- as.vector(tb$topk_ids)
    K        <- length(ids_list[[1]])
    pr_list  <- as.vector(tb$topk_probs$cast(arrow::fixed_size_list_of(arrow::float32(), K)))

    cd <- list(
      n          = N,
      x          = torch_tensor(as.vector(tb$x),                 dtype = torch_long())$view(c(N, TARGET_LEN)),
      y_hard     = torch_tensor(as.vector(tb$y_hard),            dtype = torch_long())$view(c(N, TARGET_LEN)),
      topk_ids   = torch_tensor(unlist(ids_list, use.names = FALSE), dtype = torch_int())$view(c(N, TARGET_LEN, K)),
      topk_probs = torch_tensor(unlist(pr_list, use.names = FALSE), dtype = torch_float32())$view(c(N, TARGET_LEN, K)),
      loss_mask  = torch_tensor(as.vector(tb$loss_mask),         dtype = torch_bool())$view(c(N, TARGET_LEN))
    )

    # ---- 数据校验（每次读入该 chunk 时执行一次）----
    max_id <- max(c(
      cd$x$max()$item(), cd$y_hard$max()$item(), cd$topk_ids$max()$item()
    ))
    if (max_id >= VOCAB_SIZE) {
      stop(sprintf(
        "数据文件 %s 中的 token ID 最大值 %d 超出学生词表大小 %d。\n当前数据仍是教师(Qwen)原生词表 ID，尚未投影到学生词表。\n请在 Python 侧将教师 Top-K 概率投影到学生词表（解码教师 token 文本 → 学生 BPE 重新编码）后再训练。",
        f, max_id, VOCAB_SIZE
      ))
    }
    max_p <- cd$topk_probs$max()$item()
    if (!is.nan(max_p) && max_p > 1.01) {
      stop(sprintf(
        "数据文件 %s 的 topk_probs 最大值 %g 超出归一化范围 [0,1]。\n正常应为 1.0（softmax 归一化）；若为 ~15360 说明 halffloat 读取路径未走 cast，\n若数据确系未归一化则请用最新 ts_span_align.py 重新生成。",
        f, max_p
      ))
    }

    cd
  },

  .getitem = function(i) {
    # 每个 epoch 从索引 1 重新开始 → 在此重建块级洗牌计划（保证跨 epoch 顺序不同）
    if (is.null(self$perm) || nrow(self$perm) != self$total_rows || i == 1L) {
      self$new_epoch_perm()
    }
    c   <- self$perm[i, 1L]   # chunk 索引
    loc <- self$perm[i, 2L]   # chunk 内局部样本号

    # 惰性读盘：只有跨 chunk 才加载，块内顺序访问命中缓存
    if (self$current_chunk_idx != c) {
      self$current_chunk_idx  <- c
      self$current_chunk_data <- self$load_arrow_chunk(self$chunk_files[c])
    }
    cd <- self$current_chunk_data

    # 行切片（R torch 底层零拷贝视图）
    list(
      x = list(x = cd$x[loc, ]),
      y = list(
        y_hard     = cd$y_hard[loc, ],
        topk_ids   = cd$topk_ids[loc, , ],
        topk_probs = cd$topk_probs[loc, , ],
        loss_mask  = cd$loss_mask[loc, ]
      )
    )
  },

  .length = function() {
    self$total_rows
  }
)

# 训练脚本中的 DataLoader 组装
# 兼容两种落盘位置：data/processed/（当前） 或 data/processed/chunks/（服务器侧）
chunk_dirs <- c("data/processed/chunks", "data/processed")
has_chunks <- vapply(chunk_dirs, function(d) {
  length(list.files(d, pattern = "^chunk_.*\\.arrow$")) > 0
}, logical(1))
chunk_dir <- chunk_dirs[which(has_chunks)][1]
if (is.na(chunk_dir)) {
  stop("未找到 .arrow chunk 数据文件（data/processed/chunks/ 或 data/processed/）。请先运行 ts_span_align.py 生成教师侧投影数据。")
}

train_dataset <- RtomicLazyChunksDataset(chunk_dir = chunk_dir)

train_dl <- dataloader(
  train_dataset,
  batch_size  = ENV_BATCH_SIZE,
  # 数据集内部已做块级洗牌（每个 epoch 重建 perm），此处顺序迭代即可；
  # 若 shuffle=TRUE 会产生全局随机索引，破坏块级局部性 → 逐 batch 跨 chunk 重读盘。
  shuffle     = FALSE,
  drop_last   = TRUE,
  num_workers = 0,
  pin_memory  = !is_mac
)

# =====================================================================
# 3. 带 Loss Mask 的 Top-K 蒸馏混合 Loss 函数 (原生 R torch 兼容版)
# =====================================================================
distill_loss_fn <- function(temperature = 1.0, alpha = 0.8) {
  function(output, target) {
    student_logits <- output$logits             # Shape: [B, S, Vocab]
    y_hard         <- target$y_hard             # Shape: [B, S]
    t_topk_ids     <- target$topk_ids           # Shape: [B, S, K]
    t_topk_probs   <- target$topk_probs         # Shape: [B, S, K]
    loss_mask      <- target$loss_mask          # Shape: [B, S] (torch_bool)

    # R 1-based 索引：第 3 维为 Vocab
    vocab_size <- student_logits$size(3)
    dev        <- student_logits$device

    # -------------------------------------------------------------------
    # 1. 索引边界安全钳位与 int64 类型对齐
    # -------------------------------------------------------------------
    t_topk_ids_long <- t_topk_ids$to(device = dev, dtype = torch_long())

    # 检查索引是否超出范围 [0, vocab_size - 1]
    invalid_mask <- (t_topk_ids_long < 0L) | (t_topk_ids_long >= vocab_size)

    # 替换非法索引为 1L：R torch 的 torch_gather 强制 1-based 索引，
    # 索引 0 会直接报错 "Indexing starts at 1 but found a 0"。
    # 学生词表为 1-based（PAD=1），1 是合法索引；且非法位置的教师概率已被置 0，
    # 因此该位置对 KL 的贡献为 0，不会引入噪声。
    safe_topk_ids <- torch_where(
      invalid_mask,
      torch_tensor(1L, device = dev, dtype = torch_long()),
      t_topk_ids_long
    )

    # 非法位置的教师概率置 0
    safe_topk_probs <- torch_where(
      invalid_mask,
      torch_tensor(0.0, device = dev, dtype = torch_float32()),
      t_topk_probs$to(device = dev, dtype = torch_float32())
    )

    # 防御：概率钳位到 [0,1]。数据端已归一化，正常情况无影响；
    # 防止脏数据（旧版未归一化/溢出的 probs）把 KL 项放大到数十万量级。
    safe_topk_probs <- torch_clamp(safe_topk_probs, min = 0.0, max = 1.0)

    # -------------------------------------------------------------------
    # 2. 掩码计算与 Log Softmax (显式使用 R 1-based dim = 3)
    # -------------------------------------------------------------------
    loss_mask_float <- loss_mask$to(device = dev, dtype = torch_float32())
    loss_mask_3d    <- loss_mask_float$unsqueeze(3) # 扩展至 [B, S, 1]
    t_topk_probs_masked <- safe_topk_probs * loss_mask_3d

    # 对 Student 词表维 (dim = 3) 做 Log Softmax
    student_log_probs <- nnf_log_softmax(student_logits / temperature, dim = 3)
    student_log_probs <- torch_clamp(student_log_probs, min = -100.0, max = 0.0)

    # -------------------------------------------------------------------
    # 3. 提取 Top-K 位置 log_probs (明确指定 dim = 3)
    # -------------------------------------------------------------------
    student_topk_log_probs <- torch_gather(
      student_log_probs,
      dim = 3,  # 必须使用 3 指向 Vocab 维，防止 R 转换 -1 错位
      index = safe_topk_ids
    )
    student_topk_log_probs <- torch_clamp(student_topk_log_probs, min = -100.0, max = 0.0)

    # -------------------------------------------------------------------
    # 4. KL 散度与损失融合
    # -------------------------------------------------------------------
    elem_kl <- t_topk_probs_masked * student_topk_log_probs

    # 熔断异常值
    bad_mask <- torch_isnan(elem_kl) | torch_isinf(elem_kl)
    if (bad_mask$any()$item()) {
      elem_kl <- torch_where(
        bad_mask,
        torch_tensor(0.0, device = dev, dtype = elem_kl$dtype),
        elem_kl
      )
    }

    # 对 Top-K 维 (dim = 3) 求和，得到 [B, S]
    token_kl <- - torch_sum(elem_kl, dim = 3)

    num_valid_tokens <- loss_mask_float$sum()$clamp(min = 1.0)
    kl_loss <- (token_kl * loss_mask_float)$sum() / num_valid_tokens
    kl_loss <- kl_loss * (temperature ^ 2)

    # 硬标签 Cross Entropy Loss
    logits_flat <- student_logits$view(c(-1, vocab_size))
    y_flat      <- y_hard$view(c(-1))
    mask_flat   <- loss_mask$view(c(-1))

    ce_loss     <- nnf_cross_entropy(logits_flat[mask_flat, ], y_flat[mask_flat])

    alpha * kl_loss + (1 - alpha) * ce_loss
  }
}

# =====================================================================
# 4. 优化器与 WSD 学习率调度器
# =====================================================================
custom_grouped_optim <- function(params, lr = 5e-4, weight_decay = 0.05, ...) {
  emb_names <- grep("^tok_emb|^pos_emb|^lm_head", names(params), value = TRUE)
  other_names <- setdiff(names(params), emb_names)
  grouped_params <- list(
    list(params = params[emb_names], lr = lr * 0.5),
    list(params = params[other_names], lr = lr)
  )
  optim_adamw(grouped_params, weight_decay = weight_decay, ...)
}

wsd_multiplier <- function(step, total_steps, warmup_pct = 0.05, decay_pct = 0.15) {
  current_step <- as.numeric(step) + 1
  total_steps  <- as.numeric(total_steps)

  warmup_steps <- total_steps * warmup_pct
  decay_steps  <- total_steps * decay_pct
  stable_steps <- total_steps - warmup_steps - decay_steps

  if (current_step <= warmup_steps) {
    return(current_step / warmup_steps)
  } else if (current_step <= warmup_steps + stable_steps) {
    return(1.0)
  } else {
    decay_step   <- current_step - (warmup_steps + stable_steps)
    progress     <- decay_step / decay_steps
    cosine_decay <- 0.5 * (1 + cos(pi * progress))
    return(max(0.1, cosine_decay))
  }
}

DISTILL_EPOCHS <- 3
total_steps    <- DISTILL_EPOCHS * length(train_dl)

luz_callback_clip_grad <- luz_callback(
  "clip_grad",
  initialize = function(max_norm = 1.0) {
    self$max_norm <- max_norm
  },
  on_backward_end = function() {
    torch::nn_utils_clip_grad_norm_(ctx$model$parameters, max_norm = self$max_norm)
  }
)

base_callbacks <- list(
  luz_callback_lr_scheduler(
    torch::lr_lambda,
    lr_lambda = function(step) {
      wsd_multiplier(step, total_steps = total_steps, warmup_pct = 0.05, decay_pct = 0.15)
    },
    call_on = "on_train_batch_end"
  ),
  luz_callback_clip_grad(1.0),
  luz_callback_model_checkpoint(
    path = "checkpoints/distill_model_{epoch:02d}.pt",
    save_best_only = FALSE,
    monitor = "train_loss"
  )
)

if (ENV_USE_AMP && cuda_is_available()) {
  base_callbacks <- append(base_callbacks, list(luz_callback_mixed_precision()))
}

cat(sprintf("\n[%s] 启动 15M 学生模型从头蒸馏训练 (Total Steps: %d)...\n",
            format(Sys.time(), "%H:%M:%S"), total_steps))

fitted_distill_lm <- RtomicCausalLM |>
  setup(
    loss = distill_loss_fn(temperature = 1.0, alpha = 0.8),
    optimizer = custom_grouped_optim
  ) |>
  set_hparams(
    vocab_size = VOCAB_SIZE,
    dim = DIM,
    n_layers = N_LAYERS,
    n_heads = N_HEADS,
    max_seq_len = SEQ_LEN
  ) |>
  set_opt_hparams(lr = 3e-4, weight_decay = 0.05) |>
  fit(
    data = train_dl,
    epochs = DISTILL_EPOCHS,
    accelerator = accelerator(),
    callbacks = base_callbacks,
    verbose = TRUE
  )
