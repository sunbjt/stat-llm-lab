# causal_lm/02_distill.R
Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")

# --- 兼容从 Distillation/ 子目录或仓库根目录运行 ---
if (basename(normalizePath(getwd())) == "Distillation") setwd("..")

library(torch)
library(arrow)

source("config.R")
source("causal_lm/CausalLM_model.R")

## 1. 环境专属超参配置
if (is_mac) {
  ENV_BATCH_SIZE  <- 2
  ENV_USE_AMP     <- FALSE
} else {
  ENV_BATCH_SIZE  <- 64  # 15M 小模型显存开销极小，RTX 3090 可轻松开启 32~64
  ENV_USE_AMP     <- TRUE
}

# 梯度累加系数：有效 batch = ENV_BATCH_SIZE × 系数（服务器 64 × 2 = 128）
ENV_GRAD_ACCUM <- 2L

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

# =====================================================================
# RtomicBinDataset —— 定长 .bin 蒸馏软标签数据集（双模式）
#
# 由 pack_bin.py 把 01_ts_span_align.py 的 .arrow chunk 打包成单文件定长记录：
#   每样本一条定长记录（little-endian，字段按序紧排、无对齐填充）：
#     x          [T]   int16
#     y_hard     [T]   int16
#     topk_ids   [T,K] int16
#     topk_probs [T,K] float32
#     loss_mask  [T]   uint8
#   配套 .meta（key=value）：top_k / target_len / stride / n_samples / endian。
#
# 双模式（借鉴 pure_infonce/01_pretrain.R 的全量载入模式，加内存护栏）：
#   * RAM 模式（默认）：数据总字节 ≤ 内存上限（系统 RAM × ram_frac）时，启动时把整个
#     .bin 读进一个普通 R raw 向量，.getitem 只做内存切片 + rawConnection 解码。
#     无磁盘 I/O、O(1) 随机访问，且可开 num_workers>0（持有普通向量可被 worker 共享）。
#   * STREAM 模式（兜底）：数据超出内存上限时，退回 seek + readBin 逐样本读取，
#     内存 O(batch)，任意规模都能跑；此时 num_workers 必须为 0（持有连接无法共享）。
#
# 返回结构与 RtomicLazyChunksDataset 完全一致（供 distill_loss_fn 使用）：
#   list(x = list(x = [T] long),
#        y = list(y_hard = [T] long, topk_ids = [T,K] int32,
#                 topk_probs = [T,K] float32, loss_mask = [T] bool))
# =====================================================================

# 跨平台探测物理内存（字节）。失败返回 NA → 用保守默认上限。
detect_ram_bytes <- function() {
  if (.Platform$OS.type != "unix") return(NA_real_)
  if (Sys.info()["sysname"] == "Darwin") {
    out <- suppressWarnings(system("sysctl -n hw.memsize", intern = TRUE))
    if (length(out) == 1L) as.numeric(out) else NA_real_
  } else {
    mem <- tryCatch(readLines("/proc/meminfo"), error = function(e) character(0))
    line <- grep("^MemTotal:", mem, value = TRUE)
    if (length(line) == 1L) {
      kb <- as.numeric(sub(".*:\\s*([0-9]+).*", "\\1", line))
      if (!is.na(kb)) kb * 1024 else NA_real_
    } else NA_real_
  }
}


RtomicBinDataset <- dataset(
  name = "RtomicBinDataset",

  initialize = function(bin_file, meta_file, target_len, ram_frac = 0.6) {
    if (!file.exists(bin_file) || !file.exists(meta_file)) {
      stop(sprintf(
        "缺少定长 .bin 数据（%s / %s）。\n请先运行：python pack_bin.py --chunk-dir <chunk目录> --out %s",
        bin_file, meta_file, bin_file
      ))
    }

    # ---- 解析 .meta（key=value 文本，零依赖）----
    meta <- readLines(meta_file, warn = FALSE)
    kv <- strsplit(meta, "=", fixed = TRUE)
    kv <- kv[lengths(kv) == 2L]
    m <- setNames(
      trimws(vapply(kv, function(p) p[[2]], character(1))),
      trimws(vapply(kv, function(p) p[[1]], character(1)))
    )

    self$target_len <- as.integer(target_len)
    self$K          <- as.integer(m[["top_k"]])
    self$stride     <- as.numeric(m[["stride"]])
    self$n_samples  <- as.integer(m[["n_samples"]])

    # 防御：stride 必须与 (target_len, K) 自洽，防止 --top-k 变动后旧 bin 被误用
    expect <- self$target_len * (2L + 2L + 2L * self$K + 4L * self$K + 1L)
    if (self$stride != expect) {
      stop(sprintf(
        "meta 中 stride=%d 与 target_len=%d、K=%d 推算值 %d 不一致。\n请用 pack_bin.py 重新打包。",
        self$stride, self$target_len, self$K, expect
      ))
    }

    self$raw  <- NULL   # RAM 模式：整文件 raw 向量
    self$con  <- NULL   # STREAM 模式：文件连接
    self$mode <- "ram"

    total_bytes <- self$stride * self$n_samples
    self$total_bytes <- total_bytes   # 供调用方决定 num_workers（luz/callr 会序列化拷贝整个数据集）
    ram <- detect_ram_bytes()
    cap <- if (is.na(ram)) 8e9 else ram * ram_frac
    gb  <- function(b) sprintf("%.1f GB", b / 1e9)

    if (total_bytes <= cap) {
      con <- file(bin_file, "rb")
      self$raw <- readBin(con, what = "raw", n = total_bytes)
      close(con)
      message(sprintf(
        "【RAM 模式】%d 条 × %d 字节/条 = %s ≤ 内存上限 %s：全量载入内存，切片随机访问、无读盘卡顿。\n  num_workers 由调用方按 total_bytes 决定：数据跨进程会被序列化复制，大 bin 必须保持 0。",
        self$n_samples, self$stride, gb(total_bytes), gb(cap)
      ))
    } else {
      self$mode <- "stream"
      self$con <- file(bin_file, "rb")
      message(sprintf(
        "【STREAM 模式】数据 %s 超过内存上限 %s：退回 seek+readBin 逐样本读取（内存 O(batch)），num_workers 必须保持 0。",
        gb(total_bytes), gb(cap)
      ))
    }
  },

  .getitem = function(i) {
    T <- self$target_len
    K <- self$K

    if (self$mode == "ram") {
      # O(1) 内存切片：整条记录 raw 切片 → rawConnection 解码 5 个字段
      start <- (i - 1L) * self$stride + 1L
      rec <- self$raw[start:(start + self$stride - 1L)]
      rc <- rawConnection(rec, "rb")
      on.exit(close(rc), add = TRUE)
      x   <- readBin(rc, "integer", n = T,     size = 2L, signed = TRUE,  endian = "little")
      y   <- readBin(rc, "integer", n = T,     size = 2L, signed = TRUE,  endian = "little")
      ids <- readBin(rc, "integer", n = T * K, size = 2L, signed = TRUE,  endian = "little")
      pr  <- readBin(rc, "numeric", n = T * K, size = 4L, endian = "little")
      m   <- readBin(rc, "integer", n = T,     size = 1L, signed = FALSE, endian = "little")
    } else {
      con <- self$con
      seek(con, where = (i - 1L) * self$stride, origin = "start")
      x   <- readBin(con, "integer", n = T,     size = 2L, signed = TRUE,  endian = "little")
      y   <- readBin(con, "integer", n = T,     size = 2L, signed = TRUE,  endian = "little")
      ids <- readBin(con, "integer", n = T * K, size = 2L, signed = TRUE,  endian = "little")
      pr  <- readBin(con, "numeric", n = T * K, size = 4L, endian = "little")
      m   <- readBin(con, "integer", n = T,     size = 1L, signed = FALSE, endian = "little")
    }

    list(
      x = list(x = torch_tensor(x, dtype = torch_long())),
      y = list(
        y_hard     = torch_tensor(y, dtype = torch_long()),
        topk_ids   = torch_tensor(ids, dtype = torch_int())$view(c(T, K)),
        topk_probs = torch_tensor(pr, dtype = torch_float32())$view(c(T, K)),
        loss_mask  = torch_tensor(m > 0, dtype = torch_bool())
      )
    )
  },

  .length = function() {
    self$n_samples
  }
)

# 训练脚本中的 DataLoader 组装
# ---------------------------------------------------------------------------
# 首选【定长 .bin + seek】路径（pack_bin.py 打包）：训练期 O(1) 定位、逐样本小读，
# 无整块读盘卡顿、内存 O(batch)。.bin 缺失时回退惰性 arrow chunk 路径。
# ---------------------------------------------------------------------------
BIN_FILE <- "data/processed/distill_labels.bin"
BIN_META <- "data/processed/distill_labels.meta"

if (file.exists(BIN_FILE) && file.exists(BIN_META)) {
  train_dataset <- RtomicBinDataset(
    bin_file   = BIN_FILE,
    meta_file  = BIN_META,
    target_len = TARGET_LEN
  )
  # luz/torch 的 num_workers>0 用 callr 启动子进程：把【整个数据集对象】saveRDS
  # 序列化到临时文件、每个 worker 再各自 load 一份完整拷贝。RAM 模式下 self$raw 就是
  # 整个 bin（十几 GB）：开 worker 会写一份十几 GB 临时文件 + 每个 worker 再吃十几 GB
  # 内存 —— 正是刚才服务器上 "error writing to connection" / 系统盘被打满的根因。
  # 因此只有数据足够小、拷贝几份可承受时才开 worker；否则保持 0
  # （RAM 模式本身已消除读盘卡顿，num_workers=0 不影响该收益）。
  MAX_SERIALIZE_BYTES <- 2e9   # bin ≤ 2GB 时序列化拷贝可承受
  n_workers <- if (identical(train_dataset$mode, "ram") &&
                   train_dataset$total_bytes <= MAX_SERIALIZE_BYTES) 4L else 0L
  train_dl <- dataloader(
    train_dataset,
    batch_size  = ENV_BATCH_SIZE,
    shuffle     = TRUE,    # bin 为 O(1) 随机访问，可全局洗牌
    drop_last   = TRUE,
    num_workers = n_workers,
    pin_memory  = !is_mac
  )
} else {
  # 兜底：惰性 arrow chunk（建议先跑 pack_bin.py 消除训练期读盘卡顿）
  chunk_dirs <- c("data/processed/chunks", "data/processed")
  has_chunks <- vapply(chunk_dirs, function(d) {
    length(list.files(d, pattern = "^chunk_.*\\.arrow$")) > 0
  }, logical(1))
  chunk_dir <- chunk_dirs[which(has_chunks)][1]
  if (is.na(chunk_dir)) {
    stop("未找到 .bin 或 .arrow chunk 数据。请先运行 01_ts_span_align.py 生成投影数据，再 pack_bin.py 打包。")
  }
  train_dataset <- RtomicLazyChunksDataset(chunk_dir = chunk_dir)
  message("未找到 distill_labels.bin，回退到惰性 chunk 路径（训练期会周期性读盘卡顿）。建议先运行 pack_bin.py。")
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
}

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

    # 检查索引是否超出范围。真实投影数据是 1-based surface id（∈ [1, 16383]，
    # PAD=1/UNK=2，见 01_ts_span_align.py）；<=0 与 >=vocab 一律视为非法：
    # 0-based 数据、越界 id 都被替换为 1L（概率同步置 0，对 KL 零贡献）。
    invalid_mask <- (t_topk_ids_long <= 0L) | (t_topk_ids_long >= vocab_size)

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

    total <- alpha * kl_loss + (1 - alpha) * ce_loss
    list(total = total, kl = kl_loss, ce = ce_loss)
  }
}

# =====================================================================
# 4. 原生手写训练循环
#    luz 的 fit() 封装固定每 batch step + zero_grad，且 loss 只能返回单张量：
#    —— KL/CE 无法分开观测、无法梯度累加。改为手写循环后两者都可控。
# =====================================================================
DISTILL_EPOCHS <- 3
BASE_LR        <- 3e-4
WEIGHT_DECAY   <- 0.05
ALPHA          <- 0.8
TEMPERATURE    <- 1.0
GRAD_CLIP      <- 1.0

accum_steps <- as.integer(ENV_GRAD_ACCUM)   # 梯度累加系数（env 块定义，服务器 = 2）

dir.create("checkpoints", showWarnings = FALSE, recursive = TRUE)

# ---- 设备自推导（config.R 的 device 是字符串，这里转 torch_device；MPS 不支持 non_blocking）----
device <- torch_device(if (is_mac) "mps" else if (cuda_is_available()) "cuda" else "cpu")
nb      <- device$type == "cuda"
use_amp <- isTRUE(ENV_USE_AMP) && device$type == "cuda"

# ---- batch 数与 WSD 总优化步数 ----
n_batches <- length(train_dl)   # 实测可靠返回 drop_last 后的 batch 数
if (is.na(n_batches) || n_batches < 1) {
  n_batches <- length(train_dataset) %/% ENV_BATCH_SIZE
}
if (n_batches < 1) stop("训练 batch 数为 0，请检查数据")
steps_per_epoch   <- ceiling(n_batches / accum_steps)
total_optim_steps <- DISTILL_EPOCHS * steps_per_epoch

# ---- WSD（warmup_pct=0.05, decay_pct=0.15, 余弦衰减地板 0.1）----
wsd_multiplier <- function(step, total_steps, warmup_pct = 0.05, decay_pct = 0.15) {
  current      <- as.numeric(step)
  total_steps  <- as.numeric(total_steps)
  warmup_steps <- max(1, round(total_steps * warmup_pct))
  decay_steps  <- max(1, round(total_steps * decay_pct))
  stable_steps <- max(1, total_steps - warmup_steps - decay_steps)

  if (current <= 0) return(0.0)
  if (current <= warmup_steps) return(current / warmup_steps)
  if (current <= warmup_steps + stable_steps) return(1.0)
  decay_step <- current - (warmup_steps + stable_steps)
  progress   <- min(1.0, decay_step / decay_steps)
  max(0.1, 0.5 * (1 + cos(pi * progress)))
}

# ---- 模型 ----
model <- RtomicCausalLM(
  vocab_size  = VOCAB_SIZE,
  dim         = DIM,
  n_layers    = N_LAYERS,
  n_heads     = N_HEADS,
  max_seq_len = SEQ_LEN
)$to(device = device)

# ---- 分组优化器：emb 组 LR×0.5、其余全 LR（与旧 custom_grouped_optim 一致）----
param_names <- names(model$parameters)
emb_names   <- grep("^tok_emb", param_names, value = TRUE)
other_names <- setdiff(param_names, emb_names)
all_params  <- model$parameters
grouped_params <- list(
  list(params = all_params[emb_names],   lr = BASE_LR * 0.5),
  list(params = all_params[other_names], lr = BASE_LR)
)
optimizer <- optim_adamw(grouped_params, lr = BASE_LR, weight_decay = WEIGHT_DECAY)
# param_groups 无 initial_lr 字段，WSD 分组写回需自己按组存 base lr
group_base_lrs <- vapply(optimizer$param_groups, function(pg) pg$lr, numeric(1))

# ---- Loss / AMP / 日志 ----
criterion <- distill_loss_fn(temperature = TEMPERATURE, alpha = ALPHA)
scaler    <- if (use_amp) cuda_amp_grad_scaler() else NULL

log_file <- sprintf("checkpoints/distill_loss_%s.csv", format(Sys.time(), "%H%M"))
cat("epoch,iter,kl,ce,total,lr,global_step\n", file = log_file)

optimizer$zero_grad()
global_step <- 0
init_mult   <- wsd_multiplier(1, total_steps = total_optim_steps)
for (g in seq_along(optimizer$param_groups)) {
  optimizer$param_groups[[g]]$lr <- group_base_lrs[[g]] * init_mult
}
current_lr <- group_base_lrs[2] * init_mult   # 日志用主组（全 LR）

cat(sprintf("\n[%s] 15M 学生模型蒸馏 (device=%s AMP=%s accum=%d | total_optim_steps=%d)...\n",
            format(Sys.time(), "%H:%M:%S"), device$type, use_amp, accum_steps, total_optim_steps))

for (epoch in 1:DISTILL_EPOCHS) {
  curtime    <- Sys.time()
  model$train()
  batch_idx   <- 0
  epoch_kl    <- 0; epoch_ce <- 0; epoch_total <- 0
  remainder   <- n_batches %% accum_steps

  coro::loop(for (batch in train_dl) {
    batch_idx <- batch_idx + 1

    # 梯度累加：epoch 尾部余数 batch 用部分累加，末尾强制 step
    current_accum_steps <- if (remainder != 0 && batch_idx > n_batches - remainder) {
      remainder
    } else {
      accum_steps
    }

    input_data <- list(x = batch$x$x$to(device = device, non_blocking = nb))
    target <- list(
      y_hard     = batch$y$y_hard$to(device = device, non_blocking = nb),
      topk_ids   = batch$y$topk_ids$to(device = device, non_blocking = nb),
      topk_probs = batch$y$topk_probs$to(device = device, non_blocking = nb),
      loss_mask  = batch$y$loss_mask$to(device = device, non_blocking = nb)
    )

    if (use_amp) {
      with_autocast(device_type = "cuda", {
        output <- model(input_data)
        losses <- criterion(output, target)
        loss_to_backprop <- losses$total / current_accum_steps
      })
    } else {
      output <- model(input_data)
      losses <- criterion(output, target)
      loss_to_backprop <- losses$total / current_accum_steps
    }

    # 日志用未除以累加步数的原始 batch 均值（KL/CE 分开记录）
    kl_val    <- losses$kl$item()
    ce_val    <- losses$ce$item()
    total_val <- losses$total$item()

    if (use_amp) scaler$scale(loss_to_backprop)$backward() else loss_to_backprop$backward()

    should_step <- (batch_idx %% accum_steps == 0 || batch_idx == n_batches)
    if (should_step) {
      global_step <- global_step + 1

      if (use_amp) {
        scaler$unscale_(optimizer)
        nn_utils_clip_grad_norm_(model$parameters, max_norm = GRAD_CLIP)
        scaler$step(optimizer)
        scaler$update()
      } else {
        nn_utils_clip_grad_norm_(model$parameters, max_norm = GRAD_CLIP)
        optimizer$step()
      }

      # WSD 分组 LR 写回：R 的 `for (pg in param_groups) pg$lr <- x` 是复制后
      # 修改（no-op），必须按下标写回 optimizer$param_groups[[g]]$lr。
      mult <- wsd_multiplier(min(global_step + 1, total_optim_steps), total_steps = total_optim_steps)
      for (g in seq_along(optimizer$param_groups)) {
        optimizer$param_groups[[g]]$lr <- group_base_lrs[[g]] * mult
      }
      current_lr <- group_base_lrs[2] * mult

      optimizer$zero_grad()
    }

    epoch_kl    <- epoch_kl + kl_val
    epoch_ce    <- epoch_ce + ce_val
    epoch_total <- epoch_total + total_val

    if (batch_idx %% 50 == 0 || batch_idx == 1) {
      w <- nchar(as.character(n_batches))
      cat(sprintf(
        "Epoch [%d/%d] Batch [%*d/%d] step=%*d | kl=%.4f ce=%.4f total=%.4f | lr=%.2e\n",
        epoch, DISTILL_EPOCHS, w, batch_idx, n_batches, w, global_step,
        kl_val, ce_val, total_val, current_lr
      ))
    }
    cat(sprintf("%d,%d,%.6f,%.6f,%.6f,%.6e,%d\n",
                epoch, batch_idx, kl_val, ce_val, total_val, current_lr, global_step),
        file = log_file, append = TRUE)
  })

  avg_kl    <- epoch_kl / batch_idx
  avg_ce    <- epoch_ce / batch_idx
  avg_total <- epoch_total / batch_idx
  cat(sprintf("=== Epoch %d 结束，平均 kl=%.6f ce=%.6f total=%.6f ===\n",
              epoch, avg_kl, avg_ce, avg_total))

  ckpt_path <- sprintf("checkpoints/distill_model_%02d.pt", epoch)
  torch_save(
    list(
      model       = model$state_dict(),
      optimizer   = optimizer$state_dict(),
      epoch       = epoch,
      global_step = global_step,
      kl          = avg_kl,
      ce          = avg_ce,
      total       = avg_total,
      current_lr  = current_lr,
      config = list(
        vocab_size   = VOCAB_SIZE,
        dim          = DIM,
        n_layers     = N_LAYERS,
        n_heads      = N_HEADS,
        max_seq_len  = SEQ_LEN,
        batch_size   = ENV_BATCH_SIZE,
        accum_steps  = accum_steps,
        base_lr      = BASE_LR,
        weight_decay = WEIGHT_DECAY,
        temperature  = TEMPERATURE,
        alpha        = ALPHA,
        objective    = "distill_casual_lm_native"
      )
    ),
    ckpt_path
  )
  cat(sprintf("Checkpoint saved: %s\n", ckpt_path))
  cat('耗时', round(as.numeric(difftime(Sys.time(), curtime, units = "mins")), 2), '分钟\n')
}

cat("\n蒸馏训练完成！\n")
