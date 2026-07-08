# ============================================================
# Shared configuration: environment detection + library loading
# Source this first in any script that needs torch/luz.
# Hyperparameters can be overridden after sourcing.
# ============================================================
library(torch)
library(luz)
library(R6)
library(tokenizers.bpe)
torch_manual_seed(42)

# --- Environment auto-detection ---
is_mac <- Sys.info()["sysname"] == "Darwin"
is_linux <- Sys.info()["sysname"] == "Linux"

if (is_mac) {
  cat("--- 检测到 Mac 环境：切换至【本地测试模式】---\n")
  WORK_DIR <<- "~/github/stat-llm-lab/"
  Sys.setenv(OMP_NUM_THREADS = "4")
  device = "mps"
} else if (is_linux && cuda_is_available()) {
  cat("--- 检测到 Linux + GPU 环境：切换至【服务器 GPU 训练模式】---\n")
  WORK_DIR <<- "/root/autodl-tmp/stat-llm-lab/"
  Sys.setenv(OMP_NUM_THREADS = "4")
  device = "cuda"
} else if (is_linux && !cuda_is_available()) {
  cat("--- 检测到 Linux + CPU 环境：切换至【服务器 CPU 生产/服务模式】---\n")
  # 第 3 个 Linux CPU 服务器的实际项目路径
  WORK_DIR <<- "~/github/stat-llm-lab/"
  Sys.setenv(OMP_NUM_THREADS = "2") # 生产服务器限制为 2 线程，降低 CPU 争抢与并发负载
}

setwd(WORK_DIR)

# --- Model hyperparameters (defaults, scripts may override) ---

VOCAB_SIZE   <- 2^14
DIM          <- 320
N_LAYERS     <- 8
N_HEADS      <- 8
SEQ_LEN      <- 512

BPE_MODEL_FILE <- "models/rtomic_bpe.model"
