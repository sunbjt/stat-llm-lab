# =====================================================================
# LRP 大模型纯 httpuv 服务端 (SFT 微调版)
# =====================================================================
source("config.R")

# server 端需要引入的包
library(httpuv)
library(later)
library(jsonlite)

Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")
device <- torch_device("cpu")

source("utils/BPETokenizer.R")
source("latent_residual/LRP_model.R")              # 指向 LRP 架构
source("server/generate_stream_lrp.R")     # 指向新的 LRP 异步采样流

# =====================================================================
# ==== 1. 全局状态与日志 ====
# =====================================================================
MODEL_BUSY <- FALSE
LOG_TEXT_FILE  <- "server/user_prompts_lrp.log"
LOG_SFT_JSONL  <- "server/qa_data.jsonl"

log_event <- function(type, prompt = NULL, client_ip = "UNKNOWN", err_msg = NULL) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  if (type == "PROMPT") {
    log_line <- sprintf("[%s] [PROMPT] [IP: %s] %s\n", timestamp, client_ip, gsub("\n", " ", prompt))
  } else {
    log_line <- sprintf("[%s] [ERROR] [IP: %s] %s\n", timestamp, client_ip, err_msg)
  }
  cat(log_line)
  flush(stdout())
  try(cat(log_line, file = LOG_TEXT_FILE, append = TRUE), silent = TRUE)
}

log_sft_jsonl <- function(prompt, output, client_ip) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  try({
    sft_node <- list(
      input = trimws(prompt),
      output = trimws(output),
      meta = list(timestamp = timestamp, ip = client_ip, engine = "JEPA_SFT")
    )
    cat(paste0(jsonlite::toJSON(sft_node, auto_unbox = TRUE), "\n"), file = LOG_SFT_JSONL, append = TRUE)
  }, silent = TRUE)
}

run_garbage_collection <- function() {
  gc(verbose = FALSE)
  if (cuda_is_available()) torch_cuda_empty_cache()
}

# =====================================================================
# ==== 2. 模型加载 ====
# =====================================================================
cat(sprintf("正在启动 LRP SFT 推理引擎... [设备: %s]\n", device$type))
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)
model <- RtomicLRP(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN)

WEIGHT_PATH <- "checkpoints/lrp_sft_epoch_05.pt"
ckpt <- torch_load(WEIGHT_PATH, device = "cpu")
state_dict <- if ("model" %in% names(ckpt)) ckpt$model else ckpt
model$load_state_dict(state_dict, strict = FALSE)
model <- model$to(device = device)
model$eval()
cat("JEPA 推理底座已安全就绪！\n")

# =====================================================================
# ==== 3. WebSocket 服务端 ====
# =====================================================================
app <- list(
  call = function(req) {
    path <- req$PATH_INFO
    if (path == "/" && req$REQUEST_METHOD == "GET") {
      target_file <- "server/www/index.html"
      if (!file.exists(target_file)) return(list(status=404L, body="404 Not Found"))
      return(list(
        status = 200L,
        headers = list("Content-Type" = "text/html; charset=utf-8"),
        body = rawToChar(readBin(target_file, what = "raw", n = file.info(target_file)$size))
      ))
    }
    return(list(status=404L, body="Not Found"))
  },
  
  onWSOpen = function(ws) {
    client_ip <- tryCatch(ws$request$REMOTE_ADDR, error = function(e) "UNKNOWN")
    
    ws$onMessage(function(binary, message) {
      if (MODEL_BUSY) {
        ws$send("[[ERROR]]服务器正忙，请稍后再试。")
        return()
      }
      tryCatch({
        msg_json <- jsonlite::fromJSON(message)
        prompt <- msg_json$prompt
        
        # 1. 基础的空字符检查
        prompt <- trimws(prompt)
        if (is.null(prompt) || nchar(prompt) == 0) return()
        
        # ==========================================================
        # 核心新增：Prompt 自动闭环补全逻辑
        # 检测结尾是否已经是常见标点（中文或英文的句号、问号、叹号）
        # ==========================================================
        if (!grepl("[。？！\\.\\?!]$", prompt)) {
          # 如果没有标点结尾，默认给它补一个中文句号（或问号）
          # 补句号比补问号更通用，因为类似“写一段代码”属于祈使句，补问号会有点怪
          prompt <- paste0(prompt, "。")
        }
        
        MODEL_BUSY <<- TRUE
        full_output <- ""
        
        # 记录日志时，记录补全后的 prompt
        log_event("PROMPT", prompt = prompt, client_ip = client_ip)        
        # 调用专门的 JEPA 生成逻辑
        generate_lrp_text_async(
          model       = model,
          tokenizer   = tokenizer,
          prompt      = prompt,
          max_new_tokens = 200,
          temperature = 0.3,
          top_k       = 5,
          rep_penalty = 1.2,
          on_token    = function(token_text) {
            ws$send(token_text)
            full_output <<- paste0(full_output, token_text)
          },
          on_done     = function() {
            MODEL_BUSY <<- FALSE
            ws$send("[[DONE]]")
            log_sft_jsonl(prompt, full_output, client_ip)
            run_garbage_collection()
          }
        )
      }, error = function(e) {
        MODEL_BUSY <<- FALSE
        err_msg <- conditionMessage(e)
        log_event("ERROR", client_ip = client_ip, err_msg = err_msg)
        ws$send(paste0("[[ERROR]]推理链路异常: ", err_msg))
        run_garbage_collection()
      })
    })
    
    ws$onClose(function() {
      run_garbage_collection()
    })
  }
)

# =====================================================================
# ==== 4. 启动本地监听 ====
# =====================================================================
PORT <- 8080

# 核心：就是漏了这一行！必须显式创建服务器实例
server <- startServer(host = "0.0.0.0", port = PORT, app = app)

cat(sprintf("\nLRP 全栈流式服务端启动成功！访问: http://127.0.0.1:%d/\n", PORT))
cat("按 Esc 或 Ctrl+C 停止服务\n")

while(TRUE) {
  service(100)
  Sys.sleep(0.01)
}