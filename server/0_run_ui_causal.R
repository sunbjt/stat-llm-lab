# =====================================================================
# causal 大模型纯 httpuv 服务端 (SFT 微调版)
# =====================================================================
source("config.R")

# server 端需要引入的包
library(httpuv)
library(later)
library(jsonlite)

Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")
device <- torch_device("cpu")

source("utils/BPETokenizer.R")
source("causal_lm/CausalLM_model.R")
source("server/generate_stream_causal.R")

# =====================================================================
# ==== 1. 全局状态与生产级双向日志配置 ====
# =====================================================================
# 全局模型互斥锁：彻底隔离多用户并发，防止底层显存/状态踩踏
MODEL_BUSY <- FALSE

LOG_TEXT_FILE  <- "server/user_prompts_causal.log"
LOG_SFT_JSONL  <- "server/qa_data.jsonl"

# 基础文本日志与异常记录系统
log_event <- function(type, prompt = NULL, client_ip = "UNKNOWN", err_msg = NULL) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  
  if (type == "PROMPT") {
    clean_msg <- gsub("\n", " ", prompt)
    log_line <- sprintf("[%s] [PROMPT] [IP: %s] %s\n", timestamp, client_ip, clean_msg)
  } else {
    log_line <- sprintf("[%s] [ERROR] [IP: %s] %s\n", timestamp, client_ip, err_msg)
  }
  
  cat(log_line)     # 同步输出到 R 控制台
  flush(stdout())   # 强行刷新控制台输出缓冲，保证实时可见
  
  tryCatch({
    cat(log_line, file = LOG_TEXT_FILE, append = TRUE)
  }, error = function(e) {
    cat(sprintf("[CRITICAL] 写入文本日志失败: %s\n", conditionMessage(e)))
  })
}

# 核心新增：流式生成结束后，将完整的「问题+回答」对落盘为标准 JSONL 训练集格式
log_sft_jsonl <- function(prompt, output, client_ip) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  tryCatch({
    sft_node <- list(
      input = trimws(prompt),
      output = trimws(output), # 完美捕获模型的完整输出文本
      meta = list(
        timestamp = timestamp,
        ip = client_ip,
        engine = "CausalLM"
      )
    )
    # 转化为单行不可缩进的 JSON 并追加落盘
    json_line <- jsonlite::toJSON(sft_node, auto_unbox = TRUE)
    cat(paste0(json_line, "\n"), file = LOG_SFT_JSONL, append = TRUE)
  }, error = function(e) {
    cat(sprintf("[CRITICAL] 写入 JSONL 数据集失败: %s\n", conditionMessage(e)))
  })
}

# 内存与英伟达显存双重主动碎片清理
run_garbage_collection <- function() {
  gc(verbose = FALSE)
  if (cuda_is_available()) {
    torch_cuda_empty_cache()
  }
}
# =====================================================================


# 2. 模型与分词器安全解包加载
cat("正在加载 SFT 最终微调权重...\n")
tokenizer <- RtomicBPETokenizer$new(model_file = BPE_MODEL_FILE, vocab_size = VOCAB_SIZE)
model <- RtomicCausalLM(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN)

WEIGHT_PATH <- "checkpoints/causal_sft_epoch_05.pt"
ckpt <- torch_load(WEIGHT_PATH, device = "cpu")
state_dict <- if (!is.null(ckpt$model)) ckpt$model else ckpt
model$load_state_dict(state_dict, strict = FALSE)
model <- model$to(device = device)
model$eval()
cat("Rtomic 推理底座已安全就绪！\n")


# 3. 构建 httpuv 复合服务端
app <- list(
  # ---- HTTP 通道：静态资源 + favicon ----
  call = function(req) {
    path <- req$PATH_INFO
    method <- req$REQUEST_METHOD
    
    # 前端主页
    if (path == "/" && method == "GET") {
      target_file <- "server/www/index.html"  # 此处已修正为 R 标准赋值号 <-
      if (!file.exists(target_file)) {
        return(list(
          status = 404L,
          headers = list("Content-Type" = "text/plain; charset=utf-8"),
          body = "404 Not Found"
        ))
      }
      raw_html <- readBin(target_file, what = "raw", n = file.info(target_file)$size)
      return(list(
        status = 200L,
        headers = list("Content-Type" = "text/html; charset=utf-8"),
        body = rawToChar(raw_html)
      ))
    }
    
    # favicon 请求静默忽略
    if (path == "/favicon.ico") {
      return(list(
        status = 204L,
        headers = list(),
        body = ""
      ))
    }
    
    return(list(
      status = 404L,
      headers = list("Content-Type" = "text/plain; charset=utf-8"),
      body = "Not Found"
    ))
  },
  
  # ---- WebSocket 通道：流式推理 ----
  onWSOpen = function(ws) {
    # 追踪当前客户端的 IP 来源
    client_ip <- tryCatch(ws$request$REMOTE_ADDR, error = function(e) "UNKNOWN")
    
    ws$onMessage(function(binary, message) {
      # 拦截全局并发连接，防止多用户同一时刻推理导致底层张量踩踏
      if (MODEL_BUSY) {
        ws$send("[[ERROR]]服务器正忙，其他同学正在推理中，请稍后再试。")
        return()
      }
      
      tryCatch({
        msg_json <- jsonlite::fromJSON(message)
        prompt <- msg_json$prompt
        
        if (is.null(prompt) || nchar(trimws(prompt)) == 0) {
          ws$send("[[ERROR]]请输入有效的问题。")
          return()
        }
        
        # 锁定全局状态
        MODEL_BUSY <<- TRUE
        
        # 局部作用域变量：用于在流式生成中动态把碎 Token 拼成完整句子
        full_output <- ""
        
        # 1. 成功接收到 Prompt，记录基础文本日志
        log_event("PROMPT", prompt = prompt, client_ip = client_ip)
        
        # 2. 调用异步流式生成引擎
        generate_causal_text_async(
          model       = model,
          tokenizer   = tokenizer,
          prompt      = prompt,
          max_new_tokens = 200,
          temperature = 0.15,
          rep_penalty = 1.25,
          on_token    = function(token_text) {
            ws$send(token_text)
            
            # 核心增补：将吐出来的每一个碎 Token 实时累加进内存变量
            full_output <<- paste0(full_output, token_text)
          },
          on_done     = function() {
            # 解除全局独占锁
            MODEL_BUSY <<- FALSE
            ws$send("[[DONE]]")
            
            # 核心增补：推理完美收官，将 Prompt 和刚才拼接完毕的完整回答一齐原子化写入 JSONL 
            log_sft_jsonl(prompt = prompt, output = full_output, client_ip = client_ip)
            
            # 触发内存与显存垃圾回收
            run_garbage_collection()
          }
        )
        
      }, error = function(e) {
        # 异常断开必须强制释放全局模型锁，防止整站彻底瘫痪死锁
        MODEL_BUSY <<- FALSE
        err_msg <- conditionMessage(e)
        
        # 记录异常状态日志
        log_event("ERROR", prompt = NULL, client_ip = client_ip, err_msg = err_msg)
        
        ws$send(paste0("[[ERROR]]推理链路异常: ", err_msg))
        run_garbage_collection()
      })
    })
    
    # 客户端断开连接时，预防性清理一遍资源
    ws$onClose(function() {
      run_garbage_collection()
    })
  }
)

# 4. 启动本地监听
PORT <- 8080
cat(sprintf("\n 全栈 WebSocket 推理服务端已启动！请用浏览器访问：http://127.0.0.1:%d/\n\n", PORT))
server <- startServer(host = "0.0.0.0", port = PORT, app = app)

cat("按 Esc 或 Ctrl+C 停止服务\n")
while(TRUE) {
  service(100)
  Sys.sleep(0.01)
}