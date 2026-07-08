library(RcppSimdJson)
library(data.table)
library(tidyverse)
library(stringr)
library(jsonlite)

# === 全局配置 ===
# 针对问答/指令数据大幅降低长度限制，因为此类对话通常较短
CHAR_LEN = 30   
HAN_RATIO = 0.5  

#' 基础文本规范化函数（已移除维基百科专属规则）
#' @param text_vec 输入的字符向量
#' @return 清洗完成的纯净字符向量
clean_text_vec <- function(text_vec) {
  
  # 1. HTML 标签清理（保留以防抓取数据中残留网页标签）
  text_vec <- str_replace_all(text_vec, "<[^>]*>", "")
  text_vec <- str_replace_all(text_vec, "&[a-zA-Z0-9#]+;", "")
  
  # 2. 标点符号归一化（修正单双引号嵌套引起的语法和转义错误）
  text_vec <- str_replace_all(text_vec, "「", '"')
  text_vec <- str_replace_all(text_vec, "」", '"')
  
  # 3. 空白文本与连续换行符压缩
  text_vec <- str_replace_all(text_vec, "\\n+", "\n")
  
  return(text_vec)
}

#' 具备低开销审计分流的高性能流式分块处理主函数
#' @param input_path 原始 JSONL 语料路径
#' @param output_path 清洗后的 JSONL 语料路径
#' @param audit_path 被拦截淘汰数据的审计日志路径
#' @param chunk_lines 每个分块装载的行数
process_qa_chunked_audit <- function(input_path, output_path, audit_path, chunk_lines = 100000) {
  
  con_in <- file(input_path, "r")
  con_out <- file(output_path, "w")
  con_audit <- file(audit_path, "w") 
  on.exit({ close(con_in); close(con_out); close(con_audit) })
  
  total_in <- 0; total_out <- 0; total_dropped <- 0
  chunk_id <- 0
  
  cat(sprintf("[%s] 启动问答数据流式双轨清洗流水线...\n", format(Sys.time())))
  
  repeat {
    lines <- readLines(con_in, n = chunk_lines, warn = FALSE)
    if (length(lines) == 0) break
    chunk_id <- chunk_id + 1
    total_in <- total_in + length(lines)
    
    # 硬件级极速反序列化
    parsed <- data.table::rbindlist(RcppSimdJson::fparse(lines), fill = TRUE)
    
    # 容错：如果数据行连 text 字段都没有，直接跳过
    if (!"text" %in% names(parsed)) {
      next
    }
    
    # ------ 【第一轨：初始质量门控与分流】 ------
    char_len_init <- nchar(parsed$text)
    han_count_init <- str_count(parsed$text, "\\p{Han}")
    alpha_count_init <- str_count(parsed$text, "[a-zA-Z]")
    
    # 计算文本内容的绝对中文纯度
    content_han_ratio_init <- han_count_init / (han_count_init + alpha_count_init + 1e-5)
    
    # 移除原有的 bad_title 校验，仅依赖长度和中文比例
    bad_stats <- (char_len_init < CHAR_LEN) | (content_han_ratio_init < HAN_RATIO)
    is_bad_init <- replace_na(bad_stats, TRUE)
    
    # 将初筛未通过的数据实时流式剥离至审计端
    if (any(is_bad_init)) {
      audit_df1 <- parsed[is_bad_init, ]
      audit_df1$drop_reason <- "Init_TooShort_or_LowHan"
      jsonlite::stream_out(audit_df1, con_audit, pagesize = nrow(audit_df1), verbose = FALSE)
      total_dropped <- total_dropped + nrow(audit_df1)
    }
    
    # ------ 【第二轨：核心向量化清洗与二次门控】 ------
    valid_df <- parsed[!is_bad_init, ]
    
    if (nrow(valid_df) > 0) {
      # 文本在底层进行高速矩阵级文本替换
      valid_df$text_clean <- clean_text_vec(valid_df$text)
      
      char_len_post <- nchar(valid_df$text_clean)
      han_count_post <- str_count(valid_df$text_clean, "\\p{Han}")
      alpha_count_post <- str_count(valid_df$text_clean, "[a-zA-Z]")
      
      content_han_ratio_post <- han_count_post / (han_count_post + alpha_count_post + 1e-5)
      is_bad_post <- replace_na((char_len_post < CHAR_LEN) | (content_han_ratio_post < HAN_RATIO), TRUE)
      
      # 捕获清洗后缩水过大、不符合要求的文档
      if (any(is_bad_post)) {
        audit_df2 <- valid_df[is_bad_post, ]
        audit_df2$drop_reason <- "PostClean_TooShort_or_LowHan"
        audit_df2$text <- audit_df2$text_clean 
        audit_df2$text_clean <- NULL
        jsonlite::stream_out(audit_df2, con_audit, pagesize = nrow(audit_df2), verbose = FALSE)
        total_dropped <- total_dropped + nrow(audit_df2)
      }
      
      # ------ 【最终优质语料放行写入】 ------
      final_clean <- valid_df[!is_bad_post, ]
      total_out <- total_out + nrow(final_clean)
      
      if (nrow(final_clean) > 0) {
        final_clean$text <- final_clean$text_clean
        # 只提取并输出 text 字段
        jsonlite::stream_out(
          final_clean %>% select(text), 
          con_out, 
          pagesize = nrow(final_clean), 
          verbose = FALSE
        )
      }
    }
    
    # 打印区块进度
    cat(sprintf("[%s] 区块 %02d: 读取 %6d 行 → 放行 %6d 行 | 累计拦截: %7d 行\n", 
                format(Sys.time()), chunk_id, length(lines), nrow(final_clean), total_dropped))
    
    # 强制进行垃圾回收
    rm(lines, parsed, valid_df, final_clean); gc(verbose = FALSE)
  }
  
  cat(sprintf("\n 数据工程预处理全线完成！\n总计输入语料: %d 条\n成功放行优质语料: %d 条\n整体留存率: %.2f%%\n丢弃审查日志详见: %s\n", 
              total_in, total_out, (total_out / total_in) * 100, audit_path))
}

# 调用主流水线函数
process_qa_chunked_audit(
  input_path  = "data/raw/wiki_focused_semantic.jsonl",      # 请替换为你的实际输入路径
  output_path = "data/processed/wiki_clean.jsonl",
  audit_path  = "data/processed/wiki_deleted_audit.jsonl",
  chunk_lines = 100000
)
