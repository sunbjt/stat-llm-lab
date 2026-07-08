# =====================================================================
# Rtomic 百万级预训练语料 (120万行) 工业级分块 (Chunks) 洗练流水线
# 终极完全体：降维打击繁体字、伪客服、自导自演及图书排版残留
# =====================================================================
library(tidyverse)
library(jsonlite)

# === 配置路径与核心参数 ===
input_path   <- "data/raw/pretrain_data.jsonl"
output_path  <- "data/processed/pretrain_clean.jsonl"
CHUNK_SIZE   <- 50000  # 每块吞吐 5 万行，内存稳如磐石

# 初始化：清空旧文件并打开追加流
if (file.exists(output_path)) file.remove(output_path)
con_out <- file(output_path, "ab") 

# 状态计数器
total_processed_rows <- 0
total_saved_rows     <- 0
chunk_counter        <- 0
t0 <- Sys.time()

cat("Rtomic 工业级分块流式洗练引擎 [繁体字绝对过滤版] 启动...\n")
cat(sprintf("每块吞吐量: %d 行 | 目标文件: %s\n\n", CHUNK_SIZE, input_path))

# =====================================================================
# ==== 核心：定义单块处理函数 ====
# =====================================================================
process_chunk <- function(chunk_df) {
  chunk_counter <<- chunk_counter + 1
  n_input <- nrow(chunk_df)
  total_processed_rows <<- total_processed_rows + n_input
  
  # 转换为 tibble 开始 Tidyverse 管道清洗
  processed_chunk <- chunk_df |> 
    as_tibble() |> 
    filter(!is.na(text)) |> 
    
    mutate(
      # ---- 步骤 A: 针对“自导自演”伪文章的定向手术 (大刀阔斧切头去尾) ----
      text = str_remove_all(text, "^.*?写一篇.*?(的文章|内容)[。|：|\\.]?\\s*(好的，)?以下是(为您|为您准备的|您请求的|我写的)?文章[：|。|\\s]*"),
      text = str_remove_all(text, "^(好的，)?我会为您写一篇.*?[：|。|\\s]*"),
      text = str_remove_all(text, "请(从上述文章中|把文章中的|从文章中)?提取.*$"),
      text = str_remove_all(text, "提取(出)?这篇文章的关键词.*$"),
      text = str_remove_all(text, "(给出|给定|给我|查找|设计|创建|编写|提供|生成|评价|介绍|提示|写)一(篇|种|份|个|款).*$"),
      
      # ---- 步骤 B: 图书排版与结构残留清理 ----
      # 1. 精准清除行首遗留的冒号、问号、破折号以及列表引导点
      text = str_remove(text, "^[：:\\?\\?？\\|\\s·•▪▲——\\-）\\)\\]\\}、,，\\.\\。\\;；①②③④⑤⑥⑦⑧⑨⑩]+"),
      
      # 2. 擦除文本中乱入的硬编码图书脚注或参考文献角标（如：(10) 或 [12]）
      text = str_remove_all(text, "\\(\\s*\\d+\\s*\\)|\\[\\s*\\d+\\s*\\]"),
      
      # 3. 擦除行首或段落间断层、无上下文支撑的硬编码纯数字条目号（如 1) 或 2) ）
      text = str_remove_all(text, "(?m)^\\s*\\d+[\\]\\)\\}\\.、]\\s*"),
      
      # 4. 切除图书章节前缀 (如 "16.5\n本章小结"、"17.1.1")
      text = str_remove_all(text, "(?m)^\\d+(\\.\\d+)*\\s*\\n?.*\\n?"),
      
      # 5. 擦除损坏的空白 URL 占位符、特殊废弃符号等
      text = str_remove_all(text, "请访问\\s*。"),
      text = str_remove_all(text, "（\\s*，|\\(\\s*for\\s+big\\s+data"),
      text = str_remove_all(text, "（\\s*and\\s+ethics\\s*）|\\(\\s*and\\s+ethics\\s*\\)"),
      
      # 6. 抹除项目中无意义的零碎引导列表符号
      text = str_remove_all(text, "[•▪▲\\t]"),
      
      # 7. 精准脱掉 ** 加粗外衣
      text = str_replace_all(text, "\\*\\*(.*?)\\*\\*", "\\1"),
      
      # 9. 擦除文本内部由于图书断层乱入的孤立点号分隔符（如 " . 第二章 . " 或末尾参考文献）
      text = str_replace_all(text, "\\s+\\.\\s+", " "),
      
      # 10. 顺手干掉行首或因切分残留的孤立标点符号和空格
      text = str_remove(text, "^\\s*\\.\\s*"),
      
      # 8. 收拢连续多个换行符、无意义空格，双重 trim
      text = str_replace_all(text, "\\n+", "\n"),
      text = str_replace_all(text, "\\s+", " "),
      # 擦除因 JSON 转义残留的斜杠，还原为纯文本的双引号
      text = gsub('\\\\?"', '', text),
      text = gsub('\\?"', '', text),
      text = str_trim(text)
    ) |> 
    
    # ---- 步骤 C: 智能化质量硬拦截 (漏斗过滤器) ----
  filter(
    # 过滤 1: 长度防御，太短的碎片直接扔掉
    str_length(text) > 100,
    
    # 过滤 2: 【新防线】利用极速字符集硬拦截：如果文本中包含台湾/香港常用的核心繁体字根，直接全行抹除
    # 核心拦截词包含：隨(随)、變(变)、發(发)、趨(趋)、數據(数据)、體(体)、轉(转)、廣(广) 等
    !str_detect(text, "[隨變發趨數據體轉廣響應業範劃變陣體欄優導進製]"),
    
    # 过滤 3: 强力全网通缉低密度、污染严重的“查天气/订机票/问时间”纯闲聊伪客服语料
    !str_detect(text, "帮我查一下天气|查询哪个城市|北京的天气|最高温度为|最低温度为|注意防晒|携机票|单程还是往返|现在几点了"),
    
    # 过滤 4: 剔除低信息密度的图书客客套套话与大纲纯题目提示
    # 过滤 4: 剔除低信息密度的图书客客套套话与大纲纯题目提示
    # 过滤 4: 剔除低信息密度的图书客客套套话与大纲纯题目提示
    !str_detect(text, "推荐序|副行长|车品觉先生|前言建立|目录推荐序|练习 IV|作业 \\d"),
    
    # 【终极防线 1】行首核心动词与高频疑问词熔断：直接干掉提问前缀
    !str_detect(text, "^(如何|怎样|该如何|该怎样|我应该如何|哪些|什么是).*(更有效|改善|避免|坚持|合理安排|利用|有效利用|提升|打破|阻止|积极参与|社会领域|克服)"),
    
    # 【终极防线 2】AI 套话与结构化列表熔断：只要包含这些标准回答句式，整行直接抹除
    !str_detect(text, "以下是一些方法|考虑任务优先级的几个因素|以下是考虑任务优先级|方法来更好地集中注意力|以下是一些方法可以寻求帮助|机器学习是人工智能的一个重要分支|防止网络安全攻击的最佳方法|快速响应和解决问题的步骤|在上述领域中|总的来说，人工智能可以|这些方法听起来都很实用"),
    
    # 【核心升级】利用行首特征与高频 AI 句式，直接熔断 Prompt 问答对语料
    !str_detect(text, "^(该如何|我应该|我要|我该|我想|如何|怎样|该怎样|我该?如何|我想了解).*(开发|避免|实施|提高|准备|提升|解决|利用|结合|做到|注意)"),
    !str_detect(text, "可以按照以下步骤|以下是一些建议|以下是一般的步骤|以下是一些处理|综上所述"),
    !str_detect(text, "在开发解决方案时|如何确定最佳的编程语言|处理演讲中的紧张情绪"),
    
    !str_detect(text, "写篇|写一|你好|内容： |写篇关于|题目：|基于以下角色|给你两个角色信息|作为一名AI|MOSS|实体识别|实体抽取|列出|列举"),
    
    # 过滤 5: 标点密度安全审查（精准拦截纯代码块、参数 dump 堆叠）
    (str_count(text, "[\\p{P}]") / str_length(text)) < 0.15
  )
  
  n_output <- nrow(processed_chunk)
  total_saved_rows <<- total_saved_rows + n_output
  
  # ---- 步骤 D: 块级数据即时追加落盘 ----
  if (n_output > 0) {
    jsonlite::stream_out(processed_chunk, con_out, verbose = FALSE)
  }
  
  # 实时进度条打印
  elapsed_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("[Block %03d] 已累计读取: %7d 行 | 当前块留存: %5d 行 | 累计导出: %7d 行 | 耗时: %.1fs\n", 
              chunk_counter, total_processed_rows, n_output, total_saved_rows, elapsed_time))
  
  rm(processed_chunk)
}

# =====================================================================
# ==== 运行：拉动 jsonlite 的 Chunks 大数据流大闸 ====
# =====================================================================
jsonlite::stream_in(
  con = file(input_path, "r"), 
  handler = process_chunk, 
  pagesize = CHUNK_SIZE, 
  verbose = FALSE
)

# 完工，安全关闭流链接
close(con_out)

# =====================================================================
# ==== 最终收官报告 ====
# =====================================================================
t_end <- Sys.time()
cat("\n=====================================================================\n")
cat("🎉 120 万行全量 Chunks [繁体字熔断版] 流水线洗练结束！\n")
cat(sprintf(" 累计读取原始数据 : %d 行\n", total_processed_rows))
cat(sprintf(" 沉淀黄金预训练语料 : %d 行\n", total_saved_rows))
cat(sprintf(" 数据沙里淘金留存率 : %.2f%%\n", (total_saved_rows / total_processed_rows) * 100))
cat(sprintf(" 整个流水线总共耗时 : %.2f 秒\n", as.numeric(difftime(t_end, t0, units = "secs"))))
cat("=====================================================================\n")