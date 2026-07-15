library(RcppSimdJson)

# =====================================================================
# 1. 配置文件路径
# =====================================================================
input_file <- "data/raw/pretrain_clean.jsonl"          # 原始数据
clean_file <- "data/processed/pretrain_clean.jsonl"     # 纯净输出
audit_file <- "data/processed/audit_filtered.jsonl"     # 被拦截的脏数据

file.create(clean_file)
file.create(audit_file)

# =====================================================================
# 2. 定义过滤规则池 (grepl 支持正则，易于无限扩展)
# =====================================================================
# 规则 A: 实体断崖 (汉字 + 间隔号 + 汉字)
pattern_name <- "[\\x{4e00}-\\x{9fa5}]+[\\x{00b7}\\x{2022}\\x{2027}][\\x{4e00}-\\x{9fa5}]+"

# 规则 B: 包含特定英文词汇 (利用 (?i) 实现忽略大小写)
pattern_ebook <- "(?i)ebook"

# 规则 C: 繁体中文探针 (选取高频专属繁体字，命中即判定为繁体语料)
pattern_trad <- "[護漸這們會說過對還個為與從時進學實發動網機體點廠廣車頁見長門風飛馬鳥麥東魚愛]"

# =====================================================================
# 3. 极简流式主循环
# =====================================================================
con_in <- file(input_file, "r")
con_clean <- file(clean_file, "a")
con_audit <- file(audit_file, "a")

chunk_size <- 50000 
chunk_idx <- 1

cat("启动极简流式过滤管道...\n")

while (length(lines <- readLines(con_in, n = chunk_size, warn = FALSE)) > 0) {
  
  # 1. 高速解析提取文本
  parsed <- tryCatch({
    RcppSimdJson::fparse(lines, always_list = TRUE)
  }, error = function(e) NULL)
  
  if (is.null(parsed)) next
  
  texts <- vapply(parsed, function(x) {
    if (is.list(x) && !is.null(x$text) && !is.na(x$text)) x$text else ""
  }, character(1))
  
  valid_idx <- texts != ""
  
  # 2. 并行应用过滤规则 (返回 TRUE/FALSE 向量)
  hit_name  <- grepl(pattern_name, texts, perl = TRUE)
  hit_ebook <- grepl(pattern_ebook, texts, perl = TRUE)
  hit_trad  <- grepl(pattern_trad, texts, perl = TRUE)
  hit_url <- grepl("__|以下文本|文案|原始文本|请回答问题|中文回答|根据给出|给定的文章|提取下|请根据|请编辑|改写|创作一首|进行摘要|将以下|总结并|你认为|学习贯彻|提问|一段文字|提取以下|请告诉我|祝您|当前位置|以下段落|将下面|是的|帮我|提取关键|总结一下|总结以下|根据给定|一段文本|摘要生成|以下问题|下面的文本|从一段|文本纠错|给定一|我是一个|根据给定的|生成一个|为以下|根据以下|意识形态|人工智能在医疗|更好地理解|例如，可以使用|人工智能可以通过|他们的对话|以下是|网址|来源|script|两个维护|全市|总书记|同志|市委|例子|防控|可以根据学生的学习", texts)
  
  # 3. 逻辑合并：只要命中任意一个地雷，标记为脏数据 (TRUE)
  is_dirty <- hit_name | hit_ebook | hit_trad | hit_url
  
  # 4. 构建索引分流
  keep_idx <- valid_idx & !is_dirty  # 有效且干净
  drop_idx <- valid_idx & is_dirty   # 有效但命中了过滤规则
  
  # 5. 物理落盘
  if (any(keep_idx)) writeLines(lines[keep_idx], con_clean)
  if (any(drop_idx)) writeLines(lines[drop_idx], con_audit)
  
  cat(sprintf("Chunk %04d | 放行: %-6d | 拦截: %-6d\n", 
              chunk_idx, sum(keep_idx), sum(drop_idx)))
  
  chunk_idx <- chunk_idx + 1
}

close(con_in)
close(con_clean)
close(con_audit)
