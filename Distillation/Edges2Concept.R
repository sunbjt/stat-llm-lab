
library(data.table)
library(dplyr)   # 或者 library(tidyverse)

en_edges <- readRDS('~/Downloads/en_edges.Rdata')
setDT(en_edges)

# 定义基于 data.table 的递归图谱提取函数
extract_subgraph <- function(edges_dt, seed_node, max_depth = 2) {
  
  # 初始化状态
  visited_nodes <- c(seed_node)
  current_nodes <- c(seed_node)
  results_list <- list()
  
  for (depth in 1:max_depth) {
    message(sprintf("正在提取第 %d 层节点网络...", depth))
    
    # 利用 data.table 快速匹配：找出 Source 或 Target 包含当前节点的所有边
    # 注意：知识图谱是有向的，但相关性探索通常需要双向查找
    layer_edges <- edges_dt[Source %in% current_nodes | Target %in% current_nodes]
    
    if (nrow(layer_edges) == 0) {
      message("未发现更多关联节点，提前结束递归。")
      break
    }
    
    # 标记当前发现层的深度
    layer_edges[, Depth := depth]
    results_list[[depth]] <- layer_edges
    
    # 提取新发现的节点（合并这一层所有的 Source 和 Target）
    new_nodes <- unique(c(layer_edges$Source, layer_edges$Target))
    
    # 剔除已经访问过的节点，作为下一层的搜索起点，防止网络成环死循环
    current_nodes <- setdiff(new_nodes, visited_nodes)
    visited_nodes <- c(visited_nodes, current_nodes)
    
    if (length(current_nodes) == 0) {
      break
    }
  }
  
  # 合并所有层级结果
  final_graph <- rbindlist(results_list)
  
  # 由于双向查找和多路径到达，结果可能会有重复的边，需要按照核心三元组去重
  final_graph <- unique(final_graph, by = c("Relation", "Source", "Target"))
  
  # 按照权重（如果有的话）和深度排序，把最高质量的边排在前面
  setorder(final_graph, Depth, -Weight)
  
  return(final_graph)
}

# 执行递归：例如以“人工智能”为起点，向外扩散 2 层
# ai_graph <- extract_subgraph(en_edges, "artificial intelligence", max_depth = 2)

# 定义种子节点及对应深度（可根据需要调整）
seeds <- list(
  "artificial intelligence" = 2,
  "machine learning"        = 2,
  "algorithm"               = 2,
  "data science"            = 2,
  "statistics"              = 2,
  "big data"                = 2,
  "programming"             = 1,
  "why"                     = 2,
  "what"                    = 1,
  "how"                     = 2
)

# 存储所有子图结果的列表
subgraph_list <- list()

for (i in seq_along(seeds)) {
  seed <- names(seeds)[i]
  depth <- seeds[[i]]
  message(sprintf("\n正在处理种子节点: %s (深度 %d)", seed, depth))
  
  sg <- extract_subgraph(en_edges, seed_node = seed, max_depth = depth)
  
  # 添加一列记录该边来自哪个种子节点（可选，便于溯源）
  sg[, Seed := seed]
  
  subgraph_list[[i]] <- sg
}

# 合并所有子图
combined_graph <- rbindlist(subgraph_list, fill = TRUE)

# 去重（按关键三元组：Source, Relation, Target），保留第一次出现的记录（可根据权重/深度调整）
combined_graph <- unique(combined_graph, by = c("Source", "Relation", "Target"))

# 排序（按深度和权重，综合排序）
setorder(combined_graph, Depth, -Weight)

# 查看结果
print(combined_graph)

# ConceptNet 语义关系标签 (Relations) 注释字典
# 
# 【一、 核心本体与逻辑关系】 (知识图谱骨架，价值最高)
#  - IsA: 上下位包含关系 (例: 深度学习 IsA 机器学习)
#  - PartOf: 整体与局部关系 (例: 注意力机制 PartOf Transformer)
#  - InstanceOf: 实体与概念的关系 (例: GPT-4 InstanceOf 大语言模型)
#  - DefinedAs: 提供确切的解释或等价定义
#  - dbpedia/field: 所属学科/领域 (例: 微积分 dbpedia/field 数学)
#  - dbpedia/genre: 所属流派
#  - dbpedia/influencedBy: 受...影响 (多用于技术演进脉络)
#  - dbpedia/knownFor: 以...闻名
#
# 【二、 行为、属性与因果】 (动态特征，适合生成问答解释)
#  - UsedFor: 工具或概念的用途 (例: GPU UsedFor 模型训练)
#  - CapableOf: 主体具备的能力 (例: 人工智能 CapableOf 识别图像)
#  - HasPrerequisite: 先决条件，极具推理价值 (例: 反向传播 HasPrerequisite 计算梯度)
#  - Causes: 直接的因果关系 (例: 内存泄漏 Causes 系统崩溃)
#  - HasSubevent: 完成某事需经历的子步骤 (例: 模型训练 HasSubevent 更新权重)
#  - HasProperty: 固有的特征或属性 (例: 开源软件 HasProperty 免费的)
#
# 【三、 需过滤的噪音】 (信息密度低，不加入白名单)
#  - HasContext(语境)、RelatedTo(泛相关)、Synonym(同义词)、Antonym(反义词)
#  - DerivedFrom/FormOf (词根变体)

target_relations <- c(
  "IsA", "PartOf", "InstanceOf", "DefinedAs", 
  "UsedFor", "CapableOf", "HasPrerequisite", 
  "Causes", "HasProperty", "HasSubevent", 
  "dbpedia/field", "dbpedia/genre", "dbpedia/influencedBy", "dbpedia/knownFor"
)

# 2. 从提取的 ai_graph 中过滤噪音
# 将数量庞大的 RelatedTo 和 HasContext 丢弃，确保语料纯度和逻辑性
ai_graph_clean <- combined_graph[Relation %in% target_relations & Weight >= 1]

# 3. 按关系类型查看过滤后的分布
print(table(ai_graph_clean$Relation))

# 4. 按质量排序（优先深度浅、权重高的核心知识）
setorder(ai_graph_clean, Depth, -Weight)

print(paste("过滤后的高价值硬知识节点数:", nrow(ai_graph_clean)))
ai_graph_clean_dt <- data.frame(
  concept = unique(c(ai_graph_clean$Target, ai_graph_clean$Source))
) |>
  filter(
    nchar(concept) >= 4
  )

## 从大模型直接蒸馏获取语料

# 如果没有这些包，请先安装: install.packages(c("httr2", "jsonlite", "data.table", "glue"))
library(httr2)
library(jsonlite)
library(data.table)
library(glue)

# ==========================================
# 0. 环境与代理配置
# ==========================================
# 设置代理 (本地请求国外 API 时必备，如果是国内直连 DeepSeek 可注释掉)
# Sys.setenv(https_proxy = "http://127.0.0.1:7897")
# Sys.setenv(http_proxy  = "http://127.0.0.1:7897")

OUTPUT_JSONL <- "~/Downloads/ai_pretraining_deepseek.jsonl"     # 最终输出的微调数据集

# ==========================================
# 1. 定义限流器环境 (Global State)
# ==========================================
# 使用 R 的环境 (environment) 存储全局变量，防止污染主环境
limit_env <- new.env()
limit_env$last_req_time <- Sys.time() - 100 # 初始化为过去，确保首次执行不等待

# ==========================================
# 2. 智能限流的 API 调用函数 (DeepSeek JSON 版)
# ==========================================
call_deepseek_json_safe <- function(concept, model = "deepseek-chat") {
  
  # ---------- 限流防护 (DeepSeek 限制较宽松，可适当降低间隔) ----------
  # 假设我们保守一点，每分钟请求 30 次，间隔 2 秒
  MIN_INTERVAL <- 1
  time_since_last <- as.numeric(difftime(Sys.time(), limit_env$last_req_time, units = "secs"))
  
  if (time_since_last < MIN_INTERVAL) {
    wait_time <- ceiling(MIN_INTERVAL - time_since_last)
    
    message(glue::glue("\n触发限流保护，正在冷却 {wait_time} 秒..."))
    for (i in wait_time:1) {
      cat(paste0("\r", i, " ... ")) 
      Sys.sleep(1)
    }
    cat("\rGO!   \n") 
  }
  
  # ---------- 构建 Prompt ----------
  # 【关键修改】：System Prompt 必须明确要求 JSON，并在 User Prompt 给出预期的 Key
  system_role <- "你是一个专业的学术翻译与概念专家。你必须且只能返回合法的 JSON 格式数据。"
  
  user_content <- glue::glue(
    "基于 '{concept}' 这个概念，用中文解释或科普内容（内容不要包含 '{concept}' 这个英文单词，不少于300字）。\n\n",
    "期望的 JSON 格式如下：\n",
    "{{\"explanation\": \"在这里填写你的详细解释内容\"}}"
  )
  
  # ---------- 获取 API Key ----------
  api_key <- Sys.getenv("deepseek_API_KEY")
  # 兼容大小写环境变量
  if (api_key == "") api_key <- Sys.getenv("DEEPSEEK_API_KEY") 
  if (api_key == "") stop("未找到 DEEPSEEK_API_KEY，请先使用 Sys.setenv(DEEPSEEK_API_KEY='...') 设置！")
  
  # DeepSeek 的统一入口
  url <- "https://api.deepseek.com/chat/completions"
  
  # ---------- 构建请求体 (OpenAI 兼容格式 + JSON 模式) ----------
  body <- list(
    model = model,
    messages = list(
      list(role = "system", content = system_role),
      list(role = "user", content = user_content)
    ),
    # 强制 DeepSeek 返回 JSON 对象 
    response_format = list(type = "json_object"),
    temperature = 0.1 # 调低 temperature 让信息抽取类任务更稳定、严谨
  )
  
  result_text <- tryCatch({
    # ---------- 发送 httr2 请求 ----------
    req <- request(url) |> 
      req_method("POST") |>
      req_headers(
        "Authorization" = paste("Bearer", api_key), # DeepSeek 使用 Bearer Token
        "Content-Type" = "application/json"
      ) |>
      req_body_json(body) |> 
      req_perform()
    
    # 成功执行后，更新最后请求时间
    limit_env$last_req_time <- Sys.time() 
    
    # ---------- 解析响应内容 ----------
    # 提取 DeepSeek 返回的 content 内容
    resp_data <- resp_body_json(req)
    resp_data$choices[[1]]$message$content
    
  }, error = function(e) {
    message("API 调用出错: ", e$message)
    # 即使报错，也要更新时间，防止死循环重试
    limit_env$last_req_time <- Sys.time()
    return(NULL)
  })
  
  return(result_text)
}

# ==========================================
# 3. 主执行逻辑 (单线程 + 实时断点保存)
# ==========================================

# 【这里假设你外部环境已经有了 ai_graph_clean_dt，且有一列叫做 concept】
df <- ai_graph_clean_dt

# 取前几条进行安全测试，确认无误后再放开全量执行
# df <- head(df, 25)

total_rows <- nrow(df)

# 若文件不存在则创建；若存在，后续代码会自动追加 (Append)
if (!file.exists(OUTPUT_JSONL)) file.create(OUTPUT_JSONL)

for (i in 1:total_rows) {
  # 提取当前的概念词
  current_concept <- df$concept[i]
  message(sprintf("正在处理 %d/%d: %s", i, total_rows, current_concept))
  
  # 调用 DeepSeek API
  json_str <- call_deepseek_json_safe(current_concept)
  
  if (!is.null(json_str)) {
    tryCatch({
      # 解析返回的 JSON
      parsed_json <- fromJSON(json_str, simplifyVector = FALSE)
      
      # 【关键修改】：因为现在是针对单个概念，所以重写了 instruction 的组装逻辑
      # 取消了针对 Source 和 Target 关系的判断
      record <- list(
        text = ifelse(is.null(parsed_json$explanation), "", parsed_json$explanation)
      )
      
      # 单条追加写入，防止中断导致数据全部丢失
      con <- file(OUTPUT_JSONL, open = "a", encoding = "UTF-8")
      writeLines(toJSON(record, auto_unbox = TRUE, force = TRUE), con = con)
      close(con)
      
    }, error = function(e) {
      warning(sprintf("行 %d (%s): JSON 结构解析失败, 原始返回为: %s", i, current_concept, json_str))
    })
  }
}

message(sprintf("\n全部处理完成！高质量预训练语料已保存至: %s", OUTPUT_JSONL))
