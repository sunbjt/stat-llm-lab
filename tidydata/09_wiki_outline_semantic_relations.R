library(torch)
library(safetensors)
library(tok)
library(data.table)
library(igraph)
library(tidyverse)

setwd('~/github/stat-llm-lab/tidydata/')

# ----------------------------------------------------------------------
# 1. 加载数据与提取唯一概念列表
# ----------------------------------------------------------------------
load_relations_from_rdata <- function(rdata_file = "Outline_relations.Rdata") {
  if (!file.exists(rdata_file)) stop("RData file not found: ", rdata_file)
  load(rdata_file)
  return(relations)
}

relations <- load_relations_from_rdata('Outline_relations.Rdata')

if (is.data.frame(relations)) {
  concepts <- unique(c(relations[[1]], relations[[2]]))
} else {
  concepts <- unique(as.character(relations))
}
concepts <- concepts[nchar(concepts) > 0]
N <- length(concepts)
cat(sprintf("提取到 %d 个原始概念\n", N))

# ----------------------------------------------------------------------
# 2. 加载模型与定义语义编码器
# ----------------------------------------------------------------------
tokenizer <- tok::tokenizer$from_file("~/Downloads/tokenizer.json")
weights <- safetensors::safe_load_file("~/Downloads/model.safetensors", framework = "torch")
config <- jsonlite::fromJSON("~/Downloads/config.json")

emb_layer <- nn_embedding(config$vocab_size, config$hidden_size)
emb_w <- weights[["embeddings.word_embeddings.weight"]]
with_no_grad({ emb_layer$weight$set_(emb_w) })

encode_semantic_batch <- function(text_batch) {
  with_no_grad({
    encs <- tokenizer$encode_batch(text_batch)
    ids_list <- lapply(encs, function(x) x$ids)
    lens <- lengths(ids_list)
    max_len <- max(lens)
    batch_n <- length(text_batch)
    
    padded_ids <- matrix(0L, nrow = batch_n, ncol = max_len)
    for (i in seq_len(batch_n)) {
      if (lens[i] > 0) padded_ids[i, seq_len(lens[i])] <- ids_list[[i]]
    }
    
    ids <- torch_tensor(padded_ids, dtype = torch_long()) + 1L
    vectors <- emb_layer(ids)
    return(vectors$mean(dim = 2))
  })
}

# ----------------------------------------------------------------------
# 3. 批量计算所有概念的向量
# ----------------------------------------------------------------------
cat("开始计算全量概念 Embedding...\n")
embed_batch_size <- 512
num_batches <- ceiling(N / embed_batch_size)
vec_list <- vector("list", num_batches)

for (i in seq_len(num_batches)) {
  idx <- ((i - 1) * embed_batch_size + 1):min(i * embed_batch_size, N)
  batch_vecs <- encode_semantic_batch(concepts[idx])
  batch_norm <- nnf_normalize(batch_vecs, p = 2, dim = 2)
  vec_list[[i]] <- batch_norm$to(device = "cpu")
}

all_vecs <- torch_cat(vec_list, dim = 1) # [1110000, 384]

# ----------------------------------------------------------------------
# Step 3.5：精细化过滤（加入学术消歧义白名单，召回 Transformer 等关键模型）
# ----------------------------------------------------------------------
cat("执行精细化语义与实体过滤...\n")

concepts_clean_text <- gsub("_", " ", concepts)

# 1. 维基学术/技术消歧义【白名单】（强行保留）
academic_wiki_pattern <- "\\((machine learning|computer science|mathematics|physics|biology|psychology|statistics|economics|sociology|chemistry|algorithm|model|discipline|neural network)\\b"
is_academic_wiki_tag <- grepl(academic_wiki_pattern, concepts_clean_text, ignore.case = TRUE)

# 2. 维基非学术实体【黑名单】
invalid_pattern <- "^(List of|Outline of|Category:|Index of|Glossary of)"
entity_wiki_pattern <- "\\((mathematician|physicist|chemist|biologist|philosopher|politician|historian|born|died|novel|film|album|song|city|state|country|war|[0-9]{4})\\b"

rule_invalid_idx <- grepl(invalid_pattern, concepts_clean_text, ignore.case = TRUE) | 
                    grepl(entity_wiki_pattern, concepts_clean_text, ignore.case = TRUE)

# 3. 语义锚点相似度计算
academic_prompts <- c("academic discipline", "branch of science", "field of study", "academic subject", "scientific theory")
person_prompts   <- c("famous person", "historical figure", "biography of human", "researcher name", "individual person")

academic_anchor <- nnf_normalize(encode_semantic_batch(academic_prompts)$mean(dim = 1, keepdim = TRUE), p = 2, dim = 2)
person_anchor   <- nnf_normalize(encode_semantic_batch(person_prompts)$mean(dim = 1, keepdim = TRUE), p = 2, dim = 2)

sim_academic <- as.numeric(torch_matmul(all_vecs, academic_anchor$t())$to(device = "cpu"))
sim_person   <- as.numeric(torch_matmul(all_vecs, person_anchor$t())$to(device = "cpu"))
semantic_diff <- sim_academic - sim_person

# 4. 截断分位数保持 6%
person_cutoff <- quantile(semantic_diff, probs = 0.06)

# 5. 补充 model, effect, architecture, framework, learning 等关键词保护网
academic_keywords <- "(ology|ics|graphy|metry|science|studies|theory|analysis|engineering|mechanics|dynamics|chemistry|physics|biology|math|philosophy|economics|sociology|law|pricing|capture|commercialization|management|policy|system|method|protocol|algorithm|network|model|effect|architecture|framework|learning|structure|process|concept)"
is_academic_like <- grepl(academic_keywords, concepts_clean_text, ignore.case = TRUE)

# 6. 保留逻辑：(在黑名单外 AND (语义通过 OR 命中学术词)) OR (命中学术白名单)
keep_mask <- ((!rule_invalid_idx) & ((semantic_diff > person_cutoff) | is_academic_like)) | is_academic_wiki_tag
removed_concepts <- concepts[!keep_mask]

concepts <- concepts[keep_mask]
all_vecs <- all_vecs[keep_mask, ]
N_clean <- length(concepts)

cat(sprintf("精细过滤完成：原始 %d 条 -> 保留 %d 条 (剔除 %d 条，保留率 %.2f%%)\n", 
            length(keep_mask), N_clean, length(removed_concepts), (N_clean / length(keep_mask)) * 100))

# 快速验证 Transformer 是否被拯救
cat("\n【验证】Transformer 状态:", "Transformer_(machine_learning_model)" %in% concepts, "\n")

# ----------------------------------------------------------------------
# Step 4：分块计算 Top-20 近邻 (GPU Chunked KNN)
# ----------------------------------------------------------------------
cat("\n开始计算 Top-20 近邻...\n")
device <- if (cuda_is_available()) "cuda" else "cpu"

# 将清洗后的全量向量载入设备
all_vecs_dev <- all_vecs$to(device = device)
N_clean <- length(concepts)

k <- 20
chunk_size <- 200
num_chunks <- ceiling(N_clean / chunk_size)
chunk_edges_list <- vector("list", num_chunks)

start_time <- Sys.time()

for (i in seq_len(num_chunks)) {
  start_idx <- (i - 1) * chunk_size + 1
  end_idx <- min(i * chunk_size, N_clean)
  
  chunk_vecs <- all_vecs_dev[start_idx:end_idx, ]
  
  # 分块点积计算 Cosine 相似度矩阵: [chunk_len, N_clean]
  sim_mat <- torch_matmul(chunk_vecs, all_vecs_dev$t())
  
  # 取 Top-21 (第 1 个近邻是自身，切片 2:21 排除自身)
  topk_res <- torch_topk(sim_mat, k = k + 1, dim = 2)
  topk_val <- topk_res[[1]][, 2:(k + 1)]$to(device = "cpu")
  topk_idx <- topk_res[[2]][, 2:(k + 1)]$to(device = "cpu")
  
  from_idx <- torch_arange(start_idx, end_idx, dtype = torch_long())$view(c(-1, 1))$expand_as(topk_idx)
  
  dt_chunk <- data.table(
    from = as.integer(from_idx$flatten()),
    to = as.integer(topk_idx$flatten()),
    sim = as.numeric(topk_val$flatten())
  )
  
  chunk_edges_list[[i]] <- dt_chunk
  
  if (i %% 10 == 0) {
    cat(sprintf("\rKNN 计算进度: %d / %d (%.1f%%)", i, num_chunks, (i/num_chunks)*100))
    gc(verbose = FALSE)
  }
}

edges_dt <- rbindlist(chunk_edges_list)
cat(sprintf("\n生成近邻边表共 %d 条数据，耗时: %.2f 秒\n", 
            nrow(edges_dt), as.numeric(difftime(Sys.time(), start_time, units = "secs"))))

# ----------------------------------------------------------------------
# Step 5：基于相似度分布的图剪枝 (Pruning)
# ----------------------------------------------------------------------
sim_quantiles <- quantile(edges_dt$sim, probs = seq(0, 1, 0.1))
cat("\nTop-20 相似度分布概况：\n")
print(sim_quantiles)

# 动态剪枝：剔除相似度处于后 20% 的弱关联边（保留前 80% 强语义关联边）
threshold <- quantile(edges_dt$sim, probs = 0.20) 
cat(sprintf("\n使用相似度阈值剪枝: >= %.4f\n", threshold))

pruned_edges <- edges_dt[sim >= threshold]
cat(sprintf("剪枝前: %d 条边 -> 剪枝后: %d 条边 (保留率 %.2f%%)\n", 
            nrow(edges_dt), nrow(pruned_edges), (nrow(pruned_edges)/nrow(edges_dt))*100))

# 映射回干净的概念文本名称
pruned_edges[, `:=`(
  from_concept = concepts[from],
  to_concept = concepts[to]
)]

# ----------------------------------------------------------------------
# Step 6：构建图并求解 PageRank
# ----------------------------------------------------------------------
cat("\n正在构建图并计算 PageRank...\n")

g <- graph_from_data_frame(
  d = pruned_edges[, .(from_concept, to_concept, sim)], 
  directed = TRUE
)

# 计算带相似度权重的 PageRank
pr_scores <- page_rank(g, algo = "prpack", directed = TRUE, weights = E(g)$sim)$vector

top_n <- 30000
pg <- page_rank(g)$vector
top_concepts <- data.frame(name = names(pg), pagerank = as.numeric(pr_scores)) %>%
  arrange(desc(pagerank)) %>%
  head(top_n)

# 衔接 08_translate_concept_glm.R 包加载，以及第 0、3 部分代码。