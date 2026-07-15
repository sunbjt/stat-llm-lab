
# ConceptNet 5.7 的离线三元组全量数据文件下载地址
# https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz

library(data.table)
gz_file <- '~/Downloads/conceptnet-assertions-5.7.0.csv.gz'
edges <- fread(gz_file, sep="\t", header=FALSE, 
               select=c(2, 3, 4, 5), # 仅保留核心列：关系、源节点、目标节点、元数据JSON
               col.names=c("Relation", "Source", "Target", "Metadata"))

en_edges <- edges[grepl("/c/en/", Source) & grepl("/c/en/", Target)]

# print(head(zh_edges))

library(jsonlite)

# 1. 提取元数据中的 weight 权重
# 使用 sapply 和 fromJSON 快速解析每一行的 JSON 获取 weight
en_edges[, Weight := sapply(Metadata, function(x) {
  tryCatch(fromJSON(x)$weight, error = function(e) NA)
})]

# 2. 清洗 Relation, Source 和 Target 列，去除 URI 前缀和词性后缀
en_edges[, Relation := gsub("^/r/", "", Relation)]
en_edges[, Source := gsub("^/c/en/([^/]+).*", "\\1", Source)]
en_edges[, Target := gsub("^/c/en/([^/]+).*", "\\1", Target)]
en_edges[, Source := gsub("_", " ", Source)] # 把下划线替换为空格
en_edges[, Target := gsub("_", " ", Target)]

# 3. 删除冗余的 Metadata 列
en_edges[, Metadata := NULL]
en_edges <- unique(en_edges, by = c("Relation", "Source", "Target"))

# 查看清洗后的纯净数据
print(head(en_edges))
saveRDS(en_edges, file = '~/Downloads/en_edges.Rdata')
