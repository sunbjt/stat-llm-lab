
library(arrow)
library(dplyr)
library(tidyr)
library(data.table)
library(tidyverse)

dataset_path <- "~/github/stat-llm-lab/data/processed/chunks/chunk_001.arrow"
arrow_table <- read_feather(dataset_path)

head(arrow_table$topk_ids[[1]], 10)
head(arrow_table$topk_probs[[1]], 10)
