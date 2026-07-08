# 计算模型参数量的实用函数
source("config.R")

# 1. 经典交叉熵架构
source("causal_lm/CausalLM_model.R")
causal_model <- RtomicCausalLM(
  vocab_size = VOCAB_SIZE,
  dim = DIM,
  n_layers = N_LAYERS,
  n_heads = N_HEADS,
  max_seq_len = SEQ_LEN
)

# 2. 潜在空间残差预测架构
source("latent_residual/LRP_model.R")
lrp_model <- RtomicLRP(
  vocab_size = VOCAB_SIZE,
  dim = DIM,
  n_layers = N_LAYERS,
  n_heads = N_HEADS,
  max_seq_len = SEQ_LEN
)

# 3. 潜在空间对比学习架构
source("latent_contrastive/contrastive_model.R")
Latent_model <- TokenLatentModel(
  vocab_size = VOCAB_SIZE,
  dim = DIM,
  n_layers = N_LAYERS,
  n_heads = N_HEADS,
  max_seq_len = SEQ_LEN
)

# 4.JEPA 完整实现（世界模型）— world_model
source("world_model/jepa_model.R")
world_model <- RtomicJEPA_VQ(
  vocab_size = VOCAB_SIZE,
  dim = DIM,
  n_layers = N_LAYERS,
  n_heads = N_HEADS,
  max_seq_len = SEQ_LEN
)

count_parameters <- function(model) {
  # 获取所有参数
  params <- model$parameters
  
  # 遍历计算每个参数张量的元素个数并求和
  total_params <- sum(sapply(params, function(p) p$numel()))
  
  # 也可以单独计算“可训练”的参数
  trainable_params <- sum(sapply(params, function(p) {
    if (p$requires_grad) p$numel() else 0
  }))
  
  cat(sprintf("总参数量: %s\n", format(total_params, big.mark = ",")))
  cat(sprintf("可训练参数: %s\n", format(trainable_params, big.mark = ",")))
  cat(sprintf("换算为百万级: %.2f M\n", total_params / 1e6))

  return(invisible(total_params))
}

count_parameters(causal_model)
count_parameters(lrp_model)
count_parameters(Latent_model)
count_parameters(world_model)



