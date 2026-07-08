# 加载必要的包
library(ggplot2)

# 你的 WSD 函数
wsd_multiplier <- function(step, total_steps, warmup_pct = 0.1, decay_pct = 0.1) {
  current_step <- as.numeric(step) + 1 
  total_steps <- as.numeric(total_steps)
  
  warmup_steps <- total_steps * warmup_pct
  decay_steps <- total_steps * decay_pct
  stable_steps <- total_steps - warmup_steps - decay_steps
  
  if (current_step <= warmup_steps) {
    return(current_step / warmup_steps)
  } else if (current_step <= warmup_steps + stable_steps) {
    return(1.0)
  } else {
    decay_step <- current_step - (warmup_steps + stable_steps)
    progress <- decay_step / decay_steps
    cosine_decay <- 0.5 * (1 + cos(pi * progress))
    return(max(0.1, cosine_decay))
  }
}

# ============================================================
# 配置参数
# ============================================================
total_steps <- 2100
warmup_pct <- 0.1
decay_pct <- 0.15  # 稍微调大一点让 Decay 阶段更明显

# 生成数据
steps <- 1:total_steps
multipliers <- sapply(steps, wsd_multiplier, 
                      total_steps = total_steps, 
                      warmup_pct = warmup_pct, 
                      decay_pct = decay_pct)

# 标记阶段
phase <- rep("Stable", length(steps))
phase[steps <= total_steps * warmup_pct] <- "Warmup"
phase[steps > total_steps * (1 - decay_pct)] <- "Decay"

df <- data.frame(step = steps, multiplier = multipliers, phase = phase)
# 计算各个分界点
warmup_end <- total_steps * warmup_pct
decay_start <- total_steps * (1 - decay_pct)

ggplot(df, aes(x = step, y = multiplier)) +
  geom_line(color = "#2C3E50") +
  # 添加阶段背景色块
  annotate("rect", 
           xmin = -Inf, xmax = warmup_end, 
           ymin = -Inf, ymax = Inf, 
           fill = "#E69F00", alpha = 0.1) +
  annotate("rect", 
           xmin = warmup_end, xmax = decay_start, 
           ymin = -Inf, ymax = Inf, 
           fill = "#0072B2", alpha = 0.1) +
  annotate("rect", 
           xmin = decay_start, xmax = Inf, 
           ymin = -Inf, ymax = Inf, 
           fill = "#D55E00", alpha = 0.1) +
  # 添加阶段文字标签
  annotate("text", 
           x = warmup_end / 2, y = 1.04, 
           label = "Warmup", size = 4, color = "#E69F00") +
  annotate("text", 
           x = (warmup_end + decay_start) / 2, y = 1.04, 
           label = "Stable (100%)", size = 4, color = "#0072B2") +
  annotate("text", 
           x = (decay_start + total_steps) / 2, y = 1.04, 
           label = "Decay", size = 4, color = "#D55E00") +
  # 添加垂直分割线
  geom_vline(xintercept = c(warmup_end, decay_start), 
             linetype = "dashed", color = "gray50", alpha = 0.7) +
  coord_cartesian(xlim = c(0, total_steps), ylim = c(0, 1.1)) +
  labs(
    title = "WSD 学习率调度器：三阶段可视化",
    subtitle = paste0("Warmup 线性上升 → Stable 保持满血 → Decay 余弦退火至 0.1"),
    x = "训练步数 (Step)",
    y = "学习率乘子 (Multiplier)"
  )

