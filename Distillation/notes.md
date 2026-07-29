https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF/

https://huggingface.co/mradermacher/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B.Q4_K_M.gguf



| 蒸馏范式 | 可访问资源 | 迁移内容 | 典型代表场景 |
| :--- | :--- | :--- | :--- |
| **白盒蒸馏** | 教师模型的权重、Logits、隐层表征 | 词表分布、Transformer 隐层向量、Self-Attention 权重 | 同系列开源模型压缩（如 70B 蒸馏至 8B） |
| **黑盒蒸馏** | 仅 API 输入输出文本 | 高质量文本、思维链（CoT）、合成指令集 | DeepSeek-R1 蒸馏小模型、GPT-4 提炼数据集 |
| **反馈/对齐蒸馏** | 教师模型的打分/偏好 | 偏好反馈（Reward/Preference） | 利用强模型充当 Judge 进行 DPO/RLAIF 对齐 |
