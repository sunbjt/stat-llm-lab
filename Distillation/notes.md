https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF/

https://huggingface.co/mradermacher/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B.Q4_K_M.gguf



| 蒸馏范式 | 可访问资源 | 迁移内容 | 典型代表场景 |
| :--- | :--- | :--- | :--- |
| **白盒蒸馏** | 教师模型的权重、Logits、隐层表征 | 词表分布、Transformer 隐层向量、Self-Attention 权重 | 同系列开源模型压缩（如 70B 蒸馏至 8B） |
| **黑盒蒸馏** | 仅 API 输入输出文本 | 高质量文本、思维链（CoT）、合成指令集 | DeepSeek-R1 蒸馏小模型、GPT-4 提炼数据集 |
| **反馈/对齐蒸馏** | 教师模型的打分/偏好 | 偏好反馈（Reward/Preference） | 利用强模型充当 Judge 进行 DPO/RLAIF 对齐 |


离线蒸馏流程，分成 Teacher 端（Python） 和 Student 端（R torch） 两部分。

- Teacher (Python, Qwen 2B)：负责生成 soft labels（logits 或 top‑k 概率），保存到文件。
- Student (R torch, 15M)：加载这些文件，把 teacher 的 soft labels作为训练目标，做 KL/CE 蒸馏。
- 数据流：文本 → teacher logits → 文件 → student 训练。

异构 Tokenizer 离线蒸馏的核心难点在于 序列长度不匹配（Sequence Alignment） 与 词表空间不一致（Vocabulary Mapping）。处理该架构最稳妥的路线是 在 Python 端基于字符偏移（Character Offsets）对齐序列，并将 Teacher 的 Top-$K$ 概率投影到 Student 的词表空间，最后通过 Apache Arrow (Parquet) 文件流转到 R 语言。

1. 序列与词表对齐策略由于两个 Tokenizer 分词结果不同，Teacher 文本位置与 Student 文本位置无法点对点对齐：

    - 序列对齐 (Token-to-Char Mapping)：利用 Tokenizer 的 return_offsets_mapping=True 提取每个 Token 对应的原始字符区间 $[char_{start}, char_{end}]$。Python 端先确定 Student 的 Token 序列切片，再将 Teacher 在相同字符范围内的 logits 进行加权平均或对齐。
    - 词表映射 (Vocab Projection)：Teacher 输出整个词表的 logits 开销极大（Qwen 词表约 15 万）。在 Python 端提取 Teacher 的 Top-$K$（如 $K=32$）预测 Token 文本，映射回 Student Tokenizer 的 ID；若无法映射则填入 unk_id。

2. 数据落盘协议 (Python $\to$ Parquet)不要直接存储全量 Logits，采用 Top-$K$ 稀疏存储 + Parquet 列式存储，R 语言可通过 arrow 包零拷贝高效读取。


调用本地模型，以及安装位置：

```shell
export HF_ENDPOINT=https://hf-mirror.com
# 放置模型
mkdir -p /root/autodl-tmp/cache
```