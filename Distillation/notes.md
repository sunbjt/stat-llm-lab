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

### 已实现算法（ts_span_align.py，学生词表投影）

教师与学生的词表无任何 ID 对应关系，投影必须经过**文本桥**：
`教师 token id --解码(Qwen)--> 文本 --编码(学生 YTTM BPE)--> 学生子词 id`。

每学生位置 i 的边界 $b$ = 学生 token i+1 的起始字符。取**包含 b 的教师 token k**：
- 教师预测 token k 的分布 = `logits[k-1]`（上一教师边界处的 logits）
- 目标字符偏移 $off = b - \text{start}(k)$（b 恰为 token 起点时 off=0，等价"取第一个子词"）
- 对 `logits[k-1]` 的 top-K 每个候选教师 token：解码 → 文本 → 学生 BPE → 取**覆盖字符 off 的那个学生子词** id
- 多个候选投到同一学生 id → 概率**加和** → 取 top-K（默认 16，`--top-k` 可调）**归一化**存盘

**存储格式（控制体积）**：每学生位置存 265B ≈ topk_ids(32×int32=128B) + topk_probs(32×float32=128B) 是大头。
- `TOP_K=16`：topk 存储减半；
- `topk_probs` 用 **float16** 存盘（R 的 `load_arrow_chunk` 读成 double → 转 float32，无需改 R 代码）；
- `feather.write_feather(..., compression="zstd")`，R 的 arrow 可直接解压。
三者叠加后每样本从 ~135KB → ~19KB 量级。

**训练端（`02_distill_casual_lm.R`）**：`RtomicLazyChunksDataset` 在 `initialize` 时**一次性预载全部 chunk**（K=16/f16 后单 chunk 仅几十 MB，全量可控），训练期 `.getitem` 纯内存 O(1) 随机切片，杜绝“全局 shuffle + 单 chunk 缓存 → 相邻随机索引跨 chunk → 逐 batch 重读盘”的卡顿（曾导致读 5 分钟不训练）。内存内 `topk_ids` 存 **int32**、`topk_probs` 存 **float16**（loss 内 `.to()` 自动升到 long/float32），比 int64/f32 减半内存。`load_arrow_chunk` 用 `unlist(fixed_size_list)` 直连展平，免 rbind。

> **⚠️ 血的教训（halffloat 读取坑）**：`topk_probs` 列是 `fixed_size_list<halffloat>`。R arrow 的 `as.vector()` 会把 halffloat 的**位模式按 int16 解读**——f16 的 `1.0`（0x3C00）读成 **15360**，导致 KL 项放大 ~1 万倍、loss 飙到 767027。**必须先 `tb$topk_probs$cast(arrow::fixed_size_list_of(arrow::float32(), K))` 再 `as.vector`**（走正确的 f16→f32 转换）。topk_ids 是 int32，无此问题。pyarrow 读同一文件则正确（max probs = 1.0），验证时若 R 与 pyarrow 读数不一致，先怀疑 halffloat 读取。

边界 b 在**第一个**教师 token 内部时（k=0，无 `logits[-1]`）该位置无教师信号 → `loss_mask=False`。

学生表层 id = YTTM id + 1，且 **clamp ≤ 16383**（YTTM id 16383 → 表层 16384 超出 embedding 范围）。
学生子词偏移规则（R 侧同一 C++ 库验证）：`▁`→空格；**首子词**若以空格开头而文本不以空格开头，去掉该前导空格（YTTM 在文本开头发的是边界标记而非真实空格）；文本先 `.strip()`（不改变学生 ids）。

2. 数据落盘协议 (Python $\to$ Parquet)不要直接存储全量 Logits，采用 Top-$K$ 稀疏存储 + Parquet 列式存储，R 语言可通过 arrow 包零拷贝高效读取。


调用本地模型，以及安装位置：

```shell
export HF_ENDPOINT=https://hf-mirror.com
# 放置模型
mkdir -p /root/autodl-tmp/cache
```