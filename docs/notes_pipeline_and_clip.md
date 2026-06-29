# 项目笔记：Pipeline 结构 & CLIP 原理

---

## 一、各 Notebook 做什么，它们之间的关系

### 整体流水线

整个项目是一条顺序流水线，每个 notebook 的输出是下一个的输入：

```
ImageNet val 图片（10000 张，分层随机采样覆盖全部 1000 类）
      │
      ▼
 NB01  ──── 用 DINO ViT-B/16 提取 patch 激活值
      │         产出: data/layer{4,8,12}_activations.pt
      │              data/selected_indices.pt（记录使用了哪些图片）
      │
      ▼
 NB02  ──── 在激活值上训练 SAE（d_input=768, d_hidden=3072, 30k步）
      │         产出: checkpoints/sae_layer{4,8,12}.pt
      │
      ├── NB02_5 ── 检查训练是否健康（firing rate、dead feature）→ 只产出图，无文件
      │
      ├── NB02_6 ── alpha sweep 调参（已选定 α=1e-3，无需重跑）→ 无持久文件
      │
      ▼
 NB03  ──── 用 CLIP 给每个 SAE feature 打词标签（全部 3072 个 feature）
      │         也对 top-768 方差 ViT 原始神经元做相同标注（baseline 对比）
      │         产出: checkpoints/labels_layer{4,8,12}.pt
      │              checkpoints/raw_labels_layer{4,8,12}.pt
      │
      ▼
 NB04  ──── 计算评估指标 + 生成人工评分图格
                产出: checkpoints/human_eval_grids/（150 张图，需手动打分）
                     【待做】checkpoints/human_ratings.pt
```

### NB05 和 NB06 是平行的扩展分支

NB05/06 消费 baseline 已有的 checkpoint，**不修改 baseline 任何文件**，可随时重跑：

```
checkpoints/sae_layer{4,8,12}.pt          checkpoints/labels_layer{4,8,12}.pt
data/layer{4,8,12}_activations.pt                      │
          │                                             │
          ▼                                             ▼
        NB05                                          NB06
   Spatial Coherence（Extension B）          Semantic Taxonomy（Extension A）
   算每个 feature 激活的空间方差              把 CLIP 标签归成 6 大语义类别
          │                                             │
          ▼                                             ▼
spatial_variances.pt                      semantic_categories.pt
spatial_variance_distribution.png         semantic_taxonomy_distribution.png
spatial_variance_cdf.png
```

### 各 Notebook 一句话总结

| NB | 输入 | 做什么 | 输出 |
|---|---|---|---|
| 01 | ImageNet 图片 | 跑 DINO，用 forward hook 抓 Layer 4/8/12 的 patch 激活 | `layer*_activations.pt`, `selected_indices.pt` |
| 02 | `layer*_activations.pt` | 训练 SAE（MSE + L1 loss，30k steps，α=1e-3） | `sae_layer*.pt` |
| 02_5 | `sae_layer*.pt` | 检查 firing rate、dead feature，诊断训练健康状态 | 图（诊断用） |
| 02_6 | `layer*_activations.pt` | 短训练对比不同 alpha，确定最优稀疏度系数 | 无持久文件 |
| 03 | `sae_layer*.pt` + 原始图片 | CLIP 对**全部 3072 个** SAE feature 的 top-10 patch 打词标签；同时对 top-768 方差 ViT 原始神经元做 baseline 标注 | `labels_layer*.pt`, `raw_labels_layer*.pt` |
| 04 | `labels_layer*.pt` + `sae_layer*.pt` | 计算 CLIP separation score，生成人工评分图格 | `human_eval_grids/`，待打分后生成 `human_ratings.pt` |
| **05** | `sae_layer*.pt` + `layer*_activations.pt` | Extension B：计算每个 feature 的空间激活方差（100k token 样本） | `spatial_variances.pt`，CDF 图 |
| **06** | `labels_layer*.pt` | Extension A：把 CLIP 标签归 6 类，画跨层分布图，计算熵 | `semantic_categories.pt`，stacked bar 图 |

---

## 二、CLIP 的原理

### 核心思想

CLIP（Contrastive Language-Image Pre-training，OpenAI 2021）用**对比学习**把图像和文字拉到同一个向量空间里，使得语义相关的图文对在空间中距离更近。

### 训练方式

从互联网收集 4 亿对 `(图片, 文字描述)` 配对。每个 batch 有 N 张图片和 N 段文字：

```
图片: [🐕, 🚗, 🌊, ...]   →  图像编码器  →  [v₁, v₂, v₃, ...]
文字: ["a dog", "a car", "ocean waves", ...] →  文字编码器  →  [t₁, t₂, t₃, ...]
```

构造 N×N 相似度矩阵，对角线是匹配对，其余是不匹配的负样本：

```
         t₁(dog)   t₂(car)   t₃(ocean)
v₁(🐕)  [  高        低         低    ]
v₂(🚗)  [  低        高         低    ]
v₃(🌊)  [  低        低         高    ]
```

训练目标（对比损失）：对角线余弦相似度尽量高，非对角线尽量低。图像编码器和文字编码器同时训练，互相监督。

### 推理方式（Zero-shot）

给定一张图片和一堆候选文字，找余弦相似度最高的：

```python
候选词 = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

v = image_encoder(img)              # 图片向量
[t₁, t₂, t₃] = text_encoder(候选词) # 文字向量

scores = [cos(v, t₁), cos(v, t₂), cos(v, t₃)]
预测 = 候选词[argmax(scores)]        # 不需要微调，直接用
```

### 在本项目中的用法（NB03）

NB03 用 CLIP 给每个 SAE feature 自动打标签，流程如下：

```
某个 SAE feature
      │
      ▼
找 top-10 激活 patch（激活值最高的 10 个图像区域，64×64 像素裁剪）
      │
      ▼
CLIP 图像编码器 → 10 个 patch 各自的向量 → 取平均 → 向量 v
      │
      ▼
候选词（ImageNet 1000 类 + 38 个 patch 描述词 + 37 个纹理/场景词）
→ CLIP 文字编码器 → [t_fur, t_sky, t_dog, ...]
      │
      ▼
argmax(cos(v, tᵢ)) → 标签，例如 "animal fur"
```

本质上是在问：**"这个 feature 激活的那些 patch，长得最像什么概念？"**

### CLIP 的关键属性

| 属性 | 说明 |
|---|---|
| 语义空间共享 | 图片中的"狗毛"和文字"fur"在同一空间中挨得很近，不需要手工规则 |
| Zero-shot | 从未专门训练过"给 SAE feature 打标签"这个任务，直接迁移使用 |
| 局限性（CLIP attractor）| "blurred background" 在余弦相似度上异常强，早期层大量 feature 被贴上此标签，掩盖了真实分布 |

### Separation Score 的意义（NB04）

仅靠标签还不够，还需要知道标签有多"准"。Separation Score 衡量的是：

- 取 feature **激活最高**的那批 patch，算它们和标签文字的 CLIP 相似度（高激活组）
- 取 feature **激活最低**的那批 patch，算同样的相似度（低激活组）
- `score = (mean(高激活组) - mean(低激活组)) / std(两组合并)`，标准差归一化

Score 越高 = 这个标签越能区分"该 feature 在看什么"和"该 feature 不在看什么" = feature 越可解释。

---

## 三、semantic_taxonomy.py 的分类机制

NB06 使用 `src/semantic_taxonomy.py` 将 CLIP 打出的标签归入 6 个语义大类。采用**三层查找**策略：

```
assign_category(label)
      │
      ├── Tier 1: _SUPP_MAP（75条显式映射）
      │   覆盖 clip_labeler 词表中所有 patch/纹理/场景补充词
      │   例如: "animal fur" → texture, "blurred background" → background
      │   例如: "wing" → object_part, "water surface" → scene
      │
      ├── Tier 2: _IMAGENET_CLASSES（1000条）
      │   从 torchvision 或 data/imagenet_classes.txt 加载
      │   所有 ImageNet 类名 → object
      │   关键作用: 防止 "black bear"→color, "water buffalo"→scene 等误判
      │
      └── Tier 3: 关键字正则回退（安全网，正常情况不会触发）
```

6 个类别：background / texture / color / object_part / scene / object

**注意**：color 在实验中始终为 0，因为 CLIP 词表中 ImageNet 类名的语义竞争力远强于颜色词，CLIP 几乎从不选颜色词作为 feature 标签。这是词表设计的局限，需在报告中说明。

---

## 四、实验结果

### Baseline（NB03/04）

| 指标 | Layer 4 | Layer 8 | Layer 12 |
|---|---|---|---|
| Final MSE | 0.3838 | 0.4248 | 0.4481 |
| Final L0 | 55.9 | 44.6 | 38.9 |
| Dead features | 0% | 0% | 0% |
| SAE mean cosine | 0.293 | 0.294 | 0.294 |
| SAE separation score | 0.030 | 0.069 | 0.065 |
| Raw neuron separation | 0.047 | 0.045 | 0.032 |

SAE 在 Layer 8 (+53%) 和 Layer 12 (+103%) 上 separation score 显著优于原始神经元。Layer 4 例外：SAE 反而弱于原始神经元，因为早期层原始神经元已经编码了简单的低层次特征，SAE 的稀疏分解反而引入了噪声。

### Extension A — 语义分类（NB06，3072 features/层）

| 类别 | Layer 4 | Layer 8 | Layer 12 |
|---|---|---|---|
| background | 2231 (72.6%) | 1404 (45.7%) | 1298 (42.3%) |
| texture | 330 (10.7%) | 451 (14.7%) | 382 (12.4%) |
| object_part | 55 (1.8%) | 104 (3.4%) | 98 (3.2%) |
| scene | 31 (1.0%) | 74 (2.4%) | 60 (2.0%) |
| object | 425 (13.8%) | 1039 (33.8%) | 1234 (40.2%) |
| CLIP 可解释率 | 28.3% | 55.0% | 58.3% |
| Shannon 熵（全体） | 1.247 | 1.746 | 1.697 |
| Shannon 熵（可解释） | 1.615 | 1.464 | 1.306 |

**排除 blurred background 后（可解释特征内部）：**

- texture 单调下降：38.0% → 26.7% → 21.3%
- object 单调上升：49.0% → 61.5% → 68.9%
- 与 Raghu 2021 假设一致：深层 SAE 特征更多对应具体对象语义

**Layer 8 是熵的峰值**（1.746）：background 和 object 比例最均衡，feature 类型最多样。

### Extension B — 空间方差（NB05，3072 features/层）

| Layer | 均值方差 | % Global (>10) | % Local (<3) |
|---|---|---|---|
| Layer 4 | 15.6 | 69.8% | 6.9% |
| Layer 8 | 20.2 | 85.8% | 1.7% |
| Layer 12 | 21.6 | 88.0% | 1.2% |

**方差随层数单调递增**，与 CNN 预期相反。原因：ViT 的 self-attention 从第一层就是全图级别的，DINO 自监督训练进一步鼓励全局语义表示。所有层的 feature 都高度全局化，深层更甚。% Local 接近零，说明几乎没有空间定位型 feature。

---

## 五、待完成事项

1. **人工评分**（必须）：打开 `checkpoints/human_eval_grids/` 中的 150 张图，每张评分 1-5，然后运行 NB04 最后一个 cell 保存 `human_ratings.pt`
   - 5 = 10 个 patch 明显共享同一个视觉概念
   - 1 = 完全随机，无规律

2. **报告**：根据上述结果撰写项目报告
