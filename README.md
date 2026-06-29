# Sparse Autoencoder Feature Decomposition in DINO Vision Transformers

**Course:** Explainable and Trustworthy AI — Politecnico di Torino 2025/2026  
**Goal:** Train Sparse Autoencoders (SAEs) on DINO ViT-B/16 patch activations to decompose polysemantic neurons into interpretable visual features. Evaluate with CLIP-based automated labeling and two structural extensions.

---

## What This Project Does

A ViT neuron is **polysemantic** — it responds to many unrelated visual concepts at once. An SAE learns a larger set of sparse features that reconstruct those activations. The idea is that each SAE feature is **monosemantic**: it fires only for one coherent visual concept (a texture, a body part, a scene element). We then ask CLIP "what do the image patches that activate this feature look like?" to label each feature automatically.

We do this at three depths of DINO ViT-B/16: **Layer 4** (early, low-level), **Layer 8** (mid), **Layer 12** (final, semantic).

---

## Repository Structure

```
xai-vit-sae/
├── src/
│   ├── sae.py                    # SAE model
│   ├── sae_training.py           # Training loop, evaluation, dead-neuron resampling
│   ├── activation_collector.py   # Forward hooks to extract DINO patch activations
│   ├── clip_labeler.py           # CLIP-based feature labeling + separation score
│   ├── semantic_taxonomy.py      # Extension A: maps labels → 6 semantic categories
│   └── spatial_analysis.py       # Extension B: spatial variance of feature activation
├── notebooks/
│   ├── 01_activation_collection.ipynb   # Step 1: collect activations from ImageNet val
│   ├── 02_sae_training.ipynb            # Step 2: train SAEs
│   ├── 02_5_feature_diagnosis.ipynb     # Step 2.5: sanity-check trained SAEs
│   ├── 02_6_alpha_sweep.ipynb           # Step 2.6: sweep L1 penalty (α) to pick best
│   ├── 03_clip_labeling.ipynb           # Step 3: label SAE features with CLIP
│   ├── 04_evaluation.ipynb              # Step 4: CLIP score + human interpretability
│   ├── 05_spatial_coherence.ipynb       # Extension B: spatial variance analysis
│   └── 06_semantic_taxonomy.ipynb       # Extension A: semantic category distribution
├── data/
│   ├── layer4_activations.pt            # Raw DINO activations (N×197, 768-dim)
│   ├── layer8_activations.pt
│   ├── layer12_activations.pt
│   └── selected_indices.pt              # Which ImageNet-val images were used
├── checkpoints/
│   ├── sae_layer{4,8,12}.pt             # Trained SAE weights + normalization stats
│   ├── labels_layer{4,8,12}.pt          # CLIP labels for SAE features
│   ├── raw_labels_layer{4,8,12}.pt      # CLIP labels for raw ViT neurons (baseline)
│   ├── human_eval_grids/                # 150 PNG grids (50/layer) for human rating
│   ├── spatial_variances.pt             # Output of NB05 (Extension B)
│   ├── semantic_categories.pt           # Output of NB06 (Extension A)
│   └── *.png                            # Training curves, evaluation plots
└── tests/
    ├── test_semantic_taxonomy.py
    └── test_spatial_analysis.py
```

---

## Source Files

### `src/sae.py` — Sparse Autoencoder

A single-hidden-layer autoencoder with tied bias and ReLU sparsity.

```
x (768) → encode → f (3072, sparse via ReLU) → decode → x̂ (768)
```

- `d_input=768` (DINO ViT-B/16 hidden size), `d_hidden=3072` (4× expansion)
- Loss: `MSE(x, x̂) + α * L1(f)`  — the L1 term forces most features to be zero for any given input
- `normalize_decoder()` keeps decoder columns unit-norm (prevents the decoder from collapsing features into scale)

### `src/sae_training.py` — Training Logic

- **`PilotConfig`**: dataclass holding all hyperparameters (steps=30k, warmup=10k, α swept in NB02_6)
- **`run_pilot`**: full training loop with Adam, warmup schedule on α, and optional dead-neuron resampling at step 20k
- **`resample_dead_neurons`**: if a feature never fires during training, reinitialize it from high-reconstruction-error examples — prevents feature collapse
- **`evaluate_sae`**: computes MSE, L0 (average active features per token), dead feature %, and mean encoder similarity (redundancy check)

### `src/activation_collector.py` — DINO Hook

- Registers PyTorch forward hooks on `model.blocks[layer_idx]` for layers 3, 7, 11 (0-indexed), which correspond to Layers 4, 8, 12 in the paper
- Each image produces 197 tokens: 1 CLS + 196 patch tokens from the 14×14 spatial grid
- Saves flat tensors of shape `(N_images × 197, 768)` per layer
- Uses stratified sampling to cover all 1000 ImageNet classes proportionally

### `src/clip_labeler.py` — CLIP Feature Labeling

Two-step process:

1. **Labeling** (`label_features`): For each SAE feature, find the top-10 activating patches. Crop those patches from the original images. Average their CLIP image embeddings and find the closest text in a vocabulary (ImageNet-1000 + patch-level + texture/scene supplements). The closest text becomes the feature's label.

2. **Separation score** (`compute_separation_scores`): Measures how well the label distinguishes high-activation patches from low-activation patches. Computes `mean_CLIP_sim(top patches) - mean_CLIP_sim(bottom patches)`, normalized by standard deviation. Higher = label is more informative about when the feature fires.

### `src/semantic_taxonomy.py` — Extension A

Maps each CLIP label string to one of 6 semantic categories using a 3-tier lookup:

| Tier | Mechanism | Coverage |
|---|---|---|
| 1 | `_SUPP_MAP` — explicit dict for all 75 items in clip_labeler's patch/texture vocabularies | deterministic, highest priority |
| 2 | `_IMAGENET_CLASSES` — loaded from torchvision or `data/imagenet_classes.txt` | all 1000 ImageNet class names → `object` |
| 3 | keyword fallback with word-boundary regex | safety net for unexpected labels |

| Category | Examples |
|---|---|
| `background` | "blurred background", "dark background", "shadow" |
| `object_part` | "wing", "fin", "claw", "animal eye", "bird beak", "wheel" |
| `scene` | "sky", "water surface", "grass texture", "sandy surface" |
| `texture` | "animal fur", "feather texture", "brick wall", "vertical stripe" |
| `color` | "light", "dark" |
| `object` | all ImageNet-1000 class names (e.g. "black bear", "water buffalo") |

Tier 2 is critical: it prevents compound ImageNet names like "black bear" from misfiring on color/scene keywords in Tier 3. `data/imagenet_classes.txt` is generated by NB06 cell-0 (requires torchvision).

### `src/spatial_analysis.py` — Extension B

For each SAE feature, finds the top-10 activating **patch tokens** across a 100k-token sample, maps them back to their `(row, col)` position in the 14×14 ViT patch grid, and computes:

```
spatial_variance = row.var() + col.var()
```

- **Low variance** (~0–3): feature consistently activates at the same spatial location → spatially localized (e.g. a detector for objects at image center)
- **High variance** (~10–84): feature activates all over the image → spatially global (e.g. a texture detector)

Max possible value is ~84 (uniform distribution over 14×14 grid).

---

## Notebooks (Run Order)

| Notebook | What it does | Key output |
|---|---|---|
| `01_activation_collection` | Forward-pass 10k ImageNet-val images through DINO, save patch activations | `data/layer{4,8,12}_activations.pt` |
| `02_sae_training` | Train SAEs (α=1e-3, 30k steps) on each layer | `checkpoints/sae_layer{4,8,12}.pt` |
| `02_5_feature_diagnosis` | Check firing rates, dead features, reconstruction quality | Visual sanity check |
| `02_6_alpha_sweep` | Train short pilots at α ∈ {1e-4, 5e-4, 1e-3, 2e-3} to pick best sparsity | Justified α=1e-3 |
| `03_clip_labeling` | Label all 3072 SAE features + top-768-variance raw neurons with CLIP | `labels_layer{4,8,12}.pt`, `raw_labels_layer{4,8,12}.pt` |
| `04_evaluation` | CLIP separation score + human interpretability rating | Evaluation table, `human_ratings.pt` |
| `05_spatial_coherence` | Compute spatial variance per feature per layer | `spatial_variances.pt`, CDF plot |
| `06_semantic_taxonomy` | Assign 6-category labels, stacked bar chart, Shannon entropy | `semantic_categories.pt`, bar chart |

---

## Baseline Results

| Metric | Layer 4 | Layer 8 | Layer 12 |
|---|---|---|---|
| Final MSE | 0.3838 | 0.4248 | 0.4481 |
| Final L0 | 55.9 | 44.6 | 38.9 |
| Dead features | 0% | 0% | 0% |
| SAE mean cosine score | 0.293 | 0.294 | 0.294 |
| SAE mean separation score | 0.030 | 0.069 | 0.065 |
| Raw neuron mean cosine score | 0.295 | 0.295 | 0.295 |
| Raw neuron mean separation score | 0.047 | 0.045 | 0.032 |

All metrics computed over all 3072 SAE features / top-768-variance raw neurons. L0 decreases monotonically across layers (later layers use fewer features per token). SAE features outperform raw neurons at Layer 8 (+53%) and Layer 12 (+103%) on separation score. At Layer 4 the SAE is weaker than raw neurons — early-layer raw neurons already encode simple low-level features directly, so the SAE overhead hurts rather than helps interpretability here.

---

## Extension A — Semantic Taxonomy

**Question:** Do earlier layers encode more scene-level features while later layers encode more object/part-level features? (Raghu 2021 hypothesis)

Run `notebooks/06_semantic_taxonomy.ipynb`. It:
1. Assigns all 3072 SAE features per layer to one of 6 categories
2. Plots a two-panel stacked bar chart: raw distribution and interpretable-only (excluding "blurred background" attractor)
3. Computes Shannon entropy and CLIP interpretability rate per layer

**Actual results:**

| Category | Layer 4 | Layer 8 | Layer 12 |
|---|---|---|---|
| background | 2231 (72.6%) | 1404 (45.7%) | 1298 (42.3%) |
| texture | 330 (10.7%) | 451 (14.7%) | 382 (12.4%) |
| object_part | 55 (1.8%) | 104 (3.4%) | 98 (3.2%) |
| scene | 31 (1.0%) | 74 (2.4%) | 60 (2.0%) |
| object | 425 (13.8%) | 1039 (33.8%) | 1234 (40.2%) |
| CLIP interpretability rate | 28.3% | 55.0% | 58.3% |
| Shannon entropy (all) | 1.247 | 1.746 | 1.697 |

Among interpretable features (excluding "blurred background"), `texture` decreases monotonically (38%→27%→21%) and `object` increases monotonically (49%→62%→69%) across layers — consistent with the Raghu 2021 hypothesis. The CLIP interpretability rate nearly doubles from Layer 4 to Layer 8, indicating that deeper-layer features are more semantically coherent. `color` is always 0 because ImageNet class names outcompete color terms in CLIP cosine similarity (noted as a limitation).

---

## Extension B — Spatial Coherence

**Question:** Do earlier layers encode spatially global features (whole-image patterns) while later layers encode spatially localized features (specific regions)?

Run `notebooks/05_spatial_coherence.ipynb`. It:
1. Computes spatial variance for all 3072 features at each layer (100k token sample)
2. Plots a histogram of variance distributions per layer
3. Plots a CDF for cross-layer comparison
4. Reports % of features that are "global" (variance > 10) vs "local" (variance < 3)

**Actual results:**

| Layer | Mean variance | % Global (>10) | % Local (<3) |
|---|---|---|---|
| Layer 4 | 15.6 | 69.8% | 6.9% |
| Layer 8 | 20.2 | 85.8% | 1.7% |
| Layer 12 | 21.6 | 88.0% | 1.2% |

Spatial variance **increases** monotonically with depth — the opposite of what CNNs show. This is consistent with DINO ViT-B/16's global self-attention: all layers can attend to the entire image from the first layer. Deeper layers develop features that activate across even broader spatial extents, likely because DINO's self-supervised objective encourages global semantic representations. % Local features collapse to near-zero by Layer 12, confirming that almost no SAE feature is spatially pinned to a specific image region.

---

## Human Evaluation (Required)

Grid images for 150 features (50 per layer) are in `checkpoints/human_eval_grids/`. Each image is a 2×5 grid of the top-10 activating patches for one feature.

**Rating rubric:**

| Score | Meaning |
|---|---|
| 5 | All 10 patches clearly share one visual concept, nameable in one word |
| 4 | 8–9 patches consistent, minor outliers |
| 3 | Vague direction, mixed content |
| 2 | 3–4 patches loosely related |
| 1 | Completely random, no pattern |

After rating, run cell-09 in `notebooks/04_evaluation.ipynb` to save `checkpoints/human_ratings.pt`.

---

## Setup

```bash
pip install torch torchvision transformers timm tqdm matplotlib
```

DINO model is loaded via `torch.hub`:
```python
model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
```

GPU required for activation collection, SAE training, CLIP labeling, and spatial coherence computation. Semantic taxonomy (NB06) runs on CPU.
