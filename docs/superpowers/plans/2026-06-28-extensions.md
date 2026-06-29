# SAE ViT Extensions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two extensions to the baseline SAE pipeline — spatial coherence analysis (Extension B) and semantic taxonomy analysis (Extension A) — to formally answer the paper's core research questions about cross-layer visual feature evolution in DINO ViT-B/16.

**Architecture:** Extension B computes spatial position variance of top-activating patches per SAE feature, revealing whether Layer 4 features are globally distributed (supporting Raghu 2021). Extension A maps existing CLIP labels into 6 semantic categories and plots how the texture/object/scene distribution shifts across layers 4 → 8 → 12. Both extensions reuse trained SAE checkpoints and activation tensors with no retraining.

**Tech Stack:** Python 3.10, PyTorch, matplotlib, existing `src/sae.py`, `src/clip_labeler.py`, `checkpoints/sae_layer*.pt`, `checkpoints/labels_layer*.pt`, `data/layer*_activations.pt`

## Global Constraints

- ViT-B/16 patch grid: 14×14, token offsets 1–196 (token 0 = CLS, excluded from all patch analysis)
- Token index in full tensor: `abs_idx` in `acts` of shape `(N_images * 197, 768)`
- Patch-only filtering: `patch_mask = torch.arange(N) % 197 != 0` → shape `(N_images * 196, 768)`
- Spatial row/col from patch-only index `p`: `row = (p % 196) // 14`, `col = (p % 196) % 14`
- D_INPUT = 768, D_HIDDEN = 3072
- Checkpoints always loaded with `map_location='cpu'` (machine may lack CUDA)
- All new source files go in `src/`, tests in `tests/`, notebooks in `notebooks/`
- Save all output figures to `checkpoints/` with `dpi=150`

---

## Task 1: Spatial Coherence Analysis (`src/spatial_analysis.py`)

**Files:**
- Create: `src/spatial_analysis.py`
- Create: `tests/test_spatial_analysis.py`
- Create: `notebooks/05_spatial_coherence.ipynb`

**Interfaces:**
- Produces: `compute_feature_spatial_variances(acts_path, ckpt, n_sample, k, seed, device) -> torch.Tensor` shape `(D_hidden,)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_spatial_analysis.py`:

```python
import torch
import pytest
from src.spatial_analysis import patch_idx_to_rowcol, compute_spatial_variance, compute_feature_spatial_variances


def test_patch_idx_to_rowcol_first():
    # patch-only index 0 → row 0, col 0
    assert patch_idx_to_rowcol(0) == (0, 0)


def test_patch_idx_to_rowcol_last():
    # patch-only index 195 → row 13, col 13
    assert patch_idx_to_rowcol(195) == (13, 13)


def test_patch_idx_to_rowcol_second_row():
    # patch-only index 14 → row 1, col 0
    assert patch_idx_to_rowcol(14) == (1, 0)


def test_compute_spatial_variance_identical():
    # all patches at same position → variance 0
    assert compute_spatial_variance([0, 0, 0]) == pytest.approx(0.0)


def test_compute_spatial_variance_spread():
    # patches at corners: (0,0)=idx0 and (13,13)=idx195
    var = compute_spatial_variance([0, 195])
    assert var > 0


def test_compute_spatial_variance_single():
    assert compute_spatial_variance([42]) == pytest.approx(0.0)


def test_compute_feature_spatial_variances_shape():
    # Smoke test with tiny synthetic data
    D_INPUT, D_HIDDEN = 4, 8
    N_IMAGES = 5
    N_TOKENS = N_IMAGES * 197

    acts = torch.randn(N_TOKENS, D_INPUT)

    from src.sae import SAE
    sae = SAE(d_input=D_INPUT, d_hidden=D_HIDDEN)
    fake_ckpt = {
        'state_dict': sae.state_dict(),
        'acts_mean': torch.zeros(D_INPUT),
        'acts_rms':  torch.ones(D_INPUT),
    }

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(acts, f.name)
        tmp = f.name

    result = compute_feature_spatial_variances(tmp, fake_ckpt, n_sample=N_IMAGES * 196, k=3, device='cpu')
    os.unlink(tmp)

    assert result.shape == (D_HIDDEN,)
    assert (result >= 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd xai-vit-sae
pytest tests/test_spatial_analysis.py -v
```

Expected: `ImportError: cannot import name 'patch_idx_to_rowcol' from 'src.spatial_analysis'`

- [ ] **Step 3: Write `src/spatial_analysis.py`**

```python
import torch
from src.sae import SAE


def patch_idx_to_rowcol(patch_only_idx: int) -> tuple:
    """Convert patch-only tensor index to (row, col) in the 14×14 ViT-B/16 grid.

    patch_only_idx is the position in a tensor where CLS tokens have been removed,
    so each image contributes exactly 196 entries.
      row = (patch_only_idx % 196) // 14   → 0..13
      col = (patch_only_idx % 196) % 14    → 0..13
    """
    pos_in_img = patch_only_idx % 196
    return pos_in_img // 14, pos_in_img % 14


def compute_spatial_variance(patch_only_indices: list) -> float:
    """Spatial variance (sum of row-var and col-var) for a list of patch-only indices."""
    if len(patch_only_indices) < 2:
        return 0.0
    rows = torch.tensor([(i % 196) // 14 for i in patch_only_indices], dtype=torch.float)
    cols = torch.tensor([(i % 196) % 14  for i in patch_only_indices], dtype=torch.float)
    return (rows.var(unbiased=False) + cols.var(unbiased=False)).item()


def compute_feature_spatial_variances(
    acts_path: str,
    ckpt: dict,
    n_sample: int = 100_000,
    k: int = 10,
    seed: int = 42,
    device: str = 'cuda',
) -> torch.Tensor:
    """Return a (D_hidden,) tensor of spatial variance per SAE feature.

    For each feature, top-k activating patch tokens are found and their
    positions in the 14×14 ViT patch grid are used to compute variance.
    High variance = feature activates across the whole image (global).
    Low variance  = feature activates at consistent locations (local).
    """
    acts = torch.load(acts_path, map_location='cpu', weights_only=False)

    # Remove CLS tokens (index 0 of each 197-token image sequence)
    N = acts.shape[0]
    patch_mask = torch.arange(N) % 197 != 0
    acts_patch = acts[patch_mask]  # (N_images * 196, D_INPUT)

    g = torch.Generator().manual_seed(seed)
    n_sample = min(n_sample, acts_patch.shape[0])
    perm = torch.randperm(acts_patch.shape[0], generator=g)[:n_sample]

    acts_mean = ckpt['acts_mean'].cpu() if hasattr(ckpt['acts_mean'], 'cpu') else torch.tensor(ckpt['acts_mean'])
    acts_rms  = ckpt['acts_rms'].cpu()  if hasattr(ckpt['acts_rms'],  'cpu') else torch.tensor(ckpt['acts_rms'])
    x = (acts_patch[perm].float() - acts_mean) / acts_rms.clamp_min(1e-8)

    sae = SAE(d_input=acts.shape[1], d_hidden=ckpt['state_dict']['W_e'].shape[0])
    sae.load_state_dict(ckpt['state_dict'])
    sae = sae.to(device)
    sae.eval()

    CHUNK = 4096
    parts = []
    with torch.no_grad():
        for s in range(0, x.shape[0], CHUNK):
            f, _ = sae(x[s:s + CHUNK].to(device))
            parts.append(f.cpu())
    sae_acts = torch.cat(parts)  # (n_sample, D_hidden)

    D = sae_acts.shape[1]
    spatial_vars = torch.zeros(D)

    for feat_idx in range(D):
        vals = sae_acts[:, feat_idx]
        n_fires = int((vals > 0).sum())
        if n_fires < 2:
            continue
        top_k_sample = vals.topk(min(k, n_fires)).indices  # indices into sample
        abs_patch = perm[top_k_sample].tolist()             # indices into acts_patch
        spatial_vars[feat_idx] = compute_spatial_variance(abs_patch)

    return spatial_vars
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_spatial_analysis.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```
git add src/spatial_analysis.py tests/test_spatial_analysis.py
git commit -m "feat: add spatial coherence analysis module"
```

- [ ] **Step 6: Create `notebooks/05_spatial_coherence.ipynb` and run it**

Cell 1 — imports and load:
```python
import sys; sys.path.insert(0, '..')
import torch
import matplotlib.pyplot as plt
from src.spatial_analysis import compute_feature_spatial_variances

DEVICE = 'cuda'  # change to 'cpu' if no GPU

ckpt4  = torch.load('../checkpoints/sae_layer4.pt',  map_location='cpu', weights_only=False)
ckpt8  = torch.load('../checkpoints/sae_layer8.pt',  map_location='cpu', weights_only=False)
ckpt12 = torch.load('../checkpoints/sae_layer12.pt', map_location='cpu', weights_only=False)
print('Checkpoints loaded')
```

Cell 2 — compute spatial variances (runs SAE forward pass, ~2 min per layer):
```python
print('Computing spatial variances (100k tokens per layer)...')
sv4  = compute_feature_spatial_variances('../data/layer4_activations.pt',  ckpt4,  device=DEVICE)
sv8  = compute_feature_spatial_variances('../data/layer8_activations.pt',  ckpt8,  device=DEVICE)
sv12 = compute_feature_spatial_variances('../data/layer12_activations.pt', ckpt12, device=DEVICE)

torch.save({'layer4': sv4, 'layer8': sv8, 'layer12': sv12},
           '../checkpoints/spatial_variances.pt')
print(f'Layer 4  mean_var={sv4.mean():.3f}  median={sv4.median():.3f}')
print(f'Layer 8  mean_var={sv8.mean():.3f}  median={sv8.median():.3f}')
print(f'Layer 12 mean_var={sv12.mean():.3f}  median={sv12.median():.3f}')
```

Cell 3 — plot distribution:
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
pairs = [('Layer 4', sv4, 'steelblue'), ('Layer 8', sv8, 'seagreen'), ('Layer 12', sv12, 'crimson')]

for ax, (name, sv, color) in zip(axes, pairs):
    ax.hist(sv.numpy(), bins=40, color=color, alpha=0.7)
    ax.axvline(sv.mean().item(), color='red', linestyle='--',
               label=f'Mean={sv.mean():.2f}')
    ax.set_title(f'{name} — Spatial Variance')
    ax.set_xlabel('Variance (row_var + col_var, max≈42)')
    ax.set_ylabel('Feature count')
    ax.legend()

plt.suptitle('Higher variance = feature activates globally across image\nLower variance = feature is spatially localized', fontsize=11)
plt.tight_layout()
plt.savefig('../checkpoints/spatial_variance_distribution.png', dpi=150)
plt.show()
```

Cell 4 — overlay CDF plot:
```python
import numpy as np

fig, ax = plt.subplots(figsize=(8, 5))
for name, sv, color in pairs:
    sorted_sv = np.sort(sv.numpy())
    cdf = np.arange(1, len(sorted_sv) + 1) / len(sorted_sv)
    ax.plot(sorted_sv, cdf, label=name, color=color, linewidth=2)

ax.axvline(10, color='gray', linestyle=':', label='var=10 threshold')
ax.set_xlabel('Spatial variance')
ax.set_ylabel('Cumulative fraction of features')
ax.set_title('CDF of Spatial Variance — cross-layer comparison')
ax.legend()
plt.tight_layout()
plt.savefig('../checkpoints/spatial_variance_cdf.png', dpi=150)
plt.show()
```

Cell 5 — print summary table:
```python
print(f"{'Layer':<12} {'Mean var':>10} {'% Global (>10)':>16} {'% Local (<3)':>14}")
print('-' * 55)
for name, sv in [('Layer 4', sv4), ('Layer 8', sv8), ('Layer 12', sv12)]:
    pct_global = 100 * (sv > 10).float().mean().item()
    pct_local  = 100 * (sv < 3).float().mean().item()
    print(f'{name:<12} {sv.mean().item():>10.3f} {pct_global:>16.1f}% {pct_local:>14.1f}%')
```

**Expected output direction:** Layer 4 should have higher mean variance than Layer 12, supporting Raghu 2021's claim that early ViT layers have mixed local/global attention. If the result is reversed, that is also a finding worth reporting.

- [ ] **Step 7: Commit notebook**

```
git add notebooks/05_spatial_coherence.ipynb checkpoints/spatial_variances.pt
git commit -m "feat: spatial coherence notebook and saved variances"
```

---

## Task 2: Semantic Taxonomy Analysis (`src/semantic_taxonomy.py`)

**Files:**
- Create: `src/semantic_taxonomy.py`
- Create: `tests/test_semantic_taxonomy.py`
- Create: `notebooks/06_semantic_taxonomy.ipynb`

**Interfaces:**
- Consumes: `checkpoints/labels_layer{4,8,12}.pt` → `labels_dict['labels']` (list of str, length 3072)
- Produces: `assign_category(label: str) -> str` returning one of `{'background', 'texture', 'color', 'object_part', 'scene', 'object'}`
- Produces: `get_category_distribution(labels: list) -> dict[str, int]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_semantic_taxonomy.py`:

```python
from src.semantic_taxonomy import assign_category, get_category_distribution, CATEGORIES


def test_categories_complete():
    assert set(CATEGORIES) == {'background', 'texture', 'color', 'object_part', 'scene', 'object'}


def test_background_label():
    assert assign_category('blurred background') == 'background'


def test_texture_label():
    assert assign_category('animal fur') == 'texture'
    assert assign_category('fabric texture') == 'texture'


def test_object_part_label():
    assert assign_category('muzzle') == 'object_part'
    assert assign_category('bird beak') == 'object_part'
    assert assign_category('wing') == 'object_part'


def test_scene_label():
    assert assign_category('grass texture') == 'scene'
    assert assign_category('water surface') == 'scene'


def test_imagenet_class_is_object():
    # 'patas' is an ImageNet class (patas monkey), not a texture/part/scene keyword
    assert assign_category('patas') == 'object'
    assert assign_category('disk brake') == 'object'


def test_distribution_sums():
    labels = ['blurred background', 'animal fur', 'patas', 'muzzle', 'grass texture']
    dist = get_category_distribution(labels)
    assert sum(dist.values()) == len(labels)
    assert set(dist.keys()) <= set(CATEGORIES)


def test_distribution_counts():
    labels = ['blurred background', 'blurred background', 'patas']
    dist = get_category_distribution(labels)
    assert dist['background'] == 2
    assert dist['object'] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_semantic_taxonomy.py -v
```

Expected: `ImportError: cannot import name 'assign_category' from 'src.semantic_taxonomy'`

- [ ] **Step 3: Write `src/semantic_taxonomy.py`**

```python
CATEGORIES = ['background', 'texture', 'color', 'object_part', 'scene', 'object']

# Keywords checked in priority order; first match wins.
# 'object' is the catch-all (no keywords) for ImageNet class names.
_RULES = [
    ('background', ['background', 'backdrop']),
    ('texture',    ['fur', 'feather', 'scale', 'grain', 'fabric', 'skin', 'wool',
                    'metal surface', 'wooden surface', 'stone surface',
                    'concrete surface', 'brick', 'tile', 'stripe', 'pattern',
                    'texture', 'edge', 'dot', 'grid']),
    ('color',      ['dark background', 'bright background', ' white ', ' black ',
                    ' brown ', ' dark ', ' bright ']),
    ('object_part',['eye', 'beak', 'nose', 'ear', 'paw', 'muzzle', 'wing', 'fin',
                    'tail', 'claw', 'wheel', 'window', 'roof', 'wall', 'face',
                    'slot', 'file', 'antenna', 'horn', 'tooth']),
    ('scene',      ['sky', 'water', 'ocean', 'mountain', 'forest', 'desert', 'snow',
                    'sand', 'rock', 'cloud', 'road', 'floor', 'grass', 'surface',
                    'field', 'beach', 'river', 'lake', 'soil', 'mud']),
]


def assign_category(label: str) -> str:
    """Map a CLIP label string to one of 6 semantic categories.

    Priority: background > texture > color > object_part > scene > object.
    'object' is the default catch-all for ImageNet class names.
    """
    label_lower = label.lower()
    for category, keywords in _RULES:
        if any(kw in label_lower for kw in keywords):
            return category
    return 'object'


def get_category_distribution(labels: list) -> dict:
    """Count features per category for a list of label strings."""
    dist = {cat: 0 for cat in CATEGORIES}
    for label in labels:
        dist[assign_category(label)] += 1
    return dist
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_semantic_taxonomy.py -v
```

Expected: all 8 tests PASS. If `grass texture` fails the scene test, add `'grass'` to the scene keyword list.

- [ ] **Step 5: Commit**

```
git add src/semantic_taxonomy.py tests/test_semantic_taxonomy.py
git commit -m "feat: add semantic taxonomy module"
```

- [ ] **Step 6: Create `notebooks/06_semantic_taxonomy.ipynb` and run it**

Cell 1 — imports and load:
```python
import sys; sys.path.insert(0, '..')
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from src.semantic_taxonomy import assign_category, get_category_distribution, CATEGORIES

labels4  = torch.load('../checkpoints/labels_layer4.pt',  map_location='cpu', weights_only=False)
labels8  = torch.load('../checkpoints/labels_layer8.pt',  map_location='cpu', weights_only=False)
labels12 = torch.load('../checkpoints/labels_layer12.pt', map_location='cpu', weights_only=False)
print(f'Loaded: {len(labels4["labels"])} features per layer')
```

Cell 2 — assign categories and print distribution:
```python
cats4  = [assign_category(l) for l in labels4['labels']]
cats8  = [assign_category(l) for l in labels8['labels']]
cats12 = [assign_category(l) for l in labels12['labels']]

torch.save({'layer4': cats4, 'layer8': cats8, 'layer12': cats12},
           '../checkpoints/semantic_categories.pt')

print(f"{'Category':<14} {'Layer 4':>10} {'Layer 8':>10} {'Layer 12':>10}")
print('-' * 48)
d4, d8, d12 = [get_category_distribution(c) for c in [cats4, cats8, cats12]]
for cat in CATEGORIES:
    print(f'{cat:<14} {d4[cat]:>10} {d8[cat]:>10} {d12[cat]:>10}')
```

Cell 3 — stacked bar chart:
```python
dists = [d4, d8, d12]
layer_names = ['Layer 4', 'Layer 8', 'Layer 12']
cat_colors = {
    'background':  '#999999',
    'texture':     '#4e79a7',
    'color':       '#b07aa1',
    'object_part': '#f28e2b',
    'scene':       '#59a14f',
    'object':      '#e15759',
}

n_features = len(labels4['labels'])
fig, ax = plt.subplots(figsize=(9, 6))
bottoms = np.zeros(3)

for cat in CATEGORIES:
    counts = np.array([d[cat] / n_features * 100 for d in dists])
    ax.bar(layer_names, counts, bottom=bottoms,
           label=cat, color=cat_colors[cat], width=0.5)
    bottoms += counts

ax.set_ylabel('% of SAE features')
ax.set_title('Semantic Category Distribution Across Layers\n(Raghu 2021 hypothesis: Layer 4 should have more scene-level features than Layer 12)')
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
plt.tight_layout()
plt.savefig('../checkpoints/semantic_taxonomy_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
```

Cell 4 — entropy (diversity) per layer:
```python
def entropy(dist: dict, n_total: int) -> float:
    import math
    return -sum((v / n_total) * math.log2(v / n_total + 1e-9)
                for v in dist.values() if v > 0)

print(f"\n{'Layer':<12} {'Entropy (bits)':>16}")
print('-' * 30)
for name, dist in [('Layer 4', d4), ('Layer 8', d8), ('Layer 12', d12)]:
    h = entropy(dist, n_features)
    print(f'{name:<12} {h:>16.3f}')
print('(Higher entropy = more diverse feature types)')
```

**Expected output direction:**
- Layer 4 should have more `scene` and `texture` features, fewer `object` features
- Layer 12 should have more `object` and `object_part` features
- Entropy should be highest at Layer 4 (most mixed) and lowest at Layer 12 (most specialized)

If the taxonomy mapping is off (e.g., too many features fall into 'object'), inspect `Counter(cats4).most_common(10)` and adjust `_RULES` in `src/semantic_taxonomy.py` accordingly, then re-run tests.

- [ ] **Step 7: Commit notebook**

```
git add notebooks/06_semantic_taxonomy.ipynb checkpoints/semantic_categories.pt
git commit -m "feat: semantic taxonomy notebook and saved categories"
```

---

## Self-Review

**Spec coverage:**
- Extension B (spatial coherence) → Task 1 ✓
- Extension A (semantic taxonomy) → Task 2 ✓
- Cross-layer plots → both tasks include `savefig` calls ✓
- TDD: tests written before implementation in both tasks ✓
- Raghu 2021 hypothesis explicitly framed in expected-output notes ✓

**Placeholder scan:** None found.

**Type consistency:**
- `compute_feature_spatial_variances` returns `torch.Tensor` shape `(D_hidden,)` — used as `sv4`, `sv8`, `sv12` in notebook ✓
- `assign_category` returns `str` — used in list comprehension and `get_category_distribution` ✓
- `get_category_distribution` returns `dict[str, int]` — used in table print and bar chart ✓
