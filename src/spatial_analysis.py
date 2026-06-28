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
