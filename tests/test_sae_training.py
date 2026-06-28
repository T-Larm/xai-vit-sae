import torch

from src.sae import SAE
from src.sae_training import (
    build_patch_indices,
    compute_activation_stats,
    mean_encoder_similarity,
)


def test_build_patch_indices_excludes_cls_tokens():
    acts = torch.zeros(2 * 5, 3)
    idx = build_patch_indices(acts, tokens_per_image=5)
    assert idx.tolist() == [1, 2, 3, 4, 6, 7, 8, 9]


def test_compute_activation_stats_uses_patch_tokens_only():
    acts = torch.tensor([
        [100.0, 100.0],
        [1.0, 3.0],
        [5.0, 7.0],
        [200.0, 200.0],
        [9.0, 11.0],
        [13.0, 15.0],
    ])
    idx = build_patch_indices(acts, tokens_per_image=3)
    mean, rms = compute_activation_stats(acts, idx, d_input=2, chunk=2)
    expected = acts[idx].mean(dim=0)
    assert torch.allclose(mean, expected)
    assert rms.item() > 0


def test_mean_encoder_similarity_zero_for_orthogonal_rows():
    sae = SAE(d_input=2, d_hidden=2)
    with torch.no_grad():
        sae.W_e.copy_(torch.eye(2))
    assert abs(mean_encoder_similarity(sae)) < 1e-6
