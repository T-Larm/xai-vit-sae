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
