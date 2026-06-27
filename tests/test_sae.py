import torch
import pytest
import sys
sys.path.insert(0, '.')
from src.sae import SAE

D_INPUT = 768
D_HIDDEN = 3072
BATCH = 16


@pytest.fixture
def sae():
    return SAE(d_input=D_INPUT, d_hidden=D_HIDDEN, alpha=1e-3)


def test_encode_shape(sae):
    x = torch.randn(BATCH, D_INPUT)
    f = sae.encode(x)
    assert f.shape == (BATCH, D_HIDDEN)


def test_encode_nonnegative(sae):
    x = torch.randn(BATCH, D_INPUT)
    f = sae.encode(x)
    assert (f >= 0).all()


def test_decode_shape(sae):
    f = torch.randn(BATCH, D_HIDDEN).abs()
    x_hat = sae.decode(f)
    assert x_hat.shape == (BATCH, D_INPUT)


def test_forward_shapes(sae):
    x = torch.randn(BATCH, D_INPUT)
    f, x_hat = sae(x)
    assert f.shape == (BATCH, D_HIDDEN)
    assert x_hat.shape == (BATCH, D_INPUT)


def test_loss_nonnegative(sae):
    x = torch.randn(BATCH, D_INPUT)
    f, x_hat = sae(x)
    total, mse, l1 = sae.loss(x, f, x_hat)
    assert total.item() >= 0
    assert mse.item() >= 0
    assert l1.item() >= 0


def test_decoder_normalization(sae):
    with torch.no_grad():
        sae.W_d.mul_(5.0)
    sae.normalize_decoder()
    col_norms = sae.W_d.norm(dim=0)  # (D_HIDDEN,)
    assert torch.allclose(col_norms, torch.ones(D_HIDDEN), atol=1e-5)
