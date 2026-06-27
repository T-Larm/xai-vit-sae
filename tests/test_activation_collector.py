import torch
import torch.nn as nn
import tempfile
import os
import sys
sys.path.insert(0, '.')
from src.activation_collector import save_activations, register_hooks


class FakeViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(768, 768) for _ in range(12)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def test_hooks_capture_correct_layers():
    model = FakeViT()
    x = torch.randn(2, 197, 768)
    captured = {}
    hooks = register_hooks(model, layer_indices=[3, 11], captured=captured)
    model(x)
    for h in hooks:
        h.remove()
    assert set(captured.keys()) == {3, 11}
    assert captured[3][0].shape == (2, 197, 768)
    assert captured[11][0].shape == (2, 197, 768)


def test_save_activations_shape():
    with tempfile.TemporaryDirectory() as tmpdir:
        buffers = {3: [torch.randn(4, 197, 768)], 11: [torch.randn(4, 197, 768)]}
        save_activations(buffers, output_dir=tmpdir, layer_indices=[3, 11])
        for layer_idx in [3, 11]:
            path = os.path.join(tmpdir, f'layer{layer_idx + 1}_activations.pt')
            assert os.path.exists(path)
            acts = torch.load(path, weights_only=False)
            assert acts.shape == (4 * 197, 768)
