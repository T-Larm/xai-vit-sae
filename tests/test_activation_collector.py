import torch
import torch.nn as nn
import tempfile
import os
import sys
sys.path.insert(0, '.')
from src.activation_collector import save_activations, register_hooks, _stratified_indices


class FakeDataset:
    """Minimal stand-in for ImageFolder: 100 classes × 50 images = 5000 total."""
    def __init__(self):
        self.imgs = [(f'img_{i}.jpg', i // 50) for i in range(5000)]


def test_stratified_indices_covers_all_classes():
    dataset = FakeDataset()
    indices = _stratified_indices(dataset, n_images=1000)
    assert len(indices) == 1000
    classes_covered = {dataset.imgs[i][1] for i in indices}
    assert len(classes_covered) == 100  # all 100 classes represented


def test_stratified_indices_saved_as_tensor():
    dataset = FakeDataset()
    indices = _stratified_indices(dataset, n_images=1000)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'selected_indices.pt')
        torch.save(torch.tensor(indices, dtype=torch.long), path)
        loaded = torch.load(path, weights_only=True)
    assert isinstance(loaded, torch.Tensor)
    assert loaded.dtype == torch.long
    assert loaded.tolist() == indices
    assert isinstance(loaded[0].item(), int)  # .item() must work (used in notebooks)


def test_stratified_indices_patch_mapping():
    """Verify that patch-only index correctly maps back to original image."""
    dataset = FakeDataset()
    indices = _stratified_indices(dataset, n_images=100)
    # For position pos in the sampled list, patch-only index is pos*196 + j
    for pos in range(min(5, len(indices))):
        for j in range(196):
            patch_idx = pos * 196 + j
            img_pos  = patch_idx // 196
            tok_off  = patch_idx % 196 + 1
            assert img_pos == pos
            assert 1 <= tok_off <= 196


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
