import torch
import sys
sys.path.insert(0, '.')
from PIL import Image
from src.clip_labeler import find_top_k_indices, assign_labels_from_embeddings, crop_patch


def test_find_top_k_returns_highest():
    acts = torch.zeros(100, 10)
    acts[5,  3] = 9.0
    acts[42, 3] = 7.0
    acts[1,  3] = 5.0
    top = find_top_k_indices(acts, feature_idx=3, k=2)
    assert top[0].item() == 5
    assert top[1].item() == 42


def test_crop_patch_always_fixed_size():
    img = Image.new('RGB', (224, 224))
    for tok_offset in [1, 14, 100, 183, 196]:  # corners + center + edge
        crop = crop_patch(img, tok_offset, context=64)
        assert crop.size == (64, 64), f"Got {crop.size} for tok_offset={tok_offset}"


def test_crop_patch_cls_token():
    img = Image.new('RGB', (224, 224))
    assert crop_patch(img, tok_offset=0, context=64).size == (64, 64)


def test_crop_patch_tok_offset_range():
    img = Image.new('RGB', (224, 224))
    # tok_offset=1  → top-left patch, center at (8, 8)
    # tok_offset=196 → bottom-right patch, center at (216, 216)
    # Both should return 64x64 regardless of edge proximity
    for tok_offset in [1, 196]:
        crop = crop_patch(img, tok_offset, context=64)
        assert crop.size == (64, 64)


def test_assign_labels_returns_vocab_strings():
    mean_img_embs = torch.randn(3, 512)
    mean_img_embs = mean_img_embs / mean_img_embs.norm(dim=1, keepdim=True)
    vocab = ['cat', 'dog', 'car', 'tree', 'sky']
    text_embs = torch.randn(5, 512)
    text_embs = text_embs / text_embs.norm(dim=1, keepdim=True)
    labels = assign_labels_from_embeddings(mean_img_embs, text_embs, vocab)
    assert len(labels) == 3
    assert all(l in vocab for l in labels)
