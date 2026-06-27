import torch
import sys
sys.path.insert(0, '.')
from src.clip_labeler import find_top_k_indices, assign_labels_from_embeddings


def test_find_top_k_returns_highest():
    acts = torch.zeros(100, 10)
    acts[5,  3] = 9.0
    acts[42, 3] = 7.0
    acts[1,  3] = 5.0
    top = find_top_k_indices(acts, feature_idx=3, k=2)
    assert top[0].item() == 5
    assert top[1].item() == 42


def test_assign_labels_returns_vocab_strings():
    mean_img_embs = torch.randn(3, 512)
    mean_img_embs = mean_img_embs / mean_img_embs.norm(dim=1, keepdim=True)
    vocab = ['cat', 'dog', 'car', 'tree', 'sky']
    text_embs = torch.randn(5, 512)
    text_embs = text_embs / text_embs.norm(dim=1, keepdim=True)
    labels = assign_labels_from_embeddings(mean_img_embs, text_embs, vocab)
    assert len(labels) == 3
    assert all(l in vocab for l in labels)
