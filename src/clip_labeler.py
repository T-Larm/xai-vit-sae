import torch


VOCAB = [
    # ImageNet-style objects
    'dog', 'cat', 'bird', 'fish', 'flower', 'tree', 'car', 'truck', 'airplane',
    'boat', 'person', 'face', 'hand', 'eye', 'fur', 'feather', 'wheel',
    # Textures and low-level
    'texture', 'pattern', 'edge', 'stripe', 'dot', 'grid', 'gradient',
    # Scenes / backgrounds
    'sky', 'grass', 'water', 'ground', 'sand', 'snow',
    'wall', 'floor', 'building', 'window', 'road', 'background',
]


def find_top_k_indices(activations: torch.Tensor, feature_idx: int, k: int) -> torch.Tensor:
    """Return indices of the k tokens with the highest activation for feature_idx."""
    vals = activations[:, feature_idx]
    return torch.topk(vals, k=min(k, len(vals))).indices


def assign_labels_from_embeddings(
    mean_image_embs: torch.Tensor,
    text_embs: torch.Tensor,
    vocab: list,
) -> list:
    """Assign best-matching vocab label to each feature via cosine similarity."""
    sims = mean_image_embs @ text_embs.T   # (n_features, n_vocab)
    return [vocab[i] for i in sims.argmax(dim=1).tolist()]


def label_features(
    sae_activations: torch.Tensor,
    token_to_image: dict,
    clip_model_name: str = 'openai/clip-vit-base-patch32',
    k: int = 10,
    vocab: list = None,
    device: str = 'cuda',
) -> dict:
    """
    Label SAE features with CLIP.

    Args:
        sae_activations: (N_tokens, N_features) — SAE encoded outputs
        token_to_image:  dict mapping token_idx (int) → PIL.Image
        k:               number of top-activating patches per feature
        vocab:           list of text strings; defaults to VOCAB above

    Returns:
        dict with keys 'labels' (list of str), 'scores' (list of float),
        'top_k_indices' (list of tensors)
    """
    from transformers import CLIPProcessor, CLIPModel  # lazy import — not needed for unit tests

    if vocab is None:
        vocab = VOCAB

    model     = CLIPModel.from_pretrained(clip_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model.eval()

    # Precompute text embeddings once
    text_inputs = processor(text=vocab, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(dim=1, keepdim=True)  # (V, D)

    n_features = sae_activations.shape[1]
    results = {'labels': [], 'scores': [], 'top_k_indices': []}

    for feat_idx in range(n_features):
        top_k_idx = find_top_k_indices(sae_activations, feat_idx, k)
        top_k_list = [i.item() for i in top_k_idx]
        patches    = [token_to_image[idx] for idx in top_k_list if idx in token_to_image]

        if not patches:
            results['labels'].append('unknown')
            results['scores'].append(0.0)
            results['top_k_indices'].append(top_k_idx)
            continue

        inputs = processor(images=patches, return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            img_embs  = model.get_image_features(**inputs)
            img_embs  = img_embs / img_embs.norm(dim=1, keepdim=True)
            mean_emb  = img_embs.mean(dim=0, keepdim=True)            # (1, D)

        label  = assign_labels_from_embeddings(mean_emb, text_embs, vocab)[0]
        score  = (mean_emb @ text_embs.T).max().item()

        results['labels'].append(label)
        results['scores'].append(score)
        results['top_k_indices'].append(top_k_idx)

    return results
