import torch
from PIL import Image


# Extra terms for low-level/texture features not well-covered by ImageNet class names
_TEXTURE_SCENE_VOCAB = [
    'fur', 'feather', 'scales', 'skin', 'wool', 'fabric', 'metal', 'wood',
    'stone', 'glass', 'plastic', 'brick', 'tile', 'stripe', 'pattern',
    'texture', 'edge', 'dot', 'grid', 'gradient', 'shadow', 'background',
    'sky', 'water', 'ocean', 'mountain', 'forest', 'desert', 'snow', 'sand',
    'rock', 'cloud', 'wall', 'floor', 'road', 'light', 'dark',
]


def get_vocab() -> list:
    """Return ImageNet-1000 class names plus texture/scene supplements.

    Uses torchvision's built-in label list so we never hardcode 1000 strings.
    Falls back to _TEXTURE_SCENE_VOCAB only if torchvision is unavailable.
    """
    try:
        from torchvision.models import ResNet50_Weights
        imagenet_classes = list(ResNet50_Weights.IMAGENET1K_V1.meta['categories'])
    except Exception:
        imagenet_classes = []
    # Append texture/scene terms not present in ImageNet class names
    existing = set(imagenet_classes)
    extras = [w for w in _TEXTURE_SCENE_VOCAB if w not in existing]
    return imagenet_classes + extras


# Module-level cache; built lazily on first call to label_features
VOCAB: list = []


def _normalize_rows(embs: torch.Tensor) -> torch.Tensor:
    """Normalize rows safely without producing NaNs for zero vectors."""
    return embs / embs.norm(dim=1, keepdim=True).clamp_min(1e-8)


def crop_patch(img: Image.Image, tok_offset: int, context: int = 64) -> Image.Image:
    """Return a context×context crop centered on the patch for tok_offset.

    ViT-B/16 divides a 224×224 image into a 14×14 grid of 16×16 patches.
    Token 0 is the CLS token; tokens 1-196 correspond to spatial patches.
    """
    if tok_offset == 0:
        return img.resize((context, context))
    w, h = img.size
    row = (tok_offset - 1) // 14
    col = (tok_offset - 1) % 14
    cx = col * 16 + 8
    cy = row * 16 + 8
    left = max(0, min(cx - context // 2, w - context))
    top  = max(0, min(cy - context // 2, h - context))
    return img.crop((left, top, left + context, top + context))


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
        token_to_image:  dict mapping token_idx (int) → PIL.Image (full 224×224)
        k:               number of top-activating patches per feature
        vocab:           list of text strings; defaults to VOCAB above

    Returns:
        dict with keys 'labels' (list of str), 'scores' (list of float),
        'top_k_indices' (list of tensors)
    """
    from transformers import CLIPProcessor, CLIPModel  # lazy import — not needed for unit tests

    if vocab is None:
        global VOCAB
        if not VOCAB:
            VOCAB = get_vocab()
        vocab = VOCAB

    model     = CLIPModel.from_pretrained(clip_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model.eval()

    # Precompute text embeddings once
    prompts = [f"a photo of a {w}" for w in vocab]
    text_inputs = processor(text=prompts, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        text_embs = model.get_text_features(**text_inputs)
        text_embs = _normalize_rows(text_embs)  # (V, D)

    n_features = sae_activations.shape[1]
    results = {'labels': [], 'scores': [], 'top_k_indices': []}

    for feat_idx in range(n_features):
        top_k_idx  = find_top_k_indices(sae_activations, feat_idx, k)
        top_k_list = [i.item() for i in top_k_idx]

        # Crop to the specific 64×64 patch region instead of the whole image.
        # token_to_image values may be plain PIL.Image (legacy) or (PIL.Image, tok_offset) tuples.
        patches = []
        for idx in top_k_list:
            if idx in token_to_image:
                val = token_to_image[idx]
                if isinstance(val, tuple):
                    img, tok_offset = val
                else:
                    img, tok_offset = val, idx % 197
                patches.append(crop_patch(img, tok_offset))

        if not patches:
            results['labels'].append('unknown')
            results['scores'].append(0.0)
            results['top_k_indices'].append(top_k_idx)
            continue

        inputs = processor(images=patches, return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            img_embs = model.get_image_features(**inputs)
            img_embs = _normalize_rows(img_embs)
            mean_emb = _normalize_rows(img_embs.mean(dim=0, keepdim=True))  # (1, D)

        label = assign_labels_from_embeddings(mean_emb, text_embs, vocab)[0]
        score = (mean_emb @ text_embs.T).max().item()

        results['labels'].append(label)
        results['scores'].append(score)
        results['top_k_indices'].append(top_k_idx)

    return results


def compute_correlation_scores(
    sae_activations: torch.Tensor,
    token_to_image: dict,
    labels: list,
    clip_model_name: str = 'openai/clip-vit-base-patch32',
    device: str = 'cuda',
) -> list:
    """
    Interpretability score based on top-vs-bottom activation separation.

    For each feature, compare the CLIP similarity of the feature label on the
    highest-activation tokens against the lowest-activation tokens. A larger
    positive value means the label better concentrates on tokens where the
    feature actually fires.
    """
    from transformers import CLIPProcessor, CLIPModel

    model     = CLIPModel.from_pretrained(clip_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model.eval()

    # Collect patches for all tokens present in token_to_image
    valid_indices, patches = [], []
    for i in range(sae_activations.shape[0]):
        if i in token_to_image:
            val = token_to_image[i]
            img, tok_off = val if isinstance(val, tuple) else (val, i % 197)
            patches.append(crop_patch(img, tok_off))
            valid_indices.append(i)

    if not patches:
        return [0.0] * len(labels)

    # CLIP image embeddings for all patches (one-time cost)
    BATCH = 256
    img_embs_list = []
    for start in range(0, len(patches), BATCH):
        inp = processor(images=patches[start:start+BATCH],
                        return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            e = model.get_image_features(**inp)
            e = _normalize_rows(e)
        img_embs_list.append(e.cpu())
    img_embs = torch.cat(img_embs_list)  # (n_valid, 512)

    # Text embeddings for unique labels only (much smaller than vocab)
    unique_labels = list(dict.fromkeys(labels))
    txt_inp = processor(text=[f"a photo of a {l}" for l in unique_labels],
                        return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        txt_embs = model.get_text_features(**txt_inp)
        txt_embs = _normalize_rows(txt_embs)
    label_to_emb = {l: txt_embs[i].cpu() for i, l in enumerate(unique_labels)}

    valid_acts = sae_activations[valid_indices]  # (n_valid, n_features)

    sep_scores = []
    n_valid = valid_acts.shape[0]
    k_eval = max(1, n_valid // 20)
    k_eval = min(k_eval, max(1, n_valid // 2))
    for feat_idx, label in enumerate(labels):
        t_emb     = label_to_emb[label].unsqueeze(0)
        clip_sims = (img_embs @ t_emb.T).squeeze(-1)
        sae_vals  = valid_acts[:, feat_idx]
        top_idx = torch.topk(sae_vals, k=min(k_eval, sae_vals.numel())).indices
        bot_idx = torch.topk(-sae_vals, k=min(k_eval, sae_vals.numel())).indices
        pos_sims = clip_sims[top_idx]
        neg_sims = clip_sims[bot_idx]
        pooled = torch.cat([pos_sims, neg_sims])
        pooled_std = pooled.std(unbiased=False)
        score = pos_sims.mean() - neg_sims.mean()
        if pooled_std > 0:
            score = score / (pooled_std + 1e-8)
        sep_scores.append(score.item())

    return sep_scores
