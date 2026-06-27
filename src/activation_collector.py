import torch
from pathlib import Path

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def register_hooks(model, layer_indices: list, captured: dict) -> list:
    hooks = []
    for layer_idx in layer_indices:
        captured[layer_idx] = []
        def make_hook(idx):
            def hook(module, input, output):
                captured[idx].append(output.detach().cpu())
            return hook
        hooks.append(model.blocks[layer_idx].register_forward_hook(make_hook(layer_idx)))
    return hooks


def save_activations(buffers: dict, output_dir: str, layer_indices: list):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for layer_idx in layer_indices:
        acts = torch.cat(buffers[layer_idx], dim=0)   # (N_images, 197, 768)
        flat = acts.reshape(-1, acts.shape[-1])        # (N_images * 197, 768)
        path = out / f'layer{layer_idx + 1}_activations.pt'
        torch.save(flat, path)
        print(f'Layer {layer_idx + 1}: saved {flat.shape} to {path}')


def run_collection(
    imagenet_val_dir: str,
    output_dir: str,
    layer_indices: list = [3, 11],
    n_images: int = 10000,
    batch_size: int = 64,
    device: str = 'cuda',
):
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    from tqdm import tqdm

    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    model.eval().to(device)

    captured = {}
    hooks = register_hooks(model, layer_indices, captured)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    dataset = datasets.ImageFolder(imagenet_val_dir, transform=transform)
    subset  = Subset(dataset, list(range(min(n_images, len(dataset)))))
    loader  = DataLoader(subset, batch_size=batch_size, num_workers=4, pin_memory=True)

    with torch.no_grad():
        for images, _ in tqdm(loader, desc='Collecting activations'):
            model(images.to(device))

    for hook in hooks:
        hook.remove()

    save_activations(captured, output_dir, layer_indices)
