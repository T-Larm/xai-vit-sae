from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from src.sae import SAE


@dataclass
class PilotConfig:
    d_input: int = 768
    d_hidden: int = 3072
    lr: float = 1e-4
    batch_size: int = 2048
    steps: int = 30_000
    warmup_steps: int = 10_000
    log_every: int = 2_000
    resample_step: int | None = 20_000
    eval_tokens: int = 50_000
    stat_chunk: int = 8192
    device: str = "cuda"
    seed: int = 0


def build_patch_indices(acts: torch.Tensor, tokens_per_image: int = 197) -> torch.Tensor:
    """Return flattened indices for patch tokens, excluding CLS tokens."""
    n_images = acts.shape[0] // tokens_per_image
    img_starts = torch.arange(n_images) * tokens_per_image
    patch_offsets = torch.arange(1, tokens_per_image)
    return (img_starts.unsqueeze(1) + patch_offsets.unsqueeze(0)).reshape(-1)


def compute_activation_stats(
    acts: torch.Tensor,
    patch_indices: torch.Tensor,
    d_input: int = 768,
    chunk: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and scalar RMS over patch-token activations."""
    n_tokens = len(patch_indices)
    mean = torch.zeros(d_input)
    for start in range(0, n_tokens, chunk):
        mean += acts[patch_indices[start:start + chunk]].float().sum(dim=0)
    mean /= n_tokens

    sos = 0.0
    for start in range(0, n_tokens, chunk):
        x = acts[patch_indices[start:start + chunk]].float() - mean
        sos += x.pow(2).sum().item()
    rms = torch.tensor(max((sos / (n_tokens * d_input)) ** 0.5, 1e-8))
    return mean, rms


def mean_encoder_similarity(sae: SAE) -> float:
    """Mean pairwise cosine similarity of encoder rows; lower means less redundancy."""
    with torch.no_grad():
        w = F.normalize(sae.W_e.detach().cpu(), dim=1)
        sim = w @ w.T
        sim.fill_diagonal_(0.0)
        return sim.mean().item()


def _zero_adam_state_for_dead(opt: torch.optim.Optimizer, sae: SAE, dead_mask: torch.Tensor) -> None:
    for param in (sae.W_e, sae.b_e, sae.W_d):
        state = opt.state.get(param)
        if not state or "exp_avg" not in state:
            continue
        if param is sae.W_d:
            state["exp_avg"][:, dead_mask] = 0
            state["exp_avg_sq"][:, dead_mask] = 0
        else:
            state["exp_avg"][dead_mask] = 0
            state["exp_avg_sq"][dead_mask] = 0


def resample_dead_neurons(
    sae: SAE,
    acts: torch.Tensor,
    patch_indices: torch.Tensor,
    mean: torch.Tensor,
    rms: torch.Tensor,
    opt: torch.optim.Optimizer,
    dead_mask: torch.Tensor,
    device: str,
    pool_size: int = 4096,
) -> int:
    """Reinitialize dead neurons from high-reconstruction-error examples."""
    n_dead = int(dead_mask.sum().item())
    if n_dead == 0:
        return 0

    pool = patch_indices[torch.randperm(len(patch_indices))[:pool_size]]
    x_pool = (acts[pool].float() - mean) / rms.item()

    with torch.no_grad():
        x = x_pool.to(device)
        _, x_hat = sae(x)
        errors = ((x - x_hat) ** 2).sum(dim=-1)
        probs = errors / errors.sum().clamp_min(1e-8)
        chosen = torch.multinomial(probs, n_dead, replacement=True)
        dirs = F.normalize(x[chosen], dim=-1)

        sae.W_e.data[dead_mask] = dirs * 0.2
        sae.b_e.data[dead_mask] = 0.0
        sae.W_d.data[:, dead_mask] = dirs.T
        _zero_adam_state_for_dead(opt, sae, dead_mask)
    return n_dead


def evaluate_sae(
    sae: SAE,
    acts: torch.Tensor,
    patch_indices: torch.Tensor,
    mean: torch.Tensor,
    rms: torch.Tensor,
    config: PilotConfig,
) -> dict:
    """Evaluate reconstruction, sparsity, dead features, and redundancy."""
    n_eval = min(config.eval_tokens, len(patch_indices))
    g = torch.Generator().manual_seed(config.seed + 10_000)
    sample = patch_indices[torch.randperm(len(patch_indices), generator=g)[:n_eval]]
    mean_dev = mean.to(config.device)
    rms_val = rms.item()

    total_mse = 0.0
    total_l0 = 0.0
    total_n = 0
    ever_fired = torch.zeros(config.d_hidden, dtype=torch.bool)
    sae.eval()
    with torch.no_grad():
        for start in range(0, n_eval, config.batch_size):
            batch_idx = sample[start:start + config.batch_size]
            x = (acts[batch_idx].to(config.device).float() - mean_dev) / rms_val
            f, x_hat = sae(x)
            batch_n = x.shape[0]
            total_mse += ((x - x_hat) ** 2).mean().item() * batch_n
            total_l0 += (f > 0).float().sum(dim=1).mean().item() * batch_n
            total_n += batch_n
            ever_fired |= (f > 0).any(dim=0).cpu()
    sae.train()

    return {
        "mse": total_mse / total_n,
        "l0": total_l0 / total_n,
        "dead_pct": (~ever_fired).float().mean().item() * 100,
        "mean_sim": mean_encoder_similarity(sae),
    }


def run_pilot(
    acts: torch.Tensor,
    patch_indices: torch.Tensor,
    mean: torch.Tensor,
    rms: torch.Tensor,
    alpha: float,
    label: str,
    config: PilotConfig | None = None,
    logger: Callable[[str], None] = print,
) -> dict:
    """Train a short SAE pilot and return summary metrics plus a training log."""
    config = config or PilotConfig()
    torch.manual_seed(config.seed)

    mean_dev = mean.to(config.device)
    rms_val = rms.item()
    n_patch = len(patch_indices)
    sae = SAE(config.d_input, config.d_hidden, alpha=alpha).to(config.device)
    opt = torch.optim.Adam(sae.parameters(), lr=config.lr)

    g = torch.Generator().manual_seed(config.seed)
    perm = torch.randperm(n_patch, generator=g)
    ptr = 0
    log = []

    for step in range(config.steps):
        if ptr + config.batch_size > n_patch:
            perm = torch.randperm(n_patch, generator=g)
            ptr = 0

        batch_pi = patch_indices[perm[ptr:ptr + config.batch_size]]
        ptr += config.batch_size
        x = (acts[batch_pi].to(config.device).float() - mean_dev) / rms_val

        f, x_hat = sae(x)
        alpha_eff = alpha * min(1.0, step / max(config.warmup_steps, 1))
        total, mse, _ = sae.loss(x, f, x_hat, alpha=alpha_eff)
        opt.zero_grad()
        total.backward()
        opt.step()
        sae.normalize_decoder()

        if config.resample_step is not None and step == config.resample_step:
            stats = evaluate_sae(sae, acts, patch_indices, mean, rms, config)
            dead_mask = torch.zeros(config.d_hidden, dtype=torch.bool)
            dead_count = round(stats["dead_pct"] * config.d_hidden / 100)
            if dead_count:
                # Recompute exact dead mask over the same evaluation sample.
                n_eval = min(config.eval_tokens, len(patch_indices))
                sample = patch_indices[torch.randperm(len(patch_indices), generator=g)[:n_eval]]
                ever_fired = torch.zeros(config.d_hidden, dtype=torch.bool)
                sae.eval()
                with torch.no_grad():
                    for start in range(0, n_eval, config.batch_size):
                        xb = (acts[sample[start:start + config.batch_size]].to(config.device).float() - mean_dev) / rms_val
                        ever_fired |= (sae.encode(xb) > 0).any(dim=0).cpu()
                sae.train()
                dead_mask = ~ever_fired
            n_resampled = resample_dead_neurons(
                sae, acts, patch_indices, mean, rms, opt, dead_mask, config.device
            )
            logger(f"  step {step}: resampled {n_resampled} dead neurons")

        if step % config.log_every == 0 or step == config.steps - 1:
            l0 = (f > 0).float().sum(dim=1).mean().item()
            batch_dead = ((f > 0).float().sum(dim=0) == 0).float().mean().item() * 100
            row = {
                "step": step,
                "mse": mse.item(),
                "l0": l0,
                "batch_dead_pct": batch_dead,
                "alpha_eff": alpha_eff,
            }
            log.append(row)
            logger(
                f"[{label}] step={step:5d} | mse={row['mse']:.4f} | "
                f"l0={l0:.1f} | batch_dead={batch_dead:.1f}% | alpha_eff={alpha_eff:.2e}"
            )

    summary = evaluate_sae(sae, acts, patch_indices, mean, rms, config)
    summary.update({"label": label, "alpha": alpha, "log": log})
    logger(
        f"[{label}] alpha={alpha:.1e} | mse={summary['mse']:.4f} | "
        f"l0={summary['l0']:.1f} | dead={summary['dead_pct']:.1f}% | "
        f"mean_sim={summary['mean_sim']:.4f}"
    )
    return summary
