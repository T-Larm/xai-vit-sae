import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sae_training import (
    PilotConfig,
    build_patch_indices,
    compute_activation_stats,
    run_pilot,
)


def parse_alpha_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def run_layer(layer: int, alphas: list[float], args: argparse.Namespace, config: PilotConfig) -> list[dict]:
    acts_path = args.data_dir / f"layer{layer}_activations.pt"
    print(f"\nLoading layer {layer}: {acts_path}")
    acts = torch.load(acts_path, weights_only=False)
    patch_indices = build_patch_indices(acts)
    print(f"Layer {layer}: {len(patch_indices)} patch tokens")

    print("Computing normalization stats...")
    mean, rms = compute_activation_stats(
        acts, patch_indices, d_input=config.d_input, chunk=config.stat_chunk
    )

    results = []
    for alpha in alphas:
        label = f"layer{layer}_alpha_{alpha:.0e}"
        results.append(run_pilot(acts, patch_indices, mean, rms, alpha, label, config))
    return results


def choose_winner(results: list[dict], target_sim: float, target_dead: float) -> dict:
    passing = [r for r in results if r["mean_sim"] < target_sim and r["dead_pct"] < target_dead]
    if passing:
        return max(passing, key=lambda r: r["alpha"])
    return min(results, key=lambda r: (r["mean_sim"], r["dead_pct"], r["mse"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run short SAE alpha sweeps for ViT layers 4 and 8.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/alpha_sweep_results.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-tokens", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alphas-l4", default="4e-3,6e-3,8e-3")
    parser.add_argument("--alphas-l8", default="3e-3,5e-3,6e-3")
    parser.add_argument("--target-sim", type=float, default=0.20)
    parser.add_argument("--target-dead", type=float, default=2.0)
    args = parser.parse_args()

    config = PilotConfig(
        device=args.device,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        eval_tokens=args.eval_tokens,
        seed=args.seed,
        resample_step=max(args.warmup_steps * 2, 1) if args.steps > args.warmup_steps * 2 else None,
    )

    results = {
        "config": vars(args) | {"data_dir": str(args.data_dir), "out": str(args.out)},
        "layer4": run_layer(4, parse_alpha_list(args.alphas_l4), args, config),
        "layer8": run_layer(8, parse_alpha_list(args.alphas_l8), args, config),
    }
    results["recommendations"] = {
        "layer4": choose_winner(results["layer4"], args.target_sim, args.target_dead),
        "layer8": choose_winner(results["layer8"], args.target_sim, args.target_dead),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved sweep results to {args.out}")
    print("Recommended alphas:")
    for layer, row in results["recommendations"].items():
        print(
            f"  {layer}: alpha={row['alpha']:.1e}, mse={row['mse']:.4f}, "
            f"l0={row['l0']:.1f}, dead={row['dead_pct']:.1f}%, mean_sim={row['mean_sim']:.4f}"
        )


if __name__ == "__main__":
    main()
