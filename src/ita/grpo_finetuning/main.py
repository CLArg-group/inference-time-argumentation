import sys
import yaml
import argparse
from pathlib import Path

# Repo root — relative dataset / bs_model_path / output_dir values in configs
# and CLI args are resolved against this, independent of the cwd.
REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Step 1 – parse args / load config (no heavy imports yet)
# ---------------------------------------------------------------------------

_CONFIG_DEFAULTS = {
    "seed": 0,
    "conservativeness": 1.0,
    "no_wandb": False,
    "continuous_reward": True,
    "bs_device": "cuda:0",
    "upper_threshold": None,
    "lower_threshold": None,
    "batch_size": 16,
    "epochs": 2,
    "learning_rate": 1e-5,
    "num_generatons": 2,
    "model_id": "Qwen/Qwen3-8B",
    "push_to_hub": False,
    "enable_threshold_optimization": False,
    "threshold_opt_window": 50,
    "threshold_opt_interval": 50,
    "threshold_opt_min_samples": 20,
    "threshold_opt_criterion": "balanced_accuracy",
}


def _load_config(config_path: str) -> argparse.Namespace:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file '{config_path}' is empty or invalid.")
    merged = {**_CONFIG_DEFAULTS, **cfg}
    return argparse.Namespace(**merged)


def _resolve_repo_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Make relative dataset / bs_model_path / output_dir values relative to REPO_ROOT."""
    for attr in ("dataset", "bs_model_path", "output_dir"):
        value = getattr(args, attr, None)
        if value and not Path(value).is_absolute():
            setattr(args, attr, str(REPO_ROOT / value))
    return args


def _parse_args() -> argparse.Namespace:
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        config_path = sys.argv[idx + 1]
        args = _load_config(config_path)
        print(f"[Config] Loaded from {config_path}")
        # Apply CLI-only resume overrides injected by the shell script (not in YAML)
        for flag, attr in [
            ("--resume_from_checkpoint", "resume_from_checkpoint"),
            ("--wandb_run_id", "wandb_run_id"),
        ]:
            if flag in sys.argv:
                setattr(args, attr, sys.argv[sys.argv.index(flag) + 1])
        return _resolve_repo_paths(args)
    else:
        from ita.grpo_finetuning.parse_cli import parse_arguments
        return _resolve_repo_paths(parse_arguments())


def main():
    args = _parse_args()

    import torch
    import wandb

    from ita.base_score_model.base_score_model import parse_base_score_model
    from ita.base_score_model.utils import parse_semantics
    from ita.grpo_finetuning.grpo_stage import train_transformers_model
    from ita.grpo_finetuning.util import get_dataset

    base_score_func = parse_base_score_model(args)
    gradual_semantics = parse_semantics(args.semantics, args.conservativeness)

    torch.manual_seed(args.seed)

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb_run_id = getattr(args, "wandb_run_id", None)
        wandb.init(
            project="argllms-grpo",
            config=vars(args),
            id=wandb_run_id,
            resume="allow",
        )

    train_dataset, _ = get_dataset(args.dataset)

    train_transformers_model(
        args=args,
        train_dataset=train_dataset,
        base_score_func=base_score_func,
        gradual_semantics=gradual_semantics,
        use_wandb=use_wandb,
    )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
