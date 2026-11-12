#!/usr/bin/env python3
"""
GRPO training stage — standard HuggingFace transformers + PEFT.
"""
import hashlib
import yaml
import argparse
from pathlib import Path

import torch
import wandb

from huggingface_hub import HfApi, upload_folder
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
# llm_blender (pulled in by trl.trainer.judges) references TRANSFORMERS_CACHE which
# was removed from transformers.utils.hub in newer releases. Patch it back before trl loads.
import transformers.utils.hub as _thub
if not hasattr(_thub, "TRANSFORMERS_CACHE"):
    from huggingface_hub.constants import HF_HUB_CACHE as _hf_cache
    _thub.TRANSFORMERS_CACHE = _hf_cache

from trl import GRPOConfig, GRPOTrainer

from ita.base_score_model.base_score_model import BaseScoreStrategy
from ita.grpo_finetuning.threshold_optimization import (
    ThresholdState,
    load_threshold_history,
    optimize_thresholds_from_records,
)
from ita.grpo_finetuning.util import argumentative_reward, get_dataset
from ita.semantics.gradual_semantics import GradualSemantics
from ita.grpo_finetuning.training_store import STORE, ThresholdRecord
from ita.grpo_finetuning.visualizations import generate_all_plots


# ---------------------------------------------------------------------------
# HuggingFace upload
# ---------------------------------------------------------------------------

def _sanitize_repo_id(repo_id: str, max_len: int = 96) -> str:
    """Ensure repo_id fits HF Hub's naming rules (alphanumeric/-/_/., <= max_len
    chars total, no leading/trailing '-' or '.'). Long output_dir-derived names
    are truncated and given a short content hash suffix to avoid collisions
    between different names that share a truncated prefix.
    """
    if len(repo_id) <= max_len:
        return repo_id
    namespace, sep, name = repo_id.partition("/")
    if not sep:
        namespace, name = None, namespace
    suffix = "-" + hashlib.sha1(repo_id.encode()).hexdigest()[:8]
    prefix_len = len(namespace) + 1 if namespace else 0
    budget = max_len - prefix_len - len(suffix)
    truncated = name[:budget].rstrip("-.")
    new_name = truncated + suffix
    return f"{namespace}/{new_name}" if namespace else new_name


def upload_artefacts_to_hub(local_dir: str, repo_id: str, commit_message: str = "Upload artefacts"):
    api = HfApi()
    if "/" not in repo_id:
        try:
            username = api.whoami()["name"]
            repo_id = f"{username}/{repo_id}"
        except Exception:
            pass
    repo_id = _sanitize_repo_id(repo_id)
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    except Exception as e:
        msg = str(e)
        # "already exists" is harmless — everything else (403 token perms, network, etc.) is fatal
        if "already exist" not in msg.lower() and "409" not in msg:
            raise RuntimeError(
                f"[HF] Cannot create repo '{repo_id}'. "
                f"If you see 403, your token lacks write permissions — "
                f"run `huggingface-cli login` with a Write-access token.\n"
                f"Original error: {e}"
            ) from e
        print(f"[HF] Repo {repo_id} already exists, skipping create.")
    # Artefact upload is best-effort: the checkpoint/plots/logs are already
    # safely on local disk by this point, so a Hub-side failure (auth, network,
    # Xet backend outage, etc.) must not crash the whole multi-hour training run.
    try:
        upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            ignore_patterns=["*.pyc", "__pycache__"],
        )
        print(f"[HF] Uploaded {local_dir} → {repo_id}")
    except Exception as e:
        print(f"[HF] Warning: upload to {repo_id} failed, continuing training without it: {e}")


# ---------------------------------------------------------------------------
# Trainer callbacks
# ---------------------------------------------------------------------------

class StepTrackerCallback(TrainerCallback):
    """Keeps STORE's current step in sync and records loss from trainer logs."""

    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        STORE.set_step(state.global_step)

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and "loss" in logs:
            STORE.add_loss(state.global_step, logs["loss"])


class CheckpointUploadCallback(TrainerCallback):
    """At every trainer checkpoint: regenerate plots, flush logs and, if a
    repo_id is given, upload to HF."""

    def __init__(self, output_dir: Path, repo_id: str | None = None):
        self.output_dir = output_dir
        self.repo_id = repo_id

    def _flush_and_upload(self, step: int, label: str):
        print(f"[CheckpointUpload] Flushing artefacts at {label} (step {step})")
        generate_all_plots(STORE, self.output_dir / "plots")
        STORE.save_argument_log(self.output_dir / "logs" / "argument_log.jsonl")
        STORE.save_failure_summary(self.output_dir / "logs" / "failure_summary.json")
        STORE.save_threshold_log(self.output_dir / "logs" / "threshold_log.json")
        if self.repo_id:
            upload_artefacts_to_hub(
                str(self.output_dir),
                self.repo_id,
                commit_message=f"Artefacts at {label} (step {step})",
            )

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._flush_and_upload(state.global_step, f"checkpoint-{state.global_step}")

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._flush_and_upload(state.global_step, "train_end")


class MidpointCheckpointCallback(TrainerCallback):
    """Triggers a save at a manually specified step (used for the 2nd midpoint)."""

    def __init__(self, target_step: int):
        self.target_step = target_step
        self._saved = False

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if not self._saved and state.global_step >= self.target_step:
            control.should_save = True
            self._saved = True


class ThresholdRecalibrationCallback(TrainerCallback):
    """Periodically re-optimizes the reward's upper/lower thresholds using a
    recent window of generated completions (opt-in via
    --enable-threshold-optimization). No-op unless enabled by the caller."""

    def __init__(
        self,
        threshold_state: ThresholdState,
        window: int,
        interval: int,
        min_samples: int,
        criterion: str,
        use_wandb: bool,
    ):
        self.threshold_state = threshold_state
        self.window = window
        self.interval = interval
        self.min_samples = min_samples
        self.criterion = criterion
        self.use_wandb = use_wandb

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        step = state.global_step
        if step == 0 or step % self.interval != 0:
            return

        records = STORE.get_records_since(step - self.window)
        result = optimize_thresholds_from_records(
            records, criterion=self.criterion, min_samples=self.min_samples
        )
        if result is None:
            print(
                f"[ThresholdOpt] step={step}: skipped "
                f"(fewer than {self.min_samples} valid samples in the last {self.window} steps)"
            )
            return

        new_upper, new_lower, delta, metrics = result
        self.threshold_state.update(new_upper, new_lower)
        metric_value = metrics.get(self.criterion, float("nan"))
        STORE.add_threshold_record(
            ThresholdRecord(
                step=step,
                upper=new_upper,
                lower=new_lower,
                delta=delta,
                metric_name=self.criterion,
                metric_value=metric_value,
                n_samples=len(records),
            )
        )
        print(
            f"[ThresholdOpt] step={step} upper={new_upper:.4f} lower={new_lower:.4f} "
            f"delta={delta:.4f} {self.criterion}={metric_value:.4f} (n={len(records)})"
        )
        if self.use_wandb:
            wandb.log(
                {
                    "threshold_opt/upper": new_upper,
                    "threshold_opt/lower": new_lower,
                    "threshold_opt/delta": delta,
                    f"threshold_opt/{self.criterion}": metric_value,
                    "threshold_opt/n_samples": len(records),
                },
                step=step,
            )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def save_config(args: argparse.Namespace, output_dir: Path) -> Path:
    """Snapshot the resolved run config inside the run's output directory."""
    cfg = vars(args)
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = configs_dir / f"{output_dir.name}.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[Config] Saved to {cfg_path}")
    return cfg_path


def load_config(config_path: str) -> argparse.Namespace:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return argparse.Namespace(**cfg)


# ---------------------------------------------------------------------------
# Training entry-point
# ---------------------------------------------------------------------------

def train_transformers_model(
    args: argparse.Namespace,
    train_dataset,
    base_score_func: BaseScoreStrategy,
    gradual_semantics: GradualSemantics,
    use_wandb: bool = False,
):
    """
    Train with GRPO using standard transformers + PEFT.

    Args:
        args:              Parsed CLI namespace (used for model/training hyperparams).
        train_dataset:     HuggingFace Dataset (already formatted via make_conversation).
        base_score_func:   A BaseScoreStrategy instance injected from main.py.
        gradual_semantics: A GradualSemantics instance injected from main.py.
        use_wandb:         Whether W&B logging is active.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(args, output_dir)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required for training (left-pad is for inference only)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # LoRA — mirrors FastLanguageModel.get_peft_model(r=8, lora_alpha=32, ...)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    # Gemma 4 wraps its linear layers in Gemma4ClippableLinear which PEFT's type
    # check doesn't recognise.  Unwrap to the inner nn.Linear before applying LoRA.
    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if type(child).__name__ == "Gemma4ClippableLinear" and hasattr(child, "linear"):
                setattr(parent, child_name, child.linear)

    model = get_peft_model(model, lora_config)
    # GRPOTrainer.__init__ does model.warnings_issued["estimate_tokens"] = True.
    # PEFT's __getattr__ doesn't expose class-level dicts from PreTrainedModel,
    # so set it as an instance attribute explicitly.
    model.warnings_issued = {}
    model.print_trainable_parameters()

    print(f"[Model] {args.model_id} loaded in bf16 + LoRA (r=8)")

    # Compute midpoint checkpoint steps
    total_steps = max(1, (len(train_dataset) // max(1, args.batch_size)) * args.epochs)
    mid1 = total_steps // 3
    mid2 = (2 * total_steps) // 3
    print(f"[Checkpoints] mid1={mid1}, mid2={mid2}, total={total_steps}")

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        remove_unused_columns=False,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=2,
        num_train_epochs=args.epochs,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_completion_length=2048,
        max_prompt_length=512,
        num_generations=args.num_generatons,
        report_to=["wandb"] if use_wandb else ["tensorboard"],
        logging_steps=2,
        push_to_hub=False,
        save_strategy="steps",
        save_steps=mid1,
        resume_from_checkpoint=getattr(args, "resume_from_checkpoint", None),
    )

    enable_threshold_opt = getattr(args, "enable_threshold_optimization", False)
    threshold_state = None
    if enable_threshold_opt:
        seed_upper, seed_lower = args.upper_threshold, args.lower_threshold
        threshold_log_path = output_dir / "logs" / "threshold_log.json"
        if getattr(args, "resume_from_checkpoint", None):
            prior_history = load_threshold_history(threshold_log_path)
            for record in prior_history:
                STORE.add_threshold_record(record)
            if prior_history:
                seed_upper, seed_lower = prior_history[-1].upper, prior_history[-1].lower
                print(
                    f"[ThresholdOpt] Resuming with last computed thresholds: "
                    f"upper={seed_upper:.4f} lower={seed_lower:.4f} "
                    f"({len(prior_history)} prior re-optimization events reloaded)"
                )
        else:
            # Fresh run: record the static starting values as step=0 so the
            # plot/JSON log reflect what was actually used before the first
            # recalibration event, instead of only ever showing already-
            # recomputed values.
            STORE.add_threshold_record(
                ThresholdRecord(
                    step=0,
                    upper=seed_upper,
                    lower=seed_lower,
                    delta=seed_upper - seed_lower,
                    metric_name="seed",
                    metric_value=float("nan"),
                    n_samples=0,
                )
            )
        threshold_state = ThresholdState(upper=seed_upper, lower=seed_lower)

    # Artefacts are uploaded to the HF Hub (as <user>/<output dir name>) only on request
    hub_repo_id = output_dir.name if getattr(args, "push_to_hub", False) else None

    callbacks = [
        StepTrackerCallback(),
        MidpointCheckpointCallback(mid2),
        CheckpointUploadCallback(output_dir, repo_id=hub_repo_id),
    ]
    if enable_threshold_opt:
        callbacks.append(
            ThresholdRecalibrationCallback(
                threshold_state=threshold_state,
                window=getattr(args, "threshold_opt_window", 50),
                interval=getattr(args, "threshold_opt_interval", 50),
                min_samples=getattr(args, "threshold_opt_min_samples", 20),
                criterion=getattr(args, "threshold_opt_criterion", "balanced_accuracy"),
                use_wandb=use_wandb,
            )
        )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            argumentative_reward(
                base_score_func=base_score_func,
                semantics=gradual_semantics,
                upper_threshold=args.upper_threshold,
                lower_threshold=args.lower_threshold,
                use_wandb=use_wandb,
                continuous_reward=getattr(args, "continuous_reward", False),
                threshold_state=threshold_state,
            )
        ],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    resume_ckpt = getattr(args, "resume_from_checkpoint", None)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(output_dir))

    # Final artefact flush
    generate_all_plots(STORE, output_dir / "plots")
    STORE.save_argument_log(output_dir / "logs" / "argument_log.jsonl")
    STORE.save_failure_summary(output_dir / "logs" / "failure_summary.json")
    STORE.save_threshold_log(output_dir / "logs" / "threshold_log.json")
    if hub_repo_id:
        upload_artefacts_to_hub(str(output_dir), hub_repo_id, "Final model + all artefacts")

    print("[Done] Training complete.")
