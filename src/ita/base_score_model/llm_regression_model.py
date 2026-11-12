"""
Main changes compared to the original LLM regression model:
- LoRA applies to all key linear projections in Qwen3's architecture, not just the attention ones.
- LoRA uses the FEATURE_EXTRACTION task type rather than CAUSAL_LM.
- Applies a full analyst prompt and chat template during inference to maintain consistency with base model prompting.
- Uses a less aggressive dropout of 0.1 for both the LoRA adapters and the regression head.
- Uses safetensors for checkpointing.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast, overload

import torch
import torch.nn as nn
from tqdm import tqdm
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file as st_load, save_file as st_save
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


@dataclass
class LoraAdapterConfig:
    """LoRA adapter settings applied to the transformer backbone."""

    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: Literal["none", "all", "lora_only"] = "none"


@dataclass
class RegressionHeadConfig:
    """Architecture of the two-layer MLP regression head."""

    hidden_dim: int = 256
    dropout: float = 0.1


class LLMRegressionModel(nn.Module):
    """Transformer backbone with a learned regression head for score prediction.

    Args:
        model_name (str): HuggingFace Hub identifier for the causal LM
            backbone.
        lora_config (LoraAdapterConfig | None, optional): If given, wraps the
            backbone with a PEFT LoRA adapter so that only the low-rank
            matrices and the regression head are trainable. Pass ``None`` to
            keep the backbone as-is. Defaults to None.
        head_config (RegressionHeadConfig | None, optional): Regression head
            architecture. Defaults to ``RegressionHeadConfig()``.
        gradient_checkpointing (bool, optional): Enable gradient checkpointing
            on the backbone to trade compute for lower activation memory during
            training. Defaults to False.
        device_map (str | dict, optional): Forwarded to
            ``AutoModelForCausalLM.from_pretrained``. Use ``"auto"`` for
            automatic multi-GPU placement or a device string such as
            ``"cuda:0"`` for single-device placement. Defaults to ``"auto"``.
        seed (int, optional): Random seed for LoRA and regression head initialisation.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        lora_config: LoraAdapterConfig | None = None,
        head_config: RegressionHeadConfig | None = None,
        gradient_checkpointing: bool = False,
        device_map: str | dict[str, Any] = "auto",
        seed: int = 42,
    ) -> None:
        super().__init__()

        if head_config is None:
            head_config = RegressionHeadConfig()

        self._model_name = model_name
        self._lora_used = lora_config is not None
        self._head_config = head_config

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Left-padding ensures the final real token sits at the rightmost
        # position in every sequence, making it easier to extract the
        # final hidden state
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=device_map,
        )
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        if lora_config is not None:
            peft_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                target_modules=lora_config.target_modules,
                bias=lora_config.bias,
            )
            set_seed(seed)
            self.backbone = get_peft_model(self.backbone, peft_cfg)
            self.backbone.print_trainable_parameters()

        if hasattr(self.backbone.config, "hidden_size"):
            hidden_size: int = cast(int, self.backbone.config.hidden_size)  # type: ignore[union-attr]
        else:
            hidden_size: int = cast(int, self.backbone.config.text_config.hidden_size)  # type: ignore[union-attr]

        # With device_map="auto" the backbone may span multiple devices.
        # We co-locate the regression head with the backbone's first parameter
        # (the embedding layer) so that input tensors and the head share a
        # device; hidden states are explicitly moved in forward().
        self.input_device: torch.device = next(self.backbone.parameters()).device

        # The head runs in float32 — bfloat16 lacks the precision needed for
        # stable regression output, especially near the Sigmoid saturation points.
        set_seed(seed)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, head_config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_config.dropout),
            nn.Linear(head_config.hidden_dim, 1),
            nn.Sigmoid(),
        ).to(dtype=torch.float32, device=self.input_device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute regression scores for a tokenised batch.

        Args:
            input_ids (torch.Tensor): Shape ``[batch, seq_len]``. Must be on
                ``self._input_device``.
            attention_mask (torch.Tensor): Shape ``[batch, seq_len]``, 1 for
                real tokens, 0 for padding.

        Returns:
            (torch.Tensor): Shape ``[batch]``, values in ``[0, 1]``.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # With left-padding, padding tokens occupy the leading positions so the
        # last real token is always at the final sequence position.  Indexing
        # with -1 is both correct and cheaper than computing attention-mask sums.
        final_hidden: torch.Tensor = outputs.hidden_states[
            -1
        ]  # [batch, seq_len, hidden_size]
        last_token_hidden = final_hidden[:, -1, :]  # [batch, hidden_size]

        # Upcast and move to the regression head's device before the MLP.
        last_token_hidden = last_token_hidden.to(
            dtype=torch.float32, device=self.input_device
        )
        return self.regression_head(last_token_hidden).squeeze(-1)  # [batch]

    def tokenize_forward(
        self,
        conversations: list[list[dict[str, str]]],
    ) -> torch.Tensor:
        """Tokenize a batch of conversations and run model.forward().

        Args:
            conversations (list[list[dict]]): A batch of conversations, each a
                list of ``{"role": str, "content": str}`` dicts.

        Returns:
            (torch.Tensor): Shape ``[batch]``, values in ``[0, 1]``.
        """
        # Format each conversation with the backbone's chat template.
        # Suppress thinking for Qwen3.
        texts = [
            str(
                self.tokenizer.apply_chat_template(
                    conv,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
            for conv in conversations
        ]
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.input_device)
        attention_mask = encoded["attention_mask"].to(self.input_device)
        return self.forward(input_ids, attention_mask)

    @overload
    def predict(
        self,
        input_conversations: list[dict[str, str]],
        batch_size: int = 8,
        disable_tqdm: bool = False,
    ) -> float: ...
    @overload
    def predict(
        self,
        input_conversations: list[list[dict[str, str]]],
        batch_size: int = 8,
        disable_tqdm: bool = False,
    ) -> list[float]: ...
    def predict(
        self,
        input_conversations: list[dict[str, str]] | list[list[dict[str, str]]],
        batch_size: int = 8,
        disable_tqdm: bool = False,
    ) -> float | list[float]:
        """Predict regression scores for one or more chat-format conversations.

        Args:
            input_conversations (list[dict] | list[list[dict]]): Either a single
                conversation — a ``list`` of ``{"role": str, "content": str}``
                dicts — or a batch of such conversations.
            batch_size (int, optional): Maximum number of conversations per
                forward pass. Defaults to 8.
            disable_tqdm (bool, optional): If True, disable the tqdm progress bar.
                Defaults to False.

        Returns:
            (list[float]): One score per conversation, each in ``[0, 1]``.
        """
        # Normalise a single conversation to a length-1 batch.
        if input_conversations and isinstance(input_conversations[0], dict):
            conversations = cast(list[list[dict[str, str]]], [input_conversations])
            return_list = False
        else:
            conversations = cast(list[list[dict[str, str]]], input_conversations)
            return_list = True

        scores: list[float] = []
        self.eval()
        # Preferable to torch.no_grad() for inference.
        with torch.inference_mode():
            for start in tqdm(
                range(0, len(conversations), batch_size),
                desc="Predicting scores",
                disable=disable_tqdm,
            ):
                chunk = conversations[start : start + batch_size]
                scores_tensor = self.tokenize_forward(chunk)
                scores.extend(scores_tensor.tolist())

        if return_list:
            return scores
        return scores[0]

    @overload
    def generate(
        self,
        input_conversations: list[dict[str, str]],
        batch_size: int = 8,
        seed: int | None = None,
        disable_tqdm: bool = False,
    ) -> str: ...
    @overload
    def generate(
        self,
        input_conversations: list[list[dict[str, str]]],
        batch_size: int = 8,
        seed: int | None = None,
        disable_tqdm: bool = False,
    ) -> list[str]: ...
    def generate(
        self,
        input_conversations: list[dict[str, str]] | list[list[dict[str, str]]],
        batch_size: int = 8,
        seed: int | None = None,
        disable_tqdm: bool = False,
    ) -> str | list[str]:
        if seed is not None:
            set_seed(seed)

        if input_conversations and isinstance(input_conversations[0], dict):
            conversations = cast(list[list[dict[str, str]]], [input_conversations])
            return_list = False
        else:
            conversations = cast(list[list[dict[str, str]]], input_conversations)
            return_list = True

        formatted_texts: list[str] = [
            str(
                self.tokenizer.apply_chat_template(
                    conv,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
            for conv in conversations
        ]

        outputs: list[str] = []
        with torch.inference_mode():
            for start in tqdm(
                range(0, len(formatted_texts), batch_size),
                desc="Generating outputs",
                disable=disable_tqdm,
            ):
                batch = formatted_texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch, padding=True, truncation=True, return_tensors="pt"
                )
                input_ids = encoded["input_ids"].to(self.input_device)
                attention_mask = encoded["attention_mask"].to(self.input_device)
                with torch.inference_mode():
                    output_ids = self.backbone.generate(  # type: ignore[union-attr]
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=10,
                    )
                new_ids = output_ids[:, input_ids.shape[1] :]
                outputs.extend(
                    self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)
                )

        if return_list:
            return outputs
        return outputs[0]

    def save_checkpoint(self, checkpoint_dir: str | Path) -> None:
        """Persist the model to *checkpoint_dir*.

        Directory layout:
            checkpoint_dir/
              adapter/                       ← PEFT LoRA adapter files (LoRA only)
              regression_head.safetensors    ← regression head weights
              model_config.json              ← metadata required by load_checkpoint

        Args:
            checkpoint_dir (str | Path): Target directory; created if it does
                not exist.
        """
        out = Path(checkpoint_dir)
        out.mkdir(parents=True, exist_ok=True)

        if self._lora_used:
            # save_pretrained writes only the lightweight adapter matrices,
            # not the frozen backbone weights.
            self.backbone.save_pretrained(str(out / "adapter"), safe_serialization=True)

        st_save(
            self.regression_head.state_dict(), str(out / "regression_head.safetensors")
        )

        (out / "model_config.json").write_text(
            json.dumps(
                {
                    "model_name": self._model_name,
                    "lora_used": self._lora_used,
                    "head_config": asdict(self._head_config),
                },
                indent=2,
            )
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        device_map: str | dict[str, Any] = "auto",
        lora_config: LoraAdapterConfig | None = None,
        seed: int = 42,
    ) -> "LLMRegressionModel":
        """Reconstruct a model saved with :meth:`save_checkpoint`.

        If the checkpoint already contains a LoRA adapter it is always
        restored.  Pass ``lora_config`` to additionally wrap a non-LoRA
        checkpoint's backbone with fresh LoRA adapters (e.g. to start
        end-to-end fine-tuning from a primed regression-head checkpoint).

        Args:
            checkpoint_dir (str | Path): Directory previously written by
                :meth:`save_checkpoint`.
            device_map (str | dict, optional): Forwarded to
                ``AutoModelForCausalLM.from_pretrained``. Defaults to
                ``"auto"``.
            lora_config (LoraAdapterConfig | None, optional): When given and
                the checkpoint does *not* already include a LoRA adapter,
                the backbone is wrapped with fresh LoRA adapters using this
                configuration.  Ignored if the checkpoint already has LoRA.
                Defaults to ``None``.
            seed (int, optional): Random seed for reproducibility when using LoRA.
                Ignored if not using LoRA or if the checkpoint already has LoRA.
                Defaults to 42.

        Returns:
            (LLMRegressionModel): The loaded model.
        """
        ckpt = Path(checkpoint_dir)
        meta: dict[str, Any] = json.loads((ckpt / "model_config.json").read_text())

        head_cfg = RegressionHeadConfig(**meta["head_config"])

        # Construct the model without LoRA so the backbone is a plain
        # AutoModelForCausalLM; the adapter is attached in the next step.
        model = cls(
            model_name=meta["model_name"],
            lora_config=None,
            head_config=head_cfg,
            device_map=device_map,
        )

        if meta["lora_used"]:
            model.backbone = PeftModel.from_pretrained(
                model.backbone,
                str(ckpt / "adapter"),
                torch_device=str(model.input_device),
            )
            model._lora_used = True
        elif lora_config is not None:
            peft_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                target_modules=lora_config.target_modules,
                bias=lora_config.bias,
            )
            set_seed(seed)
            model.backbone = get_peft_model(model.backbone, peft_cfg)  # type: ignore[arg-type]
            model.backbone.print_trainable_parameters()
            model._lora_used = True

        # load_file accepts a device string (e.g. "cuda:0") and maps tensors there.
        head_state = st_load(
            str(ckpt / "regression_head.safetensors"),
            device=str(model.input_device),
        )
        model.regression_head.load_state_dict(head_state)

        return model
