"""Small, dependency-free LoRA layer set for Motus Stage-3 fine-tuning.

The official Stage-3 trainer, data loader, three-expert forward pass and losses stay
unchanged.  This module only changes which parameters receive optimizer updates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize


class LoRALinear(nn.Module):
    """Drop-in replacement that preserves the original ``weight``/``bias`` keys."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        self.bias = base.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.lora_A = nn.Parameter(base.weight.new_empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = F.linear(value, self.weight, self.bias)
        update = F.linear(F.linear(self.dropout(value), self.lora_A), self.lora_B)
        return base + update * self.scaling


class LoRAWeightDelta(nn.Module):
    """LoRA parametrization for Motus' non-Linear 4-D expert QKV tensors."""

    def __init__(self, weight: torch.Tensor, rank: int, alpha: float):
        super().__init__()
        if weight.ndim != 4:
            raise ValueError(f"expected [3, heads, in, head_dim], got {tuple(weight.shape)}")
        self.shape = tuple(weight.shape)
        input_dim = self.shape[2]
        output_dim = self.shape[0] * self.shape[1] * self.shape[3]
        self.rank = int(rank)
        self.scaling = float(alpha) / self.rank
        self.lora_A = nn.Parameter(weight.new_empty(self.rank, input_dim))
        self.lora_B = nn.Parameter(weight.new_zeros(output_dim, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        delta = (self.lora_B @ self.lora_A).reshape(self.shape)
        return weight + delta * self.scaling


def _set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def _branch(name: str) -> str | None:
    if name.startswith("video_model.wan_model.blocks."):
        return "wan"
    if name.startswith("action_expert."):
        return "action"
    if name.startswith("und_expert."):
        return "understanding"
    return None


def normalize_lora_config(value: dict[str, Any] | None) -> dict[str, Any]:
    source = value or {}
    targets = source.get("targets", {})
    return {
        "rank": int(source.get("rank", 16)),
        "alpha": float(source.get("alpha", 32.0)),
        "dropout": float(source.get("dropout", 0.05)),
        "targets": {
            "wan": bool(targets.get("wan", True)),
            "action": bool(targets.get("action", True)),
            "understanding": bool(targets.get("understanding", True)),
            "expert_qkv": bool(targets.get("expert_qkv", True)),
        },
        "train_io_layers": bool(source.get("train_io_layers", True)),
        "train_norms": bool(source.get("train_norms", True)),
        "train_modulation": bool(source.get("train_modulation", True)),
    }


def _remap_parametrized_qkv_key(key: str) -> str:
    for leaf in ("wan_action_qkv", "wan_und_qkv"):
        if key.endswith("." + leaf):
            return key[: -(len(leaf))] + f"parametrizations.{leaf}.original"
    return key


def remap_stage2_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map official Stage-2 QKV keys to their parametrized Stage3-LoRA names."""
    return {_remap_parametrized_qkv_key(key): value for key, value in state.items()}


def _install_stage2_loader(model: nn.Module) -> None:
    """Keep the official partial Stage-2 initialization after QKV parametrization."""

    def load_pretrain_weights(self, path: str) -> None:
        if self.config.training_mode != "finetune":
            raise ValueError("load_pretrain_weights is only valid in finetune mode")
        root = Path(path)
        candidates = [
            root / "pytorch_model" / "mp_rank_00_model_states.pt",
            root / "mp_rank_00_model_states.pt",
        ] if root.is_dir() else [root]
        checkpoint_file = next((item for item in candidates if item.is_file()), None)
        if checkpoint_file is None:
            raise FileNotFoundError("Stage-2 checkpoint not found; tried:\n" + "\n".join(map(str, candidates)))
        payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        state = payload.get("module", payload)
        filtered = {
            key: value for key, value in state.items()
            if "action_expert.input_encoder" not in key and "action_expert.decoder" not in key
        }
        missing, unexpected = self.load_state_dict(remap_stage2_state_dict(filtered), strict=False)
        print(
            f"Stage-2 initialized for LoRA: missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    model.load_pretrain_weights = MethodType(load_pretrain_weights, model)


def apply_lora(model: nn.Module, raw_config: dict[str, Any] | None) -> dict[str, Any]:
    """Freeze Motus and inject LoRA into all three Stage-3 experts."""
    config = normalize_lora_config(raw_config)
    rank, alpha, dropout = config["rank"], config["alpha"], config["dropout"]
    targets = config["targets"]

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    wrapped: list[str] = []
    for name, module in list(model.named_modules()):
        branch = _branch(name)
        if branch is None or not targets[branch] or not isinstance(module, nn.Linear):
            continue
        # G2's 16-D state/action adapters are new in Chapter 16 and need full updates.
        if name.startswith("action_expert.input_encoder.") or name.startswith("action_expert.decoder."):
            continue
        _set_submodule(model, name, LoRALinear(module, rank, alpha, dropout))
        wrapped.append(name)

    qkv_count = 0
    if targets["expert_qkv"]:
        if targets["action"]:
            for block in model.action_expert.blocks:
                original = block.wan_action_qkv
                original.requires_grad_(False)
                parametrize.register_parametrization(
                    block, "wan_action_qkv", LoRAWeightDelta(original, rank, alpha)
                )
                qkv_count += 1
        if targets["understanding"]:
            for block in model.und_expert.blocks:
                original = block.wan_und_qkv
                original.requires_grad_(False)
                parametrize.register_parametrization(
                    block, "wan_und_qkv", LoRAWeightDelta(original, rank, alpha)
                )
                qkv_count += 1

    if config["train_io_layers"]:
        for parameter in model.action_expert.input_encoder.parameters():
            parameter.requires_grad_(True)
        for parameter in model.action_expert.decoder.parameters():
            parameter.requires_grad_(True)

    # Small Stage-3-specific scale/normalization parameters are cheap and useful.
    for name, parameter in model.named_parameters():
        branch = _branch(name)
        if branch is None or not targets.get(branch, False):
            continue
        if config["train_norms"] and ".norm" in name:
            parameter.requires_grad_(True)
        if config["train_modulation"] and (
            name.endswith(".modulation") or name.endswith(".registers")
        ):
            parameter.requires_grad_(True)

    model._chapter16_lora_config = config
    model._chapter16_lora_modules = wrapped
    _install_stage2_loader(model)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(
        "Chapter-16 Stage3-LoRA: "
        f"linear_modules={len(wrapped)} qkv_tensors={qkv_count} "
        f"trainable={trainable / 1e6:.1f}M total={total / 1e9:.2f}B "
        f"({100.0 * trainable / total:.3f}%)",
        flush=True,
    )
    return config


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only LoRA and explicitly trainable Stage-3 parameters."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_lora_adapter(model: nn.Module, directory: Path, step: int) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(getattr(model, "_chapter16_lora_config"))
    config.update({"format": "chapter16-motus-stage3-lora-v1", "step": int(step)})
    path = directory / "adapter_model.pt"
    torch.save({"adapter": adapter_state_dict(model), "lora_config": config}, path)
    (directory / "lora_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def read_lora_config(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if path.is_file() and path.name == "adapter_model.pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return payload.get("lora_config")
    roots = [path, path.parent] if path.is_dir() else [path.parent, path.parent.parent]
    for root in roots:
        candidate = root / "lora_config.json"
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def load_lora_adapter(model: nn.Module, path: Path) -> tuple[list[str], list[str]]:
    path = Path(path)
    file = path / "adapter_model.pt" if path.is_dir() else path
    payload = torch.load(file, map_location="cpu", weights_only=False)
    state = payload.get("adapter", payload)
    expected = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    absent = sorted(expected.difference(state))
    if absent:
        raise RuntimeError(
            f"LoRA adapter is incomplete: {len(absent)} trainable tensors are missing; "
            f"examples={absent[:8]}"
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    return list(missing), list(unexpected)
