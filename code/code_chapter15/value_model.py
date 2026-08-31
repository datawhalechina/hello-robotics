"""A small reimplementation of Pistar06 distributional Value stack."""

from __future__ import annotations
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import torch
from torch import nn
from config import LOCAL_GEMMA_MODEL
import torch.nn.functional as F


@dataclass
class ValueModelConfig:
    vision_model: str = "google/siglip-so400m-patch14-384"
    language_model: str = str(LOCAL_GEMMA_MODEL)
    fusion_dim: int = 512
    dropout: float = 0.1
    bins: int = 201
    value_min: float = -1.0
    value_max: float = 0.0
    dtype: str = "float32"
    freeze_vision: bool = False
    freeze_language: bool = False


def _hidden_size(model):
    cfg = model.config
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return int(cfg.text_config.hidden_size)
    raise ValueError("cannot infer hidden size")


def _vision_size(model):
    cfg = model.config
    for owner in (cfg, getattr(cfg, "vision_config", None)):
        if owner is not None:
            for key in ("projection_dim", "hidden_size"):
                if hasattr(owner, key):
                    return int(getattr(owner, key))
    raise ValueError("cannot infer vision size")


def _load_language(repo, dtype):
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    local_only = Path(repo).expanduser().is_dir()
    cfg = AutoConfig.from_pretrained(repo, local_files_only=local_only)
    arch = getattr(cfg, "architectures", None) or []
    if any(str(x).endswith("ForCausalLM") for x in arch):
        loaded = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=dtype, local_files_only=local_only
        )
        if hasattr(loaded, "model"):
            return loaded.model
    return AutoModel.from_pretrained(
        repo, torch_dtype=dtype, local_files_only=local_only
    )


class Pistar06Value(nn.Module):
    def __init__(self, cfg: ValueModelConfig):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        self.cfg = cfg
        self.model_dtype = (
            torch.bfloat16
            if cfg.dtype == "bfloat16" and torch.cuda.is_available()
            else torch.float32
        )
        self.vision = AutoModel.from_pretrained(
            cfg.vision_model, torch_dtype=self.model_dtype
        )
        self.language = _load_language(cfg.language_model, self.model_dtype)
        processor = AutoImageProcessor.from_pretrained(cfg.vision_model, use_fast=True)
        size = processor.size
        edge = (
            int(size.get("height", size.get("shortest_edge", 384)))
            if isinstance(size, dict)
            else int(size)
        )
        width = int(size.get("width", edge)) if isinstance(size, dict) else edge
        self.image_size = (edge, width)
        self.register_buffer(
            "image_mean",
            torch.tensor(processor.image_mean or [0.5] * 3).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(processor.image_std or [0.5] * 3).view(1, 3, 1, 1),
            persistent=False,
        )
        self.image_projector = nn.Sequential(
            nn.Linear(_vision_size(self.vision), cfg.fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.language_projector = nn.Sequential(
            nn.Linear(_hidden_size(self.language), cfg.fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.norm = nn.LayerNorm(cfg.fusion_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(cfg.fusion_dim * 2, cfg.fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim, cfg.bins),
        )
        if cfg.freeze_vision:
            self.vision.eval()
            for p in self.vision.parameters():
                p.requires_grad = False
        if cfg.freeze_language:
            self.language.eval()
            for p in self.language.parameters():
                p.requires_grad = False

    def _image_features(self, x):
        if hasattr(self.vision, "get_image_features"):
            return self.vision.get_image_features(pixel_values=x)
        out = self.vision(pixel_values=x, return_dict=True)
        return (
            out.pooler_output
            if getattr(out, "pooler_output", None) is not None
            else out.last_hidden_state.mean(1)
        )

    def forward(self, input_ids, attention_mask, images):
        # images: B,N,C,H,W uint8/float
        b, n = images.shape[:2]
        x = images.reshape(b * n, *images.shape[2:]).float()
        x = x / 255 if x.max() > 1 else x
        if x.shape[-2:] != self.image_size:
            x = F.interpolate(x, self.image_size, mode="bilinear", align_corners=False)
        x = (x - self.image_mean.to(x)) / (self.image_std.to(x))
        x = x.to(self.model_dtype)
        with torch.no_grad() if self.cfg.freeze_vision else nullcontext():
            vf = self._image_features(x)
        with torch.no_grad() if self.cfg.freeze_language else nullcontext():
            out = self.language(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            hidden = out.last_hidden_state
            mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
            lf = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        image = self.image_projector(vf.float()).view(b, n, -1).mean(1)
        language = self.language_projector(lf.float())
        return self.head(self.norm(torch.cat([image, language], -1)))

    def save(self, path: Path, extra: dict | None = None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        (path / "value_config.json").write_text(
            json.dumps({"model": asdict(self.cfg), "extra": extra or {}}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, device="cpu"):
        payload = json.loads((Path(path) / "value_config.json").read_text())
        model = cls(ValueModelConfig(**payload["model"]))
        model.load_state_dict(torch.load(Path(path) / "model.pt", map_location=device))
        return model.to(device), payload.get("extra", {})


def two_hot_cross_entropy(logits, targets):
    return -(targets.to(logits.dtype) * F.log_softmax(logits, -1)).sum(-1).mean()


def expected_from_logits(logits, low=-1.0, high=0.0):
    return (
        logits.softmax(-1)
        * torch.linspace(low, high, logits.shape[-1], device=logits.device)
    ).sum(-1)
