"""Reusable 16D Motus Stage-3 inference runtime for offline and simulator examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor

from data_utils import cache_name, image_tensor, load_state_dict_file, stitch_views
from lora import apply_lora, load_lora_adapter, read_lora_config
from motus_compat import configure_attention, install_motus_imports
from settings import (
    ACTION_DIM,
    INFERENCE_STEPS,
    NUM_VIDEO_FRAMES,
    QWEN_ROOT,
    STATE_DIM,
    STAGE2_ROOT,
    T5_CACHE_ROOT,
    VIDEO_ACTION_FREQ_RATIO,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    WAN_ROOT,
    stage3_prompt,
)


@dataclass
class Prediction:
    frames: np.ndarray  # [8,H,W,3], uint8
    actions: np.ndarray  # [16,16], float32


def build_model(checkpoint: Path, device: str = "cuda:0"):
    if not torch.cuda.is_available():
        raise RuntimeError("Motus/WAN inference requires an NVIDIA CUDA GPU")
    torch.cuda.set_device(torch.device(device))
    install_motus_imports()
    configure_attention()
    from models.motus import Motus, MotusConfig

    config = MotusConfig(
        wan_checkpoint_path=str(WAN_ROOT),
        vae_path=str(WAN_ROOT / "Wan2.2_VAE.pth"),
        wan_config_path=str(WAN_ROOT),
        vlm_checkpoint_path=str(QWEN_ROOT),
        video_precision="bfloat16",
        action_state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        action_expert_dim=1024,
        action_expert_ffn_dim_multiplier=4,
        action_expert_norm_eps=1.0e-5,
        und_expert_hidden_size=512,
        und_expert_ffn_dim_multiplier=4,
        und_expert_norm_eps=1.0e-5,
        vlm_adapter_input_dim=2048,
        vlm_adapter_projector_type="mlp3x_silu",
        global_downsample_rate=3,
        video_action_freq_ratio=VIDEO_ACTION_FREQ_RATIO,
        num_video_frames=NUM_VIDEO_FRAMES,
        video_height=VIDEO_HEIGHT,
        video_width=VIDEO_WIDTH,
        batch_size=1,
        video_loss_weight=1.0,
        action_loss_weight=1.0,
        training_mode="finetune",
        load_pretrained_backbones=False,
    )
    # Motus already places trainable branches in configured BF16 while keeping a few
    # numerically sensitive time layers in FP32.  Do not force-cast the whole model.
    model = Motus(config).to(device=device)
    checkpoint = Path(checkpoint)
    lora_config = read_lora_config(checkpoint)
    if lora_config is not None:
        # Rebuild the frozen Stage-2 base, inject the same adapters, then load only
        # the small Stage-3 LoRA/16-D G2 delta checkpoint.
        apply_lora(model, lora_config)
        model.load_pretrain_weights(str(STAGE2_ROOT))
        adapter_file = checkpoint / "adapter_model.pt" if checkpoint.is_dir() else checkpoint
        _, unexpected = load_lora_adapter(model, adapter_file)
        if unexpected:
            raise RuntimeError(f"unexpected LoRA adapter keys: {unexpected[:8]}")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"LoRA adapter loaded: {adapter_file}; "
            f"trainable_delta={trainable / 1e6:.1f}M"
        )
    else:
        state = load_state_dict_file(checkpoint)
        # Accelerate hooks normally save the unwrapped state_dict. Handle an occasional DDP prefix.
        if state and all(key.startswith("module.") for key in state):
            state = {key[7:]: value for key, value in state.items()}
        try:
            missing, unexpected = model.load_state_dict(state, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint shape mismatch. Use a Chapter-16 16D Stage-3 checkpoint, not the "
                "14D Motus_robotwin2 checkpoint."
            ) from exc
        critical = (
            "action_expert.input_encoder.state_encoder.0.weight",
            "action_expert.input_encoder.action_encoder.0.weight",
            "action_expert.decoder.action_head.0.weight",
        )
        absent = [name for name in critical if name in missing]
        if absent:
            raise RuntimeError(f"checkpoint is not a complete 16D Stage-3 model; missing {absent}")
        print(f"checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


def to_frame_array(predicted: torch.Tensor) -> np.ndarray:
    value = predicted.detach().float().cpu()
    if value.ndim != 5:
        raise ValueError(f"expected 5D predicted video, got {tuple(value.shape)}")
    value = value[0]
    # Official decode commonly returns [C,T,H,W]; some branches return [T,C,H,W].
    if value.shape[0] == 3:
        value = value.permute(1, 2, 3, 0)
    elif value.shape[1] == 3:
        value = value.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"cannot identify RGB axis in {tuple(value.shape)}")
    return (value.clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)


class MotusPolicy:
    def __init__(
        self, checkpoint: Path, device: str = "cuda:0",
        inference_steps: int = INFERENCE_STEPS, seed: int | None = 16016,
        reseed_each_request: bool = False,
    ):
        self.device = torch.device(device)
        self.inference_steps = int(inference_steps)
        self.seed = seed
        self.reseed_each_request = bool(reseed_each_request)
        # RoboTwin seeds an evaluation run once; it does not reset the diffusion
        # latent before every get_action() call.  Preserve that official RNG
        # behavior by default even though inference runs in a separate process.
        if self.seed is not None:
            torch.manual_seed(int(self.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.seed))
        self.model = build_model(Path(checkpoint), device)
        self.processor = AutoProcessor.from_pretrained(str(QWEN_ROOT), trust_remote_code=True)
        install_motus_imports()
        from utils.vlm_utils import preprocess_vlm_messages
        self.preprocess_vlm_messages = preprocess_vlm_messages
        self.embedding_cache: dict[str, list[torch.Tensor]] = {}

    def language_embedding(self, instruction: str) -> list[torch.Tensor]:
        prompt = stage3_prompt(instruction)
        if prompt not in self.embedding_cache:
            path = T5_CACHE_ROOT / cache_name(prompt)
            if not path.is_file():
                raise FileNotFoundError(
                    f"T5 cache missing: {path}. Run python3 prepare_data.py --encode-t5 first."
                )
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(loaded, torch.Tensor):
                loaded = [loaded]
            if not isinstance(loaded, list) or not loaded:
                raise TypeError(f"invalid T5 cache: {path}")
            self.embedding_cache[prompt] = [x.squeeze(0) if x.ndim == 3 else x for x in loaded]
        # One instruction per request.
        return [self.embedding_cache[prompt][0].to(self.device, dtype=torch.bfloat16)]

    @torch.inference_mode()
    def predict_composite(self, composite: np.ndarray, state: np.ndarray, instruction: str) -> Prediction:
        # Optional deterministic-ablation mode.  Official RoboTwin evaluation
        # advances the RNG between calls, so this is disabled by default.
        if self.reseed_each_request and self.seed is not None:
            torch.manual_seed(int(self.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.seed))
        prompt = stage3_prompt(instruction)
        image = Image.fromarray(np.asarray(composite, np.uint8), mode="RGB")
        first = image_tensor(image).unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        state_tensor = torch.as_tensor(state, dtype=torch.bfloat16, device=self.device).reshape(1, STATE_DIM)
        vlm = self.preprocess_vlm_messages(prompt, image, self.processor)
        frames, actions = self.model.inference_step(
            first_frame=first,
            state=state_tensor,
            num_inference_steps=self.inference_steps,
            language_embeddings=self.language_embedding(prompt),
            vlm_inputs=[vlm],
        )
        return Prediction(to_frame_array(frames), actions[0].detach().float().cpu().numpy())

    def predict(self, images: tuple[np.ndarray, np.ndarray, np.ndarray], state: np.ndarray, instruction: str) -> Prediction:
        return self.predict_composite(stitch_views(*images), state, instruction)
