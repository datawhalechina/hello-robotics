"""The only mapping between native G2 samples and pinned OpenPI pi0.5."""

from __future__ import annotations
import dataclasses
import os
import random
import numpy as np
from acp import clean_task, tagged_task
from config import (
    ACTION_DIM,
    ACP_DROPOUT,
    CHECKPOINT_ROOT,
    PI05_ASSET_ROOT,
    PI05_HORIZON,
    PI05_IMAGE_SIZE,
    PI05_PAD_DIM,
    PI05_TOKEN_LENGTH,
    STATE_DIM,
)


def rgb8(x):
    x = np.asarray(x)
    x = np.moveaxis(x, 0, -1) if x.ndim == 3 and x.shape[0] == 3 else x
    if np.issubdtype(x.dtype, np.floating):
        x = np.nan_to_num(x * 255 if x.size and x.max() <= 1 else x)
    return np.ascontiguousarray(np.clip(x, 0, 255).astype(np.uint8))


if os.getenv("CHAPTER15_OPENPI_READY") == "1":
    from openpi import transforms
    from openpi.models import tokenizer as tokenizer_lib
    from openpi.training import config as training_config

    @dataclasses.dataclass(frozen=True)
    class Inputs(transforms.DataTransformFn):
        def __call__(self, data):
            result = {
                "state": np.asarray(data["state"], np.float32).reshape(STATE_DIM),
                "image": {
                    "base_0_rgb": rgb8(data["images"]["head"]),
                    "left_wrist_0_rgb": rgb8(data["images"]["left_wrist"]),
                    "right_wrist_0_rgb": rgb8(data["images"]["right_wrist"]),
                },
                "image_mask": {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                    "right_wrist_0_rgb": np.True_,
                },
            }
            if "actions" in data:
                result["actions"] = np.asarray(data["actions"], np.float32)
            if "prompt" in data:
                result["prompt"] = str(data["prompt"])
            if "indicator" in data:
                result["indicator"] = np.asarray(data["indicator"]).reshape(-1)
            return result

    @dataclasses.dataclass(frozen=True)
    class Outputs(transforms.DataTransformFn):
        def __call__(self, data):
            return {
                "actions": np.asarray(data["actions"], np.float32)[..., :ACTION_DIM]
            }

    @dataclasses.dataclass(frozen=True)
    class ACPTransform(transforms.DataTransformFn):
        enabled: bool = True
        dropout: float = 0.3
        seed: int = 1000
        _rng: object = dataclasses.field(default=None, init=False, compare=False)

        def __call__(self, data):
            if not self.enabled:
                return data
            raw = np.asarray(data["indicator"]).reshape(-1)
            if raw.size != 1 or int(raw[0]) not in (0, 1):
                raise ValueError("indicator must be scalar 0/1")
            rng = self._rng
            if rng is None:
                rng = random.Random(self.seed)
                object.__setattr__(self, "_rng", rng)
            task = clean_task(data.get("prompt", ""))
            return {
                **data,
                "prompt": task
                if rng.random() < self.dropout
                else tagged_task(task, bool(raw[0])),
            }

    @dataclasses.dataclass(frozen=True)
    class G2Data(training_config.DataConfigFactory):
        acp: bool = True
        dropout: float = 0.3
        seed: int = 1000

        def create(self, assets_dirs, model):
            structure = {
                "images": {
                    "head": "observation.images.head",
                    "left_wrist": "observation.images.left_wrist",
                    "right_wrist": "observation.images.right_wrist",
                },
                "state": "observation.state",
                "actions": "action",
                "prompt": "prompt",
            }
            if self.acp:
                structure["indicator"] = "complementary_info.acp_indicator"
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model),
                repack_transforms=transforms.Group(
                    inputs=[transforms.RepackTransform(structure)]
                ),
                data_transforms=transforms.Group(
                    inputs=[Inputs(), ACPTransform(self.acp, self.dropout, self.seed)],
                    outputs=[Outputs()],
                ),
                model_transforms=transforms.Group(
                    inputs=[
                        transforms.InjectDefaultPrompt(None),
                        transforms.ResizeImages(PI05_IMAGE_SIZE, PI05_IMAGE_SIZE),
                        transforms.PadStatesAndActions(model.action_dim),
                        transforms.TokenizePrompt(
                            tokenizer_lib.PaligemmaTokenizer(model.max_token_len),
                            discrete_state_input=True,
                        ),
                    ]
                ),
                action_sequence_keys=("action",),
            )


def model_config(lora=False):
    from openpi.models import pi0_config

    return pi0_config.Pi0Config(
        pi05=True,
        action_dim=PI05_PAD_DIM,
        action_horizon=PI05_HORIZON,
        max_token_len=PI05_TOKEN_LENGTH,
        discrete_state_input=True,
        paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
        image_augmentation=False,
    )


def train_config(
    name,
    repo_id,
    initial,
    stage,
    batch_size=32,
    steps=30000,
    lora=False,
    seed=1000,
    fsdp_devices=1,
):
    from openpi.training import config, optimizer, weight_loaders

    model = model_config(lora)
    acp = stage == "acp"
    return config.TrainConfig(
        name=name,
        model=model,
        data=G2Data(
            repo_id=repo_id,
            acp=acp,
            dropout=ACP_DROPOUT if acp else 0,
            seed=seed,
            base_config=config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(f"{initial}/params"),
        freeze_filter=model.get_freeze_filter(),
        lr_schedule=optimizer.CosineDecaySchedule(1000, 2.5e-5, 30000, 2.5e-6),
        optimizer=optimizer.AdamW(
            b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.01, clip_gradient_norm=1.0
        ),
        ema_decay=None,
        assets_base_dir=str(PI05_ASSET_ROOT),
        checkpoint_base_dir=str(CHECKPOINT_ROOT),
        batch_size=batch_size,
        num_workers=2,
        drop_last=False,
        num_train_steps=steps,
        log_interval=200,
        save_interval=1000,
        keep_period=5000,
        seed=seed,
        wandb_enabled=False,
        fsdp_devices=fsdp_devices,
        policy_metadata={
            "base_model": "pi0.5",
            "contract": "g2_pi05_v1",
            "action_transform": "all_absolute",
            "state_dim": 16,
            "action_dim": 16,
            "action_horizon": 50,
            "flow_steps": 10,
        },
    )


def runtime_config(lora, meta):
    from openpi.training import config, weight_loaders

    return config.TrainConfig(
        name="chapter15_runtime",
        model=model_config(lora),
        data=G2Data(
            repo_id=".", assets=config.AssetsConfig(asset_id="."), acp=False, dropout=0
        ),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        policy_metadata=meta,
        wandb_enabled=False,
    )
