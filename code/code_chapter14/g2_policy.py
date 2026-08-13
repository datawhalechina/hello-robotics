"""G2数据布局与原生openpi π0.5之间的轻量适配。"""

from __future__ import annotations

import dataclasses

import numpy as np

try:
    from openpi import transforms
    from openpi.training import config as training_config
except ImportError:  # 允许不安装openpi时导入纯Isaac模块和运行单元测试
    transforms = None
    training_config = None


MODEL_SPECS = {
    "instruction": {"horizon": 50, "token_len": 200, "input_dim": 16, "output_dim": 16},
    "manipulation": {"horizon": 30, "token_len": 220, "input_dim": 21, "output_dim": 21},
    "finetuned": {"horizon": 50, "token_len": 200, "input_dim": 16, "output_dim": 16},
}


def _image(images: dict, key: str) -> np.ndarray:
    value = np.asarray(images[key])
    if np.issubdtype(value.dtype, np.floating):
        value = np.clip(value * 255.0, 0, 255).astype(np.uint8)
    if value.ndim == 3 and value.shape[0] == 3:
        value = np.moveaxis(value, 0, -1)
    return np.ascontiguousarray(value, dtype=np.uint8)


if transforms is not None:

    @dataclasses.dataclass(frozen=True)
    class G2Inputs(transforms.DataTransformFn):
        """三路图像和完整G2状态；在归一化前补齐到模型的32维。"""

        action_dim: int
        input_dim: int = 16
        mask_manipulation_state: bool = False

        def __call__(self, data: dict) -> dict:
            state = np.asarray(data["state"], dtype=np.float32).reshape(-1).copy()
            if state.size != self.input_dim:
                raise ValueError(f"G2 state应为{self.input_dim}维，实际为{state.size}")
            if self.mask_manipulation_state:
                state[14:16] = 0.0
                state[16:20] = 0.0
            state = transforms.pad_to_dim(state, self.action_dim)
            result = {
                "state": state,
                "image": {
                    "base_0_rgb": _image(data["images"], "top_head"),
                    "left_wrist_0_rgb": _image(data["images"], "hand_left"),
                    "right_wrist_0_rgb": _image(data["images"], "hand_right"),
                },
                "image_mask": {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                    "right_wrist_0_rgb": np.True_,
                },
            }
            if "actions" in data:
                actions = np.asarray(data["actions"], dtype=np.float32).copy()
                if actions.shape[-1] != self.input_dim:
                    raise ValueError(f"G2 action应为{self.input_dim}维")
                if self.mask_manipulation_state:
                    actions[..., 16:20] = 0.0
                result["actions"] = transforms.pad_to_dim(actions, self.action_dim)
            if "prompt" in data:
                result["prompt"] = data["prompt"]
            return result


    @dataclasses.dataclass(frozen=True)
    class G2Outputs(transforms.DataTransformFn):
        output_dim: int

        def __call__(self, data: dict) -> dict:
            return {"actions": np.asarray(data["actions"])[..., : self.output_dim]}


    @dataclasses.dataclass(frozen=True)
    class G2DataConfig(training_config.DataConfigFactory):
        """与公开G2 baseline一致：双臂delta，夹爪absolute。"""

        input_dim: int = 16
        output_dim: int = 16
        manipulation_layout: bool = False

        def create(self, assets_dirs, model_config):
            repack = transforms.Group(
                inputs=[
                    transforms.RepackTransform(
                        {
                            "images": {
                                "top_head": "observation.images.top_head",
                                "hand_left": "observation.images.hand_left",
                                "hand_right": "observation.images.hand_right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
            data = transforms.Group(
                inputs=[G2Inputs(model_config.action_dim, self.input_dim, self.manipulation_layout)],
                outputs=[G2Outputs(self.output_dim)],
            )
            # manipulation原模型还包含腰部：14臂delta、2夹爪absolute、其余delta。
            mask = (
                transforms.make_bool_mask(14, -2, 16)
                if self.manipulation_layout
                else transforms.make_bool_mask(14, -2)
            )
            data = data.push(
                inputs=[transforms.DeltaActions(mask)],
                outputs=[transforms.AbsoluteActions(mask)],
            )
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack,
                data_transforms=data,
                model_transforms=training_config.ModelTransformFactory()(model_config),
                action_sequence_keys=("action",),
            )


def make_runtime_config(model_kind: str, *, lora: bool = False):
    from openpi.models import pi0_config
    from openpi.training import config, weight_loaders

    spec = MODEL_SPECS[model_kind]
    return config.TrainConfig(
        name=f"chapter14_{model_kind}_runtime",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=spec["horizon"],
            max_token_len=spec["token_len"],
            discrete_state_input=True,
            paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
            action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
        ),
        data=G2DataConfig(
            repo_id=".",
            assets=config.AssetsConfig(asset_id="."),
            input_dim=spec["input_dim"],
            output_dim=spec["output_dim"],
            manipulation_layout=model_kind == "manipulation",
        ),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        policy_metadata={
            "model_kind": model_kind,
            "robot": "G2_omnipicker",
            "state_dim": spec["input_dim"],
            "action_dim": spec["output_dim"],
            "action_horizon": spec["horizon"],
            "action_type": "JOINT_ABS",
            "state_order": "left_arm7,right_arm7,left_gripper,right_gripper[,waist5]",
        },
        wandb_enabled=False,
    )


def make_finetune_config(
    *, repo_id: str, initial_checkpoint: str, assets_base_dir: str,
    checkpoint_base_dir: str, batch_size: int, train_steps: int, lora: bool,
):
    """为本章16维数据创建训练配置；默认从G2 instruction权重继续训练。"""
    from openpi.models import pi0_config
    from openpi.training import config, optimizer, weight_loaders

    model = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=50,
        discrete_state_input=True,
        paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
    )
    return config.TrainConfig(
        name="pi05_g2_color_blocks",
        model=model,
        data=G2DataConfig(
            repo_id=repo_id,
            input_dim=16,
            output_dim=16,
            base_config=config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(f"{initial_checkpoint}/params"),
        freeze_filter=model.get_freeze_filter(),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=min(500, max(10, train_steps // 10)),
            peak_lr=5e-5,
            decay_steps=train_steps,
            decay_lr=1e-5,
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None if lora else 0.99,
        assets_base_dir=assets_base_dir,
        checkpoint_base_dir=checkpoint_base_dir,
        batch_size=batch_size,
        num_workers=2,
        num_train_steps=train_steps,
        log_interval=50,
        save_interval=min(1000, train_steps),
        keep_period=5000,
        wandb_enabled=False,
    )
