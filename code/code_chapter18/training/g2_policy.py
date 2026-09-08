"""G2 数据与官方 ACoT-VLA 的适配和训练配置。"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

try:
    from .config import (
        DATASET_FPS,
        DATASET_VERSION,
        GRIPPER_OPEN_RAD,
        LOW_LEVEL_HZ,
        TRAIN_CONFIG_NAME,
    )
except ImportError:
    from config import (
        DATASET_FPS,
        DATASET_VERSION,
        GRIPPER_OPEN_RAD,
        LOW_LEVEL_HZ,
        TRAIN_CONFIG_NAME,
    )

try:
    from openpi import transforms
    from openpi.training import config as training_config
except ImportError:  # Isaac Sim 的纯场景代码不强制依赖训练环境
    transforms = None
    training_config = None


G2_DIM = 16
MODEL_ACTION_DIM = 32
# 30 Hz下coarse以stride 2覆盖约3.27秒，fine连续覆盖约0.97秒。
COARSE_ACTION_HORIZON = 50
ACTION_HORIZON = 30
JOINT_ACTION_SHIFTS = (2, 1)
MAX_TOKEN_LEN = 200


@dataclasses.dataclass(frozen=True)
class LowMemoryAdafactor:
    """Factored second moment optimizer for a 24 GiB single GPU."""

    decay_rate: float = 0.8
    eps: float = 1e-30
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    def create(self, lr, weight_decay_mask=None):
        import optax

        adafactor = optax.adafactor(
            learning_rate=lr,
            min_dim_size_to_factor=128,
            decay_rate=self.decay_rate,
            multiply_by_parameter_scale=False,
            clipping_threshold=1.0,
            momentum=None,
            weight_decay_rate=self.weight_decay,
            weight_decay_mask=weight_decay_mask,
            eps=self.eps,
        )
        return optax.chain(
            optax.clip_by_global_norm(self.clip_gradient_norm), adafactor
        )


def _image(images: dict, key: str) -> np.ndarray:
    value = np.asarray(images[key])
    if np.issubdtype(value.dtype, np.floating):
        value = np.clip(value * 255.0, 0, 255).astype(np.uint8)
    if value.ndim == 3 and value.shape[0] == 3:
        value = np.moveaxis(value, 0, -1)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"相机 {key} 应为 HWC/CHW RGB，实际为 {value.shape}")
    return np.ascontiguousarray(value, dtype=np.uint8)


if transforms is not None:

    @dataclasses.dataclass(frozen=True)
    class G2ACOTInputs(transforms.DataTransformFn):
        """构造三相机观测，并从同一绝对动作序列生成 coarse/fine 监督。"""

        action_dim: int = MODEL_ACTION_DIM
        input_dim: int = G2_DIM
        coarse_action_horizon: int = COARSE_ACTION_HORIZON
        action_horizon: int = ACTION_HORIZON
        joint_action_shifts: Sequence[int] = JOINT_ACTION_SHIFTS

        def __call__(self, data: dict) -> dict:
            state = np.asarray(data["state"], dtype=np.float32).reshape(-1)
            if state.size != self.input_dim:
                raise ValueError(
                    f"G2 state 应为 {self.input_dim} 维，实际为 {state.size}"
                )

            result = {
                "state": transforms.pad_to_dim(state, self.action_dim),
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
                raw_actions = np.asarray(data["actions"], dtype=np.float32).copy()
                if raw_actions.ndim != 2 or raw_actions.shape[-1] != self.input_dim:
                    raise ValueError(
                        f"G2 action 序列应为 (T, {self.input_dim})，实际为 {raw_actions.shape}"
                    )
                # 本任务不使用左夹爪。episode 0 冷启动时 PhysX 的约 1e-3 rad
                # 回弹不应成为动作监督，统一恢复为专家下发的标准张开命令。
                raw_actions[:, 14] = -GRIPPER_OPEN_RAD
                horizons = (self.coarse_action_horizon, self.action_horizon)
                for key, horizon, shift in zip(
                    ("coarse_actions", "actions"),
                    horizons,
                    self.joint_action_shifts,
                    strict=True,
                ):
                    required = (horizon - 1) * shift + 1
                    if raw_actions.shape[0] < required:
                        raise ValueError(
                            f"{key} 需要至少 {required} 帧，实际为 {raw_actions.shape[0]}"
                        )
                    sampled = raw_actions[:required:shift]
                    if sampled.shape[0] != horizon:
                        raise AssertionError(
                            f"{key} 采样后长度错误：{sampled.shape[0]}"
                        )
                    result[key] = transforms.pad_to_dim(sampled, self.action_dim)

            if "prompt" in data:
                result["prompt"] = data["prompt"]
            return result

    @dataclasses.dataclass(frozen=True)
    class G2ACOTOutputs(transforms.DataTransformFn):
        """仅将前 16 维物理动作返回给 G2，保留 coarse action 供诊断。"""

        output_dim: int = G2_DIM

        def __call__(self, data: dict) -> dict:
            return {
                key: np.asarray(data[key])[..., : self.output_dim]
                for key in ("actions", "coarse_actions")
                if key in data
            }

    @dataclasses.dataclass(frozen=True)
    class G2ACOTDataConfig(training_config.DataConfigFactory):
        """复用第十四章 16 维 G2 数据合同：手臂 delta、夹爪 absolute。"""

        input_dim: int = G2_DIM
        output_dim: int = G2_DIM
        joint_action_shifts: Sequence[int] = JOINT_ACTION_SHIFTS

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
                inputs=[
                    G2ACOTInputs(
                        action_dim=model_config.action_dim,
                        input_dim=self.input_dim,
                        coarse_action_horizon=model_config.coarse_action_horizon,
                        action_horizon=model_config.action_horizon,
                        joint_action_shifts=self.joint_action_shifts,
                    )
                ],
                outputs=[G2ACOTOutputs(self.output_dim)],
            )
            delta_mask = transforms.make_bool_mask(14, -2)
            data = data.push(
                inputs=[transforms.ACOTDeltaActions(delta_mask, (True, True))],
                outputs=[transforms.ACOTAbsoluteActions(delta_mask, (True, True))],
            )
            result = dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack,
                data_transforms=data,
                model_transforms=training_config.ModelTransformFactory()(model_config),
                action_sequence_keys=("action",),
            )
            # 官方 ACoT data loader 用这个扩展属性决定原始动作窗口长度。
            object.__setattr__(
                result, "joint_action_shifts", tuple(self.joint_action_shifts)
            )
            return result


def make_model_config(
    *,
    lora: bool = True,
    coarse_action_horizon: int = COARSE_ACTION_HORIZON,
    action_horizon: int = ACTION_HORIZON,
    max_token_len: int = MAX_TOKEN_LEN,
):
    """Pi0.5 语言主干使用 LoRA，ACoT 双动作专家及新模块全量训练。"""
    from openpi.models import acot_vla

    return acot_vla.ACOTConfig(
        pi05=True,
        discrete_state_input=True,
        action_dim=MODEL_ACTION_DIM,
        coarse_action_horizon=coarse_action_horizon,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
        coarse_action_expert_variant="gemma_300m",
        action_expert_variant="gemma_300m",
        adopt_explicit_action_reasoner=True,
        adopt_implicit_action_reasoner=True,
        downsample_based_implicit_extractor=True,
    )


def make_runtime_config(
    *,
    lora: bool = True,
    coarse_action_horizon: int = COARSE_ACTION_HORIZON,
    action_horizon: int = ACTION_HORIZON,
    max_token_len: int = MAX_TOKEN_LEN,
):
    from openpi.training import config, weight_loaders

    model = make_model_config(
        lora=lora,
        coarse_action_horizon=coarse_action_horizon,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
    )
    return config.TrainConfig(
        name=TRAIN_CONFIG_NAME,
        model=model,
        data=G2ACOTDataConfig(
            repo_id=".",
            assets=config.AssetsConfig(asset_id="."),
            base_config=config.DataConfig(prompt_from_task=False),
        ),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        policy_metadata={
            "model_kind": "acot_pi05",
            "dataset_version": DATASET_VERSION,
            "action_hz": DATASET_FPS,
            "low_level_hz": LOW_LEVEL_HZ,
            "robot": "G2_omnipicker",
            "state_dim": G2_DIM,
            "action_dim": G2_DIM,
            "action_horizon": action_horizon,
            "coarse_action_horizon": coarse_action_horizon,
            "max_token_len": max_token_len,
            "action_type": "JOINT_ABS",
            "state_order": "left_arm7,right_arm7,left_gripper,right_gripper",
        },
        wandb_enabled=False,
    )


def make_finetune_config(
    *,
    repo_id: str,
    initial_checkpoint: str,
    assets_base_dir: str,
    checkpoint_base_dir: str,
    batch_size: int,
    train_steps: int,
    lora: bool = True,
    coarse_action_horizon: int = COARSE_ACTION_HORIZON,
    action_horizon: int = ACTION_HORIZON,
    max_token_len: int = MAX_TOKEN_LEN,
    ema_decay: float | None = None,
):
    from openpi.training import config, optimizer, weight_loaders

    warmup_steps = min(2_000, max(1, train_steps // 10))
    model = make_model_config(
        lora=lora,
        coarse_action_horizon=coarse_action_horizon,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
    )
    freeze_filter = model.get_freeze_filter(
        # 冻结视觉主干，语言主干仅训练 LoRA；双动作专家和 ACoT
        # reasoner/fusion 模块全量训练。
        # 60 条固定场景示教不足以稳定更新完整视觉主干；保留预训练视觉能力。
        freeze_vision=True,
        freeze_llm=lora,
        freeze_llm_embedder=True,
        freeze_dual_ae=[False, False],
    )
    return config.TrainConfig(
        name=TRAIN_CONFIG_NAME,
        model=model,
        data=G2ACOTDataConfig(
            repo_id=repo_id,
            input_dim=G2_DIM,
            output_dim=G2_DIM,
            base_config=config.DataConfig(prompt_from_task=True),
            joint_action_shifts=JOINT_ACTION_SHIFTS,
        ),
        weight_loader=weight_loaders.ACOTCheckpointWeightLoader(
            f"{initial_checkpoint.rstrip('/')}/params"
        ),
        freeze_filter=freeze_filter,
        lr_schedule=optimizer.CosineDecaySchedule(
            # 正式 20k 使用 2k warmup；短 smoke test 自动缩短。
            warmup_steps=warmup_steps,
            peak_lr=5e-5,
            # Optax 要求 warmup 之后至少还有一个衰减步；这只影响 1-step smoke。
            decay_steps=max(train_steps, warmup_steps + 1),
            decay_lr=1e-5,
        ),
        optimizer=LowMemoryAdafactor(clip_gradient_norm=1.0),
        # 24GB 本地显卡默认不保留完整 EMA 参数副本；大显存设备可显式启用。
        ema_decay=ema_decay,
        assets_base_dir=assets_base_dir,
        checkpoint_base_dir=checkpoint_base_dir,
        batch_size=batch_size,
        num_workers=2,
        num_train_steps=train_steps,
        log_interval=min(50, max(1, train_steps // 10)),
        # 大模型 checkpoint 很大；调用方可覆盖保存间隔，默认每 5000 步保存。
        save_interval=min(5000, train_steps),
        keep_period=5000,
        wandb_enabled=False,
        policy_metadata={
            "model_kind": "acot_pi05",
            "dataset_version": DATASET_VERSION,
            "action_hz": DATASET_FPS,
            "low_level_hz": LOW_LEVEL_HZ,
            "robot": "G2_omnipicker",
            "state_dim": G2_DIM,
            "action_dim": G2_DIM,
            "action_horizon": action_horizon,
            "coarse_action_horizon": coarse_action_horizon,
            "max_token_len": max_token_len,
            "action_type": "JOINT_ABS",
            "state_order": "left_arm7,right_arm7,left_gripper,right_gripper",
        },
    )
