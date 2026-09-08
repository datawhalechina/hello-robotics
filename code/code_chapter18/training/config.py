"""ACoT-VLA 数据、训练资产和 checkpoint 配置。"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAPTER_ROOT = HERE
PROJECT_ROOT = HERE.parent
OPENPI_ROOT = PROJECT_ROOT / "acotvla"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"

PI05_BASE_CHECKPOINT = CHECKPOINT_ROOT / "base/pi05_base"

DATASET_VERSION = "30hz_synced_hold_v2"
RAW_DATA_DIR = PROJECT_ROOT / "data/raw"
LEROBOT_HOME = PROJECT_ROOT / "data/lerobot"
REPO_ID = f"acotvla_g2_blocks/g2_blocks_{DATASET_VERSION}"
TRAIN_ASSETS_DIR = PROJECT_ROOT / "assets"
TRAIN_CHECKPOINT_DIR = CHECKPOINT_ROOT
TRAIN_CONFIG_NAME = "acot_pi05_base_g2_blocks_lora"

DATASET_FPS = 30
LOW_LEVEL_HZ = 120
EXPECTED_EPISODES = 60
EXPECTED_EPISODE_FRAMES = 255
GRIPPER_OPEN_RAD = 0.785


def checkpoint_path(custom: str) -> Path:
    """解析训练权重并检查参数目录。"""
    checkpoint_root = CHECKPOINT_ROOT.resolve()
    raw = Path(custom).expanduser()
    if raw.is_absolute():
        path = raw.resolve()
    else:
        cwd_path = raw.resolve()
        project_path = (PROJECT_ROOT / raw).resolve()
        path = cwd_path if cwd_path.is_relative_to(checkpoint_root) else project_path
    if not path.is_relative_to(checkpoint_root):
        raise ValueError(f"checkpoint 必须位于 {checkpoint_root}，实际为：{path}")
    if not (path / "params").is_dir():
        raise FileNotFoundError(
            f"找不到模型参数：{path / 'params'}。请指定具体的训练 step 目录。"
        )
    return path
