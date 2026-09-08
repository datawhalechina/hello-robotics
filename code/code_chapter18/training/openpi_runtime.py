"""集中处理官方 ACoT-VLA 路径、LeRobot 数据和 checkpoint 辅助逻辑。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from .config import CHAPTER_ROOT, LEROBOT_HOME, OPENPI_ROOT
except ImportError:
    from config import CHAPTER_ROOT, LEROBOT_HOME, OPENPI_ROOT


def _inside(path: str | Path, root: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        return False


def prepare_openpi(*, lerobot_home: Path | None = None) -> None:
    """加载官方 ACoT-VLA 源码并设置 LeRobot 数据目录。"""
    source = OPENPI_ROOT / "src/openpi/__init__.py"
    client = OPENPI_ROOT / "packages/openpi-client/src/openpi_client/__init__.py"
    if not source.is_file() or not client.is_file():
        raise FileNotFoundError(f"ACoT-VLA/OpenPI 源码不完整：{OPENPI_ROOT}")

    # 如果当前进程已经导入了外部OpenPI，直接报错，避免静默混用两套代码。
    for package in ("openpi", "openpi_client"):
        module = sys.modules.get(package)
        module_file = getattr(module, "__file__", None) if module is not None else None
        if module_file and not _inside(module_file, OPENPI_ROOT):
            raise RuntimeError(
                f"检测到其他 {package}：{module_file}；应使用 {OPENPI_ROOT}"
            )

    # 显式设置数据根目录，避免继承 shell 中不匹配的 HF_LEROBOT_HOME。
    dataset_root = LEROBOT_HOME if lerobot_home is None else Path(lerobot_home)
    os.environ["HF_LEROBOT_HOME"] = str(dataset_root.resolve())
    client_src = OPENPI_ROOT / "packages/openpi-client/src"
    for path in (CHAPTER_ROOT, OPENPI_ROOT / "src", client_src):
        value = str(path.resolve())
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


def find_norm_stats(checkpoint: Path) -> Path:
    """兼容发布权重assets/norm_stats.json和训练权重assets/<id>/norm_stats.json。"""
    files = sorted((checkpoint / "assets").rglob("norm_stats.json"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"{checkpoint / 'assets'}中应有且只有一个norm_stats.json，实际找到{len(files)}个"
        )
    return files[0].parent


def checkpoint_uses_lora(checkpoint: Path) -> bool:
    """从Orbax参数元数据判断checkpoint是否包含LoRA层。"""
    metadata_dir = checkpoint / "params" / "array_metadatas"
    for metadata in metadata_dir.glob("*"):
        if metadata.is_file() and b"lora_" in metadata.read_bytes():
            return True
    return False
