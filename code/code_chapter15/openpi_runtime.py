"""Load only this chapter's pinned OpenPI source tree and validate checkpoints."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from config import DATA_ROOT, LEROBOT_ROOT, OPENPI_ROOT


def prepare_openpi():
    required = (
        OPENPI_ROOT / "src/openpi/__init__.py",
        OPENPI_ROOT / "packages/openpi-client/src/openpi_client/__init__.py",
    )
    if not all(x.is_file() for x in required):
        raise FileNotFoundError(f"incomplete OpenPI snapshot: {OPENPI_ROOT}")
    os.environ.update(
        HF_LEROBOT_HOME=str(LEROBOT_ROOT),
        HF_HOME=str(DATA_ROOT / "hf_cache"),
        CHAPTER15_OPENPI_READY="1",
    )
    for path in (OPENPI_ROOT / "src", OPENPI_ROOT / "packages/openpi-client/src"):
        value = str(path)
        sys.path.remove(value) if value in sys.path else None
        sys.path.insert(0, value)


def _load_package_from_openpi_venv(package: str) -> None:
    """Load one small client dependency without exposing the full venv to Isaac Sim."""
    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    package_dir = OPENPI_ROOT / ".venv" / "lib" / python_dir / "site-packages" / package
    init_file = package_dir / "__init__.py"
    if not init_file.is_file():
        raise ModuleNotFoundError(
            f"{package!r} is missing for Isaac Sim Python. Run ./setup_env.sh first; "
            f"expected {init_file}"
        )

    spec = importlib.util.spec_from_file_location(
        package,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {package!r} from {init_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(package, None)
        raise


def prepare_openpi_client() -> None:
    """Prepare the OpenPI websocket client inside either OpenPI or Isaac Python."""
    prepare_openpi()
    try:
        importlib.import_module("msgpack")
    except ModuleNotFoundError as exc:
        if exc.name != "msgpack":
            raise
        _load_package_from_openpi_venv("msgpack")


def metadata(checkpoint):
    path = Path(checkpoint) / "chapter15_policy.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint metadata: {path}")
    return json.loads(path.read_text())


def norm_directory(checkpoint):
    files = list((Path(checkpoint) / "assets").rglob("norm_stats.json"))
    if len(files) != 1:
        raise RuntimeError(
            f"checkpoint must contain exactly one G2 norm_stats.json, got {len(files)}"
        )
    return files[0].parent


def latest(run_dir):
    choices = [x for x in Path(run_dir).glob("*") if x.is_dir() and x.name.isdigit()]
    if not choices:
        raise FileNotFoundError(f"no numeric checkpoint under {run_dir}")
    return max(choices, key=lambda x: int(x.name))
