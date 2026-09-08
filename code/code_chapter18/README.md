# ACoT-VLA G2 三色物块代码

本目录提供 G2 Omnipicker 三色物块入盒任务的完整运行流程。

| 模块 | 文件 | 作用 |
|---|---|---|
| 统一配置 | `config.py` | 场景、频率、关节顺序和数据路径 |
| 数据采集 | `collect_demos.py` | 采集 RGB 各 20 条成功专家轨迹 |
| 采集验收 | `smoke_collect.py` | RGB 小规模试采、数据检查和动作回放 |
| 数据转换 | `training/convert_dataset.py` | NPZ 转 LeRobot |
| 统计计算 | `training/compute_norm_stats.py` | 计算 state/coarse/fine 统计 |
| 模型适配 | `training/g2_policy.py` | 连接 G2 数据合同与 ACoT-VLA |
| 模型训练 | `training/train.py` | LoRA 与 ACoT 新模块微调 |
| 模型服务 | `training/serve_model.py` | 加载 checkpoint 并启动 WebSocket 服务 |
| 交互推理 | `run_inference.py` | 可视化执行单色任务 |
| 独立评测 | `evaluate_acot_policy.py` | RGB 单任务成功率评测 |

完整环境配置、命令和数据合同见 `../../docs/chapter18/第十八章 ACoT-VLA 视觉语言动作模型部署.md`。

运行前将官方 ACoT-VLA 仓库克隆到 `acotvla/`，再按照教程安装训练环境。
