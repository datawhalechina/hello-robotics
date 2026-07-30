"""三个第七章示例共用的命令行参数与仿真运行时。"""

import argparse
from dataclasses import replace

try:
    from .camera import G2RGBCamera
    from .config import CAMERAS, SimulationConfig
    from .scan_controller import VisionScanController
    from .simulation import G2VisionSimulation
except ImportError:
    from camera import G2RGBCamera
    from config import CAMERAS, SimulationConfig
    from scan_controller import VisionScanController
    from simulation import G2VisionSimulation


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--headless", action="store_true", help="无窗口运行并保存最终结果")
    parser.add_argument(
        "--scene", choices=("warehouse", "room"), default="warehouse",
        help="warehouse 目标更丰富；room 与第四章场景一致",
    )
    parser.add_argument("--camera", choices=CAMERAS, default="head_front", help="选择 G2 机载相机")
    parser.add_argument("--max-frames", type=int, default=900, help="最多运行多少个物理帧")
    parser.add_argument("--process-every", type=int, default=4, help="每隔多少物理帧处理一次图像")
    parser.add_argument("--spin-speed", type=float, default=0.0, help="扫描角速度，rad/s；0 表示静止")
    parser.add_argument("--physics-hz", type=int, default=120)
    parser.add_argument("--rendering-hz", type=int, default=30)


class G2VisionDemoRuntime:
    """统一加载场景、G2、G2 机载相机和底盘扫描控制。"""

    def __init__(self, args: argparse.Namespace) -> None:
        if args.max_frames <= 0:
            raise ValueError("max_frames 必须大于 0")
        if args.process_every <= 0:
            raise ValueError("process_every 必须大于 0")

        config = replace(
            SimulationConfig(),
            headless=args.headless,
            physics_hz=args.physics_hz,
            rendering_hz=args.rendering_hz,
            scene=args.scene,
        )
        self.config = config
        self.simulation = G2VisionSimulation(config)
        self.camera = None
        self.scan = None
        self.spin_speed = float(args.spin_speed)
        try:
            self.camera = G2RGBCamera(CAMERAS[args.camera], frequency=config.rendering_hz)
            self.scan = VisionScanController(self.simulation.robot)
            self.camera.wait_for_bgr(self.simulation)
        except Exception:
            self.simulation.close()
            raise

    def step(self) -> None:
        self.scan.update(self.config.physics_dt, self.spin_speed)
        self.simulation.step(render=True)

    def close(self) -> None:
        try:
            if self.scan is not None:
                self.scan.stop()
        finally:
            self.simulation.close()
