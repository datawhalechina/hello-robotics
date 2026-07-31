"""示例 2：双雷达 3D XYZI 体素建图、回环检测与位姿图优化。"""

import argparse

try:
    from .demo_common import add_common_arguments, run_mapping
except ImportError:
    from demo_common import add_common_arguments, run_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, "g2_dual_lidar_3d_map")
    run_mapping(parser.parse_args(), build_3d=True, build_2d=False)


if __name__ == "__main__":
    main()
