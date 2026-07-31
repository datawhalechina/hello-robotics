"""示例 3：双雷达投影为 2D 射线，生成 ROS/Nav2 格式占据栅格。"""

import argparse

try:
    from .demo_common import add_common_arguments, run_mapping
except ImportError:
    from demo_common import add_common_arguments, run_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, "g2_dual_lidar_2d_map")
    run_mapping(parser.parse_args(), build_3d=False, build_2d=True)


if __name__ == "__main__":
    main()
