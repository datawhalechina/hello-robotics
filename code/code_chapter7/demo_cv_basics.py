"""示例 1：读取 G2 相机，并演示常用 OpenCV 基础处理。"""

import argparse

try:
    from .config import DEFAULT_OUTPUT_DIR
    from .cv_utils import BasicCVProcessor, ImageWindow, make_cv_panel, save_image
    from .demo_common import G2VisionDemoRuntime, add_common_arguments
except ImportError:
    from config import DEFAULT_OUTPUT_DIR
    from cv_utils import BasicCVProcessor, ImageWindow, make_cv_panel, save_image
    from demo_common import G2VisionDemoRuntime, add_common_arguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "opencv_basics.jpg"),
        help="退出时保存的结果图",
    )
    parser.add_argument("--canny-low", type=int, default=80)
    parser.add_argument("--canny-high", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = BasicCVProcessor(canny_low=args.canny_low, canny_high=args.canny_high)
    runtime = G2VisionDemoRuntime(args)
    window = ImageWindow("Chapter 7 - OpenCV Basics", enabled=not args.headless)
    last_panel = None

    try:
        print("\n=== 第七章示例 1：OpenCV 基础处理 ===")
        print("依次展示：原图、灰度、滤波、边缘、二值化和轮廓。按 q/Esc 退出。\n")
        for frame in range(1, args.max_frames + 1):
            if not runtime.simulation.is_running():
                break
            runtime.step()
            if frame % args.process_every:
                continue

            image = runtime.camera.capture_bgr()
            if image is None:
                continue
            result = processor.process(image)
            last_panel = make_cv_panel(image, result)
            if not window.show(last_panel):
                break
    finally:
        if last_panel is not None:
            save_image(args.output, last_panel)
        window.close()
        runtime.close()


if __name__ == "__main__":
    main()
