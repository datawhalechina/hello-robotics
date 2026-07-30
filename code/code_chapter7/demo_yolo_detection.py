"""示例 2：使用 YOLO 对 G2 相机图像进行实时目标检测。"""

import argparse
from pathlib import Path

try:
    from .config import DEFAULT_DETECTION_MODEL, DEFAULT_OUTPUT_DIR, InferenceConfig
    from .cv_utils import ImageWindow, save_image
    from .demo_common import G2VisionDemoRuntime, add_common_arguments
    from .yolo_vision import YOLOVision, draw_detections
except ImportError:
    from config import DEFAULT_DETECTION_MODEL, DEFAULT_OUTPUT_DIR, InferenceConfig
    from cv_utils import ImageWindow, save_image
    from demo_common import G2VisionDemoRuntime, add_common_arguments
    from yolo_vision import YOLOVision, draw_detections


def parse_args() -> argparse.Namespace:
    defaults = InferenceConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.set_defaults(process_every=6)
    parser.add_argument("--model", default=str(DEFAULT_DETECTION_MODEL), help="YOLO 检测权重")
    parser.add_argument("--conf", type=float, default=defaults.confidence)
    parser.add_argument("--iou", type=float, default=defaults.iou)
    parser.add_argument("--imgsz", type=int, default=defaults.image_size)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "yolo_detection.jpg"),
        help="退出时保存的检测图",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = G2VisionDemoRuntime(args)
    window = ImageWindow("Chapter 7 - YOLO Detection", enabled=not args.headless)
    last_annotated = None

    try:
        model = YOLOVision(args.model)
        print("\n=== 第七章示例 2：YOLO 目标检测 ===")
        print(f"模型：{Path(args.model).expanduser().resolve()}")
        print("按 q/Esc 退出。\n")
        for frame in range(1, args.max_frames + 1):
            if not runtime.simulation.is_running():
                break
            runtime.step()
            if frame % args.process_every:
                continue

            image = runtime.camera.capture_bgr()
            if image is None:
                continue
            prediction = model.predict(image, args.conf, args.iou, args.imgsz)
            last_annotated = draw_detections(image, prediction)
            names = [item.class_name for item in prediction.detections]
            print(
                f"[YOLO] frame={frame:04d}, count={len(names)}, classes={names[:8]}",
                flush=True,
            )
            if not window.show(last_annotated):
                break
    finally:
        if last_annotated is not None:
            save_image(args.output, last_annotated)
        window.close()
        runtime.close()


if __name__ == "__main__":
    main()
