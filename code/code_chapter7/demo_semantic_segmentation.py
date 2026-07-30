"""示例 3：默认使用 yolo26s-seg 进行分割，并生成逐像素语义图。"""

import argparse
from pathlib import Path

import cv2

try:
    from .config import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_SEGMENTATION_MODEL,
        InferenceConfig,
    )
    from .cv_utils import ImageWindow, save_image
    from .demo_common import G2VisionDemoRuntime, add_common_arguments
    from .segmentation import (
        HSVSemanticSegmenter,
        colorize_semantic_map,
        draw_legend,
        overlay_semantic,
        yolo_to_semantic_map,
    )
    from .yolo_vision import YOLOVision
except ImportError:
    from config import DEFAULT_OUTPUT_DIR, DEFAULT_SEGMENTATION_MODEL, InferenceConfig
    from cv_utils import ImageWindow, save_image
    from demo_common import G2VisionDemoRuntime, add_common_arguments
    from segmentation import (
        HSVSemanticSegmenter,
        colorize_semantic_map,
        draw_legend,
        overlay_semantic,
        yolo_to_semantic_map,
    )
    from yolo_vision import YOLOVision


def parse_args() -> argparse.Namespace:
    defaults = InferenceConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.set_defaults(process_every=6)
    parser.add_argument(
        "--method",
        choices=("color", "yolo"),
        default="yolo",
        help="默认 yolo（yolo26s-seg.pt）；color 仅用于学习传统 HSV 分割",
    )
    parser.add_argument("--model", default=str(DEFAULT_SEGMENTATION_MODEL))
    parser.add_argument("--conf", type=float, default=defaults.confidence)
    parser.add_argument("--iou", type=float, default=defaults.iou)
    parser.add_argument("--imgsz", type=int, default=defaults.image_size)
    parser.add_argument("--mask-threshold", type=float, default=defaults.mask_threshold)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "semantic_segmentation.jpg"),
        help="退出时保存的原图/语义图/叠加图",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = G2VisionDemoRuntime(args)
    window = ImageWindow("Chapter 7 - Semantic Segmentation", enabled=not args.headless)
    last_panel = None

    try:
        color_model = HSVSemanticSegmenter() if args.method == "color" else None
        yolo_model = YOLOVision(args.model) if args.method == "yolo" else None
        print("\n=== 第七章示例 3：语义分割 ===")
        print(f"方法：{args.method}")
        if yolo_model is not None:
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

            if color_model is not None:
                semantic_map = color_model.predict(image)
                class_names = color_model.class_names
                class_colors = color_model.class_colors
            else:
                prediction = yolo_model.predict(image, args.conf, args.iou, args.imgsz)
                semantic_map, class_names = yolo_to_semantic_map(
                    prediction, args.mask_threshold
                )
                class_colors = None

            color_map = colorize_semantic_map(semantic_map, class_colors)
            overlay = overlay_semantic(image, semantic_map, class_colors=class_colors)
            overlay = draw_legend(overlay, class_names, semantic_map, class_colors)
            last_panel = cv2.hconcat((image, color_map, overlay))

            present = [class_names.get(int(i), str(i)) for i in set(semantic_map.flat) if i]
            print(
                f"[Segmentation] frame={frame:04d}, classes={sorted(present)}",
                flush=True,
            )
            if not window.show(last_panel):
                print("[Segmentation] 用户结束显示。", flush=True)
                break
    except Exception as exc:
        # Isaac Sim 快速关闭时可能来不及显示完整 traceback，先输出明确错误。
        print(f"[Segmentation] 运行失败：{type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        if last_panel is not None:
            save_image(args.output, last_panel)
        window.close()
        runtime.close()


if __name__ == "__main__":
    main()
