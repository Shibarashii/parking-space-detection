"""
Model Export Script for Nail Feature Object Detection
Export trained models to various formats (ONNX, TorchScript, TFLite, etc.)
"""
from utils.logger import get_logger
from config.model_config import ModelConfig
from models.yolo_detector import YOLODetector
import sys
from pathlib import Path
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Export YOLO Model")

    # Model arguments
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights"
    )

    # Export arguments
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript", "engine", "coreml", "saved_model",
                 "pb", "tflite", "edgetpu", "tfjs", "paddle"],
        help="Export format"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--half", action="store_true",
                        help="FP16 quantization")
    parser.add_argument("--int8", action="store_true",
                        help="INT8 quantization")
    parser.add_argument("--dynamic", action="store_true",
                        help="Dynamic axes (ONNX/TensorRT)")
    parser.add_argument("--simplify", action="store_true",
                        help="Simplify ONNX model")
    parser.add_argument("--opset", type=int, default=None,
                        help="ONNX opset version")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size for export")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")

    # Output arguments
    parser.add_argument("--verbose", action="store_true",
                        default=True, help="Verbose output")

    return parser.parse_args()


def main():
    """Main export function"""
    args = parse_args()

    # Setup logger
    logger = get_logger("export", ModelConfig.LOG_DIR)

    logger.info("="*60)
    logger.info("Starting Model Export")
    logger.info("="*60)
    logger.info(f"Weights: {args.weights}")
    logger.info(f"Format: {args.format}")
    logger.info(f"Image size: {args.imgsz}")
    logger.info(f"Device: {args.device or ModelConfig.DEVICE}")

    # Check if weights exist
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Weights file not found: {args.weights}")
        sys.exit(1)

    # Initialize detector with trained weights
    detector = YOLODetector(
        model_name=str(weights_path),
        device=args.device or ModelConfig.DEVICE,
        verbose=args.verbose
    )

    # Prepare export kwargs
    export_kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "half": args.half,
        "int8": args.int8,
        "dynamic": args.dynamic,
        "batch": args.batch
    }

    # Format-specific kwargs
    if args.format == "onnx":
        export_kwargs["simplify"] = args.simplify
        if args.opset:
            export_kwargs["opset"] = args.opset

    logger.info("Export configuration:")
    for key, value in export_kwargs.items():
        logger.info(f"  {key}: {value}")

    try:
        # Export model
        export_path = detector.export(**export_kwargs)

        logger.info("="*60)
        logger.info("Export completed successfully!")
        logger.info("="*60)
        logger.info(f"Exported model saved to: {export_path}")

        # Print file size
        export_file = Path(export_path)
        if export_file.exists():
            size_mb = export_file.stat().st_size / (1024 * 1024)
            logger.info(f"File size: {size_mb:.2f} MB")

    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
