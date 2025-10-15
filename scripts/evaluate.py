"""
Evaluation Script for Nail Feature Object Detection
"""
from utils.logger import get_logger
from config.model_config import ModelConfig
from models.yolo_detector import YOLODetector
import sys
from pathlib import Path
import argparse
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO Object Detection Model")

    # Model arguments
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights"
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "config" / "data.yaml"),
        help="Path to data.yaml file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate"
    )

    # Evaluation arguments
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=300,
                        help="Maximum detections per image")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu/mps)")

    # Output arguments
    parser.add_argument("--save-json", action="store_true",
                        help="Save results to JSON")
    parser.add_argument("--save-hybrid", action="store_true",
                        help="Save hybrid labels")
    parser.add_argument("--verbose", action="store_true",
                        default=True, help="Verbose output")

    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()

    # Setup logger
    logger = get_logger("evaluation", ModelConfig.LOG_DIR)

    logger.info("="*60)
    logger.info("Starting Model Evaluation")
    logger.info("="*60)
    logger.info(f"Weights: {args.weights}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Split: {args.split}")
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

    # Get validation configuration
    val_config = ModelConfig.get_validation_config(
        batch=args.batch,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        split=args.split,
        save_json=args.save_json,
        save_hybrid=args.save_hybrid
    )

    logger.info("Evaluation configuration:")
    for key, value in val_config.items():
        logger.info(f"  {key}: {value}")

    try:
        # Evaluate model
        results = detector.validate(
            data_yaml=args.data,
            **val_config
        )

        logger.info("="*60)
        logger.info("Evaluation Results")
        logger.info("="*60)

        # Print detailed metrics
        if hasattr(results, 'box'):
            metrics = results.box
            logger.info(f"\nOverall Metrics:")

            # These are arrays, so we need to take the mean
            if hasattr(metrics, 'p') and metrics.p is not None:
                precision = metrics.p.mean() if hasattr(metrics.p, 'mean') else metrics.p
                logger.info(f"  Precision: {precision:.4f}")

            if hasattr(metrics, 'r') and metrics.r is not None:
                recall = metrics.r.mean() if hasattr(metrics.r, 'mean') else metrics.r
                logger.info(f"  Recall:    {recall:.4f}")

            logger.info(f"  mAP50:     {metrics.map50:.4f}")
            logger.info(f"  mAP50-95:  {metrics.map:.4f}")

            # Per-class metrics
            if hasattr(metrics, 'maps'):
                logger.info(f"\nPer-Class mAP50-95:")

                # Load class names
                with open(args.data, 'r') as f:
                    data_config = yaml.safe_load(f)
                    class_names = data_config.get('names', [])

                for i, (cls_map, cls_name) in enumerate(zip(metrics.maps, class_names)):
                    logger.info(f"  {cls_name:30s}: {cls_map:.4f}")

        logger.info("\n" + "="*60)
        logger.info("Evaluation completed successfully!")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
