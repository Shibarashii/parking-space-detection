"""
Training Script for Nail Feature Object Detection
"""
from utils.logger import get_logger
from config.model_config import ModelConfig
from models.yolo_detector import YOLODetector
import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train YOLO Object Detection Model")

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        choices=list(ModelConfig.MODELS.keys()),
        help="Model architecture to use"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to pretrained weights (optional)"
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "config" / "data.yaml"),
        help="Path to data.yaml file"
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu/mps)")

    # Hyperparameters
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience")
    parser.add_argument("--save-period", type=int, default=-1,
                        help="Save checkpoint every n epochs")

    # Augmentation
    parser.add_argument("--augment", action="store_true",
                        help="Use augmentation")
    parser.add_argument("--mosaic", type=float, default=1.0,
                        help="Mosaic augmentation probability")
    parser.add_argument("--mixup", type=float, default=0.0,
                        help="Mixup augmentation probability")
    parser.add_argument("--copy-paste", type=float, default=0.0,
                        help="Copy-paste augmentation probability")
    parser.add_argument("--degrees", type=float, default=0.0,
                        help="Image rotation (+/- deg)")
    parser.add_argument("--translate", type=float, default=0.1,
                        help="Image translation (+/- fraction)")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Image scale (+/- gain)")
    parser.add_argument("--shear", type=float, default=0.0,
                        help="Image shear (+/- deg)")
    parser.add_argument("--perspective", type=float,
                        default=0.0, help="Image perspective (+/- fraction)")
    parser.add_argument("--flipud", type=float, default=0.0,
                        help="Flip up-down probability")
    parser.add_argument("--fliplr", type=float, default=0.5,
                        help="Flip left-right probability")
    parser.add_argument("--hsv-h", type=float, default=0.015,
                        help="HSV-Hue augmentation (fraction)")
    parser.add_argument("--hsv-s", type=float, default=0.7,
                        help="HSV-Saturation augmentation (fraction)")
    parser.add_argument("--hsv-v", type=float, default=0.4,
                        help="HSV-Value augmentation (fraction)")
    parser.add_argument("--erasing", type=float, default=0.0,
                        help="Random erasing probability")

    # Output arguments
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(ModelConfig.MODEL_DIR),
        help="Project directory"
    )

    # Other arguments
    parser.add_argument("--resume", action="store_true",
                        help="Resume training")
    parser.add_argument("--verbose", action="store_true",
                        default=True, help="Verbose output")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    # Create directories
    ModelConfig.create_directories()

    # Setup logger
    logger = get_logger("training", ModelConfig.LOG_DIR)

    # Generate experiment name if not provided
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"{args.model}_{timestamp}"

    logger.info(f"Starting training experiment: {args.name}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Device: {args.device or ModelConfig.DEVICE}")

    # Get model path
    if args.weights:
        model_path = args.weights
        logger.info(f"Loading custom weights: {model_path}")
    else:
        model_path = ModelConfig.get_model_path(args.model)
        logger.info(f"Using pretrained weights: {model_path}")

    # Initialize detector
    detector = YOLODetector(
        model_name=model_path,
        device=args.device or ModelConfig.DEVICE,
        verbose=args.verbose
    )

    # Get training configuration
    train_config = ModelConfig.get_training_config(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        patience=args.patience,
        save_period=args.save_period,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        erasing=args.erasing,
        seed=args.seed,
        resume=args.resume
    )

    logger.info("Training configuration:")
    for key, value in train_config.items():
        logger.info(f"  {key}: {value}")

    try:
        # Train model
        results = detector.train(
            data_yaml=args.data,
            project=args.project,
            name=args.name,
            **train_config
        )

        logger.info("Training completed successfully!")
        logger.info(
            f"Best model saved to: {results.save_dir / 'weights' / 'best.pt'}")
        logger.info(
            f"Last model saved to: {results.save_dir / 'weights' / 'last.pt'}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
