"""
Batch Inference Script
Process multiple images and generate a summary report
"""
from utils.logger import get_logger
from config.model_config import ModelConfig
from models.yolo_detector import YOLODetector
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Batch Inference with Summary Report")

    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained weights")
    parser.add_argument("--source", type=str, required=True,
                        help="Directory containing images")
    parser.add_argument("--data", type=str,
                        default=str(PROJECT_ROOT / "config" / "data.yaml"))
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save results to CSV")
    parser.add_argument("--save-json", action="store_true",
                        help="Save results to JSON")
    parser.add_argument("--device", type=str, default=None, help="Device")

    return parser.parse_args()


def main():
    """Main batch inference function"""
    args = parse_args()

    # Setup logger
    logger = get_logger("batch_inference", ModelConfig.LOG_DIR)

    # Output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ModelConfig.PREDICTION_DIR / f"batch_{timestamp}"
    else:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("Starting Batch Inference")
    logger.info("="*60)
    logger.info(f"Weights: {args.weights}")
    logger.info(f"Source: {args.source}")
    logger.info(f"Output: {output_dir}")

    # Load class names
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
        class_names = data_config.get('names', [])

    # Initialize detector
    detector = YOLODetector(
        model_name=args.weights,
        device=args.device or ModelConfig.DEVICE,
        verbose=False
    )

    # Get all images
    source_dir = Path(args.source)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(source_dir.glob(f'*{ext}'))
        image_files.extend(source_dir.glob(f'*{ext.upper()}'))

    logger.info(f"Found {len(image_files)} images")

    # Process images
    results_data = []

    for img_path in image_files:
        logger.info(f"Processing: {img_path.name}")

        # Predict
        results = detector.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            save=True,
            project=str(output_dir),
            name="predictions",
            verbose=False
        )

        # Extract results
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            # Image-level summary
            img_data = {
                'image_name': img_path.name,
                'num_detections': len(boxes),
                'detections': []
            }

            # Count per class
            for cls_idx, cls_name in enumerate(class_names):
                count = (boxes.cls.cpu().numpy() == cls_idx).sum()
                img_data[cls_name] = int(count)

            # Detection details
            for box in boxes:
                detection = {
                    'class': class_names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy.cpu().numpy().tolist()[0]
                }
                img_data['detections'].append(detection)

            results_data.append(img_data)

    # Generate summary statistics
    logger.info("\n" + "="*60)
    logger.info("Summary Statistics")
    logger.info("="*60)

    total_detections = sum(d['num_detections'] for d in results_data)
    avg_detections = total_detections / \
        len(results_data) if results_data else 0

    logger.info(f"Total images processed: {len(results_data)}")
    logger.info(f"Total detections: {total_detections}")
    logger.info(f"Average detections per image: {avg_detections:.2f}")

    # Per-class statistics
    logger.info("\nPer-Class Detection Counts:")
    class_totals = {cls: 0 for cls in class_names}
    for data in results_data:
        for cls in class_names:
            class_totals[cls] += data.get(cls, 0)

    for cls, count in class_totals.items():
        logger.info(f"  {cls:30s}: {count:5d}")

    # Save results
    if args.save_csv:
        # Create DataFrame for CSV
        df_data = []
        for data in results_data:
            row = {'image_name': data['image_name'],
                   'num_detections': data['num_detections']}
            for cls in class_names:
                row[cls] = data.get(cls, 0)
            df_data.append(row)

        df = pd.DataFrame(df_data)
        csv_path = output_dir / "batch_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"\n✓ Results saved to CSV: {csv_path}")

    if args.save_json:
        json_path = output_dir / "batch_results.json"
        with open(json_path, 'w') as f:
            json.dump({
                'summary': {
                    'total_images': len(results_data),
                    'total_detections': total_detections,
                    'avg_detections_per_image': avg_detections,
                    'class_totals': class_totals
                },
                'results': results_data
            }, f, indent=2)
        logger.info(f"✓ Results saved to JSON: {json_path}")

    logger.info("\n" + "="*60)
    logger.info("Batch inference completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
