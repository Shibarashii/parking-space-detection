"""
Prediction Script for Object Detection
Supports: images, videos, directories, and webcam/camera
Compatible with both standard detection and OBB models
Includes parking space summary generation
"""
from utils.logger import get_logger
from config.model_config import ModelConfig
from models.yolo_detector import YOLODetector
import sys
from pathlib import Path
import argparse
from datetime import datetime
import cv2
import yaml
import time
import json
from collections import defaultdict
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Predict with YOLO Object Detection Model")

    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained model weights")
    parser.add_argument("--source", type=str, required=True,
                        help="Image, directory, video, or camera (0, 1, 2...)")
    parser.add_argument("--data", type=str,
                        default=str(PROJECT_ROOT / "config" / "data.yaml"))
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=300,
                        help="Maximum detections per image")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu/mps)")
    parser.add_argument("--half", action="store_true",
                        help="Use FP16 half-precision")
    parser.add_argument("--line-width", type=int, default=2,
                        help="Bounding box line width")
    parser.add_argument("--show-labels", action="store_true",
                        default=True, help="Show labels")
    parser.add_argument("--show-conf", action="store_true",
                        default=True, help="Show confidence")
    parser.add_argument("--hide-labels", dest="show_labels",
                        action="store_false")
    parser.add_argument("--hide-conf", dest="show_conf", action="store_false")
    parser.add_argument("--project", type=str,
                        default=str(ModelConfig.PREDICTION_DIR))
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name")
    parser.add_argument("--save", action="store_true",
                        default=True, help="Save results")
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument("--save-txt", action="store_true",
                        help="Save results to txt")
    parser.add_argument("--save-conf", action="store_true",
                        help="Save confidence scores")
    parser.add_argument("--save-crop", action="store_true",
                        help="Save cropped predictions")
    parser.add_argument("--view-img", action="store_true",
                        help="Display results")
    parser.add_argument("--verbose", action="store_true",
                        default=True, help="Verbose output")
    parser.add_argument("--camera-width", type=int,
                        default=640, help="Camera width")
    parser.add_argument("--camera-height", type=int,
                        default=480, help="Camera height")
    parser.add_argument("--show-fps", action="store_true",
                        default=True, help="Show FPS")
    parser.add_argument("--save-summary", action="store_true",
                        default=True, help="Save detection summary")
    parser.add_argument("--no-summary", dest="save_summary",
                        action="store_false", help="Don't save summary")

    return parser.parse_args()


def is_camera_source(source):
    """Check if source is a camera index"""
    try:
        int(source)
        return True
    except ValueError:
        return False


def is_obb_model(model):
    """Check if the model is an OBB model"""
    try:
        # Check model task type
        if hasattr(model, 'model') and hasattr(model.model, 'task'):
            return model.model.task == 'obb'
        # Fallback: check model name
        model_name = str(model.model_name) if hasattr(
            model, 'model_name') else ''
        return 'obb' in model_name.lower()
    except:
        return False


def get_detections_info(result, is_obb=False):
    """
    Get detection information from result
    Compatible with both standard and OBB models

    Args:
        result: YOLO result object
        is_obb: Whether this is an OBB model

    Returns:
        tuple: (num_detections, classes, boxes_or_obb, confidences)
    """
    if is_obb:
        # OBB model uses .obb instead of .boxes
        if hasattr(result, 'obb') and result.obb is not None:
            num_detections = len(result.obb)
            classes = result.obb.cls.cpu().numpy() if num_detections > 0 else []
            confs = result.obb.conf.cpu().numpy() if num_detections > 0 else []
            return num_detections, classes, result.obb, confs
        return 0, [], None, []
    else:
        # Standard detection model uses .boxes
        if hasattr(result, 'boxes') and result.boxes is not None:
            num_detections = len(result.boxes)
            classes = result.boxes.cls.cpu().numpy() if num_detections > 0 else []
            confs = result.boxes.conf.cpu().numpy() if num_detections > 0 else []
            return num_detections, classes, result.boxes, confs
        return 0, [], None, []


class ParkingSummary:
    """Class to track and summarize parking space detections"""

    def __init__(self, class_names):
        self.class_names = class_names
        self.results = []
        self.total_stats = defaultdict(int)
        self.start_time = datetime.now()

    def add_result(self, image_name, num_detections, classes, confidences):
        """Add detection result for an image"""
        class_counts = {}
        class_confidences = {}

        for cls_idx in range(len(self.class_names)):
            count = int((classes == cls_idx).sum())
            class_counts[self.class_names[cls_idx]] = count
            self.total_stats[self.class_names[cls_idx]] += count

            # Get average confidence for this class
            if count > 0:
                cls_confs = confidences[classes == cls_idx]
                class_confidences[self.class_names[cls_idx]
                                  ] = float(np.mean(cls_confs))

        self.results.append({
            'image': image_name,
            'total_spaces': num_detections,
            'detections': class_counts,
            'avg_confidence': class_confidences,
            'timestamp': datetime.now().isoformat()
        })

        self.total_stats['total_images'] += 1
        self.total_stats['total_detections'] += num_detections

    def get_summary_dict(self):
        """Get complete summary as dictionary"""
        total_spaces = self.total_stats.get('total_detections', 0)
        empty_spaces = self.total_stats.get('empty', 0)
        occupied_spaces = self.total_stats.get('occupied', 0)

        # Calculate occupancy rate
        occupancy_rate = (occupied_spaces / total_spaces *
                          100) if total_spaces > 0 else 0
        availability_rate = (empty_spaces / total_spaces *
                             100) if total_spaces > 0 else 0

        summary = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_images_processed': self.total_stats.get('total_images', 0),
                'processing_duration': str(datetime.now() - self.start_time)
            },
            'overall_statistics': {
                'total_parking_spaces': total_spaces,
                'empty_spaces': empty_spaces,
                'occupied_spaces': occupied_spaces,
                'occupancy_rate': round(occupancy_rate, 2),
                'availability_rate': round(availability_rate, 2)
            },
            'per_class_statistics': {
                class_name: count
                for class_name, count in self.total_stats.items()
                if class_name not in ['total_images', 'total_detections']
            },
            'per_image_results': self.results
        }

        return summary

    def save_json(self, output_dir):
        """Save summary as JSON"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / 'detection_summary.json'
        summary = self.get_summary_dict()

        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return json_path

    def save_txt(self, output_dir):
        """Save summary as formatted text"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        txt_path = output_dir / 'detection_summary.txt'
        summary = self.get_summary_dict()

        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PARKING SPACE DETECTION SUMMARY\n")
            f.write("="*70 + "\n\n")

            # Metadata
            f.write("PROCESSING INFORMATION\n")
            f.write("-"*70 + "\n")
            f.write(f"Timestamp: {summary['metadata']['timestamp']}\n")
            f.write(
                f"Total Images: {summary['metadata']['total_images_processed']}\n")
            f.write(
                f"Processing Duration: {summary['metadata']['processing_duration']}\n")
            f.write("\n")

            # Overall statistics
            stats = summary['overall_statistics']
            f.write("OVERALL STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Total Parking Spaces: {stats['total_parking_spaces']}\n")
            f.write(f"Empty Spaces: {stats['empty_spaces']}\n")
            f.write(f"Occupied Spaces: {stats['occupied_spaces']}\n")
            f.write(f"Occupancy Rate: {stats['occupancy_rate']}%\n")
            f.write(f"Availability Rate: {stats['availability_rate']}%\n")
            f.write("\n")

            # Per-image results
            if summary['per_image_results']:
                f.write("PER-IMAGE RESULTS\n")
                f.write("-"*70 + "\n")
                for result in summary['per_image_results']:
                    f.write(f"\nImage: {result['image']}\n")
                    f.write(f"  Total Spaces: {result['total_spaces']}\n")
                    for class_name, count in result['detections'].items():
                        avg_conf = result['avg_confidence'].get(class_name, 0)
                        f.write(
                            f"  {class_name}: {count} (avg conf: {avg_conf:.2f})\n")

            f.write("\n" + "="*70 + "\n")

        return txt_path

    def save_csv(self, output_dir):
        """Save per-image results as CSV"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / 'detection_results.csv'

        with open(csv_path, 'w') as f:
            # Header
            headers = ['image', 'total_spaces']
            for class_name in self.class_names:
                headers.extend([class_name, f'{class_name}_confidence'])
            f.write(','.join(headers) + '\n')

            # Data rows
            for result in self.results:
                row = [
                    result['image'],
                    str(result['total_spaces'])
                ]
                for class_name in self.class_names:
                    count = result['detections'].get(class_name, 0)
                    conf = result['avg_confidence'].get(class_name, 0)
                    row.extend([str(count), f"{conf:.4f}"])
                f.write(','.join(row) + '\n')

        return csv_path

    def create_visualization(self, output_dir):
        """Create a summary visualization image"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create summary image
        img_width = 800
        img_height = 600
        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

        summary = self.get_summary_dict()
        stats = summary['overall_statistics']

        # Title (FONT_HERSHEY_DUPLEX instead of FONT_HERSHEY_BOLD)
        cv2.putText(img, "PARKING DETECTION SUMMARY", (50, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3)

        y_offset = 120

        # Main statistics
        cv2.putText(img, f"Total Spaces: {stats['total_parking_spaces']}",
                    (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        y_offset += 50

        cv2.putText(img, f"Empty: {stats['empty_spaces']}",
                    (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        y_offset += 45

        cv2.putText(img, f"Occupied: {stats['occupied_spaces']}",
                    (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        y_offset += 45

        cv2.putText(img, f"Occupancy: {stats['occupancy_rate']:.1f}%",
                    (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        y_offset += 45

        cv2.putText(img, f"Availability: {stats['availability_rate']:.1f}%",
                    (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

        # Draw occupancy bar
        bar_y = 350
        bar_height = 60
        bar_width = 700
        bar_x = 50

        # Background bar
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      (200, 200, 200), -1)

        # Occupied portion (red)
        if stats['total_parking_spaces'] > 0:
            occupied_width = int(
                (stats['occupied_spaces'] / stats['total_parking_spaces']) * bar_width)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + occupied_width, bar_y + bar_height),
                          (0, 0, 255), -1)

        # Labels
        cv2.putText(img, "Occupancy Visualization", (bar_x, bar_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, f"Occupied ({stats['occupancy_rate']:.1f}%)",
                    (bar_x + 10, bar_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Footer
        cv2.putText(img, f"Processed: {summary['metadata']['total_images_processed']} images",
                    (50, img_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        cv2.putText(img, summary['metadata']['timestamp'][:19],
                    (50, img_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Save
        viz_path = output_dir / 'summary_visualization.jpg'
        cv2.imwrite(str(viz_path), img)

        return viz_path

    def save_all(self, output_dir):
        """Save all summary formats"""
        output_dir = Path(output_dir)

        json_path = self.save_json(output_dir)
        txt_path = self.save_txt(output_dir)
        csv_path = self.save_csv(output_dir)
        viz_path = self.create_visualization(output_dir)

        return {
            'json': json_path,
            'txt': txt_path,
            'csv': csv_path,
            'visualization': viz_path
        }

    def print_summary(self):
        """Print summary to console"""
        summary = self.get_summary_dict()
        stats = summary['overall_statistics']

        print("\n" + "="*70)
        print("PARKING SPACE DETECTION SUMMARY")
        print("="*70)
        print(f"\nTotal Parking Spaces: {stats['total_parking_spaces']}")
        print(f"Empty Spaces: {stats['empty_spaces']}")
        print(f"Occupied Spaces: {stats['occupied_spaces']}")
        print(f"Occupancy Rate: {stats['occupancy_rate']:.2f}%")
        print(f"Availability Rate: {stats['availability_rate']:.2f}%")
        print("\nImages Processed: {0}".format(
            summary['metadata']['total_images_processed']))
        print("="*70 + "\n")


def plot_with_custom_colors(result, class_names, line_width=2, show_labels=False, show_conf=False):
    """
    Plot detections with custom colors (Green for empty, Red for occupied)
    Works with both standard detection and OBB models

    Args:
        result: YOLO result object
        class_names: List of class names
        line_width: Box line width
        show_labels: Whether to show class labels
        show_conf: Whether to show confidence scores

    Returns:
        Annotated image (BGR format)
    """
    img = result.orig_img.copy()

    # Check if OBB or standard detection
    is_obb = hasattr(result, 'obb') and result.obb is not None

    if is_obb:
        # OBB model
        if len(result.obb) == 0:
            return img

        boxes = result.obb.xyxyxyxy.cpu().numpy()  # Rotated box corners
        classes = result.obb.cls.cpu().numpy().astype(int)
        confs = result.obb.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):
            # Determine color based on class name
            if cls < len(class_names):
                class_name = class_names[cls].lower()
                if class_name == 'empty':
                    color = (0, 255, 0)  # Green (BGR)
                elif class_name == 'occupied':
                    color = (0, 0, 255)  # Red (BGR)
                else:
                    color = (255, 255, 0)  # Yellow (BGR)
            else:
                color = (255, 255, 255)  # White

            # Draw rotated rectangle
            points = box.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img, [points], isClosed=True,
                          color=color, thickness=line_width)

            # Add label if requested
            if show_labels and cls < len(class_names):
                label = class_names[cls]
                if show_conf:
                    label += f' {conf:.2f}'

                label_pos = tuple(points[0][0])
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    img, label_pos, (label_pos[0] + label_w, label_pos[1] - label_h - 5), color, -1)
                cv2.putText(img, label, (label_pos[0], label_pos[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        # Standard detection model
        if len(result.boxes) == 0:
            return img

        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):
            # Determine color based on class name
            if cls < len(class_names):
                class_name = class_names[cls].lower()
                if class_name == 'empty':
                    color = (0, 255, 0)  # Green (BGR)
                elif class_name == 'occupied':
                    color = (0, 0, 255)  # Red (BGR)
                else:
                    color = (255, 255, 0)  # Yellow (BGR)
            else:
                color = (255, 255, 255)  # White

            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)

            # Add label if requested
            if show_labels and cls < len(class_names):
                label = class_names[cls]
                if show_conf:
                    label += f' {conf:.2f}'

                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - label_h - 5),
                              (x1 + label_w, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def predict_camera(detector, camera_idx, args, class_names):
    """Handle camera/webcam prediction with live view"""
    logger = get_logger("prediction")

    # Check if OBB model
    is_obb = is_obb_model(detector.model)
    model_type = "OBB" if is_obb else "Detection"

    logger.info("\n" + "="*60)
    logger.info(f"CAMERA MODE - LIVE VIEW ({model_type})")
    logger.info("="*60)
    logger.info(f"Camera index: {camera_idx}")
    logger.info(
        "Controls: 'q'=Quit, 's'=Save, '+'=More confident, '-'=Less confident")
    logger.info("="*60)

    cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_idx}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"\n✓ Camera opened: {actual_width}x{actual_height}")
    logger.info("▶️  Starting live detection... Press 'q' to quit\n")

    frame_count = 0
    start_time = time.time()
    conf_threshold = args.conf

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                break

            frame_count += 1

            # Predict on current frame
            try:
                results = detector.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    save=False,
                    verbose=False
                )
                if results and len(results) > 0:
                    result = results[0]
                    # Use custom plotting with green/red colors
                    annotated_frame = plot_with_custom_colors(
                        result,
                        class_names,
                        line_width=args.line_width,
                        show_labels=args.show_labels,
                        show_conf=args.show_conf
                    )

                    # Get detections
                    num_detections, classes, _, confs = get_detections_info(
                        result, is_obb)

                    # Display detection counts on frame
                    if num_detections > 0 and len(class_names) > 0:
                        y_offset = 30
                        for cls_idx in range(len(class_names)):
                            count = (classes == cls_idx).sum()
                            if count > 0:
                                # Use matching colors for text
                                if class_names[cls_idx].lower() == 'empty':
                                    text_color = (0, 255, 0)  # Green
                                else:
                                    text_color = (0, 0, 255)  # Red

                                text = f"{class_names[cls_idx]}: {count}"
                                cv2.putText(annotated_frame, text, (10, y_offset),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                                y_offset += 25
                else:
                    annotated_frame = frame

            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                annotated_frame = frame

            # Show FPS
            if args.show_fps:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}",
                            (actual_width - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show confidence threshold
            cv2.putText(annotated_frame, f"Conf: {conf_threshold:.2f}",
                        (actual_width - 150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show model type
            cv2.putText(annotated_frame, model_type,
                        (10, actual_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Display live
            cv2.imshow(
                f'Detection - Live Camera ({model_type})', annotated_frame)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # 'q' or ESC
                logger.info("\nQuitting...")
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"camera_capture_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                logger.info(f"📸 Saved: {filename}")
            elif key == ord('+') or key == ord('='):
                conf_threshold = min(0.95, conf_threshold + 0.05)
                logger.info(f"Confidence: {conf_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                conf_threshold = max(0.05, conf_threshold - 0.05)
                logger.info(f"Confidence: {conf_threshold:.2f}")

    except KeyboardInterrupt:
        logger.info("\nStopped by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        logger.info(
            f"\n✅ Processed {frame_count} frames in {elapsed:.1f}s (Avg FPS: {avg_fps:.2f})")


def main():
    args = parse_args()
    logger = get_logger("prediction", ModelConfig.LOG_DIR)

    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"predict_{timestamp}"

    logger.info("="*60)
    logger.info("Starting Prediction")
    logger.info("="*60)
    logger.info(f"Weights: {args.weights}")
    logger.info(f"Source: {args.source}")
    logger.info(f"Confidence: {args.conf}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Weights file not found: {args.weights}")
        sys.exit(1)

    # Load model
    detector = YOLODetector(model_name=str(weights_path),
                            device=args.device, verbose=args.verbose)
    logger.info(f"Using device: {detector.device}")

    # Check model type
    is_obb = is_obb_model(detector.model)
    model_type = "OBB" if is_obb else "Detection"
    logger.info(f"Model type: {model_type}")

    # Load class names
    class_names = []
    if Path(args.data).exists():
        with open(args.data, 'r') as f:
            data_config = yaml.safe_load(f)
            class_names = data_config.get('names', [])
            logger.info(f"Classes: {class_names}")

    # Check if camera source
    if is_camera_source(args.source):
        camera_idx = int(args.source)
        predict_camera(detector, camera_idx, args, class_names)
        return

    # Handle files/directories/videos
    source_path = Path(args.source)
    if not source_path.exists():
        logger.error(f"Source not found: {args.source}")
        sys.exit(1)

    predict_config = ModelConfig.get_prediction_config(
        imgsz=args.imgsz, conf=args.conf, iou=args.iou, max_det=args.max_det,
        half=args.half, save=args.save, save_txt=args.save_txt,
        save_conf=args.save_conf, save_crop=args.save_crop,
        line_width=args.line_width, show_labels=args.show_labels,
        show_conf=args.show_conf
    )

    try:
        results = detector.predict(
            source=args.source,
            project=args.project,
            name=args.name,
            save=False,  # ← Disable YOLO's default save
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save_crop=args.save_crop,
            **{k: v for k, v in predict_config.items() if k not in ['save', 'save_txt', 'save_conf', 'save_crop']}
        )

        # Initialize parking summary if enabled
        parking_summary = None
        if args.save_summary and len(class_names) > 0:
            parking_summary = ParkingSummary(class_names)

        # Create save directory
        if args.save:
            save_dir = Path(args.project) / args.name
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving results to: {save_dir}")

        total_detections = 0
        for i, result in enumerate(results):
            # Get detections
            num_detections, classes, _, confs = get_detections_info(
                result, is_obb)
            total_detections += num_detections

            # Add to parking summary
            if parking_summary and num_detections > 0:
                image_name = Path(result.path).name if hasattr(
                    result, 'path') else f"image_{i+1}"
                parking_summary.add_result(
                    image_name, num_detections, classes, confs)

            if num_detections > 0:
                logger.info(f"\nImage {i+1}: {num_detections} detections")
                if len(class_names) > 0:
                    for cls_idx in range(len(class_names)):
                        count = (classes == cls_idx).sum()
                        if count > 0:
                            logger.info(f"  {class_names[cls_idx]}: {count}")

            # Plot with custom colors
            plotted = plot_with_custom_colors(
                result,
                class_names,
                line_width=args.line_width,
                show_labels=args.show_labels,
                show_conf=args.show_conf
            )

            # Save the custom-colored image
            if args.save:
                image_name = Path(result.path).name if hasattr(
                    result, 'path') else f"image_{i+1}.jpg"
                output_path = save_dir / image_name
                cv2.imwrite(str(output_path), plotted)
                logger.info(f"  Saved: {output_path}")

            # Display if requested
            if args.view_img:
                cv2.imshow(f"Prediction {i+1}", plotted)
                cv2.waitKey(0)

        # Save parking summary if enabled
        if parking_summary and args.save_summary:
            if not args.save:
                save_dir = Path(args.project) / args.name
            summary_paths = parking_summary.save_all(save_dir)
            parking_summary.print_summary()

            logger.info("\n📊 Summary files saved:")
            for file_type, file_path in summary_paths.items():
                logger.info(f"  {file_type}: {file_path}")

        logger.info(f"\nTotal detections: {total_detections}")
        if len(results) > 0:
            logger.info(
                f"Average per image: {total_detections/len(results):.2f}")
        if args.save:
            logger.info(f"\n✅ Results saved to: {save_dir}")

        logger.info("\n✅ Prediction completed!")
        if args.view_img:
            cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
