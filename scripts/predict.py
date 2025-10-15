"""
Prediction Script for Object Detection
Supports: images, videos, directories, and webcam/camera
Compatible with both standard detection and OBB models
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
        tuple: (num_detections, classes, boxes_or_obb)
    """
    if is_obb:
        # OBB model uses .obb instead of .boxes
        if hasattr(result, 'obb') and result.obb is not None:
            num_detections = len(result.obb)
            classes = result.obb.cls.cpu().numpy() if num_detections > 0 else []
            return num_detections, classes, result.obb
        return 0, [], None
    else:
        # Standard detection model uses .boxes
        if hasattr(result, 'boxes') and result.boxes is not None:
            num_detections = len(result.boxes)
            classes = result.boxes.cls.cpu().numpy() if num_detections > 0 else []
            return num_detections, classes, result.boxes
        return 0, [], None


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

                # Get annotated frame
                if results and len(results) > 0:
                    result = results[0]
                    annotated_frame = result.plot()

                    # Get detections (compatible with both OBB and standard)
                    num_detections, classes, _ = get_detections_info(
                        result, is_obb)

                    # Display detection counts
                    if num_detections > 0 and len(class_names) > 0:
                        y_offset = 30
                        for cls_idx in range(len(class_names)):
                            count = (classes == cls_idx).sum()
                            if count > 0:
                                text = f"{class_names[cls_idx]}: {count}"
                                cv2.putText(annotated_frame, text, (10, y_offset),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
            source=args.source, project=args.project, name=args.name, **predict_config)

        total_detections = 0
        for i, result in enumerate(results):
            # Get detections (compatible with both OBB and standard)
            num_detections, classes, _ = get_detections_info(result, is_obb)
            total_detections += num_detections

            if num_detections > 0:
                logger.info(f"\nImage {i+1}: {num_detections} detections")
                if len(class_names) > 0:
                    for cls_idx in range(len(class_names)):
                        count = (classes == cls_idx).sum()
                        if count > 0:
                            logger.info(f"  {class_names[cls_idx]}: {count}")

            if args.view_img and result.orig_img is not None:
                cv2.imshow(f"Prediction {i+1}", result.plot())
                cv2.waitKey(0)

        logger.info(f"\nTotal detections: {total_detections}")
        if len(results) > 0:
            logger.info(
                f"Average per image: {total_detections/len(results):.2f}")
        if args.save and results:
            logger.info(f"\nResults saved to: {results[0].save_dir}")

        logger.info("\n✅ Prediction completed!")
        if args.view_img:
            cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
