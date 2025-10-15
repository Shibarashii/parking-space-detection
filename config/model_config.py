"""
Model Configuration for Nail Feature Object Detection
"""
from pathlib import Path
from typing import Dict, Any


class ModelConfig:
    """Configuration for YOLO models"""

    # Base paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    MODEL_DIR = OUTPUT_DIR / "models"
    LOG_DIR = OUTPUT_DIR / "logs"
    PREDICTION_DIR = OUTPUT_DIR / "predictions"

    # Model settings
    MODELS = {
        # YOLOv11 (Recommended - Latest and best)
        "yolov11n": {
            "name": "yolo11n.pt",
            "description": "YOLOv11 Nano - Fastest, improved over v8n"
        },
        "yolov11s": {
            "name": "yolo11s.pt",
            "description": "YOLOv11 Small - Best balance (Recommended)"
        },
        "yolov11m": {
            "name": "yolo11m.pt",
            "description": "YOLOv11 Medium - High accuracy"
        },
        "yolov11l": {
            "name": "yolo11l.pt",
            "description": "YOLOv11 Large - Very high accuracy"
        },
        "yolov11x": {
            "name": "yolo11x.pt",
            "description": "YOLOv11 Extra Large - Maximum accuracy"
        },
        # YOLOv8 (Still good, proven)
        "yolov8n": {
            "name": "yolov8n.pt",
            "description": "YOLOv8 Nano - Fast and lightweight"
        },
        "yolov8s": {
            "name": "yolov8s.pt",
            "description": "YOLOv8 Small - Balanced"
        },
        "yolov8m": {
            "name": "yolov8m.pt",
            "description": "YOLOv8 Medium - More accurate"
        },
        "yolov8l": {
            "name": "yolov8l.pt",
            "description": "YOLOv8 Large - High accuracy"
        },
        "yolov8x": {
            "name": "yolov8x.pt",
            "description": "YOLOv8 Extra Large - Highest accuracy"
        }
    }

    # Training hyperparameters
    TRAIN_CONFIG = {
        "epochs": 100,
        "batch": 16,
        "imgsz": 640,
        "optimizer": "auto",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "label_smoothing": 0.0,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }

    # Validation settings
    VAL_CONFIG = {
        "batch": 16,
        "imgsz": 640,
        "conf": 0.001,
        "iou": 0.6,
        "max_det": 300,
        "split": "val"
    }

    # Prediction settings
    PREDICT_CONFIG = {
        "conf": 0.25,
        "iou": 0.45,
        "imgsz": 640,
        "max_det": 300,
        "half": False,
        "save": True,
        "save_txt": True,
        "save_conf": True,
        "save_crop": False,
        "show_labels": True,
        "show_conf": True,
        "line_width": 2
    }

    # Device settings
    DEVICE = "cuda:0"  # or "cpu", "mps" for Mac

    @classmethod
    def get_model_path(cls, model_name: str) -> str:
        """Get the model file path"""
        if model_name in cls.MODELS:
            return cls.MODELS[model_name]["name"]
        raise ValueError(
            f"Model {model_name} not found. Available: {list(cls.MODELS.keys())}")

    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_training_config(cls, **kwargs) -> Dict[str, Any]:
        """Get training configuration with optional overrides"""
        config = cls.TRAIN_CONFIG.copy()
        config.update(kwargs)
        return config

    @classmethod
    def get_validation_config(cls, **kwargs) -> Dict[str, Any]:
        """Get validation configuration with optional overrides"""
        config = cls.VAL_CONFIG.copy()
        config.update(kwargs)
        return config

    @classmethod
    def get_prediction_config(cls, **kwargs) -> Dict[str, Any]:
        """Get prediction configuration with optional overrides"""
        config = cls.PREDICT_CONFIG.copy()
        config.update(kwargs)
        return config
