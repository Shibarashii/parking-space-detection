"""
YOLO Detector Wrapper
Modular wrapper for YOLO object detection models
"""
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from ultralytics import YOLO
import torch
import yaml


class YOLODetector:
    """Wrapper class for YOLO object detection models"""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize YOLO detector

        Args:
            model_name: Name of the YOLO model or path to weights
            device: Device to run on ('cuda', 'cpu', 'mps'). Auto-detects if None.
            verbose: Whether to print verbose output
        """
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda:0"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.verbose = verbose
        self.model = None
        self.class_names = None

        self._load_model()

    def _load_model(self):
        """Load the YOLO model"""
        try:
            self.model = YOLO(self.model_name)
            if self.verbose:
                print(f"✓ Loaded model: {self.model_name}")
                print(f"✓ Device: {self.device}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model {self.model_name}: {str(e)}")

    def train(
        self,
        data_yaml: Union[str, Path],
        epochs: int = 100,
        batch: int = 16,
        imgsz: int = 640,
        project: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model

        Args:
            data_yaml: Path to data.yaml configuration
            epochs: Number of training epochs
            batch: Batch size
            imgsz: Image size
            project: Project directory
            name: Experiment name
            **kwargs: Additional training arguments

        Returns:
            Training results dictionary
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting Training: {name or 'experiment'}")
            print(f"{'='*60}\n")

        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=self.device,
            project=project,
            name=name,
            verbose=self.verbose,
            **kwargs
        )

        # Load class names from data.yaml
        self._load_class_names(data_yaml)

        if self.verbose:
            print(f"\n✓ Training completed!")
            print(f"✓ Model saved to: {results.save_dir}")

        return results

    def validate(
        self,
        data_yaml: Optional[Union[str, Path]] = None,
        split: str = "val",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate the model

        Args:
            data_yaml: Path to data.yaml (if not already set)
            split: Dataset split to validate on
            **kwargs: Additional validation arguments

        Returns:
            Validation results dictionary
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting Validation on {split} set")
            print(f"{'='*60}\n")

        val_args = {
            "device": self.device,
            "split": split,
            "verbose": self.verbose,
            **kwargs
        }

        if data_yaml:
            val_args["data"] = str(data_yaml)
            self._load_class_names(data_yaml)

        results = self.model.val(**val_args)

        if self.verbose:
            print(f"\n✓ Validation completed!")
            self._print_metrics(results)

        return results

    def predict(
        self,
        source: Union[str, Path, List],
        conf: float = 0.25,
        iou: float = 0.45,
        save: bool = True,
        save_txt: bool = True,
        save_conf: bool = True,
        project: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Make predictions on images/videos

        Args:
            source: Image path, directory, or list of images
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save: Save results
            save_txt: Save results to txt
            save_conf: Save confidence scores
            project: Project directory
            name: Experiment name
            **kwargs: Additional prediction arguments

        Returns:
            Prediction results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting Prediction")
            print(f"{'='*60}\n")

        # Remove verbose from kwargs if it exists to avoid duplicate
        kwargs.pop('verbose', None)

        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            device=self.device,
            save=save,
            save_txt=save_txt,
            save_conf=save_conf,
            project=project,
            name=name,
            verbose=self.verbose,
            **kwargs
        )

        if self.verbose:
            print(f"\n✓ Prediction completed!")
            if save and results:
                print(f"✓ Results saved to: {results[0].save_dir}")

        return results

    def export(
        self,
        format: str = "onnx",
        **kwargs
    ) -> str:
        """
        Export model to different formats

        Args:
            format: Export format (onnx, torchscript, tflite, etc.)
            **kwargs: Additional export arguments

        Returns:
            Path to exported model
        """
        if self.verbose:
            print(f"\nExporting model to {format} format...")

        export_path = self.model.export(format=format, **kwargs)

        if self.verbose:
            print(f"✓ Model exported to: {export_path}")

        return export_path

    def _load_class_names(self, data_yaml: Union[str, Path]):
        """Load class names from data.yaml"""
        try:
            with open(data_yaml, 'r') as f:
                data = yaml.safe_load(f)
                self.class_names = data.get('names', [])
        except Exception as e:
            print(f"Warning: Could not load class names: {str(e)}")

    def _print_metrics(self, results):
        """Print validation metrics"""
        if hasattr(results, 'box'):
            metrics = results.box
            print(f"\nMetrics Summary:")
            print(f"  mAP50-95: {metrics.map:.4f}")
            print(f"  mAP50:    {metrics.map50:.4f}")
            print(f"  mAP75:    {metrics.map75:.4f}")

            if hasattr(metrics, 'maps') and self.class_names:
                print(f"\nPer-Class mAP50-95:")
                for i, (cls_map, cls_name) in enumerate(zip(metrics.maps, self.class_names)):
                    print(f"  {cls_name:30s}: {cls_map:.4f}")

    def load_weights(self, weights_path: Union[str, Path]):
        """Load custom weights"""
        self.model = YOLO(str(weights_path))
        if self.verbose:
            print(f"✓ Loaded weights from: {weights_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "class_names": self.class_names,
            "parameters": sum(p.numel() for p in self.model.model.parameters()) if self.model else None
        }
