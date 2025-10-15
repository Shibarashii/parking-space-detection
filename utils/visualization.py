"""
Visualization utilities for object detection
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import seaborn as sns


class Visualizer:
    """Visualization utilities for object detection results"""

    def __init__(self, class_names: List[str]):
        """
        Initialize visualizer

        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.colors = self._generate_colors(len(class_names))

    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for each class"""
        colors = []
        for i in range(n):
            hue = i / n
            rgb = plt.cm.hsv(hue)[:3]
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors

    def draw_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        classes: np.ndarray,
        scores: np.ndarray,
        show_labels: bool = True,
        show_conf: bool = True,
        line_width: int = 2
    ) -> np.ndarray:
        """
        Draw bounding boxes on image

        Args:
            image: Input image (BGR or RGB)
            boxes: Bounding boxes in [x1, y1, x2, y2] format
            classes: Class indices
            scores: Confidence scores
            show_labels: Whether to show class labels
            show_conf: Whether to show confidence scores
            line_width: Line width for boxes

        Returns:
            Image with drawn boxes
        """
        img = image.copy()

        for box, cls, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = map(int, box)
            cls_idx = int(cls)

            # Get color for this class
            color = self.colors[cls_idx]

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)

            # Prepare label
            if show_labels or show_conf:
                label_parts = []
                if show_labels and cls_idx < len(self.class_names):
                    label_parts.append(self.class_names[cls_idx])
                if show_conf:
                    label_parts.append(f"{score:.2f}")

                label = " ".join(label_parts)

                # Draw label background
                (label_w, label_h), _ = cv2.getTextAndSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    img,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w, y1),
                    color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

        return img

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Plot confusion matrix

        Args:
            confusion_matrix: Confusion matrix
            save_path: Path to save the plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_pr_curve(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        class_idx: Optional[int] = None,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot precision-recall curve

        Args:
            precision: Precision values
            recall: Recall values
            class_idx: Class index (None for overall)
            save_path: Path to save the plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        if class_idx is not None and class_idx < len(self.class_names):
            plt.title(f'Precision-Recall Curve: {self.class_names[class_idx]}')
        else:
            plt.title('Precision-Recall Curve')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_class_distribution(
        self,
        class_counts: Dict[str, int],
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot class distribution

        Args:
            class_counts: Dictionary of class names and counts
            save_path: Path to save the plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)

        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        plt.bar(range(len(classes)), counts, color=self.colors[:len(classes)])
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
