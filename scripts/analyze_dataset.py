"""
Dataset Analysis Script
Analyze your dataset to determine optimal training configuration
"""
import sys
from pathlib import Path
from collections import defaultdict
import yaml
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def analyze_dataset(data_yaml_path):
    """Analyze dataset and provide training recommendations"""

    # Load data config
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    data_root = Path(data_config['path'])
    class_names = data_config['names']

    print("="*80)
    print("DATASET ANALYSIS")
    print("="*80)

    # Analyze each split
    splits = ['train', 'val', 'test']
    split_stats = {}

    for split in splits:
        split_name = 'valid' if split == 'val' else split
        labels_dir = data_root / 'labels' / split_name
        images_dir = data_root / 'images' / split_name

        if not labels_dir.exists():
            print(
                f"\n⚠ Warning: {split} labels directory not found: {labels_dir}")
            continue

        # Count images and labels
        label_files = list(labels_dir.glob('*.txt'))
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend(images_dir.glob(f'*{ext}'))

        # Class distribution
        class_counts = defaultdict(int)
        total_objects = 0
        bbox_sizes = []
        aspect_ratios = []
        objects_per_image = []

        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                objects_per_image.append(len(lines))

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        class_counts[cls_id] += 1
                        total_objects += 1

                        # Parse bbox (YOLO format: x_center, y_center, width, height)
                        w, h = float(parts[3]), float(parts[4])
                        bbox_sizes.append((w, h))
                        aspect_ratios.append(w / h if h > 0 else 1.0)

        # Image size analysis
        image_sizes = []
        if image_files:
            sample_size = min(100, len(image_files))
            for img_file in image_files[:sample_size]:
                img = cv2.imread(str(img_file))
                if img is not None:
                    image_sizes.append(
                        (img.shape[1], img.shape[0]))  # width, height

        split_stats[split] = {
            'num_images': len(image_files),
            'num_labels': len(label_files),
            'total_objects': total_objects,
            'class_counts': dict(class_counts),
            'objects_per_image': objects_per_image,
            'bbox_sizes': bbox_sizes,
            'aspect_ratios': aspect_ratios,
            'image_sizes': image_sizes
        }

    # Print statistics
    print(
        f"\n{'Split':<10} {'Images':<10} {'Labels':<10} {'Objects':<10} {'Obj/Img':<10}")
    print("-" * 80)
    for split in splits:
        if split in split_stats:
            stats = split_stats[split]
            avg_obj = np.mean(stats['objects_per_image']
                              ) if stats['objects_per_image'] else 0
            print(f"{split:<10} {stats['num_images']:<10} {stats['num_labels']:<10} "
                  f"{stats['total_objects']:<10} {avg_obj:<10.2f}")

    # Class distribution
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION")
    print("="*80)

    if 'train' in split_stats:
        train_stats = split_stats['train']
        print(
            f"\n{'Class ID':<10} {'Class Name':<35} {'Count':<10} {'Percentage':<10}")
        print("-" * 80)

        total = train_stats['total_objects']
        for cls_id, count in sorted(train_stats['class_counts'].items()):
            cls_name = class_names[cls_id] if cls_id < len(
                class_names) else f"Class {cls_id}"
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{cls_id:<10} {cls_name:<35} {count:<10} {percentage:<10.2f}%")

    # Image size analysis
    print("\n" + "="*80)
    print("IMAGE SIZE ANALYSIS")
    print("="*80)

    if 'train' in split_stats and split_stats['train']['image_sizes']:
        sizes = split_stats['train']['image_sizes']
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]

        print(
            f"\nAverage size: {np.mean(widths):.0f} x {np.mean(heights):.0f}")
        print(f"Min size: {np.min(widths):.0f} x {np.min(heights):.0f}")
        print(f"Max size: {np.max(widths):.0f} x {np.max(heights):.0f}")
        print(
            f"Median size: {np.median(widths):.0f} x {np.median(heights):.0f}")

    # Object size analysis
    print("\n" + "="*80)
    print("OBJECT SIZE ANALYSIS")
    print("="*80)

    if 'train' in split_stats and split_stats['train']['bbox_sizes']:
        bbox_sizes = split_stats['train']['bbox_sizes']
        widths = [s[0] for s in bbox_sizes]
        heights = [s[1] for s in bbox_sizes]
        areas = [w * h for w, h in bbox_sizes]

        print(
            f"\nAverage bbox size: {np.mean(widths):.3f} x {np.mean(heights):.3f} (normalized)")
        print(f"Average bbox area: {np.mean(areas):.3f} (normalized)")
        print(
            f"Small objects (<0.02 area): {sum(1 for a in areas if a < 0.02)} ({sum(1 for a in areas if a < 0.02)/len(areas)*100:.1f}%)")
        print(
            f"Medium objects (0.02-0.1 area): {sum(1 for a in areas if 0.02 <= a < 0.1)} ({sum(1 for a in areas if 0.02 <= a < 0.1)/len(areas)*100:.1f}%)")
        print(
            f"Large objects (>0.1 area): {sum(1 for a in areas if a >= 0.1)} ({sum(1 for a in areas if a >= 0.1)/len(areas)*100:.1f}%)")

    # Recommendations
    print("\n" + "="*80)
    print("TRAINING RECOMMENDATIONS")
    print("="*80)

    recommendations = generate_recommendations(split_stats, class_names)

    for category, recs in recommendations.items():
        print(f"\n{category}:")
        for rec in recs:
            print(f"  • {rec}")

    # Generate training command
    print("\n" + "="*80)
    print("RECOMMENDED TRAINING COMMANDS")
    print("="*80)

    generate_training_commands(split_stats, recommendations)


def generate_recommendations(split_stats, class_names):
    """Generate training recommendations based on dataset analysis"""
    recs = defaultdict(list)

    if 'train' not in split_stats:
        return recs

    train_stats = split_stats['train']
    num_images = train_stats['num_images']
    total_objects = train_stats['total_objects']
    class_counts = train_stats['class_counts']

    # Dataset size recommendations
    if num_images < 500:
        recs['Model Selection'].append("Small dataset: Use YOLOv8n or YOLOv8s")
        recs['Model Selection'].append("Consider data augmentation heavily")
        recs['Training'].append("Use pretrained weights (default)")
        recs['Training'].append("Lower learning rate: --lr0 0.001")
    elif num_images < 2000:
        recs['Model Selection'].append(
            "Medium dataset: Use YOLOv8s or YOLOv8m")
        recs['Training'].append("Standard augmentation should work well")
    else:
        recs['Model Selection'].append(
            "Large dataset: YOLOv8m or YOLOv8l recommended")
        recs['Training'].append("Can use higher learning rates")

    # Class imbalance
    if class_counts:
        counts = list(class_counts.values())
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / \
            min_count if min_count > 0 else float('inf')

        if imbalance_ratio > 10:
            recs['Class Imbalance'].append(
                f"High imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
            recs['Class Imbalance'].append(
                "Consider using class weights or oversampling")
            recs['Class Imbalance'].append(
                "Use --close-mosaic 0 to reduce mosaic impact on rare classes")
        elif imbalance_ratio > 3:
            recs['Class Imbalance'].append(
                f"Moderate imbalance (ratio: {imbalance_ratio:.1f}:1)")
            recs['Augmentation'].append(
                "Increase augmentation for better generalization")

    # Objects per image
    if train_stats['objects_per_image']:
        avg_objects = np.mean(train_stats['objects_per_image'])

        if avg_objects < 2:
            recs['Training'].append(
                "Few objects per image: Standard batch size (16-32)")
        else:
            recs['Training'].append(
                f"Multiple objects per image (avg: {avg_objects:.1f})")
            recs['Training'].append(
                "Consider larger batch size if GPU allows (32-64)")

    # Object size recommendations
    if train_stats['bbox_sizes']:
        areas = [w * h for w, h in train_stats['bbox_sizes']]
        small_pct = sum(1 for a in areas if a < 0.02) / len(areas) * 100

        if small_pct > 30:
            recs['Image Size'].append(f"Many small objects ({small_pct:.1f}%)")
            recs['Image Size'].append(
                "Use larger image size: --imgsz 1280 or 1024")
        else:
            recs['Image Size'].append("Standard image size works: --imgsz 640")

    # Epochs recommendation
    if num_images < 500:
        recs['Training'].append("Small dataset: 150-300 epochs recommended")
        recs['Training'].append("Use early stopping: --patience 50")
    elif num_images < 2000:
        recs['Training'].append("Medium dataset: 100-200 epochs")
        recs['Training'].append("Use early stopping: --patience 50")
    else:
        recs['Training'].append(
            "Large dataset: 50-100 epochs usually sufficient")
        recs['Training'].append("Use early stopping: --patience 30")

    # Batch size
    recs['Batch Size'].append("Start with --batch 16")
    recs['Batch Size'].append("Increase to 32 or 64 if GPU memory allows")
    recs['Batch Size'].append("Reduce to 8 if you get OOM errors")

    # Augmentation
    recs['Augmentation'].append("Always use data augmentation")
    recs['Augmentation'].append("Standard: --mosaic 1.0 --mixup 0.0")
    if num_images < 1000:
        recs['Augmentation'].append(
            "Small dataset: --mixup 0.15 for extra augmentation")

    return recs


def generate_training_commands(split_stats, recommendations):
    """Generate specific training commands"""

    if 'train' not in split_stats:
        return

    num_images = split_stats['train']['num_images']

    print("\n1. QUICK TEST (Fast training to verify setup):")
    print("-" * 80)
    print("""python scripts/train.py \\
    --model yolov8n \\
    --data config/data.yaml \\
    --epochs 10 \\
    --batch 16 \\
    --imgsz 640 \\
    --name quick_test""")

    print("\n2. BASELINE TRAINING (Good starting point):")
    print("-" * 80)

    if num_images < 500:
        cmd = """python scripts/train.py \\
    --model yolov8n \\
    --data config/data.yaml \\
    --epochs 200 \\
    --batch 16 \\
    --imgsz 640 \\
    --lr0 0.001 \\
    --patience 50 \\
    --mosaic 1.0 \\
    --mixup 0.15 \\
    --name baseline_small_dataset"""
    elif num_images < 2000:
        cmd = """python scripts/train.py \\
    --model yolov8s \\
    --data config/data.yaml \\
    --epochs 150 \\
    --batch 32 \\
    --imgsz 640 \\
    --lr0 0.01 \\
    --patience 50 \\
    --mosaic 1.0 \\
    --name baseline_medium_dataset"""
    else:
        cmd = """python scripts/train.py \\
    --model yolov8m \\
    --data config/data.yaml \\
    --epochs 100 \\
    --batch 32 \\
    --imgsz 640 \\
    --lr0 0.01 \\
    --patience 30 \\
    --mosaic 1.0 \\
    --name baseline_large_dataset"""

    print(cmd)

    print("\n3. PRODUCTION TRAINING (Best accuracy):")
    print("-" * 80)

    if num_images < 500:
        cmd = """python scripts/train.py \\
    --model yolov8s \\
    --data config/data.yaml \\
    --epochs 300 \\
    --batch 16 \\
    --imgsz 640 \\
    --lr0 0.001 \\
    --patience 100 \\
    --mosaic 1.0 \\
    --mixup 0.2 \\
    --save-period 20 \\
    --name production_v1"""
    else:
        cmd = """python scripts/train.py \\
    --model yolov8m \\
    --data config/data.yaml \\
    --epochs 150 \\
    --batch 32 \\
    --imgsz 640 \\
    --lr0 0.01 \\
    --patience 50 \\
    --mosaic 1.0 \\
    --save-period 20 \\
    --name production_v1"""

    print(cmd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze dataset and get training recommendations")
    parser.add_argument("--data", type=str,
                        default="config/data.yaml", help="Path to data.yaml")
    args = parser.parse_args()

    analyze_dataset(args.data)
