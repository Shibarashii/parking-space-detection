"""
Convert mixed annotation formats to OBB (Oriented Bounding Box) format for YOLO
Handles: standard boxes, polygons, and mixed formats
"""
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import shutil


def box_to_obb(class_id, x_center, y_center, width, height):
    """
    Convert standard bounding box to OBB format (4 corners)

    Args:
        class_id: Class ID
        x_center, y_center: Center coordinates (normalized)
        width, height: Box dimensions (normalized)

    Returns:
        List with [class_id, x1, y1, x2, y2, x3, y3, x4, y4]
    """
    # Calculate corners (clockwise from top-left)
    x1, y1 = x_center - width/2, y_center - height/2  # top-left
    x2, y2 = x_center + width/2, y_center - height/2  # top-right
    x3, y3 = x_center + width/2, y_center + height/2  # bottom-right
    x4, y4 = x_center - width/2, y_center + height/2  # bottom-left

    return [class_id, x1, y1, x2, y2, x3, y3, x4, y4]


def polygon_to_obb(class_id, points):
    """
    Convert polygon points to OBB format
    Ensures we have exactly 4 corners

    Args:
        class_id: Class ID
        points: List of coordinate pairs

    Returns:
        List with [class_id, x1, y1, x2, y2, x3, y3, x4, y4]
    """
    if len(points) == 8:
        # Already 4 corners
        return [class_id] + points

    elif len(points) == 10:
        # 5 points with first point repeated - remove last point
        return [class_id] + points[:8]

    elif len(points) > 8:
        # More than 4 corners - fit minimum area rotated rectangle
        points_array = np.array(points).reshape(-1, 2)

        # Simple approach: use first 4 unique points
        unique_points = []
        for point in points_array:
            if len(unique_points) == 0 or not any(np.allclose(point, p) for p in unique_points):
                unique_points.append(point)
                if len(unique_points) == 4:
                    break

        if len(unique_points) == 4:
            coords = [coord for point in unique_points for coord in point]
            return [class_id] + coords
        else:
            # Fallback: use bounding box
            x_coords = points_array[:, 0]
            y_coords = points_array[:, 1]
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            return [class_id, x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]

    else:
        # Less than 4 corners - shouldn't happen, but handle it
        raise ValueError(f"Invalid polygon with {len(points)} coordinates")


def convert_annotation_line(line):
    """
    Convert a single annotation line to OBB format

    Args:
        line: Annotation line string

    Returns:
        OBB format line string
    """
    parts = line.strip().split()

    if len(parts) < 5:
        return None  # Invalid annotation

    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]

    # Determine format based on number of values
    if len(coords) == 4:
        # Standard box format: x_center y_center width height
        obb = box_to_obb(class_id, *coords)

    elif len(coords) >= 8:
        # Polygon format: x1 y1 x2 y2 x3 y3 x4 y4 [x1 y1]
        obb = polygon_to_obb(class_id, coords)

    else:
        print(f"Warning: Unknown format with {len(coords)} coordinates")
        return None

    # Format as string
    return ' '.join([str(int(obb[0]))] + [f"{x:.6f}" for x in obb[1:]])


def convert_label_file(input_path, output_path):
    """
    Convert a single label file to OBB format

    Args:
        input_path: Path to input label file
        output_path: Path to output label file
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()

    converted_lines = []
    for line in lines:
        if line.strip():
            converted_line = convert_annotation_line(line)
            if converted_line:
                converted_lines.append(converted_line + '\n')

    with open(output_path, 'w') as f:
        f.writelines(converted_lines)


def convert_dataset(data_root, backup=True):
    """
    Convert entire dataset to OBB format

    Args:
        data_root: Root directory of dataset
        backup: Whether to backup original labels
    """
    data_root = Path(data_root)

    # Find all label directories
    label_dirs = []
    for split in ['train', 'valid', 'val', 'test']:
        label_dir = data_root / split / 'labels'
        if label_dir.exists():
            label_dirs.append(label_dir)

    if not label_dirs:
        print("No label directories found!")
        return

    print(f"Found {len(label_dirs)} label directories")

    # Process each directory
    for label_dir in label_dirs:
        print(f"\nProcessing: {label_dir}")

        # Backup original labels
        if backup:
            backup_dir = label_dir.parent / 'labels_original'
            if not backup_dir.exists():
                print(f"Creating backup: {backup_dir}")
                shutil.copytree(label_dir, backup_dir)

        # Get all label files
        label_files = list(label_dir.glob('*.txt'))
        print(f"Found {len(label_files)} label files")

        # Convert each file
        successful = 0
        failed = 0

        for label_file in tqdm(label_files, desc="Converting"):
            try:
                convert_label_file(label_file, label_file)
                successful += 1
            except Exception as e:
                print(f"\nError converting {label_file.name}: {e}")
                failed += 1

        print(f"✓ Converted: {successful}")
        if failed > 0:
            print(f"✗ Failed: {failed}")

    print("\n" + "="*70)
    print("Conversion completed!")
    print("="*70)


def validate_obb_format(label_file):
    """
    Validate that a label file is in correct OBB format

    Args:
        label_file: Path to label file

    Returns:
        tuple: (is_valid, message)
    """
    with open(label_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        parts = line.strip().split()

        if len(parts) != 9:
            return False, f"Line {i}: Expected 9 values (class + 8 coords), got {len(parts)}"

        try:
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]

            # Check if coordinates are normalized (0-1)
            if not all(0 <= x <= 1 for x in coords):
                return False, f"Line {i}: Coordinates not normalized (should be 0-1)"

        except ValueError as e:
            return False, f"Line {i}: Invalid number format - {e}"

    return True, "Valid OBB format"


def validate_dataset(data_root):
    """
    Validate entire dataset is in OBB format

    Args:
        data_root: Root directory of dataset
    """
    data_root = Path(data_root)

    print("Validating OBB format...")
    print("="*70)

    for split in ['train', 'valid', 'val', 'test']:
        label_dir = data_root / split / 'labels'
        if not label_dir.exists():
            continue

        print(f"\n{split.upper()}:")
        label_files = list(label_dir.glob('*.txt'))

        valid_count = 0
        invalid_files = []

        for label_file in tqdm(label_files, desc=f"Validating {split}"):
            is_valid, message = validate_obb_format(label_file)
            if is_valid:
                valid_count += 1
            else:
                invalid_files.append((label_file.name, message))

        print(f"  Valid: {valid_count}/{len(label_files)}")

        if invalid_files:
            print(f"  Invalid files:")
            for filename, message in invalid_files[:5]:  # Show first 5
                print(f"    - {filename}: {message}")
            if len(invalid_files) > 5:
                print(f"    ... and {len(invalid_files) - 5} more")

    print("\n" + "="*70)


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Convert parking space annotations to OBB format"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Root directory of the dataset (e.g., /path/to/data)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backing up original labels"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate format without converting"
    )

    args = parser.parse_args()

    # Verify data root exists
    data_root_path = Path(args.data)
    if not data_root_path.exists():
        print(f"❌ Error: Data root not found: {args.data}")
        exit(1)

    print("="*70)
    print("OBB Annotation Converter")
    print("="*70)
    print(f"Data root: {args.data}")
    print("="*70)

    if args.validate_only:
        # Only validate
        validate_dataset(args.data)
    else:
        # Convert dataset
        convert_dataset(args.data, backup=not args.no_backup)

        # Validate conversion
        validate_dataset(args.data)

        print("\n✓ Ready for training with OBB model!")
        if not args.no_backup:
            print("  Original labels backed up to: */labels_original/")
