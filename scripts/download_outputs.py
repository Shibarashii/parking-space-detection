#!/usr/bin/env python3
"""
Download training outputs as a zip file (for Google Colab)
"""

import shutil
import argparse
from pathlib import Path
from datetime import datetime


def download_all_outputs(experiment_name=None, output_path=None):
    """
    Download all training outputs as a single zip file

    Args:
        experiment_name: Specific experiment folder name, or None for all
        output_path: Custom output path for the zip file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if experiment_name:
        source_dir = Path(f"outputs/models/{experiment_name}")
        zip_name = f"{experiment_name}_{timestamp}.zip"
    else:
        source_dir = Path("outputs")
        zip_name = f"all_outputs_{timestamp}.zip"

    if not source_dir.exists():
        print(f"❌ Directory not found: {source_dir}")
        return

    # Set custom output path if provided
    if output_path:
        zip_name = str(Path(output_path) / zip_name)

    print(f"📦 Creating zip file: {zip_name}")
    print(f"📁 Source: {source_dir}")

    # Create archive
    shutil.make_archive(
        zip_name.replace('.zip', ''),
        'zip',
        source_dir.parent,
        source_dir.name
    )

    # Download in Colab environment
    try:
        from google.colab import files
        print(f"⬇️  Downloading {zip_name}...")
        files.download(zip_name)
        print("✅ Download complete!")
    except ImportError:
        print(f"✅ Zip created: {zip_name}")
        print("💡 Not in Colab - file saved locally")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Download YOLO training outputs as zip"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Specific experiment name to download (default: all outputs)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output directory for zip file"
    )

    args = parser.parse_args()
    download_all_outputs(args.experiment, args.output)


if __name__ == "__main__":
    main()
