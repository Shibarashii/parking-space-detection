"""
Download a model (weights, configs, etc.) from Hugging Face Hub.

Usage:
    python download_model.py --repo-id Shibarashii/nail-feature-detection --subdir outputs/models/nail_v2_test2 --local-dir ./models
"""

from huggingface_hub import snapshot_download
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Download YOLO/PyTorch model from Hugging Face")
    parser.add_argument("--repo-id", type=str, default="",
                        help="Hugging Face repo ID, e.g., /")
    parser.add_argument("--subdir", type=str, default="models",
                        help="Subdirectory inside repo (e.g., 'outputs/models/nail_v2_test2')")
    parser.add_argument("--local-dir", type=str, default="outputs/",
                        help="Where to store the downloaded files")
    args = parser.parse_args()

    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Downloading from {args.repo_id}/{args.subdir or '[root]'} ...")
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=local_dir,
        repo_type="model",
        allow_patterns=[f"{args.subdir}/*"] if args.subdir else None,
    )

    print(f"✅ Model files downloaded to: {path}")


if __name__ == "__main__":
    main()
