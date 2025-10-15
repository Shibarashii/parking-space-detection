#!/usr/bin/env python3
"""
Upload training outputs to Hugging Face Hub
"""

import argparse
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, login


def upload_to_huggingface(
    experiment_name=None,
    repo_id=None,
    token=None,
    repo_type="model",
    private=False,
    commit_message=None
):
    """
    Upload training outputs directly to Hugging Face Hub

    Args:
        experiment_name: Specific experiment folder name, or None for all outputs
        repo_id: Repository ID on HF Hub (e.g., "username/parking-detection-model")
        token: HuggingFace token with write access (or None to use saved token)
        repo_type: Type of repository ("model", "dataset", or "space")
        private: Whether to create a private repository
        commit_message: Custom commit message
    """

    # Login to Hugging Face
    print("🔐 Logging into Hugging Face...")
    try:
        if token:
            login(token=token)
        else:
            login()  # Will use saved token or prompt for login
    except Exception as e:
        print(f"❌ Login failed: {e}")
        print("💡 Run 'huggingface-cli login' first or provide token with --token")
        return

    # Determine source directory
    if experiment_name:
        source_dir = Path(f"outputs/models/{experiment_name}")
        default_repo_name = f"parking-detection-{experiment_name}"
    else:
        source_dir = Path("outputs")
        default_repo_name = "parking-detection-outputs"

    if not source_dir.exists():
        print(f"❌ Directory not found: {source_dir}")
        return

    # Set repository ID
    if repo_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        repo_id = f"{default_repo_name}-{timestamp}"
        print(f"⚠️  No repo_id provided. Using: {repo_id}")
        print(f"   Specify with: --repo-id username/repo-name")

    # Set commit message
    if commit_message is None:
        commit_message = f"Upload training outputs from {experiment_name or 'all experiments'}"

    # Initialize HF API
    api = HfApi()

    try:
        # Create repository if it doesn't exist
        print(f"📦 Creating/checking repository: {repo_id}")
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True
        )

        # Upload folder to Hub
        print(f"⬆️  Uploading {source_dir} to Hugging Face Hub...")
        print(f"📁 Source: {source_dir}")
        print(f"🌐 Destination: https://huggingface.co/{repo_id}")

        api.upload_folder(
            folder_path=str(source_dir),
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message
        )

        print(f"✅ Upload complete!")
        print(f"🔗 View your model at: https://huggingface.co/{repo_id}")

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("💡 Make sure you have:")
        print("   1. A Hugging Face account")
        print("   2. Generated a write token at https://huggingface.co/settings/tokens")
        print("   3. Run 'pip install huggingface_hub' if not installed")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Upload YOLO training outputs to Hugging Face Hub"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Specific experiment name to upload (default: all outputs)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        required=False,
        help="Repository ID (e.g., 'username/parking-detection-yolo11s')"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token with write access"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="model",
        choices=["model", "dataset", "space"],
        help="Type of repository (default: model)"
    )
    parser.add_argument(
        "--message",
        type=str,
        default=None,
        help="Custom commit message"
    )

    args = parser.parse_args()

    upload_to_huggingface(
        experiment_name=args.experiment,
        repo_id=args.repo_id,
        token=args.token,
        repo_type=args.repo_type,
        private=args.private,
        commit_message=args.message
    )


if __name__ == "__main__":
    main()
