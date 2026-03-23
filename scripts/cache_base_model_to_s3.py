"""
Bootstrap script: download Qwen2.5-3B-Instruct from HuggingFace and cache to S3 Gold.

Run once before the first pipeline execution. The training step mounts this S3 path
as SM_CHANNEL_MODEL, so no HuggingFace access is needed at training time.

Usage:
    python scripts/cache_base_model_to_s3.py

Requirements:
    - AWS credentials configured with s3:PutObject on the Gold bucket
    - pip install huggingface_hub (already a transitive dep of transformers)
"""

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ID = "Qwen/Qwen2.5-3B-Instruct"
S3_BUCKET = "chitrakatha-gold-152141418178"
S3_PREFIX = "base-models/qwen2.5-3b-instruct"


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not found. Install with: pip install huggingface_hub")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Downloading {REPO_ID} to {tmp_dir} ...")
        local_path = snapshot_download(
            repo_id=REPO_ID,
            local_dir=tmp_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Download complete. Syncing to s3://{S3_BUCKET}/{S3_PREFIX}/ ...")
        result = subprocess.run(
            [
                "aws", "s3", "sync", local_path,
                f"s3://{S3_BUCKET}/{S3_PREFIX}/",
                "--sse", "aws:kms",
                "--no-progress",
            ],
            check=False,
        )
        if result.returncode != 0:
            print("ERROR: aws s3 sync failed. Check your AWS credentials and bucket permissions.")
            sys.exit(result.returncode)

    print(f"\nDone. Model cached at s3://{S3_BUCKET}/{S3_PREFIX}/")
    print("Verify with:")
    print(f"  aws s3 ls s3://{S3_BUCKET}/{S3_PREFIX}/ | head -20")


if __name__ == "__main__":
    main()
