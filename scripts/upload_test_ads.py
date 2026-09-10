"""
Upload the test ad creatives to S3 so Vision Lab has something real to analyse.

    python scripts/upload_test_ads.py            # upload Ads_Video/*.mp4
    python scripts/upload_test_ads.py --list     # what is already up there
    python scripts/upload_test_ads.py --dry-run

WHERE THEY GO
    s3://<AWS_S3_BUCKET>/vision-lab-test/

    Deliberately NOT under AWS_S3_PREFIX (`admin`), which is the main app's own
    prefix for product media. Our test creatives and, later, our heatmap output
    belong in paths of our own so nobody has to guess which artefacts are whose,
    and so a lifecycle rule can be applied to ours alone.

WHY NOT KEEP THEM IN THE REPO
    Four ads are ~600 MB. Git cannot cleanly forget a pushed binary that size, so
    Ads_Video/ is gitignored and the videos live in S3 like every other creative.

MULTIPART IS AUTOMATIC
    boto3's upload_file switches to multipart above 8 MB and retries parts on its
    own, which matters on a flaky connection with a 180 MB file.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dotenv import load_dotenv  # noqa: E402

try:
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    sys.exit("boto3 is not installed. Run: pip install -r requirements-dev.txt")

load_dotenv(os.path.join(REPO, ".env"))

BUCKET = os.environ.get("AWS_S3_BUCKET")
REGION = os.environ.get("AWS_REGION")
DEST_PREFIX = os.environ.get("VL_TEST_PREFIX", "vision-lab-test")
SOURCE_DIR = os.path.join(REPO, "Ads_Video")

CONTENT_TYPES = {
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp",
}


class Progress:
    """One line per file, overwritten in place."""

    def __init__(self, name: str, total: int):
        self.name = name
        self.total = total
        self.seen = 0
        self.lock = threading.Lock()

    def __call__(self, chunk: int) -> None:
        with self.lock:
            self.seen += chunk
            pct = (self.seen / self.total) * 100 if self.total else 100
            sys.stdout.write(
                f"\r  {self.name:<14} {self.seen / 1048576:>7.1f} / "
                f"{self.total / 1048576:.1f} MB  {pct:5.1f}%")
            sys.stdout.flush()


def client():
    missing = [n for n in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                           "AWS_S3_BUCKET", "AWS_REGION") if not os.environ.get(n)]
    if missing:
        sys.exit(f"Missing in .env: {', '.join(missing)}")
    return boto3.client("s3", region_name=REGION)


def already_there(s3, key: str, size: int) -> bool:
    """Same key, same byte count - do not spend the bandwidth again."""
    try:
        head = s3.head_object(Bucket=BUCKET, Key=key)
        return head["ContentLength"] == size
    except ClientError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="re-upload even if the object already matches")
    args = parser.parse_args()

    s3 = client()

    if args.list:
        response = s3.list_objects_v2(Bucket=BUCKET, Prefix=DEST_PREFIX + "/")
        contents = response.get("Contents") or []
        if not contents:
            print(f"Nothing under s3://{BUCKET}/{DEST_PREFIX}/")
            return 0
        print(f"s3://{BUCKET}/{DEST_PREFIX}/")
        for item in contents:
            print(f"  {item['Key']:<45} {item['Size'] / 1048576:>8.1f} MB")
        return 0

    if not os.path.isdir(SOURCE_DIR):
        sys.exit(f"No such directory: {SOURCE_DIR}")

    files = sorted(f for f in os.listdir(SOURCE_DIR)
                   if os.path.splitext(f)[1].lower() in CONTENT_TYPES)
    if not files:
        sys.exit(f"No media files in {SOURCE_DIR}")

    total = sum(os.path.getsize(os.path.join(SOURCE_DIR, f)) for f in files)
    print(f"\n{len(files)} file(s), {total / 1048576:.0f} MB total")
    print(f"  -> s3://{BUCKET}/{DEST_PREFIX}/\n")

    config = TransferConfig(multipart_threshold=8 * 1024 * 1024,
                            max_concurrency=4)

    for name in files:
        path = os.path.join(SOURCE_DIR, name)
        size = os.path.getsize(path)
        key = f"{DEST_PREFIX}/{name}"
        content_type = CONTENT_TYPES[os.path.splitext(name)[1].lower()]

        if args.dry_run:
            print(f"  would upload {name:<14} {size / 1048576:>7.1f} MB  as {content_type}")
            continue

        if not args.force and already_there(s3, key, size):
            print(f"  {name:<14} already uploaded, skipping")
            continue

        try:
            s3.upload_file(path, BUCKET, key,
                           ExtraArgs={"ContentType": content_type},
                           Callback=Progress(name, size), Config=config)
            print("  done")
        except (ClientError, NoCredentialsError) as err:
            print()
            sys.exit(f"Upload failed for {name}: {err}")

    if args.dry_run:
        return 0

    print(f"\nPresign one with:\n"
          f"  python scripts/presign.py {DEST_PREFIX}/{files[0]}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
