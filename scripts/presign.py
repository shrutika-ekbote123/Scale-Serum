"""
Generate a presigned S3 GET URL for testing Vision Lab in Postman.

WHY THIS EXISTS
    The AWS CLI reads ~/.aws/credentials or real environment variables. It does
    NOT read this repo's .env file, so credentials sitting in .env do nothing for
    `aws s3 presign`. This reads .env the same way app.py does and signs with
    boto3, so there is one place credentials live and no second setup to keep in
    step.

WHAT PRESIGNING ACTUALLY IS
    A local HMAC computation. No network call is made, nothing is uploaded, and
    AWS is not contacted - which also means it will happily sign a URL for an
    object that does not exist. You find that out with a 404 when something
    finally fetches it.

USAGE
    python scripts/presign.py test/di_board_seat_v4_final.mp4
    python scripts/presign.py test/ad.mp4 --expires 7200
    python scripts/presign.py --list                 # what is in the bucket
    python scripts/presign.py test/ad.mp4 --quiet    # URL only, for piping

THE URL IS A CREDENTIAL
    Anyone holding it can read that object until it expires. Do not paste it into
    a ticket, a commit, or a chat log. Regenerate rather than share.
"""
from __future__ import annotations

import argparse
import os
import sys
from urllib.parse import urlparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit("python-dotenv is not installed. Run: pip install -r requirements.txt")

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
except ImportError:
    sys.exit("boto3 is not installed. Run: pip install -r requirements-dev.txt")

load_dotenv(os.path.join(REPO, ".env"))

BUCKET = os.environ.get("AWS_S3_BUCKET")
REGION = os.environ.get("AWS_REGION")
PREFIX = os.environ.get("AWS_S3_PREFIX", "")
DEFAULT_TTL = int(os.environ.get("AWS_S3_URL_TTL_SECONDS", 3600))


def client():
    missing = [name for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                                 "AWS_S3_BUCKET", "AWS_REGION")
               if not os.environ.get(name)]
    if missing:
        sys.exit(f"Missing in .env: {', '.join(missing)}")
    return boto3.client("s3", region_name=REGION)


def resolve_key(s3, key: str) -> tuple[str, bool]:
    """Use the key exactly as given; fall back to AWS_S3_PREFIX only if that
    misses.

    AWS_S3_PREFIX is the MAIN APP's prefix (`admin`), not ours - Vision Lab's own
    objects live under vision-lab-test/ and vision-lab/. Prepending it silently
    would make `vision-lab-test/Ads1.mp4` resolve to `admin/vision-lab-test/...`
    and 404. So the literal key wins, and the prefix is a fallback that says so.
    """
    key = key.lstrip("/")
    if exists(s3, key):
        return key, True
    if PREFIX:
        prefixed = f"{PREFIX.rstrip('/')}/{key}"
        if exists(s3, prefixed):
            print(f"  (not found as '{key}' - using '{prefixed}')", file=sys.stderr)
            return prefixed, True
    return key, False


def exists(s3, key: str) -> bool:
    """presign never checks, so a typo signs cleanly and 404s later."""
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError:
        return False


def check_region(s3) -> None:
    """A presign signed against the wrong region fails with a signature mismatch
    rather than anything that names the real problem. Catch it here instead."""
    try:
        location = s3.get_bucket_location(Bucket=BUCKET).get("LocationConstraint")
        actual = location or "us-east-1"
        if actual != REGION:
            print(f"  WARNING: AWS_REGION is {REGION} but the bucket is in {actual}.",
                  file=sys.stderr)
            print(f"           Set AWS_REGION={actual} in .env or the URL will not work.",
                  file=sys.stderr)
    except (ClientError, BotoCoreError):
        pass          # not fatal - presigning is local and may still be correct


def list_objects(s3, prefix: str = "", limit: int = 100) -> None:
    kwargs = {"Bucket": BUCKET, "MaxKeys": limit}
    if prefix:
        kwargs["Prefix"] = prefix
    try:
        response = s3.list_objects_v2(**kwargs)
    except (ClientError, NoCredentialsError) as err:
        sys.exit(f"Could not list the bucket: {err}")

    contents = response.get("Contents") or []
    if not contents:
        print(f"No objects under s3://{BUCKET}/{prefix}")
        return
    print(f"s3://{BUCKET}/{prefix}")
    for item in contents:
        size_mb = item["Size"] / (1024 * 1024)
        print(f"  {item['Key']:<60} {size_mb:>8.1f} MB   {item['LastModified']:%Y-%m-%d}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("key", nargs="?", help="object key, e.g. test/ad.mp4")
    parser.add_argument("--expires", type=int, default=DEFAULT_TTL,
                        help=f"seconds until the URL dies (default {DEFAULT_TTL})")
    parser.add_argument("--list", action="store_true", help="list objects and exit")
    parser.add_argument("--prefix", default="", help="limit --list to this prefix")
    parser.add_argument("--quiet", action="store_true", help="print only the URL")
    args = parser.parse_args()

    s3 = client()

    if args.list:
        list_objects(s3, args.prefix)
        return 0
    if not args.key:
        parser.error("give an object key, or --list to see what is there")

    check_region(s3)
    key, found = resolve_key(s3, args.key)

    url = s3.generate_presigned_url(
        "get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=args.expires)

    if args.quiet:
        print(url)
        return 0

    host = urlparse(url).hostname
    print()
    print(f"  object   s3://{BUCKET}/{key}")
    print(f"  exists   {'yes' if found else 'NO - this URL will 404 when fetched'}")
    print(f"  expires  in {args.expires}s ({args.expires // 60} min)")
    print()
    print("  Put this host in .env so the URL passes the SSRF allowlist:")
    print(f"    VL_ALLOWED_URL_HOSTS={host}")
    print()
    print("  Paste this into Postman's `creative_url` (Current value):")
    print()
    print(url)
    print()
    print("  This URL is a credential. It expires, but do not paste it into a")
    print("  ticket or a commit - regenerate instead of sharing.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
