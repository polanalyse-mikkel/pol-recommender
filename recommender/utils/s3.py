import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _get_credentials():
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not aws_access_key_id or not aws_secret_access_key:
        raise EnvironmentError(
            "AWS credentials are not defined in the environment variables. "
            "Please set 'AWS_ACCESS_KEY_ID' and 'AWS_SECRET_ACCESS_KEY'."
        )

    return aws_access_key_id, aws_secret_access_key


def _get_bucket(bucket: str | None = None) -> str:
    """Get bucket name from argument or environment variable."""
    if bucket is not None:
        return bucket

    bucket_from_env = os.environ.get("AWS_S3_BUCKET")
    if not bucket_from_env:
        raise EnvironmentError(
            "Bucket name not provided and 'AWS_S3_BUCKET' environment variable is not set. "
            "Please provide a bucket name or set the 'AWS_S3_BUCKET' environment variable."
        )

    return bucket_from_env


def upload_to_s3(key: str | Path, body, bucket: str | None = None):
    """Uploads raw bytes to S3 under bucket/key."""
    bucket_name = _get_bucket(bucket)
    aws_access_key_id, aws_secret_access_key = _get_credentials()
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="eu-west-1",
    )
    response = s3.put_object(Bucket=bucket_name, Key=key, Body=body)
    if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
        raise Exception(
            f"Failed to upload to S3. HTTP Status Code: {response['ResponseMetadata']['HTTPStatusCode']}"
        )
    return response


def download_from_s3(key: str, bucket: str | None = None):
    """Downloads raw bytes from S3 under bucket/key."""
    bucket_name = _get_bucket(bucket)
    aws_access_key_id, aws_secret_access_key = _get_credentials()
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="eu-west-1",
    )
    response = s3.get_object(Bucket=bucket_name, Key=key)
    body = response["Body"].read()
    return body


def list_files_in_bucket(prefix: str = "", bucket: str | None = None):
    """Lists files in an S3 bucket."""
    bucket_name = _get_bucket(bucket)
    aws_access_key_id, aws_secret_access_key = _get_credentials()
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="eu-west-1",
    )

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if "Contents" in response:
        files = [item["Key"] for item in response["Contents"]]
    else:
        files = []

    return files


def delete_file_from_s3(key: str, bucket: str | None = None):
    """Deletes a file from S3 under bucket/key."""
    bucket_name = _get_bucket(bucket)
    aws_access_key_id, aws_secret_access_key = _get_credentials()
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="eu-west-1",
    )
    response = s3.delete_object(Bucket=bucket_name, Key=key)
    return response


def fetch_pending_recommendation_collections(
    bucket: str | None = None,
) -> list[tuple[str, bytes]]:
    """Fetches all files under the 'pending/' prefix in the S3 bucket and returns their keys and contents."""
    bucket_name = _get_bucket(bucket)
    aws_access_key_id, aws_secret_access_key = _get_credentials()
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="eu-west-1",
    )

    # List all files under the 'pending/' prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix="pending/")
    collections = []
    if "Contents" in response:
        for item in response["Contents"]:
            key = item["Key"]
            body = download_from_s3(key, bucket=bucket_name)
            collections.append((key, body))
    else:
        print("No files found under 'pending/' prefix.")

    if not isinstance(collections, list):
        raise ValueError(
            "Expected collections to be a list, got: {}".format(type(collections))
        )
    return collections


def delete_all_pending(bucket: str | None = None):
    """Deletes all files under the 'pending/' prefix in the S3 bucket."""
    bucket_name = _get_bucket(bucket)
    aws_access_key_id, aws_secret_access_key = _get_credentials()
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="eu-west-1",
    )

    # List all files under the 'pending/' prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix="pending/")
    if "Contents" in response:
        for item in response["Contents"]:
            key = item["Key"]
            s3.delete_object(Bucket=bucket_name, Key=key)
            print(f"Deleted {key} from bucket {bucket_name}")
    else:
        print("No files found under 'pending/' prefix.")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="S3 utility CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List files in bucket")
    list_parser.add_argument("--prefix", default="", help="Prefix to filter files")
    list_parser.add_argument("--bucket", help="S3 bucket name")

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload file to S3")
    upload_parser.add_argument("key", help="S3 key (path)")
    upload_parser.add_argument("file", help="Local file path to upload")
    upload_parser.add_argument("--bucket", help="S3 bucket name")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download file from S3")
    download_parser.add_argument("key", help="S3 key (path)")
    download_parser.add_argument("output", help="Local file path to save")
    download_parser.add_argument("--bucket", help="S3 bucket name")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete file from S3")
    delete_parser.add_argument("key", help="S3 key (path)")
    delete_parser.add_argument("--bucket", help="S3 bucket name")

    # Fetch pending command
    fetch_parser = subparsers.add_parser(
        "fetch-pending", help="Fetch pending recommendation collections"
    )
    fetch_parser.add_argument("--bucket", help="S3 bucket name")

    # Delete all pending command
    delete_pending_parser = subparsers.add_parser(
        "delete-pending", help="Delete all pending files"
    )
    delete_pending_parser.add_argument("--bucket", help="S3 bucket name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "list":
            files = list_files_in_bucket(prefix=args.prefix, bucket=args.bucket)
            for f in files:
                print(f)
            print(f"\nTotal: {len(files)} files")

        elif args.command == "upload":
            with open(args.file, "rb") as f:
                body = f.read()
            upload_to_s3(args.key, body, bucket=args.bucket)
            print(f"Uploaded {args.file} to s3://{args.bucket or 'default'}/{args.key}")

        elif args.command == "download":
            body = download_from_s3(args.key, bucket=args.bucket)
            with open(args.output, "wb") as f:
                f.write(body)
            print(f"Downloaded s3://{args.bucket or 'default'}/{args.key} to {args.output}")

        elif args.command == "delete":
            delete_file_from_s3(args.key, bucket=args.bucket)
            print(f"Deleted s3://{args.bucket or 'default'}/{args.key}")

        elif args.command == "fetch-pending":
            collections = fetch_pending_recommendation_collections(bucket=args.bucket)
            print(f"Fetched {len(collections)} pending collections:")
            for key, body in collections:
                print(f"  - {key} ({len(body)} bytes)")

        elif args.command == "delete-pending":
            delete_all_pending(bucket=args.bucket)

    except FileNotFoundError as e:
        print(f"File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except EnvironmentError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
