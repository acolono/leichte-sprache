#!/usr/bin/env python3
"""Download ML models from GitHub Releases and verify checksums."""

import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "MODEL_MANIFEST.json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNK_SIZE = 65536  # 64 KB


def load_manifest() -> dict:
    """Load and return parsed MODEL_MANIFEST.json."""
    if not MANIFEST_PATH.exists():
        print(
            f"Error: MODEL_MANIFEST.json not found at {MANIFEST_PATH}. "
            "Are you in the project root?"
        )
        sys.exit(1)
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def build_download_url(repo: str, tag: str, tarball_name: str) -> str:
    """Return the GitHub Releases download URL for the tarball."""
    return f"https://github.com/{repo}/releases/download/{tag}/{tarball_name}"


def download_file(url: str, dest: Path) -> None:
    """Download URL to dest path with a Rich progress bar."""
    try:
        response = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(
                "Error: Release not found. Has the release been created? "
                "Run: scripts/upload_release.sh"
            )
            sys.exit(1)
        raise
    except urllib.error.URLError as e:
        print(f"Network error: {e}. Check your internet connection.")
        sys.exit(1)

    total_size = int(response.headers.get("Content-Length", 0))

    with Progress(
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Downloading models", total=total_size)
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                progress.update(task, advance=len(chunk))


def verify_checksum(filepath: Path, expected_sha256: str) -> bool:
    """Compute SHA256 of file and compare to expected hash."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest() == expected_sha256


def check_existing_files(manifest: dict) -> bool:
    """Check if all model files exist locally with valid checksums."""
    all_valid = True
    for filepath, info in manifest["files"].items():
        full_path = PROJECT_ROOT / filepath
        if not full_path.exists():
            print(f"  MISSING: {filepath}")
            all_valid = False
        elif verify_checksum(full_path, info["sha256"]):
            print(f"  OK:      {filepath}")
        else:
            print(f"  INVALID: {filepath} (checksum mismatch)")
            all_valid = False
    return all_valid


def extract_tarball(tarball_path: Path, project_root: Path) -> None:
    """Extract tarball to project root directory."""
    print("Extracting models...")
    with tarfile.open(tarball_path, "r:gz") as tar:
        for member in tar.getmembers():
            # Safety: prevent path traversal
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                print(f"  SKIPPED (unsafe path): {member.name}")
                continue
            print(f"  {member.name}")
        # Reset and extract
        tar.extractall(path=project_root, filter="data")


def verify_all_files(manifest: dict) -> bool:
    """Verify all model files exist and checksums match after extraction."""
    print("Verifying checksums...")
    all_valid = True
    for filepath, info in manifest["files"].items():
        full_path = PROJECT_ROOT / filepath
        if not full_path.exists():
            print(f"  FAIL: {filepath} (not found after extraction)")
            all_valid = False
        elif verify_checksum(full_path, info["sha256"]):
            print(f"  PASS: {filepath}")
        else:
            print(f"  FAIL: {filepath} (checksum mismatch)")
            all_valid = False
    return all_valid


def main() -> None:
    """Download, extract, and verify ML models from GitHub Releases."""
    parser = argparse.ArgumentParser(
        description="Download ML models from GitHub Releases and verify checksums."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files exist and checksums match",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="GitHub repo override (default: from MODEL_MANIFEST.json)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Release tag override (default: from MODEL_MANIFEST.json)",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    repo = args.repo or manifest["github_repo"]
    tag = args.tag or manifest["github_release_tag"]
    tarball_name = manifest.get("tarball_name", "models.tar.gz")

    # Check existing files unless --force
    if not args.force:
        print("Checking existing model files...")
        if check_existing_files(manifest):
            print("\nAll models present and verified. Use --force to re-download.")
            sys.exit(0)
        print()

    url = build_download_url(repo, tag, tarball_name)
    print(f"Downloading from: {url}")

    # Download to temp file
    tmp_file = None
    try:
        tmp_fd = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
        tmp_file = Path(tmp_fd.name)
        tmp_fd.close()

        download_file(url, tmp_file)

        extract_tarball(tmp_file, PROJECT_ROOT)
    except KeyboardInterrupt:
        print("\nDownload interrupted.")
        sys.exit(1)
    finally:
        if tmp_file and tmp_file.exists():
            tmp_file.unlink()

    # Verify after extraction
    if verify_all_files(manifest):
        print("\nAll models downloaded and verified successfully.")
        sys.exit(0)
    else:
        print("\nError: Some model files failed verification after download.")
        sys.exit(1)


if __name__ == "__main__":
    main()
