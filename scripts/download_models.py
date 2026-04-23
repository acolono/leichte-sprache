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


def _build_request(url: str, token: str | None) -> urllib.request.Request:
    """Build an authenticated request for GitHub release assets.

    Release assets on private repos need `Accept: application/octet-stream`
    plus a bearer token; `api.github.com/repos/.../releases/assets/{id}` is
    the auth-safe endpoint. The public `github.com/.../releases/download/...`
    URL redirects to signed S3, but authenticated clients must hit the API
    endpoint directly — see https://docs.github.com/en/rest/releases/assets.
    """
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/octet-stream")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
    return req


def _resolve_asset_api_url(repo: str, tag: str, tarball_name: str, token: str) -> str:
    """Translate (repo, tag, asset-name) into the asset API URL for private repos."""
    api_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    req = urllib.request.Request(api_url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    for asset in data.get("assets", []):
        if asset["name"] == tarball_name:
            return asset["url"]  # canonical asset API URL
    raise RuntimeError(
        f"Asset {tarball_name!r} not found in release {tag} on {repo}"
    )


def download_file(url: str, dest: Path, token: str | None = None) -> None:
    """Download URL to dest path with a Rich progress bar.

    If a token is provided and the initial URL is the public download URL,
    resolve to the API asset URL (which accepts bearer auth for private
    repos). The public URL returns 404 for anonymous clients on private
    repos, and 302 → S3 for public or authenticated clients.
    """
    try:
        req = _build_request(url, token)
        response = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            if not token:
                print(
                    "Error: Release not found. The repo may be private — "
                    "set GITHUB_TOKEN (or GH_TOKEN) with contents:read and retry. "
                    "If the repo is public, the release may not have been created yet "
                    "(run scripts/upload_release.sh)."
                )
            else:
                print(
                    "Error: Release not found even with auth. Check the tag "
                    "and repo slug in MODEL_MANIFEST.json."
                )
            sys.exit(1)
        if e.code in (401, 403):
            print(
                f"Error: GitHub returned {e.code}. The GITHUB_TOKEN may be "
                "missing, expired, or lack contents:read scope on this repo."
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
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    # Check existing files unless --force
    if not args.force:
        print("Checking existing model files...")
        if check_existing_files(manifest):
            print("\nAll models present and verified. Use --force to re-download.")
            sys.exit(0)
        print()

    if token:
        # Resolve to the API asset URL — the only URL that accepts bearer auth
        # for private-repo release assets.
        try:
            url = _resolve_asset_api_url(repo, tag, tarball_name, token)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                print(
                    f"Error: GitHub returned {e.code} resolving asset URL. "
                    "Check GITHUB_TOKEN scope (contents:read) and the repo slug + tag."
                )
                sys.exit(1)
            raise
        print(f"Downloading (authenticated) from: {url}")
    else:
        url = build_download_url(repo, tag, tarball_name)
        print(f"Downloading (anonymous) from: {url}")

    # Download to temp file
    tmp_file = None
    try:
        tmp_fd = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
        tmp_file = Path(tmp_fd.name)
        tmp_fd.close()

        download_file(url, tmp_file, token)

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
