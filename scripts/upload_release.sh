#!/usr/bin/env bash
set -euo pipefail

# Upload ML model tarball as a GitHub Release.
# Usage: ./scripts/upload_release.sh [path/to/models.tar.gz]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MANIFEST="$PROJECT_ROOT/MODEL_MANIFEST.json"

TARBALL_PATH="${1:-/tmp/leichte-sprache-models/models.tar.gz}"

# Check prerequisites
if ! command -v gh &>/dev/null; then
    echo "Error: GitHub CLI (gh) is not installed."
    echo "Install it: https://cli.github.com/"
    exit 1
fi

if ! gh auth status &>/dev/null; then
    echo "Error: Not authenticated with GitHub CLI."
    echo "Run: gh auth login"
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "Error: MODEL_MANIFEST.json not found at $MANIFEST"
    exit 1
fi

if [[ ! -f "$TARBALL_PATH" ]]; then
    echo "Error: Tarball not found at $TARBALL_PATH"
    echo "Expected the model tarball at: $TARBALL_PATH"
    exit 1
fi

# Read tag from manifest
TAG=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['github_release_tag'])")
REPO=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['github_repo'])")

echo "Creating release '$TAG' in $REPO..."
echo "Uploading: $TARBALL_PATH ($(du -h "$TARBALL_PATH" | cut -f1))"

gh release create "$TAG" "$TARBALL_PATH" \
    --repo "$REPO" \
    --title "ML Models $TAG" \
    --notes "Custom-trained ML model files for Leichte Sprache rules. Download automatically with: python scripts/download_models.py"

echo ""
echo "Release created successfully!"
echo "URL: https://github.com/$REPO/releases/tag/$TAG"
