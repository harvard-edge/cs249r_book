#!/bin/bash
# ============================================================================
# TinyTorch Release Script
# ============================================================================
#
# USAGE
# -----
#   ./scripts/release.sh 0.1.5
#   ./scripts/release.sh 0.1.5 --dry-run
#
# WHAT THIS SCRIPT DOES
# ---------------------
#   1. Updates version in pyproject.toml (single source of truth)
#   2. Updates version in settings.ini (for nbdev compatibility)
#   3. Creates a git commit with the version bump
#   4. Creates a git tag: tinytorch-v{VERSION}
#   5. Pushes commit and tag to origin
#
# SINGLE SOURCE OF TRUTH
# ----------------------
#   pyproject.toml is THE source of truth for version.
#   Other files read from it at runtime:
#     - tinytorch/__init__.py  → reads pyproject.toml
#     - tito/main.py           → reads pyproject.toml
#     - install.sh             → fetches from GitHub tags API
#     - README.md badge        → dynamic shields.io badge
#
#   settings.ini is updated for nbdev compatibility but is not authoritative.
#
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

print_step() { echo -e "${BLUE}→${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_warning() { echo -e "${YELLOW}!${NC} $1"; }

# ============================================================================
# Parse Arguments
# ============================================================================
VERSION=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -*)
            print_error "Unknown option: $1"
            echo "Usage: $0 VERSION [--dry-run]"
            exit 1
            ;;
        *)
            if [ -z "$VERSION" ]; then
                VERSION="$1"
            else
                print_error "Unexpected argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$VERSION" ]; then
    print_error "Version required"
    echo ""
    echo "Usage: $0 VERSION [--dry-run]"
    echo ""
    echo "Examples:"
    echo "  $0 0.1.5"
    echo "  $0 0.2.0 --dry-run"
    exit 1
fi

# Validate version format (semver-ish)
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    print_error "Invalid version format: $VERSION"
    echo "Expected format: MAJOR.MINOR.PATCH (e.g., 0.1.5)"
    exit 1
fi

# ============================================================================
# Verify we're in the right directory
# ============================================================================
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found"
    echo "Run this script from the tinytorch/ directory"
    exit 1
fi

# Get current version
CURRENT_VERSION=$(grep -E "^version" pyproject.toml | head -1 | sed 's/.*= *"\([^"]*\)".*/\1/')
if [ -z "$CURRENT_VERSION" ]; then
    print_error "Could not read current version from pyproject.toml"
    exit 1
fi

TAG_NAME="tinytorch-v${VERSION}"

# ============================================================================
# Show plan
# ============================================================================
echo ""
echo -e "${BOLD}TinyTorch Release${NC}"
echo ""
echo "  Current version: ${CURRENT_VERSION}"
echo "  New version:     ${VERSION}"
echo "  Tag:             ${TAG_NAME}"
echo ""

if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN - No changes will be made"
    echo ""
fi

# ============================================================================
# Check for uncommitted changes
# ============================================================================
if ! git diff --quiet || ! git diff --cached --quiet; then
    print_warning "You have uncommitted changes"
    git status --short
    echo ""
    if [ "$DRY_RUN" = false ]; then
        read -p "Continue anyway? [y/N] " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted"
            exit 1
        fi
    fi
fi

# ============================================================================
# Check if tag already exists
# ============================================================================
if git rev-parse "$TAG_NAME" >/dev/null 2>&1; then
    print_error "Tag ${TAG_NAME} already exists"
    exit 1
fi

# ============================================================================
# Update versions
# ============================================================================
print_step "Updating pyproject.toml..."
if [ "$DRY_RUN" = false ]; then
    sed -i.bak "s/^version = \".*\"/version = \"${VERSION}\"/" pyproject.toml
    rm -f pyproject.toml.bak
    print_success "Updated pyproject.toml"
else
    echo "  Would update: version = \"${VERSION}\""
fi

print_step "Updating settings.ini..."
if [ "$DRY_RUN" = false ]; then
    sed -i.bak "s/^version = .*/version = ${VERSION}/" settings.ini
    rm -f settings.ini.bak
    print_success "Updated settings.ini"
else
    echo "  Would update: version = ${VERSION}"
fi

# ============================================================================
# Git commit and tag
# ============================================================================
print_step "Creating git commit..."
if [ "$DRY_RUN" = false ]; then
    git add pyproject.toml settings.ini
    git commit -m "release: tinytorch v${VERSION}"
    print_success "Created commit"
else
    echo "  Would commit: release: tinytorch v${VERSION}"
fi

print_step "Creating git tag ${TAG_NAME}..."
if [ "$DRY_RUN" = false ]; then
    git tag -a "$TAG_NAME" -m "TinyTorch v${VERSION}"
    print_success "Created tag ${TAG_NAME}"
else
    echo "  Would create tag: ${TAG_NAME}"
fi

# ============================================================================
# Push
# ============================================================================
print_step "Pushing to origin..."
if [ "$DRY_RUN" = false ]; then
    git push origin HEAD
    git push origin "$TAG_NAME"
    print_success "Pushed commit and tag"
else
    echo "  Would push: HEAD and ${TAG_NAME}"
fi

# ============================================================================
# Done
# ============================================================================
echo ""
if [ "$DRY_RUN" = false ]; then
    print_success "Release v${VERSION} complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Create GitHub release: https://github.com/harvard-edge/cs249r_book/releases/new?tag=${TAG_NAME}"
    echo "  2. Students can update: tito system update"
else
    print_warning "Dry run complete - no changes made"
fi
echo ""
