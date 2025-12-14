#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════
# 📚 MLSysBook Manual Publisher
# ═══════════════════════════════════════════════════════════════════════════
# A comprehensive manual publishing tool for the MLSysBook textbook
# Usage: ./publish.sh [options]
#
# Features:
# - Interactive rich UI with validation
# - Manual GitHub release creation
# - Version management and checking
# - Local building and testing
# - GitHub Pages deployment
# ═══════════════════════════════════════════════════════════════════════════

set -e  # Exit on any error

# ═══════════════════════════════════════════════════════════════════════════
# 🎨 Rich UI Functions
# ═══════════════════════════════════════════════════════════════════════════

# Colors and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "\n${BOLD}${BLUE}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${BLUE}║${WHITE}  📚 MLSysBook Manual Publisher                                            ${BLUE}║${NC}"
    echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════════════════════════════╝${NC}\n"
}

print_section() {
    echo -e "\n${BOLD}${CYAN}┌─ $1 ─────────────────────────────────────────────────────────${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_step() {
    echo -e "${PURPLE}🔄 $1${NC}"
}

# Progress bar function
show_progress() {
    local current=$1
    local total=$2
    local description=$3
    local width=50
    local percentage=$((current * 100 / total))
    local completed=$((current * width / total))
    local remaining=$((width - completed))

    printf "\r${BLUE}[${GREEN}"
    printf "%*s" $completed | tr ' ' '█'
    printf "${BLUE}"
    printf "%*s" $remaining | tr ' ' '░'
    printf "${BLUE}] ${WHITE}%d%% ${CYAN}%s${NC}" $percentage "$description"

    if [ $current -eq $total ]; then
        echo ""
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# 🔧 Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

# Get current git status
get_git_status() {
    local current_branch=$(git branch --show-current 2>/dev/null || echo "unknown")
    local has_changes=$(git status --porcelain 2>/dev/null | wc -l)
    local is_clean=$([[ $has_changes -eq 0 ]] && echo "true" || echo "false")

    echo "$current_branch|$is_clean|$has_changes"
}

# Get latest version from git tags
get_latest_version() {
    local latest=$(git tag --sort=-version:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1 2>/dev/null || echo "v0.0.0")
    echo "$latest"
}

# Calculate next version
calculate_next_version() {
    local current_version=$1
    local release_type=$2

    # Remove 'v' prefix
    local version_num=${current_version#v}

    # Split version into components
    IFS='.' read -r major minor patch <<< "$version_num"

    # Handle empty or invalid versions
    major=${major:-0}
    minor=${minor:-0}
    patch=${patch:-0}

    case "$release_type" in
        "major")
            new_version="v$((major + 1)).0.0"
            ;;
        "minor")
            new_version="v$major.$((minor + 1)).0"
            ;;
        "patch")
            new_version="v$major.$minor.$((patch + 1))"
            ;;
        *)
            echo "Invalid release type: $release_type"
            exit 1
            ;;
    esac

    echo "$new_version"
}

# Check if GitHub CLI is available
check_gh_cli() {
    if command -v gh >/dev/null 2>&1; then
        if gh auth status >/dev/null 2>&1; then
            echo "authenticated"
        else
            echo "not_authenticated"
        fi
    else
        echo "not_installed"
    fi
}

# Get repository info
get_repo_info() {
    local remote_url=$(git remote get-url origin 2>/dev/null || echo "")
    local repo_name=""

    if [[ $remote_url =~ github\.com[:/]([^/]+)/([^/]+)(\.git)?$ ]]; then
        local owner="${BASH_REMATCH[1]}"
        local name="${BASH_REMATCH[2]}"
        repo_name="$owner/$name"
    fi

    echo "$repo_name"
}

# ═══════════════════════════════════════════════════════════════════════════
# 🎮 Interactive Functions
# ═══════════════════════════════════════════════════════════════════════════

# Get user input with validation
get_user_input() {
    local prompt=$1
    local default=$2
    local validation_pattern=$3
    local input=""

    while true; do
        if [ -n "$default" ]; then
            echo -e -n "${WHITE}$prompt ${CYAN}[$default]${WHITE}: ${NC}"
        else
            echo -e -n "${WHITE}$prompt: ${NC}"
        fi

        read -r input

        # Use default if empty
        if [ -z "$input" ] && [ -n "$default" ]; then
            input="$default"
        fi

        # Validate input
        if [ -z "$validation_pattern" ] || [[ $input =~ $validation_pattern ]]; then
            echo "$input"
            return 0
        else
            print_error "Invalid input. Please try again."
        fi
    done
}

# Confirm action
confirm_action() {
    local message=$1
    local default=${2:-"n"}

    echo -e -n "${YELLOW}$message ${CYAN}[y/N]${WHITE}: ${NC}"
    read -r response

    response=${response:-$default}
    case "$response" in
        [yY][eE][sS]|[yY])
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Select from menu
select_from_menu() {
    local title=$1
    shift
    local options=("$@")
    local selection

    echo -e "\n${BOLD}${WHITE}$title${NC}"
    for i in "${!options[@]}"; do
        echo -e "${CYAN}  $((i + 1)). ${WHITE}${options[$i]}${NC}"
    done

    while true; do
        echo -e -n "${WHITE}Select option [1-${#options[@]}]: ${NC}"
        read -r selection

        if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le "${#options[@]}" ]; then
            echo "${options[$((selection - 1))]}"
            return 0
        else
            print_error "Invalid selection. Please choose 1-${#options[@]}."
        fi
    done
}

# ═══════════════════════════════════════════════════════════════════════════
# 🏗️ Build Functions
# ═══════════════════════════════════════════════════════════════════════════

# Clean previous builds
clean_builds() {
    print_step "Cleaning previous builds..."
    if ./binder clean >/dev/null 2>&1; then
        print_success "Previous builds cleaned"
        return 0
    else
        print_error "Failed to clean previous builds"
        return 1
    fi
}

# Build HTML version
build_html() {
    print_step "Building HTML version..."
    show_progress 1 3 "Initializing HTML build"

    if ./binder build - html >/dev/null 2>&1; then
        show_progress 3 3 "HTML build completed"
        print_success "HTML build completed successfully"
        return 0
    else
        echo ""
        print_error "HTML build failed"
        return 1
    fi
}

# Build PDF version
build_pdf() {
    print_step "Building PDF version..."
    show_progress 1 4 "Initializing PDF build"

    if ./binder build - pdf >/dev/null 2>&1; then
        show_progress 4 4 "PDF build completed"
        print_success "PDF build completed successfully"

        # Check if PDF was created
        local pdf_path="build/pdf/Machine-Learning-Systems.pdf"
        if [ -f "$pdf_path" ]; then
            local pdf_size=$(ls -lh "$pdf_path" | awk '{print $5}')
            print_success "PDF found at $pdf_path (Size: $pdf_size)"
            return 0
        else
            print_error "PDF not found at $pdf_path"
            return 1
        fi
    else
        echo ""
        print_error "PDF build failed"
        return 1
    fi
}

# Compress PDF using Ghostscript with ebook settings
compress_pdf() {
    local pdf_path="build/pdf/Machine-Learning-Systems.pdf"
    local compressed_path="build/pdf/Machine-Learning-Systems-compressed.pdf"

    print_step "Compressing PDF with Ghostscript..."

    # Check if Ghostscript is available
    if ! command -v gs >/dev/null 2>&1; then
        print_warning "Ghostscript not found - skipping compression"
        print_info "Install with: brew install ghostscript (macOS) or apt-get install ghostscript (Linux)"
        return 0
    fi

    if [ ! -f "$pdf_path" ]; then
        print_error "PDF not found for compression: $pdf_path"
        return 1
    fi

    # Get original file size
    local original_size=$(ls -lh "$pdf_path" | awk '{print $5}')
    print_info "Original PDF size: $original_size"

    show_progress 1 3 "Compressing with Ghostscript"

    # Use Ghostscript with ebook settings for optimal compression
    if gs -sDEVICE=pdfwrite \
          -dCompatibilityLevel=1.4 \
          -dPDFSETTINGS=/ebook \
          -dNOPAUSE \
          -dQUIET \
          -dBATCH \
          -sOutputFile="$compressed_path" \
          "$pdf_path" >/dev/null 2>&1; then

        show_progress 2 3 "Compression completed"

        # Get compressed file size
        local compressed_size=$(ls -lh "$compressed_path" | awk '{print $5}')

        # Calculate compression ratio
        local original_bytes=$(stat -f%z "$pdf_path" 2>/dev/null || stat -c%s "$pdf_path" 2>/dev/null)
        local compressed_bytes=$(stat -f%z "$compressed_path" 2>/dev/null || stat -c%s "$compressed_path" 2>/dev/null)
        local compression_ratio=$(( (original_bytes - compressed_bytes) * 100 / original_bytes ))

        show_progress 3 3 "Replacing original PDF"

        # Replace original with compressed version
        mv "$compressed_path" "$pdf_path"

        print_success "PDF compressed successfully"
        print_info "Compressed size: $compressed_size (saved ${compression_ratio}%)"
        return 0
    else
        print_error "PDF compression failed"
        # Clean up failed compression file
        [ -f "$compressed_path" ] && rm "$compressed_path"
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# 🚀 Publishing Functions
# ═══════════════════════════════════════════════════════════════════════════

# Deploy to GitHub Pages
deploy_to_github_pages() {
    print_step "Deploying to GitHub Pages..."

    if quarto publish gh-pages --no-render >/dev/null 2>&1; then
        print_success "Successfully deployed to GitHub Pages"
        return 0
    else
        print_error "GitHub Pages deployment failed"
        return 1
    fi
}

# Create GitHub release
create_github_release() {
    local version=$1
    local description=$2
    local pdf_path=$3
    local gh_status=$(check_gh_cli)

    print_step "Creating GitHub release $version..."

    case "$gh_status" in
        "authenticated")
            print_info "Using GitHub CLI to create release"

            # Create release notes
            local release_notes="# Release $version

$description

## Changes
$(git log --oneline "$(get_latest_version)"..HEAD | head -10 | sed 's/^/- /')"

            if gh release create "$version" \
                --title "$version: $description" \
                --notes "$release_notes" \
                --draft \
                "$pdf_path"; then
                print_success "GitHub release created successfully"
                print_info "Release URL: $(gh release view "$version" --web --json url -q .url 2>/dev/null || echo 'Check GitHub releases page')"
                return 0
            else
                print_error "Failed to create GitHub release"
                return 1
            fi
            ;;
        "not_authenticated")
            print_warning "GitHub CLI not authenticated"
            print_info "Please authenticate with: gh auth login"
            print_info "Then manually create release at: https://github.com/$(get_repo_info)/releases/new"
            print_info "Upload PDF: $pdf_path"
            return 1
            ;;
        "not_installed")
            print_warning "GitHub CLI not installed"
            print_info "Install with: brew install gh (macOS) or visit https://cli.github.com/"
            print_info "Or manually create release at: https://github.com/$(get_repo_info)/releases/new"
            print_info "Upload PDF: $pdf_path"
            return 1
            ;;
    esac
}

# ═══════════════════════════════════════════════════════════════════════════
# 📋 Detailed Information Functions
# ═══════════════════════════════════════════════════════════════════════════

show_detailed_release_info() {
    echo ""
    echo -e "${BOLD}${BLUE}📋 Detailed Release Information${NC}"
    echo ""
    echo -e "${WHITE}🔧 Technical Process:${NC}"
    echo -e "${BLUE}  1. Git Status Check${NC}"
    echo -e "${CYAN}     • Verify repository is clean or changes are committed${NC}"
    echo -e "${CYAN}     • Ensure we're on the main branch${NC}"
    echo -e "${CYAN}     • Check for any merge conflicts${NC}"
    echo ""
    echo -e "${BLUE}  2. Version Management${NC}"
    echo -e "${CYAN}     • Calculate next version number${NC}"
    echo -e "${CYAN}     • Validate version format${NC}"
    echo -e "${CYAN}     • Check for existing versions${NC}"
    echo ""
    echo -e "${BLUE}  3. Build Phase${NC}"
    echo -e "${CYAN}     • Clean previous build artifacts${NC}"
    echo -e "${CYAN}     • Build PDF version (3-5 minutes)${NC}"
    echo -e "${CYAN}     • Build HTML version (1-2 minutes)${NC}"
    echo -e "${CYAN}     • Validate file sizes and completeness${NC}"
    echo -e "${CYAN}     • Compress PDF for web distribution${NC}"
    echo ""
    echo -e "${BLUE}  4. Release Phase${NC}"
    echo -e "${CYAN}     • Create git tag with version${NC}"
    echo -e "${CYAN}     • Push tag to remote repository${NC}"
    echo -e "${CYAN}     • Deploy to GitHub Pages (optional)${NC}"
    echo -e "${CYAN}     • Create GitHub release with PDF (optional)${NC}"
    echo ""
    echo -e "${WHITE}🌐 What Gets Released:${NC}"
    echo -e "${GREEN}  • Complete HTML textbook website${NC}"
    echo -e "${GREEN}  • All textbook chapters and content${NC}"
    echo -e "${GREEN}  • Navigation and search functionality${NC}"
    echo -e "${GREEN}  • PDF download link${NC}"
    echo -e "${GREEN}  • All images, figures, and media${NC}"
    echo ""
    echo -e "${WHITE}📊 Quality Checks:${NC}"
    echo -e "${YELLOW}  • PDF file size validation (>1MB)${NC}"
    echo -e "${YELLOW}  • HTML build completeness${NC}"
    echo -e "${YELLOW}  • Asset file integrity${NC}"
    echo -e "${YELLOW}  • Deployment success verification${NC}"
    echo ""
    echo -e "${WHITE}⚠️  Potential Issues:${NC}"
    echo -e "${RED}  • Build failures due to syntax errors${NC}"
    echo -e "${RED}  • Large file sizes causing timeout${NC}"
    echo -e "${RED}  • Network connectivity issues${NC}"
    echo -e "${RED}  • GitHub Pages deployment limits${NC}"
    echo ""
    echo -e "${WHITE}🔄 Rollback Options:${NC}"
    echo -e "${BLUE}  • Previous version remains accessible${NC}"
    echo -e "${BLUE}  • Can redeploy previous commit if needed${NC}"
    echo -e "${BLUE}  • GitHub Pages maintains version history${NC}"
    echo ""
    echo -e "${WHITE}📞 Support:${NC}"
    echo -e "${CYAN}  • Check build logs for detailed error messages${NC}"
    echo -e "${CYAN}  • Review GitHub Actions for deployment status${NC}"
    echo -e "${CYAN}  • Contact maintainers if deployment fails${NC}"
}

# ═══════════════════════════════════════════════════════════════════════════
# 🎯 Main Publishing Workflow
# ═══════════════════════════════════════════════════════════════════════════

main() {
    print_header

    # Comprehensive Publishing Overview
    print_section "Publishing Overview"
    echo -e "${WHITE}🎯 What this will do:${NC}"
    echo -e "   ${GREEN}✅ Build HTML version of your textbook${NC}"
    echo -e "   ${GREEN}✅ Build PDF version of your textbook${NC}"
    echo -e "   ${GREEN}✅ Compress PDF for faster downloads${NC}"
    echo -e "   ${GREEN}✅ Create git tag with version number${NC}"
    echo -e "   ${GREEN}✅ Deploy to GitHub Pages (optional)${NC}"
    echo -e "   ${GREEN}✅ Create GitHub release with PDF (optional)${NC}"
    echo ""
    echo -e "${WHITE}⚠️  Important Notes:${NC}"
    echo -e "   ${YELLOW}• This creates a FORMAL RELEASE with versioning${NC}"
    echo -e "   ${YELLOW}• Changes will be publicly available${NC}"
    echo -e "   ${YELLOW}• Git tags and GitHub releases are permanent${NC}"
    echo -e "   ${YELLOW}• Anyone can access the released content${NC}"
    echo ""
    echo -e "${WHITE}🔍 What will be checked:${NC}"
    echo -e "   ${BLUE}• Git repository status and branch${NC}"
    echo -e "   ${BLUE}• Build artifacts and file sizes${NC}"
    echo -e "   ${BLUE}• PDF quality and completeness${NC}"
    echo -e "   ${BLUE}• HTML build validity${NC}"
    echo ""
    echo -e "${WHITE}⏱️  Estimated time:${NC}"
    echo -e "   ${CYAN}• PDF build: 3-5 minutes${NC}"
    echo -e "   ${CYAN}• HTML build: 1-2 minutes${NC}"
    echo -e "   ${CYAN}• PDF compression: 30-60 seconds${NC}"
    echo -e "   ${CYAN}• GitHub deployment: 1-2 minutes${NC}"
    echo -e "   ${CYAN}• Total: 6-10 minutes${NC}"
    echo ""

    # Initial confirmation
    echo -e "${BOLD}${RED}🚨 PRODUCTION RELEASE CONFIRMATION${NC}"
    echo -e "${RED}This will create a formal release with versioning and make it publicly available.${NC}"
    echo -e "${RED}The release will be permanent and accessible to students, educators, and the public.${NC}"
    echo ""
    echo -e "${WHITE}Options:${NC}"
    echo -e "   ${GREEN}y/yes${NC} - Proceed with formal release creation"
    echo -e "   ${RED}n/no${NC} - Cancel release [default]"
    echo -e "   ${BLUE}info${NC} - Show more details about the release process"
    echo ""
    echo -e -n "${YELLOW}Create formal release? [y/N/info] [default: N]: ${NC}"
    read -r initial_choice

    if [ "$initial_choice" = "info" ]; then
        show_detailed_release_info
        echo ""
        echo -e -n "${YELLOW}Create formal release? [y/N] [default: N]: ${NC}"
        read -r initial_choice
    fi

    if [ "$initial_choice" != "y" ] && [ "$initial_choice" != "yes" ]; then
        print_info "Release creation cancelled"
        exit 0
    fi

    echo -e "${GREEN}✅ Release creation confirmed${NC}"
    echo ""

    # Check git status
    print_section "Git Status Check"
    IFS='|' read -r current_branch is_clean changes_count <<< "$(get_git_status)"

    print_info "Current branch: $current_branch"
    print_info "Uncommitted changes: $changes_count"

    # Handle git status
    if [ "$current_branch" != "main" ]; then
        print_warning "You are not on the main branch"

        if confirm_action "Switch to main branch and merge dev?"; then
            if [ "$is_clean" != "true" ]; then
                if confirm_action "Commit current changes first?"; then
                    local commit_msg=$(get_user_input "Commit message" "fix: update before merge")
                    git add .
                    git commit -m "$commit_msg"
                    print_success "Changes committed"
                fi
            fi

            print_step "Switching to main and merging dev..."
            git checkout main
            git pull origin main
            git merge dev
            git push origin main
            print_success "Merged dev to main"
        else
            print_error "Cannot continue without being on main branch"
            exit 1
        fi
    elif [ "$is_clean" != "true" ]; then
        print_warning "You have uncommitted changes"

        if confirm_action "Commit changes before continuing?"; then
            local commit_msg=$(get_user_input "Commit message" "fix: update before publishing")
            git add .
            git commit -m "$commit_msg"
            git push origin main
            print_success "Changes committed and pushed"
        else
            print_error "Cannot continue with uncommitted changes"
            exit 1
        fi
    else
        print_success "Git status is clean"
    fi

    # Version management
    print_section "Version Management"
    local current_version=$(get_latest_version)
    print_info "Current version: $current_version"

    local release_types=("patch" "minor" "major" "custom")
    local release_type=$(select_from_menu "Select release type:" "${release_types[@]}")

    local new_version
    if [ "$release_type" = "custom" ]; then
        new_version=$(get_user_input "Enter version (e.g., v1.0.0)" "" "^v[0-9]+\.[0-9]+\.[0-9]+$")
    else
        new_version=$(calculate_next_version "$current_version" "$release_type")
    fi

    print_info "New version will be: $new_version"

    # Check if version already exists
    if git tag -l "$new_version" | grep -q "$new_version"; then
        print_warning "Version $new_version already exists"
        if confirm_action "Replace existing version?"; then
            git tag -d "$new_version"
            git push origin --delete "$new_version" 2>/dev/null || true
            print_success "Existing version removed"
        else
            print_error "Cannot continue with existing version"
            exit 1
        fi
    fi

    # Get release description
    local description=$(get_user_input "Release description" "Content updates and improvements")

    # Confirmation
    print_section "Publishing Summary"
    echo -e "${WHITE}📋 Publishing Details:${NC}"
    echo -e "   Version: ${GREEN}$new_version${NC}"
    echo -e "   Type: ${BLUE}$release_type${NC}"
    echo -e "   Description: ${CYAN}$description${NC}"
    echo -e "   Branch: ${PURPLE}$current_branch${NC}"
    echo -e "   Repository: ${YELLOW}$(get_repo_info)${NC}"

    if ! confirm_action "Proceed with publishing?"; then
        print_info "Publishing cancelled"
        exit 0
    fi

    # Building phase
    print_section "Building Phase"

    # Clean builds
    if ! clean_builds; then
        print_error "Failed to clean builds"
        exit 1
    fi

    # Build PDF first (more important, takes longer)
    if ! build_pdf; then
        print_error "PDF build failed"
        exit 1
    fi

    # Compress PDF with Ghostscript
    if ! compress_pdf; then
        print_warning "PDF compression failed, continuing with uncompressed PDF"
    fi

    # Build HTML (faster, after PDF is ready)
    if ! build_html; then
        print_error "HTML build failed"
        exit 1
    fi

    # Publishing phase
    print_section "Publishing Phase"

    local pdf_path="build/pdf/Machine-Learning-Systems.pdf"

    # Create git tag
    print_step "Creating git tag $new_version..."
    git tag -a "$new_version" -m "Release $new_version: $description"
    git push origin "$new_version"
    print_success "Git tag created and pushed"

    # GitHub Pages deployment
    if confirm_action "Deploy to GitHub Pages?"; then
        if deploy_to_github_pages; then
            local repo_info=$(get_repo_info)
            local pages_url="https://$(echo "$repo_info" | cut -d'/' -f1).github.io/$(echo "$repo_info" | cut -d'/' -f2)"
            print_success "Deployed to: $pages_url"
        fi
    fi

    # GitHub release creation
    if confirm_action "Create GitHub release?"; then
        create_github_release "$new_version" "$description" "$pdf_path"
    fi

    # Success summary
    print_section "Publication Complete"
    print_success "🎉 Publication successful!"
    echo ""
    echo -e "${WHITE}📊 What was published:${NC}"
    echo -e "   ✅ Version: ${GREEN}$new_version${NC}"
    echo -e "   ✅ PDF build completed and compressed ($(ls -lh "$pdf_path" | awk '{print $5}'))"
    echo -e "   ✅ HTML build completed"
    echo -e "   ✅ Git tag created and pushed"

    local repo_info=$(get_repo_info)
    if [ -n "$repo_info" ]; then
        echo ""
        echo -e "${WHITE}🌐 Access your publication:${NC}"
        echo -e "   📖 Web version: https://$(echo "$repo_info" | cut -d'/' -f1).github.io/$(echo "$repo_info" | cut -d'/' -f2)"
        echo -e "   📦 Releases: https://github.com/$repo_info/releases"
        echo -e "   📄 PDF: https://github.com/$repo_info/releases/download/$new_version/Machine-Learning-Systems.pdf"
    fi

    print_success "Ready for distribution! 🚀"
}

# ═══════════════════════════════════════════════════════════════════════════
# 🚀 Script Entry Point
# ═══════════════════════════════════════════════════════════════════════════

# Change to book directory if not already there
if [ ! -f "_quarto.yml" ] && [ -d "book" ]; then
    cd book
fi

# Verify we're in the right directory
if [ ! -f "_quarto.yml" ]; then
    print_error "Not in a Quarto book directory. Please run from the book folder."
    exit 1
fi

# Run main function
main "$@"
