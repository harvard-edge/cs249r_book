#!/usr/bin/env bash
# TinyTorch Demo Studio - One script for everything
# Interactive: validate → calibrate → generate

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Banner
show_banner() {
    echo ""
    echo -e "${CYAN}${BOLD}"
    echo "  ╔═══════════════════════════════════════╗"
    echo "  ║   🔥 TinyTorch Demo Studio 🎬        ║"
    echo "  ╚═══════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
}

# Check VHS installed
check_vhs() {
    if ! command -v vhs &> /dev/null; then
        echo -e "${RED}❌ VHS is not installed${NC}"
        echo ""
        echo "Install VHS:"
        echo "  macOS:  brew install vhs"
        echo "  Linux:  go install github.com/charmbracelet/vhs@latest"
        echo ""
        exit 1
    fi
}

# ============================================================================
# VALIDATION
# ============================================================================

test_command() {
    local test_name="$1"
    local command="$2"
    local check_output="$3"
    local collect_timing="${4:-false}"
    local show_live="${5:-true}"

    echo -e "${BLUE}⏳ Testing: ${test_name}${NC}"
    echo ""

    local start=$(date +%s%N)
    
    # Show live output while capturing to file
    if [ "$show_live" = "true" ]; then
        if eval "$command" 2>&1 | tee /tmp/test_output.txt | sed 's/^/  │ /'; then
            local end=$(date +%s%N)
            local duration_ns=$((end - start))
            local duration_ms=$((duration_ns / 1000000))
            local duration_s=$(echo "scale=2; $duration_ms / 1000" | bc)
            
            echo ""
            if [ -n "$check_output" ]; then
                if grep -q "$check_output" /tmp/test_output.txt; then
                    echo -e "  ${GREEN}✓ PASS${NC} (${duration_s}s)"
                    ((PASSED++))
                else
                    echo -e "  ${YELLOW}⚠ WARNING: Output doesn't match expected pattern${NC} (${duration_s}s)"
                    ((WARNINGS++))
                fi
            else
                echo -e "  ${GREEN}✓ PASS${NC} (${duration_s}s)"
                ((PASSED++))
            fi
            
            # Store timing if requested (bash 3.2 compatible)
            if [ "$collect_timing" = "true" ]; then
                TIMING_NAMES+=("$test_name")
                TIMING_VALUES+=("$duration_ms")
            fi
        else
            echo ""
            echo -e "  ${RED}✗ FAIL${NC}"
            echo -e "  ${RED}Error:${NC}"
            tail -5 /tmp/test_output.txt | sed 's/^/    /'
            ((FAILED++))
        fi
    else
        # Silent mode (for quick tests)
        if eval "$command" > /tmp/test_output.txt 2>&1; then
            local end=$(date +%s%N)
            local duration_ns=$((end - start))
            local duration_ms=$((duration_ns / 1000000))
            local duration_s=$(echo "scale=2; $duration_ms / 1000" | bc)
            
            if [ -n "$check_output" ]; then
                if grep -q "$check_output" /tmp/test_output.txt; then
                    echo -e "  ${GREEN}✓ PASS${NC} (${duration_s}s)"
                    ((PASSED++))
                else
                    echo -e "  ${YELLOW}⚠ WARNING: Output doesn't match expected pattern${NC} (${duration_s}s)"
                    ((WARNINGS++))
                fi
            else
                echo -e "  ${GREEN}✓ PASS${NC} (${duration_s}s)"
                ((PASSED++))
            fi
            
            if [ "$collect_timing" = "true" ]; then
                TIMING_NAMES+=("$test_name")
                TIMING_VALUES+=("$duration_ms")
            fi
        else
            echo -e "  ${RED}✗ FAIL${NC}"
            echo -e "  ${RED}Error:${NC}"
            tail -5 /tmp/test_output.txt | sed 's/^/    /'
            ((FAILED++))
        fi
    fi
    echo ""
}

validate() {
    local collect_timing="${1:-false}"
    local skip_clone="${2:-false}"
    
    if [ "$collect_timing" = "true" ]; then
        echo -e "${CYAN}${BOLD}📋 Step 1: Validation + Timing Collection${NC}"
    else
        echo -e "${CYAN}${BOLD}📋 Step 1: Validation${NC}"
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    if [ "$skip_clone" = "true" ]; then
        echo -e "${YELLOW}⚡ Debug mode: Skipping git clone (using local copy)${NC}"
        echo ""
    fi
    
    echo "Testing all demo workflows..."
    echo ""

    PASSED=0
    FAILED=0
    WARNINGS=0
    
    # Timing data (bash 3.2 compatible - no associative arrays)
    TIMING_NAMES=()
    TIMING_VALUES=()

    echo -e "${CYAN}Testing Demo 01: Zero to Ready${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    if [ "$skip_clone" = "false" ]; then
        # Clean slate and clone
        echo -e "${YELLOW}🧹 Cleaning /tmp/TinyTorch_validate...${NC}"
        rm -rf /tmp/TinyTorch_validate
        cd /tmp
        echo ""
        
        test_command "git clone" \
            "git clone https://github.com/mlsysbook/TinyTorch.git TinyTorch_validate" \
            "Cloning into" \
            "$collect_timing" \
            "true"

        cd TinyTorch_validate
    else
        # Use existing directory or create from current repo
        if [ -d "/tmp/TinyTorch_validate" ]; then
            echo -e "${GREEN}✓ Using existing /tmp/TinyTorch_validate${NC}"
            echo ""
            cd /tmp/TinyTorch_validate
        else
            echo -e "${YELLOW}Creating /tmp/TinyTorch_validate from current repo...${NC}"
            # Copy current repo to /tmp for testing
            cp -r "$(git rev-parse --show-toplevel)" /tmp/TinyTorch_validate 2>/dev/null || {
                echo -e "${RED}❌ Not in a git repo. Please run from TinyTorch directory.${NC}"
                return 1
            }
            echo -e "${GREEN}✓ Created from local repo${NC}"
            echo ""
            cd /tmp/TinyTorch_validate
        fi
    fi

    test_command "setup-environment.sh" \
        "./setup-environment.sh" \
        "Setup complete" \
        "$collect_timing" \
        "true"

    test_command "activate.sh exists" \
        "test -f activate.sh" \
        "" \
        "false" \
        "false"

    test_command "source activate.sh" \
        "source activate.sh && echo 'Environment activated'" \
        "" \
        "false" \
        "false"

    test_command "tito module status" \
        "source activate.sh && tito module status" \
        "Module" \
        "$collect_timing" \
        "true"

    test_command "tito logo" \
        "source activate.sh && tito logo" \
        "TinyTorch" \
        "$collect_timing" \
        "true"

    echo ""
    echo -e "${CYAN}Testing Demo 02: Build, Test, Ship${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    test_command "tito module reset 01" \
        "source activate.sh && tito module reset 01 --force --no-backup" \
        "" \
        "false" \
        "true"

    test_command "Check src/01_tensor exists" \
        "test -d src/01_tensor" \
        "" \
        "false" \
        "false"

    test_command "jupyter lab --help" \
        "source activate.sh && jupyter lab --help" \
        "jupyter-lab" \
        "false" \
        "false"

    test_command "tito module test 01" \
        "source activate.sh && tito module test 01" \
        "" \
        "false" \
        "true"

    echo ""
    echo -e "${CYAN}Testing Demo 03: Milestones${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    test_command "tito milestones progress" \
        "source activate.sh && tito milestones progress" \
        "" \
        "false" \
        "true"

    test_command "tito milestones list" \
        "source activate.sh && tito milestones list" \
        "" \
        "false" \
        "true"

    echo ""
    echo -e "${CYAN}Testing Demo 04: Share Journey${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    test_command "tito module status (demo 04)" \
        "source activate.sh && tito module status" \
        "Module" \
        "false" \
        "false"

    echo ""
    echo "========================================"
    echo -e "${CYAN}${BOLD}📊 Validation Results${NC}"
    echo "========================================"
    echo ""
    echo -e "  ${GREEN}Passed:   $PASSED${NC}"
    echo -e "  ${YELLOW}Warnings: $WARNINGS${NC}"
    echo -e "  ${RED}Failed:   $FAILED${NC}"
    echo ""

    # Show timing summary if collected
    if [ "$collect_timing" = "true" ] && [ $FAILED -eq 0 ] && [ ${#TIMING_NAMES[@]} -gt 0 ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo -e "${CYAN}${BOLD}⏱  Timing Summary${NC}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        printf "%-30s %12s\n" "Command" "Time (s)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Loop through parallel arrays (bash 3.2 compatible)
        for i in "${!TIMING_NAMES[@]}"; do
            local name="${TIMING_NAMES[$i]}"
            local ms="${TIMING_VALUES[$i]}"
            local sec=$(echo "scale=2; $ms / 1000" | bc)
            printf "%-30s %10ss\n" "$name" "$sec"
        done
        echo ""
        
        echo -e "${CYAN}💡 VHS wait syntax for tape files:${NC}"
        echo "   Wait+Line@10ms /profvjreddi/"
        echo ""
    fi

    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}✅ All tests passed!${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}❌ Some tests failed${NC}"
        echo ""
        echo "Debug:"
        echo "  cd /tmp/TinyTorch_validate"
        echo "  source activate.sh"
        echo "  # Run failing command manually"
        echo ""
        return 1
    fi
}

# ============================================================================
# GENERATION
# ============================================================================

get_demo_name() {
    case $1 in
        00) echo "00-welcome" ;;
        02) echo "02-build-test-ship" ;;
        03) echo "03-milestone-unlocked" ;;
        04) echo "04-share-journey" ;;
        05) echo "05-logo" ;;
        *) echo "" ;;
    esac
}

generate() {
    local tape_num="$1"
    local demo_name=$(get_demo_name "$tape_num")

    if [ -z "$demo_name" ]; then
        echo -e "${RED}❌ Invalid demo number${NC}"
        return 1
    fi

    echo -e "${CYAN}${BOLD}🎬 Step 2: Generate Demo${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo -e "${BLUE}📹 Recording Demo ${tape_num}: ${demo_name}${NC}"
    echo ""

    local tape_file="docs/_static/demos/tapes/${demo_name}.tape"
    local output_gif="${demo_name}.gif"

    if [ ! -f "$tape_file" ]; then
        echo -e "${RED}❌ Tape file not found: $tape_file${NC}"
        return 1
    fi

    # Clean temp directory
    echo -e "${YELLOW}⏳ Step 2.1: Cleaning /tmp/TinyTorch...${NC}"
    rm -rf /tmp/TinyTorch
    echo -e "  ${GREEN}✓ Clean${NC}"
    echo ""
    
    # Reset all modules to clean state for demo
    echo -e "${YELLOW}⏳ Step 2.2: Resetting all modules to clean state...${NC}"
    if tito module reset --all --force --no-backup > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓ All modules reset to pristine state${NC}"
    else
        echo -e "  ${YELLOW}⚠ Could not reset modules (continuing)${NC}"
    fi
    echo ""

    # Delete old GIF if it exists (prevent appending)
    if [ -f "docs/_static/demos/gifs/${demo_name}.gif" ]; then
        echo -e "${YELLOW}🧹 Step 2.3: Removing old GIF to prevent appending...${NC}"
        rm -f "docs/_static/demos/gifs/${demo_name}.gif"
        echo -e "  ${GREEN}✓ Removed old GIF${NC}"
        echo ""
    fi

    # Record
    echo -e "${YELLOW}⏳ Step 2.4: Recording with VHS (1-2 minutes)...${NC}"
    echo ""

    local start=$(date +%s)
    
    if vhs "$tape_file" 2>&1 | while read line; do
        echo "  $line"
    done; then
        local end=$(date +%s)
        local duration=$((end - start))
        
        if [ -f "$output_gif" ]; then
            local size=$(du -h "$output_gif" | cut -f1)
            echo ""
            echo -e "${GREEN}✅ Recording complete!${NC} (took ${duration}s)"
            echo ""
            echo -e "${YELLOW}⏳ Step 2.5: Moving to docs/_static/demos/gifs/${NC}"
            mv "$output_gif" "docs/_static/demos/gifs/$output_gif"
            echo -e "  ${GREEN}✓ Saved: docs/_static/demos/gifs/$output_gif ($size)${NC}"
            echo ""
            echo -e "${BLUE}💡 Preview with:${NC}"
            echo "  open docs/_static/demos/gifs/$output_gif"
            echo ""
            return 0
        else
            echo -e "${RED}❌ Failed to create GIF${NC}"
            return 1
        fi
    else
        echo ""
        echo -e "${RED}❌ VHS recording failed${NC}"
        return 1
    fi
}

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

interactive() {
    show_banner

    # Step 1: What to do?
    echo -e "${BOLD}What would you like to do?${NC}"
    echo ""
    echo "  1) Validate only (test all commands work)"
    echo "  2) Generate demo GIF only"
    echo "  3) Full workflow (validate + timing + generate) ${GREEN}← Recommended${NC}"
    echo "  4) Exit"
    echo ""
    read -p "Choose [1-4]: " choice
    echo ""

    # Ask about skipping clone for options 1 and 3
    local skip_clone="false"
    if [ "$choice" = "1" ] || [ "$choice" = "3" ]; then
        echo -e "${YELLOW}Skip git clone? (faster for debugging with slow internet)${NC}"
        read -p "Skip clone? [y/N]: " skip_response
        if [[ "$skip_response" =~ ^[Yy]$ ]]; then
            skip_clone="true"
        fi
        echo ""
    fi

    case $choice in
        1)
            validate false "$skip_clone"
            ;;
        2)
            echo -e "${BOLD}Which demo to generate?${NC}"
            echo ""
            echo "  00) Welcome (Quick test)"
            echo "  02) Build, Test, Ship"
            echo "  03) Milestone Unlocked"
            echo "  04) Share Journey"
            echo "  05) TinyTorch Logo & Story"
            echo ""
            read -p "Choose demo [00,02-05]: " demo_num
            echo ""
            
            cd "$(dirname "$0")/../../.."
            generate "$demo_num"
            ;;
        3)
            echo -e "${CYAN}${BOLD}🔥 Full Workflow: Validate → Time → Generate${NC}"
            echo ""
            
            # Step 1: Validate + collect timing
            if validate true "$skip_clone"; then
                echo ""
                
                # Step 2: Generate
                echo -e "${BOLD}Which demo to generate?${NC}"
                echo ""
                echo "  00) Welcome (Quick test)"
                echo "  02) Build, Test, Ship"
                echo "  03) Milestone Unlocked"
                echo "  04) Share Journey"
                echo "  05) TinyTorch Logo & Story"
                echo ""
                read -p "Choose demo [00,02-05]: " demo_num
                echo ""
                
                cd "$(dirname "$0")/../../.."
                
                if generate "$demo_num"; then
                    echo ""
                    echo -e "${GREEN}${BOLD}🎉 Complete! All steps done successfully.${NC}"
                    echo ""
                fi
            else
                echo -e "${RED}Validation failed. Fix issues before generating.${NC}"
                exit 1
            fi
            ;;
        4)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac
}

# ============================================================================
# COMMAND LINE MODE (optional)
# ============================================================================

usage() {
    show_banner
    cat << USAGE
Usage:
  $0                         Interactive mode (recommended)
  $0 validate [--skip-clone] Validate only (no timing)
  $0 generate <num>          Generate demo <num>
  $0 full <num> [--skip-clone] Full workflow: validate + timing + generate

Options:
  --skip-clone               Skip git clone (use local copy, faster for debugging)

Examples:
  $0                         # Interactive Q&A (easiest)
  $0 validate                # Just validate
  $0 validate --skip-clone   # Validate without cloning (debug mode)
  $0 generate 01             # Just generate demo 01
  $0 full 01                 # Full workflow for demo 01 (recommended)
  $0 full 01 --skip-clone    # Full workflow, skip clone (debug mode)

USAGE
}

# ============================================================================
# MAIN
# ============================================================================

check_vhs

# No arguments = interactive
if [ $# -eq 0 ]; then
    interactive
    exit 0
fi

# Command line mode
case "$1" in
    validate)
        skip_clone="false"
        if [ "$2" = "--skip-clone" ]; then
            skip_clone="true"
        fi
        validate false "$skip_clone"
        ;;
    generate)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: generate requires demo number${NC}"
            usage
            exit 1
        fi
        cd "$(dirname "$0")/../../.."
        generate "$2"
        ;;
    full)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: full requires demo number${NC}"
            usage
            exit 1
        fi
        
        skip_clone="false"
        if [ "$3" = "--skip-clone" ]; then
            skip_clone="true"
        fi
        
        echo -e "${CYAN}${BOLD}🔥 Full Workflow: Validate → Time → Generate${NC}"
        echo ""
        
        if validate true "$skip_clone"; then
            echo ""
            cd "$(dirname "$0")/../../.."
            
            if generate "$2"; then
                echo ""
                echo -e "${GREEN}${BOLD}🎉 Complete! All steps done successfully.${NC}"
                echo ""
            fi
        else
            exit 1
        fi
        ;;
    --help|-h)
        usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        usage
        exit 1
        ;;
esac

