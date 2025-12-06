#!/bin/bash
# TinyTorch Demo Recorder
# Single script for calibration and demo generation

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Helper function to get demo name from number
get_demo_name() {
    case $1 in
        00) echo "00-test" ;;
        01) echo "01-zero-to-ready" ;;
        02) echo "02-build-test-ship" ;;
        03) echo "03-milestone-unlocked" ;;
        04) echo "04-share-journey" ;;
        *) echo "" ;;
    esac
}

# Check if VHS is installed
check_vhs() {
    if ! command -v vhs &> /dev/null; then
        echo -e "${RED}❌ Error: VHS is not installed${NC}"
        echo ""
        echo "Install VHS:"
        echo "  macOS:  brew install vhs"
        echo "  Linux:  go install github.com/charmbracelet/vhs@latest"
        echo ""
        exit 1
    fi
}

# Calibration mode
calibrate() {
    echo -e "${CYAN}🔥 TinyTorch Demo Calibration${NC}"
    echo "======================================"
    echo ""
    echo "Measuring actual command execution times..."
    echo ""

    # Output file
    TIMINGS_FILE="docs/_static/demos/.timings.json"

    # Clean slate
    echo -e "${YELLOW}🧹 Cleaning /tmp/TinyTorch...${NC}"
    rm -rf /tmp/TinyTorch
    echo ""

    # Start timing collection
    echo -e "${BLUE}📊 Measuring Command Timings${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Initialize JSON
    cat > "$TIMINGS_FILE" << 'EOF'
{
  "calibration_date": "",
  "system": {
    "os": "",
    "arch": ""
  },
  "timings": {}
}
EOF

    # Helper function to time a command
    time_command() {
        local name="$1"
        local cmd="$2"

        echo -e "${CYAN}⏱  Timing: $name${NC}"

        local start=$(date +%s%N)
        eval "$cmd" > /dev/null 2>&1 || true
        local end=$(date +%s%N)

        local duration_ns=$((end - start))
        local duration_ms=$((duration_ns / 1000000))
        local duration_s=$(echo "scale=2; $duration_ms / 1000" | bc)

        echo -e "   ${GREEN}✓${NC} Duration: ${YELLOW}${duration_s}s${NC} (${duration_ms}ms)"
        echo ""

        # Store in JSON (we'll use jq if available)
        if command -v jq &> /dev/null; then
            jq ".timings[\"$name\"] = {\"ms\": $duration_ms, \"seconds\": $duration_s}" "$TIMINGS_FILE" > "${TIMINGS_FILE}.tmp"
            mv "${TIMINGS_FILE}.tmp" "$TIMINGS_FILE"
        fi

        echo "$duration_ms"
    }

    # Measure git clone
    cd /tmp
    GIT_CLONE_TIME=$(time_command "git_clone" "git clone https://github.com/mlsysbook/TinyTorch.git")

    # Measure setup
    cd TinyTorch
    SETUP_TIME=$(time_command "setup_environment" "./setup-environment.sh")

    # Activate and measure TITO commands
    source .venv/bin/activate 2>/dev/null || source activate.sh

    MODULE_STATUS_TIME=$(time_command "tito_module_status" "tito module status")
    LOGO_TIME=$(time_command "tito_logo" "tito logo")

    # Add system info
    if command -v jq &> /dev/null; then
        jq ".calibration_date = \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\" | .system.os = \"$(uname -s)\" | .system.arch = \"$(uname -m)\"" "$TIMINGS_FILE" > "${TIMINGS_FILE}.tmp"
        mv "${TIMINGS_FILE}.tmp" "$TIMINGS_FILE"
    fi

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✅ Calibration Complete!${NC}"
    echo ""

    # Summary
    echo -e "${CYAN}📋 Timing Summary${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    printf "%-30s %10s\n" "Command" "Time (s)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "%-30s %10s\n" "git clone" "$(echo "scale=2; $GIT_CLONE_TIME / 1000" | bc)"
    printf "%-30s %10s\n" "./setup-environment.sh" "$(echo "scale=2; $SETUP_TIME / 1000" | bc)"
    printf "%-30s %10s\n" "tito module status" "$(echo "scale=2; $MODULE_STATUS_TIME / 1000" | bc)"
    printf "%-30s %10s\n" "tito logo" "$(echo "scale=2; $LOGO_TIME / 1000" | bc)"
    echo ""

    # Recommendations
    echo -e "${CYAN}💡 Recommended VHS Wait Times${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Update your tape files with these timeouts:"
    echo ""

    # Add 20% buffer to measured times
    GIT_TIMEOUT=$((GIT_CLONE_TIME * 120 / 100 / 1000))
    SETUP_TIMEOUT=$((SETUP_TIME * 120 / 100 / 1000))

    cat << RECOMMENDATIONS
Git clone:
  Wait+Line@10ms /🔥/ ${GIT_TIMEOUT}s

Setup:
  Wait+Line@10ms /🔥/ ${SETUP_TIMEOUT}s

TITO commands (fast):
  Wait+Line@10ms /🔥/  # No timeout needed
RECOMMENDATIONS

    echo ""
    echo -e "${BLUE}Timings saved to: $TIMINGS_FILE${NC}"
    echo ""
}

# Generate mode
generate() {
    local tape_num="$1"
    local demo_name=$(get_demo_name "$tape_num")

    if [ -z "$demo_name" ]; then
        echo -e "${RED}❌ Invalid demo number: $tape_num${NC}"
        echo "Valid demos: 00 (test), 01, 02, 03, 04"
        exit 1
    fi

    local tape_file="docs/_static/demos/tapes/${demo_name}.tape"
    local output_gif="${demo_name}.gif"

    echo -e "${BLUE}🎬 TinyTorch Demo Generator${NC}"
    echo "======================================"
    echo ""
    echo -e "${BLUE}📹 Demo ${tape_num}: ${demo_name}${NC}"
    echo ""

    # Check if tape file exists
    if [ ! -f "$tape_file" ]; then
        echo -e "${RED}❌ Tape file not found: $tape_file${NC}"
        exit 1
    fi

    # Clean up /tmp/TinyTorch
    echo -e "${YELLOW}🧹 Cleaning /tmp/TinyTorch...${NC}"
    rm -rf /tmp/TinyTorch

    # Generate the GIF
    echo -e "${GREEN}🎬 Recording demo (this may take 1-2 minutes)...${NC}"
    echo ""

    # Run VHS with the tape file
    if vhs "$tape_file" 2>&1; then
        # Check if GIF was created
        if [ -f "$output_gif" ]; then
            local size=$(du -h "$output_gif" | cut -f1)
            echo ""
            echo -e "${GREEN}✅ Success!${NC}"
            echo -e "${GREEN}   Created: $output_gif ($size)${NC}"

            # Move to docs/_static/demos/ for website use
            mv "$output_gif" "docs/_static/demos/$output_gif"
            echo -e "${GREEN}   Moved to: docs/_static/demos/$output_gif${NC}"
            echo ""
            echo -e "${BLUE}💡 Preview:${NC}"
            echo "  open docs/_static/demos/$output_gif"
        else
            echo -e "${RED}❌ Failed to create GIF${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ VHS recording failed${NC}"
        exit 1
    fi
}

# Show usage
usage() {
    cat << USAGE
TinyTorch Demo Recorder

Usage:
  $0 --calibrate                 Measure command execution times
  $0 --tape <number>             Generate a specific demo GIF

Options:
  --calibrate                    Run timing calibration
  --tape 00|01|02|03|04          Generate demo GIF for specified tape

Examples:
  $0 --calibrate                 # Measure timings on your machine
  $0 --tape 00                   # Generate test demo
  $0 --tape 01                   # Generate "Zero to Ready" demo

Available demos:
  00  Quick test (5 seconds)
  01  Zero to Ready (clone → setup → activate)
  02  Build, Test, Ship (module completion)
  03  Milestone Unlocked (achievement system)
  04  Share Your Journey (community features)

USAGE
}

# Main execution
check_vhs

# Parse arguments
if [ $# -eq 0 ]; then
    usage
    exit 0
fi

case "$1" in
    --calibrate)
        calibrate
        ;;
    --tape)
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Error: --tape requires a demo number${NC}"
            echo ""
            usage
            exit 1
        fi
        generate "$2"
        ;;
    --help|-h)
        usage
        ;;
    *)
        echo -e "${RED}❌ Unknown option: $1${NC}"
        echo ""
        usage
        exit 1
        ;;
esac
