#!/bin/bash
# VHS to Terminalizer Converter
# Converts VHS tape files to Terminalizer YAML configs for polished output

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🎬 VHS → Terminalizer Converter${NC}"
echo "======================================"
echo ""

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Usage: $0 <tape-file> [output-yml]${NC}"
    echo ""
    echo "Examples:"
    echo "  $0 docs/_static/demos/tapes/01-zero-to-ready.tape"
    echo "  $0 docs/_static/demos/tapes/01-zero-to-ready.tape 01-custom.yml"
    exit 1
fi

TAPE_FILE="$1"
OUTPUT_YML="${2:-$(basename "$TAPE_FILE" .tape).yml}"

# Check if tape file exists
if [ ! -f "$TAPE_FILE" ]; then
    echo -e "${RED}❌ Tape file not found: $TAPE_FILE${NC}"
    exit 1
fi

echo -e "${BLUE}📄 Input:  $TAPE_FILE${NC}"
echo -e "${BLUE}📝 Output: $OUTPUT_YML${NC}"
echo ""

# Check if terminalizer is installed
if ! command -v terminalizer &> /dev/null; then
    echo -e "${YELLOW}⚠️  Terminalizer not installed${NC}"
    echo ""
    echo "Install with: npm install -g terminalizer"
    echo ""
    echo -e "${BLUE}For now, creating a template config...${NC}"
fi

# Extract metadata from VHS tape
WIDTH=$(grep "^Set Width" "$TAPE_FILE" | awk '{print $3}')
HEIGHT=$(grep "^Set Height" "$TAPE_FILE" | awk '{print $3}')
THEME=$(grep "^Set Theme" "$TAPE_FILE" | sed -n 's/Set Theme "\(.*\)"/\1/p')
OUTPUT_GIF=$(grep "^Output" "$TAPE_FILE" | awk '{print $2}' | tr -d '"')

# Convert Catppuccin Mocha to Terminalizer theme
TERM_THEME="monokai"  # Default fallback
case "$THEME" in
    "Catppuccin Mocha")
        TERM_THEME="monokai"
        ;;
    "Dracula")
        TERM_THEME="dracula"
        ;;
esac

echo -e "${GREEN}📊 Extracted Settings:${NC}"
echo "  Dimensions: ${WIDTH}x${HEIGHT}"
echo "  Theme: $THEME → $TERM_THEME"
echo "  Output: $OUTPUT_GIF"
echo ""

# Parse VHS commands and convert to Terminalizer format
echo -e "${BLUE}🔄 Converting commands...${NC}"

# Create Terminalizer config header
cat > "$OUTPUT_YML" << EOF
# Terminalizer Config
# Auto-generated from VHS tape: $TAPE_FILE
# Date: $(date +"%Y-%m-%d %H:%M:%S")

config:
  # Recording settings
  command: bash -l
  cwd: /tmp

  # Terminal dimensions
  cols: $((WIDTH / 10))
  rows: $((HEIGHT / 20))

  # Rendering
  repeat: 0
  quality: 100
  frameDelay: auto
  maxIdleTime: 2000

  # Style
  theme:
    background: "transparent"
    foreground: "#abb2bf"
    cursor: "#c678dd"

  # Watermark
  watermark:
    imagePath: null
    style:
      position: absolute
      right: 15px
      bottom: 15px
      width: 100px
      opacity: 0.9

  # Cursor style
  cursorStyle: block

  # Font
  fontFamily: "JetBrains Mono, Monaco, Menlo, monospace"
  fontSize: 18
  lineHeight: 1.2
  letterSpacing: 0

# Recording frames
records:
EOF

# Parse VHS tape and convert commands
# This is a simplified converter - it extracts Type and Enter commands
# and converts Sleep to delays

FRAME_COUNT=0
DELAY_MS=0

while IFS= read -r line; do
    # Skip comments and empty lines
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "$line" ]] && continue

    # Parse Type commands
    if [[ "$line" =~ ^Type[[:space:]]+"(.+)"$ ]]; then
        TEXT="${BASH_REMATCH[1]}"
        if [ $FRAME_COUNT -gt 0 ]; then
            # Add accumulated delay to previous frame
            sed -i '' "s/delay: 0$/delay: $DELAY_MS/" "$OUTPUT_YML"
            DELAY_MS=0
        fi

        # Add typing frame
        cat >> "$OUTPUT_YML" << FRAME
  - delay: 0
    content: "$TEXT"
FRAME
        ((FRAME_COUNT++))

    # Parse Enter commands
    elif [[ "$line" =~ ^Enter ]]; then
        if [ $FRAME_COUNT -gt 0 ]; then
            sed -i '' "s/delay: 0$/delay: $DELAY_MS/" "$OUTPUT_YML"
            DELAY_MS=0
        fi

        cat >> "$OUTPUT_YML" << FRAME
  - delay: 0
    content: "\\r"
FRAME
        ((FRAME_COUNT++))

    # Parse Sleep commands and accumulate
    elif [[ "$line" =~ ^Sleep[[:space:]]+([0-9]+)(ms|s) ]]; then
        SLEEP_VAL="${BASH_REMATCH[1]}"
        SLEEP_UNIT="${BASH_REMATCH[2]}"

        if [ "$SLEEP_UNIT" = "s" ]; then
            DELAY_MS=$((DELAY_MS + SLEEP_VAL * 1000))
        else
            DELAY_MS=$((DELAY_MS + SLEEP_VAL))
        fi

    # Parse Wait commands and convert to reasonable delay
    elif [[ "$line" =~ ^Wait[[:space:]]+([0-9]+)?s?[[:space:]]*Line ]]; then
        WAIT_TIMEOUT="${BASH_REMATCH[1]:-5}"
        # Convert Wait to a reasonable delay (half the timeout)
        DELAY_MS=$((DELAY_MS + (WAIT_TIMEOUT * 1000 / 2)))
    fi

done < "$TAPE_FILE"

# Apply final delay if any
if [ $DELAY_MS -gt 0 ] && [ $FRAME_COUNT -gt 0 ]; then
    sed -i '' "s/delay: 0$/delay: $DELAY_MS/" "$OUTPUT_YML"
fi

echo -e "${GREEN}✅ Converted $FRAME_COUNT frames${NC}"
echo ""

# Add usage instructions
cat << USAGE

${BLUE}📋 Next Steps:${NC}

1. ${YELLOW}Review the generated config:${NC}
   cat $OUTPUT_YML

2. ${YELLOW}Record with Terminalizer:${NC}
   terminalizer record $OUTPUT_YML -c $OUTPUT_YML

   ${BLUE}Or manually replay commands:${NC}
   The config contains the command sequence.
   You may need to run commands manually in the terminal.

3. ${YELLOW}Render to GIF:${NC}
   terminalizer render $OUTPUT_YML -o ${OUTPUT_GIF}

4. ${YELLOW}Preview:${NC}
   terminalizer play $OUTPUT_YML

${BLUE}💡 Tips:${NC}
- Terminalizer configs may need manual adjustment
- The converter provides a starting point
- Check timing and add missing pauses
- Terminalizer creates smoother, more polished GIFs

${YELLOW}⚠️  Note:${NC}
Terminalizer requires manual execution, unlike VHS.
The converted config shows the command sequence,
but you'll need to type them during recording.

For fully automated workflow, stick with VHS.
Use Terminalizer only for final polish.

USAGE

echo -e "${GREEN}✅ Conversion complete!${NC}"
