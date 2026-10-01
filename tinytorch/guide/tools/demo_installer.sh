#!/bin/bash
# Authentic TinyTorch Installer Showcase Output
# Reproduces the official install.sh flow with realistic terminal timing

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

echo ""
echo -e "  ${BOLD}Tiny${NC}${YELLOW}🔥Torch${NC} ${DIM}v0.1.19${NC}"
echo -e "  ${DIM}Don't import it. Build it.${NC}"
echo ""

sleep 0.3
echo -e "${GREEN}✓${NC} Git 2.39.5"
sleep 0.15
echo -e "${GREEN}✓${NC} Python 3.12.9 (python3.12)"
sleep 0.15
echo -e "${GREEN}✓${NC} Python venv module"
sleep 0.15
echo -e "${GREEN}✓${NC} GitHub reachable"
sleep 0.25

echo -e "${BLUE}[1/4]${NC} Downloading from GitHub..."
sleep 0.4
echo -e "${GREEN}✓${NC} Downloaded TinyTorch ${DIM}(a732692)${NC}"
sleep 0.2

echo -e "${BLUE}[2/4]${NC} Creating Python virtual environment (.venv)..."
sleep 0.3
echo -e "${GREEN}✓${NC} Virtual environment ready"
sleep 0.2

echo -e "${BLUE}[3/4]${NC} Installing dependencies & tito CLI..."
sleep 0.4
echo -e "${GREEN}✓${NC} Dependencies installed (NumPy, Jupyter, Tito CLI)"
sleep 0.2

echo -e "${BLUE}[4/4]${NC} Verifying installation..."
sleep 0.3
echo -e "${GREEN}✓${NC} All checks passed!"
echo ""
echo -e "${BOLD}🎉 Installation complete!${NC} Run ${CYAN}tito setup${NC} to initialize your profile."
echo ""
