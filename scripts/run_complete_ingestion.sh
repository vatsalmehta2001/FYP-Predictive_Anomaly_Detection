#!/bin/bash
# Complete data ingestion pipeline with verification
#
# Usage:
#   ./scripts/run_complete_ingestion.sh [--clean]

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}================================${NC}"
echo -e "${CYAN}Grid Guardian: Data Ingestion${NC}"
echo -e "${CYAN}================================${NC}\n"

# Parse arguments
CLEAN_MODE=false
if [[ "$1" == "--clean" ]]; then
    CLEAN_MODE=true
    echo -e "${YELLOW}Clean mode enabled: Will delete existing data${NC}\n"
fi

# Step 1: Check current status
echo -e "${CYAN}[1/6] Checking current status...${NC}"
poetry run python scripts/cleanup_and_verify.py --check-only

# Step 2: Cleanup if requested
if $CLEAN_MODE; then
    echo -e "\n${YELLOW}[2/6] Cleaning up existing data...${NC}"
    poetry run python scripts/cleanup_and_verify.py --cleanup all --force
else
    echo -e "\n${CYAN}[2/6] Skipping cleanup (use --clean to force)${NC}"
fi

# Step 3: Ingest LCL with caffeinate
echo -e "\n${CYAN}[3/6] Ingesting LCL dataset...${NC}"
echo -e "${YELLOW}This may take several hours for 8.54GB / 167M records${NC}"
echo -e "${YELLOW}Progress will be saved every 10k records${NC}"
echo -e "${YELLOW}Using caffeinate to prevent laptop sleep${NC}\n"

caffeinate -i bash -c "cd $(pwd) && poetry run python -m fyp.ingestion.cli lcl \
    --input-root data/raw/lcl \
    --output-root data/processed"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}LCL ingestion complete${NC}"
else
    echo -e "${RED}LCL ingestion failed${NC}"
    exit 1
fi

# Step 4: Ingest UK-DALE with 30-min downsampling
echo -e "\n${CYAN}[4/6] Ingesting UK-DALE dataset (with 30-min downsampling)...${NC}"

caffeinate -i bash -c "cd $(pwd) && poetry run python -m fyp.ingestion.cli ukdale \
    --input-root data/raw/ukdale \
    --output-root data/processed"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}UK-DALE ingestion complete${NC}"
else
    echo -e "${RED}UK-DALE ingestion failed${NC}"
    exit 1
fi

# Step 5: Verify UK-DALE intervals
echo -e "\n${CYAN}[5/6] Verifying UK-DALE has 30-minute data...${NC}"
poetry run python scripts/verify_ukdale_intervals.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}UK-DALE verification passed${NC}"
else
    echo -e "${RED}UK-DALE verification failed${NC}"
    exit 1
fi

# Step 6: Final standardization check
echo -e "\n${CYAN}[6/6] Final standardization verification...${NC}"
poetry run python scripts/cleanup_and_verify.py --check-only

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}COMPLETE: All datasets ingested and standardized!${NC}"
echo -e "${GREEN}================================${NC}\n"

echo -e "${CYAN}Next steps:${NC}"
echo -e "1. Run exploratory notebooks: jupyter notebook notebooks/"
echo -e "2. Generate figures: Execute all cells in each notebook"
echo -e "3. Start baseline models: python -m fyp.runner forecast --dataset lcl\n"

