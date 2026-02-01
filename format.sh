#!/bin/bash
# Code formatting CLI for CGrad
# Uses clang-format for clean and uniform C code formatting

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

show_help() {
    cat << EOF
CGrad Code Formatter

USAGE:
    ./format.sh [OPTIONS]

OPTIONS:
    -h, --help          Show this help message
    -c, --check         Check formatting without modifying files (dry-run)
    -f, --fix           Format all C files in place (default)
    -v, --verbose       Show detailed output

EXAMPLES:
    ./format.sh                 # Format all files
    ./format.sh --check         # Check if formatting is needed
    ./format.sh --verbose       # Format with detailed output

EOF
}

check_clang_format() {
    if ! command -v clang-format &> /dev/null; then
        echo -e "${RED}Error: clang-format is not installed${NC}"
        echo ""
        echo "Install it using:"
        echo "  macOS:   brew install clang-format"
        echo "  Ubuntu:  sudo apt-get install clang-format"
        echo "  Fedora:  sudo dnf install clang-tools-extra"
        echo ""
        exit 1
    fi
}

format_files() {
    local check_mode=$1
    local verbose=$2
    local files_found=0
    local files_formatted=0

    # Find all C source files
    while IFS= read -r -d '' file; do
        ((files_found++))

        if [ "$check_mode" = true ]; then
            # Check mode: see if file needs formatting
            if ! clang-format --dry-run --Werror "$file" &> /dev/null; then
                echo -e "${YELLOW}Would format: $file${NC}"
                ((files_formatted++))
            elif [ "$verbose" = true ]; then
                echo -e "${GREEN}OK: $file${NC}"
            fi
        else
            # Format mode: actually format the file
            if [ "$verbose" = true ]; then
                echo "Formatting: $file"
            fi
            clang-format -i "$file"
            ((files_formatted++))
        fi
    done < <(find cgrad -type f \( -name "*.c" -o -name "*.h" \) -print0)

    echo ""
    echo -e "${GREEN}Summary:${NC}"
    echo "  Files scanned: $files_found"

    if [ "$check_mode" = true ]; then
        if [ $files_formatted -eq 0 ]; then
            echo -e "  ${GREEN}All files are properly formatted!${NC}"
            exit 0
        else
            echo -e "  ${YELLOW}Files needing formatting: $files_formatted${NC}"
            echo -e "  ${YELLOW}Run './format.sh --fix' to format them${NC}"
            exit 1
        fi
    else
        echo -e "  ${GREEN}Files formatted: $files_formatted${NC}"
    fi
}

# Parse arguments
CHECK_MODE=false
VERBOSE=false
ACTION="format"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--check)
            CHECK_MODE=true
            shift
            ;;
        -f|--fix)
            CHECK_MODE=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --install)
            install_clang_format
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
echo -e "${GREEN}CGrad Code Formatter${NC}"
echo ""

check_clang_format
format_files $CHECK_MODE $VERBOSE
