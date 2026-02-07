#!/bin/bash
#
# FRD Evaluation Script
# Evaluates Frechet Radiomics Distance on sweep output folders
#
# Usage: ./evaluate_frd.sh /path/to/sweep_folder [--workers N]
#

set -e

# Default values
NUM_WORKERS=""
SWEEP_FOLDER=""
# Default to chest X-ray optimized config
SCRIPT_DIR_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR_DEFAULT/evaluation/configs/chest_xray.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 /path/to/sweep_folder [--config /path/to/config.yaml]"
            echo ""
            echo "Arguments:"
            echo "  sweep_folder        Path to the sweep output folder"
            echo "  --config FILE       Path to PyRadiomics config YAML file"
            echo "                      Default: evaluation/configs/chest_xray.yaml"
            echo ""
            echo "Example:"
            echo "  $0 /mnt/data/graph/outputs/sweep_20260107_094119"
            exit 0
            ;;
        *)
            if [[ -z "$SWEEP_FOLDER" ]]; then
                SWEEP_FOLDER="$1"
            else
                echo "Error: Unknown argument '$1'"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate sweep folder argument
if [[ -z "$SWEEP_FOLDER" ]]; then
    echo "Error: Sweep folder path is required"
    echo "Usage: $0 /path/to/sweep_folder [--workers N]"
    exit 1
fi

if [[ ! -d "$SWEEP_FOLDER" ]]; then
    echo "Error: Sweep folder does not exist: $SWEEP_FOLDER"
    exit 1
fi

# Get absolute path
SWEEP_FOLDER=$(realpath "$SWEEP_FOLDER")

# Script directory (for locating frd module)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRD_DIR="$SCRIPT_DIR/frd-score/frd_v1"
FRD_SCRIPT="$FRD_DIR/compute_frd.py"

if [[ ! -f "$FRD_SCRIPT" ]]; then
    echo "Error: FRD script not found at: $FRD_SCRIPT"
    exit 1
fi

# Configuration
# Dynamically discover method directories (any subdir containing pathology folders with results)
METHODS=()
for dir in "$SWEEP_FOLDER"/*/; do
    dir_name=$(basename "$dir")
    # Skip known non-method directories
    if [[ "$dir_name" == "plots" || "$dir_name" == "logs" || "$dir_name" == "__pycache__" ]]; then
        continue
    fi
    # Check if this directory contains pathology subdirs with image folders
    for subdir in "$dir"/*/; do
        if [[ -d "$subdir/original" ]] || [[ -d "$subdir/reconstructed" ]]; then
            METHODS+=("$dir_name")
            break
        fi
    done
done

if [[ ${#METHODS[@]} -eq 0 ]]; then
    echo "Error: No valid method directories found in $SWEEP_FOLDER"
    exit 1
fi

# Dynamically discover conditions (pathologies) from method directories
CONDITIONS=()
for method in "${METHODS[@]}"; do
    for pathology_dir in "$SWEEP_FOLDER/$method"/*/; do
        if [[ -d "$pathology_dir" ]]; then
            pathology_name=$(basename "$pathology_dir")
            # Add if not already in array
            if [[ ! " ${CONDITIONS[*]} " =~ " ${pathology_name} " ]]; then
                CONDITIONS+=("$pathology_name")
            fi
        fi
    done
done

COMPARISONS=(
    "original:reconstructed"
    "original:increase"
    "original:decrease"
    "reconstructed:increase"
    "reconstructed:decrease"
)

# Output file
OUTPUT_CSV="$SWEEP_FOLDER/frd_results.csv"

# Validate and build config argument
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi
CONFIG_FILE=$(realpath "$CONFIG_FILE")
CONFIG_ARG="--params_file $CONFIG_FILE"

echo "=========================================="
echo "FRD Evaluation Script"
echo "=========================================="
echo "Sweep folder: $SWEEP_FOLDER"
echo "Output file:  $OUTPUT_CSV"
echo "Config:       $CONFIG_FILE"
echo "Methods:      ${METHODS[*]}"
echo "Conditions:   ${CONDITIONS[*]}"
echo "=========================================="
echo ""

# Initialize CSV with header
echo "method,condition,comparison,frd_score" > "$OUTPUT_CSV"

# Counter for progress
TOTAL=$((${#METHODS[@]} * ${#CONDITIONS[@]} * ${#COMPARISONS[@]}))
CURRENT=0

# Iterate over methods
for method in "${METHODS[@]}"; do
    METHOD_DIR="$SWEEP_FOLDER/$method"
    
    if [[ ! -d "$METHOD_DIR" ]]; then
        echo "Warning: Method directory not found, skipping: $METHOD_DIR"
        continue
    fi
    
    # Iterate over conditions
    for condition in "${CONDITIONS[@]}"; do
        CONDITION_DIR="$METHOD_DIR/$condition"
        
        if [[ ! -d "$CONDITION_DIR" ]]; then
            echo "Warning: Condition directory not found, skipping: $CONDITION_DIR"
            continue
        fi
        
        # Iterate over comparisons
        for comparison in "${COMPARISONS[@]}"; do
            CURRENT=$((CURRENT + 1))
            
            # Parse comparison (format: "dir1:dir2")
            DIR1="${comparison%%:*}"
            DIR2="${comparison##*:}"
            
            PATH1="$CONDITION_DIR/$DIR1"
            PATH2="$CONDITION_DIR/$DIR2"
            
            # Check if both directories exist and have images
            if [[ ! -d "$PATH1" ]]; then
                echo "[$CURRENT/$TOTAL] Skipping $method/$condition ($DIR1 vs $DIR2): $DIR1 directory not found"
                continue
            fi
            
            if [[ ! -d "$PATH2" ]]; then
                echo "[$CURRENT/$TOTAL] Skipping $method/$condition ($DIR1 vs $DIR2): $DIR2 directory not found"
                continue
            fi
            
            # Check for images in directories
            IMG_COUNT1=$(find "$PATH1" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
            IMG_COUNT2=$(find "$PATH2" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
            
            if [[ $IMG_COUNT1 -eq 0 ]]; then
                echo "[$CURRENT/$TOTAL] Skipping $method/$condition ($DIR1 vs $DIR2): No images in $DIR1"
                continue
            fi
            
            if [[ $IMG_COUNT2 -eq 0 ]]; then
                echo "[$CURRENT/$TOTAL] Skipping $method/$condition ($DIR1 vs $DIR2): No images in $DIR2"
                continue
            fi
            
            echo "[$CURRENT/$TOTAL] Computing FRD: $method/$condition ($DIR1 vs $DIR2)..."
            echo "           $DIR1: $IMG_COUNT1 images, $DIR2: $IMG_COUNT2 images"
            
            # Run FRD computation (frd_v1 uses --image_folder1/2 args, must run from its directory)
            FRD_OUTPUT=$(cd "$FRD_DIR" && python compute_frd.py --image_folder1 "$PATH1" --image_folder2 "$PATH2" $CONFIG_ARG 2>&1) || {
                echo "           Error computing FRD, skipping..."
                continue
            }
            
            # Extract FRD value from output (format: "FRD = <value>")
            FRD_VALUE=$(echo "$FRD_OUTPUT" | grep -oP 'FRD\s*=\s*\K[-\d.]+' | tail -1)
            
            if [[ -z "$FRD_VALUE" ]]; then
                echo "           Warning: Could not parse FRD value from output"
                echo "           Output was: $FRD_OUTPUT"
                continue
            fi
            
            echo "           FRD = $FRD_VALUE"
            
            # Append to CSV
            echo "$method,$condition,${DIR1}_vs_${DIR2},$FRD_VALUE" >> "$OUTPUT_CSV"
        done
    done
done

echo ""
echo "=========================================="
echo "FRD Evaluation Complete"
echo "=========================================="
echo "Results saved to: $OUTPUT_CSV"
echo ""
echo "Summary:"
cat "$OUTPUT_CSV"

