#!/bin/bash
# Template Ablation Sweep for Text-Based Counterfactual Generation
# Compares different prompt template styles (default, freeform, detailed)
# Uses text_ddpm method for each template set across specified pathologies

set -e  # Exit on error

# ===========================================
# Configurable Environment Variables
# ===========================================
PATHOLOGIES="${PATHOLOGIES:-Atelectasis Consolidation Infiltration Pneumothorax Edema Emphysema Fibrosis Effusion Pneumonia Pleural_Thickening Cardiomegaly Nodule Mass Hernia Fracture }"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/template_ablation_$(date +%Y%m%d_%H%M%S)}"

# Template sets to compare (space-separated): default freeform detailed
TEMPLATE_SETS="${TEMPLATE_SETS:-default freeform detailed}"

# Paths
IMAGE_ROOT="${IMAGE_ROOT:-data/PNG/PNG}"

# Generation parameters
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-15}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-100}"
SKIP="${SKIP:-36}"

# Classifier models for evaluation (space-separated)
CLASSIFIER_MODELS="${CLASSIFIER_MODELS:-densenet121-res224-all jfhealthcare}"

# ===========================================
# Main Script
# ===========================================
echo "=========================================="
echo "Template Ablation Sweep"
echo "=========================================="
echo "Output directory: ${OUTPUT_BASE}"
echo "Pathologies: ${PATHOLOGIES}"
echo "Template sets: ${TEMPLATE_SETS}"
echo "Device: ${DEVICE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Guidance scale: ${GUIDANCE_SCALE}"
echo "Inference steps: ${NUM_INFERENCE_STEPS}"
echo "Skip: ${SKIP}"
echo "Classifier models: ${CLASSIFIER_MODELS}"
echo ""

mkdir -p "${OUTPUT_BASE}"

# Save experiment config
cat > "${OUTPUT_BASE}/config.json" << EOF
{
    "experiment": "template_ablation",
    "pathologies": "${PATHOLOGIES}",
    "template_sets": "${TEMPLATE_SETS}",
    "method": "text_ddpm_fixed",
    "guidance_scale": ${GUIDANCE_SCALE},
    "num_inference_steps": ${NUM_INFERENCE_STEPS},
    "skip": ${SKIP},
    "classifier_models": "${CLASSIFIER_MODELS}"
}
EOF

for template_set in ${TEMPLATE_SETS}; do
    echo ""
    echo "=========================================="
    echo "Template Set: ${template_set}"
    echo "=========================================="
    
    for pathology in ${PATHOLOGIES}; do
        echo ""
        echo "[${template_set}] Running ${pathology}..."
        python -m evaluation.counterfactual_evaluator_text \
            --method text_ddpm_fixed \
            --template_set "${template_set}" \
            --image_root "${IMAGE_ROOT}" \
            --target_pathology "${pathology}" \
            --device "${DEVICE}" \
            --batch_size "${BATCH_SIZE}" \
            --guidance_scale "${GUIDANCE_SCALE}" \
            --num_inference_steps "${NUM_INFERENCE_STEPS}" \
            --skip "${SKIP}" \
            --classifier_models ${CLASSIFIER_MODELS} \
            --save_images \
            --output_dir "${OUTPUT_BASE}/${template_set}/${pathology}"
        
        echo "Completed: ${template_set}/${pathology}"
    done
done

echo ""
echo "=========================================="
echo "Template Ablation Sweep Complete!"
echo "=========================================="
echo "Results saved to: ${OUTPUT_BASE}"
echo ""

# Print results summary
echo "Results by template set:"
for template_set in ${TEMPLATE_SETS}; do
    echo ""
    echo "  ${template_set}:"
    for pathology in ${PATHOLOGIES}; do
        echo "    - ${OUTPUT_BASE}/${template_set}/${pathology}/results.json"
    done
done
echo ""

# Generate comparison summary if jq is available
if command -v jq &> /dev/null; then
    echo "Generating comparison summary..."
    
    SUMMARY_FILE="${OUTPUT_BASE}/summary.csv"
    echo "template_set,pathology,flip_rate_increase,flip_rate_decrease,intended_flip_rate_increase,intended_flip_rate_decrease,lpips_mean,ssim_mean" > "${SUMMARY_FILE}"
    
    for template_set in ${TEMPLATE_SETS}; do
        for pathology in ${PATHOLOGIES}; do
            RESULTS_FILE="${OUTPUT_BASE}/${template_set}/${pathology}/results.json"
            if [ -f "${RESULTS_FILE}" ]; then
                # Extract metrics using jq
                FLIP_INCREASE=$(jq -r '.evaluation_increase.classifier.aggregated.target_metrics.flip_rate // "N/A"' "${RESULTS_FILE}" 2>/dev/null || echo "N/A")
                FLIP_DECREASE=$(jq -r '.evaluation_decrease.classifier.aggregated.target_metrics.flip_rate // "N/A"' "${RESULTS_FILE}" 2>/dev/null || echo "N/A")
                INTENDED_INCREASE=$(jq -r '.evaluation_increase.classifier.aggregated.target_metrics.intended_flip_rate // "N/A"' "${RESULTS_FILE}" 2>/dev/null || echo "N/A")
                INTENDED_DECREASE=$(jq -r '.evaluation_decrease.classifier.aggregated.target_metrics.intended_flip_rate // "N/A"' "${RESULTS_FILE}" 2>/dev/null || echo "N/A")
                LPIPS=$(jq -r '.evaluation_combined.image.aggregated.lpips.mean // "N/A"' "${RESULTS_FILE}" 2>/dev/null || echo "N/A")
                SSIM=$(jq -r '.evaluation_combined.image.aggregated.ssim.mean // "N/A"' "${RESULTS_FILE}" 2>/dev/null || echo "N/A")
                
                echo "${template_set},${pathology},${FLIP_INCREASE},${FLIP_DECREASE},${INTENDED_INCREASE},${INTENDED_DECREASE},${LPIPS},${SSIM}" >> "${SUMMARY_FILE}"
            fi
        done
    done
    
    echo "Summary saved to: ${SUMMARY_FILE}"
    echo ""
    cat "${SUMMARY_FILE}"
fi

