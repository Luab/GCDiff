#!/bin/bash
# Sweep over 5 key CheXpert pathologies for graph, text, and hybrid models
# Uses DDPM inversion method for best quality
#
# Available methods:
#   graph       - Graph-based DDPM with ControlNet (shared noise)
#   text        - Text-based DDPM (shared noise)  
#   hybrid      - Hybrid graph+text DDPM (shared noise)
#   graph_fixed - Graph-based DDPM with independent noise (per "Edit Friendly DDPM" paper Eq. 6)
#   text_fixed  - Text-based DDPM with independent noise
#   hybrid_fixed - Hybrid with independent noise
#
# Template sets (for text methods):
#   default  - Structured ("Chest x-ray showing X")
#   freeform - Simple ("X is present")
#   detailed - Clinical ("...with enlarged cardiac silhouette")
#
# Example:
#   METHODS="graph_fixed text_fixed" ./run_pathology_sweep.sh
#   TEMPLATE_SET="detailed" METHODS="text_fixed" ./run_pathology_sweep.sh

set -e  # Exit on error

# ===========================================
# Configurable Environment Variables
# ===========================================
PATHOLOGIES="${PATHOLOGIES:-Atelectasis Consolidation Infiltration Pneumothorax Edema Emphysema Fibrosis Effusion Pneumonia Pleural_Thickening Cardiomegaly Nodule Mass Hernia Fracture }"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/sweep_$(date +%Y%m%d_%H%M%S)}"

# Methods to run (space-separated): graph text hybrid graph_fixed text_fixed hybrid_fixed
# Set to subset if you only want specific methods
METHODS="${METHODS:-graph_fixed text_fixed hybrid_fixed}"

# Paths
CONTROLNET_PATH="${CONTROLNET_PATH:-checkpoints/controlnet-linear/final/controlnet.pth}"
EMBEDDER_PATH="${EMBEDDER_PATH:-/mnt/data/diffusion_graph/graph2vec_embeddings.pkl}"
GRAPHS_PATH="${GRAPHS_PATH:-/mnt/data/diffusion_graph/reports_processed_graphs.pkl}"
IMAGE_ROOT="${IMAGE_ROOT:-data/PNG/PNG}"

# Generation parameters
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-15}"
CONTROLNET_SCALE="${CONTROLNET_SCALE:-1}"  # Default same as guidance_scale
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-100}"
SKIP="${SKIP:-36}"

# Classifier models for evaluation (space-separated)
# Options: densenet121-res224-all jfhealthcare resnet50-res512-all
CLASSIFIER_MODELS="${CLASSIFIER_MODELS:-densenet121-res224-all jfhealthcare}"

# Template set for text-based methods (default, freeform, detailed)
TEMPLATE_SET="${TEMPLATE_SET:-default}"

# ===========================================
# Main Script
# ===========================================
echo "=========================================="
echo "Pathology Sweep - Graph, Text & Hybrid Models"
echo "=========================================="
echo "Output directory: ${OUTPUT_BASE}"
echo "Pathologies: ${PATHOLOGIES}"
echo "Methods: ${METHODS}"
echo "Device: ${DEVICE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Guidance scale (text): ${GUIDANCE_SCALE}"
echo "ControlNet scale (graph): ${CONTROLNET_SCALE}"
echo "Inference steps: ${NUM_INFERENCE_STEPS}"
echo "Template set: ${TEMPLATE_SET}"
echo "Classifier models: ${CLASSIFIER_MODELS}"
echo ""

mkdir -p "${OUTPUT_BASE}"

# Save sweep configuration as JSON for reference
cat > "${OUTPUT_BASE}/sweep_config.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "pathologies": "${PATHOLOGIES}",
    "methods": "${METHODS}",
    "paths": {
        "controlnet_path": "${CONTROLNET_PATH}",
        "embedder_path": "${EMBEDDER_PATH}",
        "graphs_path": "${GRAPHS_PATH}",
        "image_root": "${IMAGE_ROOT}",
        "output_base": "${OUTPUT_BASE}"
    },
    "generation_params": {
        "device": "${DEVICE}",
        "batch_size": ${BATCH_SIZE},
        "guidance_scale": ${GUIDANCE_SCALE},
        "controlnet_scale": ${CONTROLNET_SCALE},
        "num_inference_steps": ${NUM_INFERENCE_STEPS},
        "skip": ${SKIP},
        "template_set": "${TEMPLATE_SET}"
    },
    "classifier_models": "${CLASSIFIER_MODELS}"
}
EOF
echo "Saved sweep config to: ${OUTPUT_BASE}/sweep_config.json"
echo ""

for pathology in ${PATHOLOGIES}; do
    echo ""
    echo "=========================================="
    echo "Processing: ${pathology}"
    echo "=========================================="
    
    # Graph model (DDPM)
    if [[ " ${METHODS} " == *" graph "* ]]; then
        echo ""
        echo "[GRAPH] Running ${pathology}..."
        python -m evaluation.counterfactual_evaluator_graph \
            --method ddpm \
            --controlnet_path "${CONTROLNET_PATH}" \
            --embedder_path "${EMBEDDER_PATH}" \
            --graphs_path "${GRAPHS_PATH}" \
            --image_root "${IMAGE_ROOT}" \
            --target_pathology "${pathology}" \
            --device "${DEVICE}" \
            --batch_size "${BATCH_SIZE}" \
            --guidance_scale "${GUIDANCE_SCALE}" \
            --controlnet_scale "${CONTROLNET_SCALE}" \
            --num_inference_steps "${NUM_INFERENCE_STEPS}" \
            --skip "${SKIP}" \
            --classifier_models ${CLASSIFIER_MODELS} \
            --save_images \
            --output_dir "${OUTPUT_BASE}/graph/${pathology}"
    fi
    
    # Text model (DDPM)
    if [[ " ${METHODS} " == *" text "* ]]; then
        echo ""
        echo "[TEXT] Running ${pathology}..."
        python -m evaluation.counterfactual_evaluator_text \
            --method text_ddpm \
            --image_root "${IMAGE_ROOT}" \
            --target_pathology "${pathology}" \
            --device "${DEVICE}" \
            --batch_size "${BATCH_SIZE}" \
            --guidance_scale "${GUIDANCE_SCALE}" \
            --num_inference_steps "${NUM_INFERENCE_STEPS}" \
            --skip "${SKIP}" \
            --template_set "${TEMPLATE_SET}" \
            --classifier_models ${CLASSIFIER_MODELS} \
            --save_images \
            --output_dir "${OUTPUT_BASE}/text/${pathology}"
    fi
    
    # Hybrid model (Graph ControlNet + Text conditioning)
    if [[ " ${METHODS} " == *" hybrid "* ]]; then
        echo ""
        echo "[HYBRID] Running ${pathology}..."
        python -m evaluation.counterfactual_evaluator_hybrid \
            --controlnet_path "${CONTROLNET_PATH}" \
            --embedder_path "${EMBEDDER_PATH}" \
            --graphs_path "${GRAPHS_PATH}" \
            --image_root "${IMAGE_ROOT}" \
            --target_pathology "${pathology}" \
            --device "${DEVICE}" \
            --batch_size "${BATCH_SIZE}" \
            --guidance_scale "${GUIDANCE_SCALE}" \
            --controlnet_scale "${CONTROLNET_SCALE}" \
            --num_inference_steps "${NUM_INFERENCE_STEPS}" \
            --skip "${SKIP}" \
            --template_set "${TEMPLATE_SET}" \
            --classifier_models ${CLASSIFIER_MODELS} \
            --save_images \
            --output_dir "${OUTPUT_BASE}/hybrid/${pathology}"
    fi
    
    # ===========================================
    # Fixed methods (independent noise sampling)
    # Per "An Edit Friendly DDPM Noise Space" paper Eq. 6
    # ===========================================
    
    # Graph model with fixed DDPM (independent noise)
    if [[ " ${METHODS} " == *" graph_fixed "* ]]; then
        echo ""
        echo "[GRAPH_FIXED] Running ${pathology} (independent noise)..."
        python -m evaluation.counterfactual_evaluator_graph \
            --method ddpm_fixed \
            --controlnet_path "${CONTROLNET_PATH}" \
            --embedder_path "${EMBEDDER_PATH}" \
            --graphs_path "${GRAPHS_PATH}" \
            --image_root "${IMAGE_ROOT}" \
            --target_pathology "${pathology}" \
            --device "${DEVICE}" \
            --batch_size "${BATCH_SIZE}" \
            --guidance_scale "${GUIDANCE_SCALE}" \
            --controlnet_scale "${CONTROLNET_SCALE}" \
            --num_inference_steps "${NUM_INFERENCE_STEPS}" \
            --skip "${SKIP}" \
            --classifier_models ${CLASSIFIER_MODELS} \
            --save_images \
            --output_dir "${OUTPUT_BASE}/graph_fixed/${pathology}"
    fi
    
    # Text model with fixed DDPM (independent noise)
    if [[ " ${METHODS} " == *" text_fixed "* ]]; then
        echo ""
        echo "[TEXT_FIXED] Running ${pathology} (independent noise)..."
        python -m evaluation.counterfactual_evaluator_text \
            --method text_ddpm_fixed \
            --image_root "${IMAGE_ROOT}" \
            --target_pathology "${pathology}" \
            --device "${DEVICE}" \
            --batch_size "${BATCH_SIZE}" \
            --guidance_scale "${GUIDANCE_SCALE}" \
            --num_inference_steps "${NUM_INFERENCE_STEPS}" \
            --skip "${SKIP}" \
            --template_set "${TEMPLATE_SET}" \
            --classifier_models ${CLASSIFIER_MODELS} \
            --save_images \
            --output_dir "${OUTPUT_BASE}/text_fixed/${pathology}"
    fi
    
    # Hybrid model with fixed DDPM (independent noise)
    if [[ " ${METHODS} " == *" hybrid_fixed "* ]]; then
        echo ""
        echo "[HYBRID_FIXED] Running ${pathology} (independent noise)..."
        python -m evaluation.counterfactual_evaluator_hybrid \
            --controlnet_path "${CONTROLNET_PATH}" \
            --embedder_path "${EMBEDDER_PATH}" \
            --graphs_path "${GRAPHS_PATH}" \
            --image_root "${IMAGE_ROOT}" \
            --target_pathology "${pathology}" \
            --device "${DEVICE}" \
            --batch_size "${BATCH_SIZE}" \
            --guidance_scale "${GUIDANCE_SCALE}" \
            --controlnet_scale "${CONTROLNET_SCALE}" \
            --num_inference_steps "${NUM_INFERENCE_STEPS}" \
            --skip "${SKIP}" \
            --template_set "${TEMPLATE_SET}" \
            --classifier_models ${CLASSIFIER_MODELS} \
            --save_images \
            --use_independent_noise \
            --output_dir "${OUTPUT_BASE}/hybrid_fixed/${pathology}"
    fi
    
    echo ""
    echo "Completed: ${pathology}"
done

echo ""
echo "=========================================="
echo "Sweep Complete!"
echo "=========================================="
echo "Results saved to: ${OUTPUT_BASE}"
echo ""

if [[ " ${METHODS} " == *" graph "* ]]; then
    echo "Graph results:"
    for pathology in ${PATHOLOGIES}; do
        echo "  - ${OUTPUT_BASE}/graph/${pathology}/results.json"
    done
    echo ""
fi

if [[ " ${METHODS} " == *" text "* ]]; then
    echo "Text results:"
    for pathology in ${PATHOLOGIES}; do
        echo "  - ${OUTPUT_BASE}/text/${pathology}/results.json"
    done
    echo ""
fi

if [[ " ${METHODS} " == *" hybrid "* ]]; then
    echo "Hybrid results:"
    for pathology in ${PATHOLOGIES}; do
        echo "  - ${OUTPUT_BASE}/hybrid/${pathology}/results.json"
    done
    echo ""
fi

if [[ " ${METHODS} " == *" graph_fixed "* ]]; then
    echo "Graph_fixed results (independent noise):"
    for pathology in ${PATHOLOGIES}; do
        echo "  - ${OUTPUT_BASE}/graph_fixed/${pathology}/results.json"
    done
    echo ""
fi

if [[ " ${METHODS} " == *" text_fixed "* ]]; then
    echo "Text_fixed results (independent noise):"
    for pathology in ${PATHOLOGIES}; do
        echo "  - ${OUTPUT_BASE}/text_fixed/${pathology}/results.json"
    done
    echo ""
fi

if [[ " ${METHODS} " == *" hybrid_fixed "* ]]; then
    echo "Hybrid_fixed results (independent noise):"
    for pathology in ${PATHOLOGIES}; do
        echo "  - ${OUTPUT_BASE}/hybrid_fixed/${pathology}/results.json"
    done
    echo ""
fi
