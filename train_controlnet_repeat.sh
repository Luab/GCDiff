#!/bin/bash
#
# Training script for ControlNet with graph conditioning
# Strategy: REPEAT - simple repetition/stacking (no additional params)
#
# Requires: new_embeddings_768.pkl (original [768] embeddings)
#
# Usage: ./train_controlnet_repeat.sh
#

set -e  # Exit on error

# Set HTTP proxy for downloading models
#export HTTP_PROXY="http://admin:Haruhi_123@192.168.1.1:1338"
#export HTTPS_PROXY="http://admin:Haruhi_123@192.168.1.1:1338"

# Navigate to project directory
cd /home/luab/graph

# Activate virtual environment
source graph/bin/activate

# Check if data files exist
if [ ! -f "reports_processed.csv" ]; then
    echo "Error: reports_processed.csv not found!"
    exit 1
fi

if [ ! -f "new_embeddings_768.pkl" ]; then
    echo "Error: new_embeddings_768.pkl not found!"
    echo "This strategy requires original [768] embeddings."
    exit 1
fi

if [ ! -d "/mnt/data/CheXpert/PNG" ]; then
    echo "Error: CheXpert image directory not found at /mnt/data/CheXpert/PNG"
    exit 1
fi

echo "================================"
echo "ControlNet Training Configuration"
echo "================================"
echo "Strategy: repeat (simple repetition)"
echo "CSV path: reports_processed.csv"
echo "Graph embeddings: new_embeddings_768.pkl"
echo "Image root: /mnt/data/CheXpert/PNG"
echo "Output dir: checkpoints/controlnet-repeat"
echo "Note: No additional parameters"
echo "================================"
echo ""

# Run training
python train_controlnet.py \
    --csv_path reports_processed.csv \
    --graph_embeddings new_embeddings_768.pkl \
    --image_root /mnt/data/CheXpert/PNG \
    --output_dir checkpoints/controlnet-repeat \
    --embedding_strategy repeat \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --num_workers 10 \
    --log_every_n_steps 10 \
    --save_every_n_epochs 1 \
    --wandb_project controlnet-graph-conditioning \
    --wandb_run_name repeat-simple \
    --seed 42

echo ""
echo "✅ Training complete!"


