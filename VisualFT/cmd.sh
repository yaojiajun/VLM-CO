#!/bin/bash

# Unified LoRA model merging script
# Usage: ./cmd.sh <model_checkpoint_dir> [output_dir]
# Example: ./cmd.sh ../output_cvrp_vision/checkpoint-5000 ../merged_cvrp_vision

# Check if model directory argument is provided
if [ -z "$1" ]; then
    echo "Error: Model checkpoint directory not specified!"
    echo "Usage: $0 <model_checkpoint_dir> [output_dir]"
    echo "Example: $0 ../output_cvrp_vision/checkpoint-5000 ../merged_cvrp_vision"
    exit 1
fi

MODEL_DIR="$1"

# Check if model directory exists
if [ ! -d "${MODEL_DIR}" ]; then
    echo "Error: Model directory '${MODEL_DIR}' does not exist!"
    exit 1
fi

# Set output directory (default: model_dir + "_merged")
if [ -z "$2" ]; then
    SAVE_DIR="${MODEL_DIR}_merged"
else
    SAVE_DIR="$2"
fi

echo "=================================="
echo "Model Merging Configuration"
echo "=================================="
echo "Model checkpoint: ${MODEL_DIR}"
echo "Output directory: ${SAVE_DIR}"
echo "=================================="

# Remove output directory if it exists
if [ -d "${SAVE_DIR}" ]; then
    echo "Warning: Output directory '${SAVE_DIR}' already exists."
    read -p "Remove it and continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing ${SAVE_DIR} directory..."
        rm -rf "${SAVE_DIR}"
    else
        echo "Aborted."
        exit 1
    fi
fi

# Create a temporary Python script
cat > merge_model_temp.py << EOF
import os
import torch
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from unsloth import FastVisionModel

MODEL_DIR = "${MODEL_DIR}"
SAVE_DIR  = "${SAVE_DIR}"

print(f"Loading vision model from {MODEL_DIR} ...")
model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    load_in_4bit=False,
    local_files_only=True,
)

# Merge LoRA adapters and save
print(f"Merging LoRA adapters into base model...")
model.save_pretrained_merged(
    SAVE_DIR,
    tokenizer,
    save_method="merged_16bit",
)
print(f"Merged model saved to: {SAVE_DIR}")
print("Done.")
EOF

# Execute the Python script
echo ""
echo "Starting model merge..."
if python merge_model_temp.py; then
    echo ""
    echo "=================================="
    echo "Model merging completed successfully!"
    echo "Merged model location: ${SAVE_DIR}"
    echo "=================================="
else
    echo ""
    echo "Error: Model merging failed!"
    rm merge_model_temp.py
    exit 1
fi

# Clean up the temporary Python script
rm merge_model_temp.py

echo "All operations completed successfully!"
