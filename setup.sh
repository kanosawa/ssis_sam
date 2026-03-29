#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== SSISv2 + SAM2 Video Pipeline Setup ==="

# --- 1. Clone SSISv2 ---
if [ ! -d "SSIS" ]; then
    echo "[1/4] Cloning SSISv2..."
    git clone https://github.com/stevewongv/SSIS.git
else
    echo "[1/4] SSISv2 already cloned."
fi

# --- 2. Clone SAM2 ---
if [ ! -d "sam2" ]; then
    echo "[2/4] Cloning SAM2..."
    git clone https://github.com/facebookresearch/sam2.git
else
    echo "[2/4] SAM2 already cloned."
fi

# --- 3. Download SSISv2 weights ---
SSIS_WEIGHT_DIR="SSIS/tools/output/SSISv2_MS_R_101_bifpn_with_offset_class_maskiouv2_da_bl"
SSIS_WEIGHT_PATH="$SSIS_WEIGHT_DIR/model_ssisv2_final.pth"
if [ ! -f "$SSIS_WEIGHT_PATH" ]; then
    echo "[3/4] SSISv2 weights not found."
    echo "  Please download manually from Google Drive:"
    echo "  https://drive.google.com/drive/folders/1MKxyq3R6AUeyLai9i9XWzG2C_n5f0ppP"
    echo "  Place model_ssisv2_final.pth in:"
    echo "  $SSIS_WEIGHT_DIR/"
    mkdir -p "$SSIS_WEIGHT_DIR"
else
    echo "[3/4] SSISv2 weights found."
fi

# --- 4. Download SAM2 checkpoint ---
SAM2_CKPT_DIR="sam2/checkpoints"
SAM2_CKPT_PATH="$SAM2_CKPT_DIR/sam2.1_hiera_large.pt"
if [ ! -f "$SAM2_CKPT_PATH" ]; then
    echo "[4/4] Downloading SAM2.1 hiera_large checkpoint..."
    mkdir -p "$SAM2_CKPT_DIR"
    wget -q --show-progress -O "$SAM2_CKPT_PATH" \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
else
    echo "[4/4] SAM2 checkpoint found."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Install SSISv2:  cd SSIS && pip install -r requirement.txt && python setup.py build develop"
echo "  2. Install SAM2:    cd sam2 && pip install -e ."
echo "  3. Download SOBA annotations (needed for SSISv2 dataset registration):"
echo "     https://drive.google.com/drive/folders/1MKxyq3R6AUeyLai9i9XWzG2C_n5f0ppP"
echo "     Extract SOBA_v2.zip to: $SCRIPT_DIR/dataset/SOBA/"
echo "  4. Run pipeline:    python pipeline.py --video <input_video.mp4> --output <output_dir>"
