#!/bin/bash
# CINDERGRACE Model Manager Startup Script

set -e

echo "=========================================="
echo "  CINDERGRACE Model Manager"
echo "=========================================="
echo ""

# Check for Network Volume
if [ -d "/workspace/models" ]; then
    echo "[OK] Models directory: /workspace/models"

    # Create standard folders if they don't exist
    mkdir -p /workspace/models/checkpoints
    mkdir -p /workspace/models/clip
    mkdir -p /workspace/models/vae
    mkdir -p /workspace/models/unet
    mkdir -p /workspace/models/diffusion_models
    mkdir -p /workspace/models/loras
    mkdir -p /workspace/models/audio_encoders
    mkdir -p /workspace/models/controlnet
    mkdir -p /workspace/models/upscale_models

    echo "[OK] All model folders ready"
else
    echo "[WARN] /workspace/models not found"
    echo "       Please attach a Network Volume"
fi

echo ""
echo "=========================================="
echo "  Starting Model Manager on port 7860"
echo "=========================================="
echo ""

# Start Gradio app
exec python app.py
