# CINDERGRACE Model Manager

Web UI for managing ComfyUI models on RunPod Network Volumes.

## Features

- **Overview**: Scan and display all models across folders with sizes
- **Download**: Download models using direct URLs (HuggingFace, CivitAI, etc.)
- **Manage**: Move or delete models between folders
- **Quick Links**: Pre-configured URLs for popular models (Wan 2.2, Flux)

## Usage on RunPod

### Option 1: Standalone Template

1. Create a new RunPod template:
   - **Container Image**: `ghcr.io/goettemar/cindergrace-model-manager:latest`
   - **Container Start Command**: (leave empty)
   - **HTTP Port**: 7860

2. Attach your Network Volume (mount path: `/workspace`)

3. Deploy a CPU pod (no GPU needed)

4. Access via the RunPod proxy URL

### Option 2: With ComfyUI

Run alongside the CINDERGRACE ComfyUI template on the same Network Volume.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_ROOT` | `/workspace/models` | Root directory for models |

## Model Folder Structure

```
/workspace/models/
├── checkpoints/      # Full model checkpoints
├── clip/             # Text encoders (CLIP, T5, UMT5)
├── vae/              # VAE models
├── unet/             # UNet models
├── diffusion_models/ # Main diffusion models (Wan, Flux)
├── loras/            # LoRA adapters
├── audio_encoders/   # Audio models
├── controlnet/       # ControlNet models
└── upscale_models/   # Upscaling models
```

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

## Building Docker Image

```bash
docker build -t cindergrace-model-manager .
docker run -p 7860:7860 -v /path/to/models:/workspace/models cindergrace-model-manager
```

## License

MIT License
