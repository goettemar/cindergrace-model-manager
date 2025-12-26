# CINDERGRACE Model Manager

Web UI for managing ComfyUI models on RunPod Network Volumes.

## Features

- **Overview**: Scan and display all models across folders with sizes
- **Download**: Download models using direct URLs (HuggingFace, CivitAI, etc.)
- **Manage**: Move or delete models between folders
- **Quick Links**: Pre-configured URLs for popular models (Wan 2.2, Flux)

## RunPod Template Einstellungen

| Feld | Wert |
|------|------|
| **Template Name** | CINDERGRACE Model Manager |
| **Container Image** | `ghcr.io/goettemar/cindergrace-model-manager:latest` |
| **Container Disk** | 5 GB |
| **Volume Mount Path** | `/workspace` |
| **HTTP Port** | `7860` |
| **GPU** | Nicht nötig (CPU Pod) |

### Template README (für RunPod):
```
CINDERGRACE Model Manager - Organize Your AI Models

Simple web UI to download, move, and manage ComfyUI models on your Network Volume.

FEATURES:
• Download models via URL (HuggingFace, CivitAI, etc.)
• View all models with sizes
• Move/delete models between folders
• Quick links for popular models (Wan 2.2, Flux)

SETUP:
1. Attach your Network Volume at /workspace
2. Deploy CPU pod (no GPU needed!)
3. Open web UI: https://<POD_ID>-7860.proxy.runpod.net
4. Download models using the Quick Links tab

SUPPORTED FOLDERS:
clip, vae, diffusion_models, unet, loras, checkpoints, controlnet, upscale_models

LINKS:
• GitHub: https://github.com/goettemar/cindergrace-model-manager
• Discord: Coming soon
```

## Quick Start

### 1. Template erstellen

1. RunPod → My Templates → New Template
2. Einstellungen wie oben eintragen
3. Save Template

### 2. CPU Pod deployen

1. Pods → + Deploy
2. Template: **CINDERGRACE Model Manager**
3. Pod Type: **CPU** (kein GPU nötig!)
4. Network Volume: Dein Model-Volume anhängen
5. Deploy

### 3. Web UI öffnen

1. Pod starten
2. URL: `https://<POD_ID>-7860.proxy.runpod.net`
3. Modelle downloaden über Quick Links oder eigene URLs

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

## Popular Models (Quick Links)

### Wan 2.2 (Video Generation)

| Model | Size | Folder |
|-------|------|--------|
| UMT5-XXL FP8 | 6.7 GB | clip |
| Wan 2.1 VAE | 254 MB | vae |
| Wan 2.2 I2V HighNoise FP8 | 14.3 GB | diffusion_models |
| Wan 2.2 I2V LowNoise FP8 | 14.3 GB | diffusion_models |

### Flux (Image Generation)

| Model | Size | Folder |
|-------|------|--------|
| CLIP-L | 235 MB | clip |
| T5-XXL FP16 | 9.2 GB | clip |
| Flux VAE | 335 MB | vae |
| Flux Krea Dev | ~24 GB | diffusion_models |

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

## Links

- [CINDERGRACE GUI](https://github.com/goettemar/cindergrace_gui)
- [CINDERGRACE ComfyUI RunPod](https://github.com/goettemar/cindergrace-comfyui-runpod)
- [Docker Image (GHCR)](https://ghcr.io/goettemar/cindergrace-model-manager)
- Discord: Coming soon

## License

MIT License
