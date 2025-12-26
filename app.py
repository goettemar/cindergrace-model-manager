#!/usr/bin/env python3
"""
CINDERGRACE Model Manager
Web UI for managing ComfyUI models on RunPod Network Volumes
"""

import gradio as gr
import os
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import json

# Configuration
MODELS_ROOT = os.environ.get("MODELS_ROOT", "/workspace/models")
MODEL_FOLDERS = [
    "checkpoints",
    "clip",
    "vae",
    "unet",
    "diffusion_models",
    "loras",
    "audio_encoders",
    "controlnet",
    "upscale_models",
]

def get_folder_size(path: str) -> str:
    """Get human-readable folder size"""
    try:
        result = subprocess.run(
            ["du", "-sh", path],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout.split()[0]
    except:
        pass
    return "?"

def get_disk_usage() -> str:
    """Get disk usage for models root"""
    try:
        result = subprocess.run(
            ["df", "-h", MODELS_ROOT],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 4:
                    return f"{parts[2]} / {parts[1]} used ({parts[4]})"
    except:
        pass
    return "Unknown"

def scan_models() -> str:
    """Scan all model folders and return markdown overview"""
    output = []
    output.append(f"## Model Storage Overview")
    output.append(f"**Root:** `{MODELS_ROOT}`")
    output.append(f"**Disk Usage:** {get_disk_usage()}")
    output.append("")

    total_files = 0

    for folder in MODEL_FOLDERS:
        folder_path = Path(MODELS_ROOT) / folder

        if folder_path.exists():
            files = list(folder_path.glob("*.safetensors")) + \
                    list(folder_path.glob("*.ckpt")) + \
                    list(folder_path.glob("*.pt")) + \
                    list(folder_path.glob("*.pth")) + \
                    list(folder_path.glob("*.bin")) + \
                    list(folder_path.glob("*.gguf"))

            folder_size = get_folder_size(str(folder_path))
            output.append(f"### {folder}/ ({len(files)} files, {folder_size})")

            if files:
                for f in sorted(files, key=lambda x: x.name):
                    size_mb = f.stat().st_size / (1024 * 1024)
                    if size_mb > 1024:
                        size_str = f"{size_mb/1024:.1f} GB"
                    else:
                        size_str = f"{size_mb:.0f} MB"
                    output.append(f"- `{f.name}` ({size_str})")
                total_files += len(files)
            else:
                output.append("- *(empty)*")
            output.append("")
        else:
            output.append(f"### {folder}/ *(not created)*")
            output.append("")

    output.append(f"---")
    output.append(f"**Total:** {total_files} model files")

    return "\n".join(output)

def get_folder_choices():
    """Get list of folder choices for dropdown"""
    return MODEL_FOLDERS

def download_model(url: str, folder: str, filename: str, hf_token: str, progress=gr.Progress()) -> str:
    """Download model using aria2c or wget"""
    if not url:
        return "Error: URL is required"
    if not folder:
        return "Error: Target folder is required"
    if not filename:
        # Extract filename from URL
        filename = url.split("/")[-1].split("?")[0]
        if not filename:
            return "Error: Could not determine filename from URL"

    # Ensure filename has extension
    if not any(filename.endswith(ext) for ext in [".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"]):
        filename += ".safetensors"

    target_dir = Path(MODELS_ROOT) / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename

    if target_path.exists():
        return f"Error: File already exists: {target_path}"

    progress(0, desc="Starting download...")

    # Build download command
    if shutil.which("aria2c"):
        cmd = ["aria2c", "-x", "16", "-s", "16", "-d", str(target_dir), "-o", filename]
        if hf_token:
            cmd.extend(["--header", f"Authorization: Bearer {hf_token}"])
        cmd.append(url)
    else:
        cmd = ["wget", "-O", str(target_path)]
        if hf_token:
            cmd.extend(["--header", f"Authorization: Bearer {hf_token}"])
        cmd.append(url)

    try:
        progress(0.1, desc="Downloading...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode == 0:
            if target_path.exists():
                size_mb = target_path.stat().st_size / (1024 * 1024)
                progress(1.0, desc="Complete!")
                return f"Success! Downloaded {filename} ({size_mb:.1f} MB) to {folder}/"
            else:
                return f"Error: Download completed but file not found"
        else:
            return f"Error: Download failed\n{result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: Download timed out (1 hour limit)"
    except Exception as e:
        return f"Error: {str(e)}"

def delete_model(folder: str, filename: str) -> str:
    """Delete a model file"""
    if not folder or not filename:
        return "Error: Folder and filename required"

    target_path = Path(MODELS_ROOT) / folder / filename

    if not target_path.exists():
        return f"Error: File not found: {target_path}"

    try:
        target_path.unlink()
        return f"Deleted: {folder}/{filename}"
    except Exception as e:
        return f"Error deleting file: {e}"

def move_model(source_folder: str, filename: str, target_folder: str) -> str:
    """Move model to different folder"""
    if not all([source_folder, filename, target_folder]):
        return "Error: All fields required"

    source_path = Path(MODELS_ROOT) / source_folder / filename
    target_dir = Path(MODELS_ROOT) / target_folder
    target_path = target_dir / filename

    if not source_path.exists():
        return f"Error: Source file not found"

    if target_path.exists():
        return f"Error: File already exists in target folder"

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(target_path))
        return f"Moved {filename} from {source_folder}/ to {target_folder}/"
    except Exception as e:
        return f"Error moving file: {e}"

def get_files_in_folder(folder: str):
    """Get list of files in a folder for dropdown update"""
    if not folder:
        return gr.Dropdown(choices=[], value=None)

    folder_path = Path(MODELS_ROOT) / folder
    if not folder_path.exists():
        return gr.Dropdown(choices=[], value=None)

    files = []
    for ext in ["*.safetensors", "*.ckpt", "*.pt", "*.pth", "*.bin", "*.gguf"]:
        files.extend([f.name for f in folder_path.glob(ext)])

    sorted_files = sorted(files)
    return gr.Dropdown(choices=sorted_files, value=sorted_files[0] if sorted_files else None)

def create_folders() -> str:
    """Create all model folders"""
    created = []
    for folder in MODEL_FOLDERS:
        folder_path = Path(MODELS_ROOT) / folder
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            created.append(folder)

    if created:
        return f"Created folders: {', '.join(created)}"
    return "All folders already exist"

# Disclaimer text
DISCLAIMER_TEXT = """
## Terms of Use & Disclaimer

By using CINDERGRACE Model Manager, you agree to the following:

### 1. Model Licenses
- **You are solely responsible** for complying with the licenses of any models you download
- Many AI models have specific license terms (commercial use, attribution, etc.)
- **Check the license** on HuggingFace/CivitAI before downloading and using any model
- The developer of this tool is not responsible for license violations

### 2. No Warranty
- This software is provided **"AS IS"** without warranty of any kind
- The developer is **not liable** for any damages, data loss, or issues arising from use
- Use at your own risk

### 3. Security - HuggingFace Token
- **DELETE your HuggingFace token** from your HuggingFace account settings after use
- Or create a **temporary token** with limited permissions for downloads only
- Never share your token with others
- The token is only used locally for authentication and is **not stored**

### 4. Content Responsibility
- You are responsible for the content you create with downloaded models
- Do not use models to create illegal, harmful, or deceptive content
- Respect the rights of others

---

**By clicking "I Accept", you confirm that you have read, understood, and agree to these terms.**
"""

def accept_disclaimer():
    """Handle disclaimer acceptance"""
    return gr.update(visible=False), gr.update(visible=True)

# Gradio UI
with gr.Blocks(title="CINDERGRACE Model Manager", theme=gr.themes.Soft()) as app:

    # Disclaimer overlay (shown first)
    with gr.Column(visible=True) as disclaimer_section:
        gr.Markdown("# CINDERGRACE Model Manager")
        gr.Markdown(DISCLAIMER_TEXT)
        accept_btn = gr.Button("I Accept / Ich akzeptiere", variant="primary", size="lg")

    # Main app (hidden until disclaimer accepted)
    with gr.Column(visible=False) as main_section:
        gr.Markdown("""
        # CINDERGRACE Model Manager
        Manage your ComfyUI models on RunPod Network Volumes
        """)

        with gr.Tab("Overview"):
            overview_output = gr.Markdown(value=scan_models)
            with gr.Row():
                refresh_btn = gr.Button("Refresh", variant="secondary")
                create_folders_btn = gr.Button("Create All Folders", variant="secondary")
            create_folders_output = gr.Textbox(label="Status", interactive=False)

            refresh_btn.click(fn=scan_models, outputs=overview_output)
            create_folders_btn.click(fn=create_folders, outputs=create_folders_output)

        with gr.Tab("Download"):
            gr.Markdown("""
            ### Download Model
            Enter the download URL and target folder. For HuggingFace models that require authentication,
            enter your HuggingFace token.

            **Important:** You are responsible for complying with model licenses.

            **Security:** Delete your HuggingFace token after use or use a temporary read-only token!
            """)

            with gr.Row():
                with gr.Column():
                    download_url = gr.Textbox(
                        label="Download URL",
                        placeholder="https://huggingface.co/.../model.safetensors"
                    )
                    download_folder = gr.Dropdown(
                        choices=MODEL_FOLDERS,
                        label="Target Folder",
                        value="diffusion_models"
                    )
                    download_filename = gr.Textbox(
                        label="Filename (optional, auto-detected from URL)",
                        placeholder="model.safetensors"
                    )
                    hf_token = gr.Textbox(
                        label="HuggingFace Token (optional) - DELETE AFTER USE!",
                        placeholder="hf_...",
                        type="password"
                    )
                    download_btn = gr.Button("Download", variant="primary")

                with gr.Column():
                    download_output = gr.Textbox(label="Status", lines=5, interactive=False)

            download_btn.click(
                fn=download_model,
                inputs=[download_url, download_folder, download_filename, hf_token],
                outputs=download_output
            )

        with gr.Tab("Manage"):
            gr.Markdown("### Move or Delete Models")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Delete Model")
                    delete_folder = gr.Dropdown(choices=MODEL_FOLDERS, label="Folder")
                    delete_file = gr.Dropdown(choices=[], label="File")
                    delete_btn = gr.Button("Delete", variant="stop")
                    delete_output = gr.Textbox(label="Status", interactive=False)

                    delete_folder.change(
                        fn=get_files_in_folder,
                        inputs=delete_folder,
                        outputs=delete_file
                    )
                    delete_btn.click(
                        fn=delete_model,
                        inputs=[delete_folder, delete_file],
                        outputs=delete_output
                    )

                with gr.Column():
                    gr.Markdown("#### Move Model")
                    move_source_folder = gr.Dropdown(choices=MODEL_FOLDERS, label="Source Folder")
                    move_file = gr.Dropdown(choices=[], label="File")
                    move_target_folder = gr.Dropdown(choices=MODEL_FOLDERS, label="Target Folder")
                    move_btn = gr.Button("Move", variant="secondary")
                    move_output = gr.Textbox(label="Status", interactive=False)

                    move_source_folder.change(
                        fn=get_files_in_folder,
                        inputs=move_source_folder,
                        outputs=move_file
                    )
                    move_btn.click(
                        fn=move_model,
                        inputs=[move_source_folder, move_file, move_target_folder],
                        outputs=move_output
                    )

        with gr.Tab("Quick Links"):
            gr.Markdown("""
            ### Popular Model Downloads

            Copy these URLs to the Download tab. Remember to add your HuggingFace token for gated models.

            **Check model licenses before downloading!**

            #### Wan 2.2 (Video Generation)
            | Model | Size | URL |
            |-------|------|-----|
            | UMT5-XXL FP8 (Text Encoder) | 6.7 GB | `https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` |
            | Wan 2.1 VAE | 254 MB | `https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors` |
            | Wan 2.2 I2V HighNoise FP8 | 14.3 GB | `https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` |
            | Wan 2.2 I2V LowNoise FP8 | 14.3 GB | `https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` |

            #### Flux (Image Generation)
            | Model | Size | URL |
            |-------|------|-----|
            | CLIP-L | 235 MB | `https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors` |
            | T5-XXL FP16 | 9.2 GB | `https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors` |
            | Flux VAE | 335 MB | `https://huggingface.co/ffxvs/vae-flux/resolve/main/ae.safetensors` |
            | Flux Dev FP8 | 11.9 GB | `https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8-e4m3fn.safetensors` |
            | Flux Krea Dev | ~24 GB | `https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev/resolve/main/flux1-krea-dev.safetensors` |

            #### Target Folders
            - **clip/** - Text encoders (CLIP, T5, UMT5)
            - **vae/** - VAE models
            - **diffusion_models/** - Main diffusion models (Wan, Flux)
            - **unet/** - Alternative location for some models
            - **loras/** - LoRA adapters
            - **audio_encoders/** - Audio models for speech-to-video
            """)

    # Connect disclaimer accept button
    accept_btn.click(
        fn=accept_disclaimer,
        outputs=[disclaimer_section, main_section]
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
