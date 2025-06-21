# FluxZayn: FLUX LayerDiffuse Extension for Stable Diffusion WebUI Forge

This extension integrates FLUX.1 image generation with LayerDiffuse capabilities (using TransparentVAE) into SD WebUI Forge.

This repo is a Forge extension implementation of LayerDiffuse-Flux (https://github.com/RedAIGC/Flux-version-LayerDiffuse)

## Features

-   FLUX.1-dev and FLUX.1-schnell Model Support (Text-to-Image and Image-to-Image).
-   **Layer Separation using TransparentVAE:**
    -   Decodes final latents through a custom TransparentVAE for RGBA output.
    -   For Img2Img, can encode RGBA input through TransparentVAE for layered diffusion.
-   Support for LayerLoRA.
-   Configurable generation parameters.

## Installation

1.  **Download and Place:**
    Place the `flux-layerdiffuse` folder (extracted from the provided ZIP) into your `stable-diffusion-webui-forge/extensions/` directory.
    The key file will be `extensions/flux-layerdiffuse/scripts/flux_layerdiffuse_main.py`.

2.  **Dependencies:**
    The `install.py` script (located in `extensions/flux-layerdiffuse/`) will attempt to install `diffusers`, `transformers`, `safetensors`, `accelerate`, and `opencv-python-headless`. Restart Forge after the first launch with the extension to ensure dependencies are loaded.

3.  **Models:**
    *   **FLUX Base Model:**
        *   In the UI ("FLUX Model Directory/ID"), provide a **path to a local FLUX model directory** (e.g., a full download of `black-forest-labs/FLUX.1-dev`) OR a **HuggingFace Model ID**.
        *   **Important:** This should NOT be a path to a single `.safetensors` file for the base FLUX model.
    *   **TransparentVAE Weights:**
        *   Download `TransparentVAE.safetensors` (or a compatible `.pth` file). I have converted the original TransparentVAE from (https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse)
        *   It's recommended to place it in `stable-diffusion-webui-forge/models/LayerDiffuse/`. The UI will default to looking here.
        *   Provide the full path to this file in the UI ("TransparentVAE Weights Path").
    *   **Layer LoRA (Optional but Recommended for Best Layer Effects):**
        *   Download the `layerlora.safetensors` file compatible with FLUX and LayerDiffuse principles (https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse/tree/main)
        *   Provide its path in the UI ("LayerLoRA Path").

4.  **Restart Stable Diffusion WebUI Forge.**

## Usage

1.  Open the "FLUX LayerDiffuse" tab in the WebUI Forge interface.
2.  **Setup Models:**
    *   Verify "FLUX Model Directory/ID" points to a valid FLUX model directory or a HuggingFace repository ID.
    *   Set "TransparentVAE Weights Path" to your `TransparentVAE.safetensors` or `.pth` file.
    *   Optionally, set "Layer LoRA Path" and adjust its strength.
3.  **Generation Parameters:** Configure prompt, negative prompt, image dimensions, inference steps, CFG scale, sampler, and seed.
    *   *Tip:* FLUX models often perform well with fewer inference steps (e.g., 20-30) and lower CFG scales (e.g., 3.0-5.0) compared to standard Stable Diffusion models.
4.  **Image-to-Image (Optional):**
    *   Upload an input image. For best results with TransparentVAE's encoding capabilities (to preserve and diffuse existing alpha/layers), provide an **RGBA** image.
    *   Adjust "Denoising Strength".
5.  Click the "Generate Images" button.
6.  The output gallery should display RGBA images if TransparentVAE was successfully used for decoding.

## Troubleshooting & Notes

-   **"FLUX Model Directory/ID" Errors:** This path *must* be to a folder containing the complete diffusers model structure for FLUX (with `model_index.json`, subfolders like `transformer`, `vae`, etc.), or a valid HuggingFace ID. It cannot be a single `.safetensors` file for the base model.
-   **Layer Quality/Separation:** The effectiveness of layer separation heavily depends on the quality of the TransparentVAE weights and the compatibility/effectiveness of the chosen Layer LoRA.
-   **Img2Img with RGBA:** If using Img2Img and you want to properly utilize TransparentVAE's encoding for layered input, ensure your uploaded image is in RGBA format. The script attempts to handle this, but native RGBA input is best.
-   **Console Logs:** Check the WebUI Forge console for `[FLUX Script]` messages. They provide verbose logging about the model loading and generation process, which can be helpful for debugging.
-   This integration is advanced. If issues arise, carefully check paths and console output.
