import gradio as gr
import torch
import torch.nn as nn 
from PIL import Image, ImageDraw 
from PIL.PngImagePlugin import PngInfo
import numpy as np
import os
import sys
import traceback 
import random 
import html 
import cv2 
import torchvision.transforms as transforms 
from typing import Optional, List, Union, Dict, Any
from modules import script_callbacks, shared, paths
import datetime 

# --- START: Dependencies Check and Dummy Definitions ---
try:
    from diffusers import FluxPipeline, FluxImg2ImgPipeline, AutoencoderKL 
    from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block 
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution 
    from diffusers.configuration_utils import ConfigMixin, register_to_config 
    from diffusers.models.modeling_utils import ModelMixin 
    from diffusers import (
        EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
        LMSDiscreteScheduler, DDIMScheduler,
    )
    DIFFUSERS_AVAILABLE = True
    try:
        from diffusers import FluxSchnellScheduler
        FLUX_SCHNELL_SCHEDULER_AVAILABLE = True
    except ImportError:
        print("[FLUX LayerDiffuse Script] WARNING: FluxSchnellScheduler not found. Consider updating the 'diffusers' library for optimal performance with 'schnell' models.")
        FluxSchnellScheduler = None
        FLUX_SCHNELL_SCHEDULER_AVAILABLE = False
except ImportError as e:
    print(f"[FLUX LayerDiffuse Script] CRITICAL IMPORT ERROR (Diffusers components): {e}. Install.py should run.")
    DIFFUSERS_AVAILABLE = False
    FLUX_SCHNELL_SCHEDULER_AVAILABLE = False
    class DummyPipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs): raise ImportError("Diffusers not available for DummyPipeline")
        def to(self, *args, **kwargs): pass
        def enable_model_cpu_offload(self, *args, **kwargs): pass
        def load_lora_weights(self, *args, **kwargs): pass
        @property
        def vae(self): 
            class DummyVAE:
                class DummyVAEConfig:
                    latent_channels = 4 
                    scaling_factor = 0.18215 
                    shift_factor = 0 
                config = DummyVAEConfig()
                dtype = torch.float32 
            return DummyVAE()
        @property
        def scheduler(self):
            class DummyScheduler:
                class DummyConfig: pass
                config = DummyConfig()
                @classmethod
                def from_config(cls, *args, **kwargs): return cls()
            return DummyScheduler()
        def __call__(self, *args, **kwargs):
            class DummyOutput: images = [Image.new("RGB", (kwargs.get('width',512),kwargs.get('height',512)), "gray")]
            return DummyOutput()
        def _unpack_latents(self, latents, height, width, vae_scale_factor): return latents 
    FluxPipeline = FluxImg2ImgPipeline = DummyPipeline
    AutoencoderKL = object
    UNetMidBlock2D = get_down_block = get_up_block = object
    DiagonalGaussianDistribution = object
    ConfigMixin = object
    ModelMixin = object 
    register_to_config = lambda x: x
    EulerDiscreteScheduler = LMSDiscreteScheduler = DDIMScheduler = object 
    FluxSchnellScheduler = None

try:
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    print(f"[FLUX LayerDiffuse Script] CRITICAL IMPORT ERROR: Safetensors not found. Install.py should run.")
    SAFETENSORS_AVAILABLE = False
    def load_safetensors(*args, **kwargs): raise ImportError("Safetensors not available")

if 'shared' not in globals():
    print("[FLUX Script] `shared` not found, creating dummy object for path determination.")
    class DummyShared:
        class DummyOpts:
            outdir_txt2img_samples = None
            outdir_img2img_samples = None
        opts = DummyOpts()
    shared = DummyShared()

if 'paths' not in globals():
    print("[FLUX Script] `paths` not found, creating dummy object.")
    class DummyPaths:
        script_path = os.getcwd() 
        models_path = os.path.join(os.getcwd(), "models")
    paths = DummyPaths()
# --- END: Dependencies Check ---


# --- START: TransparentVAE and related classes/functions ---
def zero_module(module): 
    for p in module.parameters():
        p.detach().zero_()
    return module

class LatentTransparencyOffsetEncoder(torch.nn.Module): 
    def __init__(self, latent_c=4, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.blocks = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=3, padding=1, stride=1), nn.SiLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1), nn.SiLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), nn.SiLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1), nn.SiLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), nn.SiLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1), nn.SiLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2), nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.SiLU(),
            zero_module(torch.nn.Conv2d(256, latent_c, kernel_size=3, padding=1, stride=1)),
        )
    def forward(self, x): return self.blocks(x)

# MODIFIED: Removed the problematic decorator
class UNet1024(ModelMixin, ConfigMixin): 
    @register_to_config
    def __init__(
        self, in_channels: int = 3, out_channels: int = 4, 
        down_block_types: tuple = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: tuple = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels: tuple = (32, 32, 64, 128, 256, 512, 512), layers_per_block: int = 2,
        mid_block_scale_factor: float = 1, downsample_padding: int = 1, downsample_type: str = "conv",
        upsample_type: str = "conv", dropout: float = 0.0, act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8, norm_num_groups: int = 4, 
        norm_eps: float = 1e-5, latent_c: int = 4,
    ):
        super().__init__()
        if not DIFFUSERS_AVAILABLE:
            print("[UNet1024] Diffusers not available, U-Net cannot be properly initialized.")
            return 
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))
        self.latent_conv_in = zero_module(nn.Conv2d(latent_c, block_out_channels[2], kernel_size=1))
        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block( down_block_type, num_layers=layers_per_block, in_channels=input_channel,
                out_channels=output_channel, temb_channels=None, add_downsample=not is_final_block, resnet_eps=norm_eps,
                resnet_act_fn=act_fn, resnet_groups=norm_num_groups, attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                downsample_padding=downsample_padding, resnet_time_scale_shift="default", downsample_type=downsample_type, dropout=dropout,
            )
            self.down_blocks.append(down_block)
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1], temb_channels=None, dropout=dropout, resnet_eps=norm_eps, resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor, resnet_time_scale_shift="default",
            attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
            resnet_groups=norm_num_groups, attn_groups=None, add_attention=True,
        )
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            is_final_block = i == len(block_out_channels) - 1
            up_block = get_up_block(
                up_block_type, num_layers=layers_per_block + 1, in_channels=input_channel, out_channels=output_channel,
                prev_output_channel=prev_output_channel, temb_channels=None, add_upsample=not is_final_block,
                resnet_eps=norm_eps, resnet_act_fn=act_fn, resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                resnet_time_scale_shift="default", upsample_type=upsample_type, dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x, latent):
        if not DIFFUSERS_AVAILABLE: 
            raise RuntimeError("UNet1024 called but diffusers components are not available.")
        sample_latent = self.latent_conv_in(latent)
        sample = self.conv_in(x)
        emb = None
        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            if i == 3: sample = sample + sample_latent
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples
        sample = self.mid_block(sample, emb)
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            sample = upsample_block(sample, res_samples, emb)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample

def checkerboard(shape): return np.indices(shape).sum(axis=0) % 2 

def build_alpha_pyramid(color, alpha, dk=1.2): 
    pyramid = []
    current_premultiplied_color = color * alpha
    current_alpha = alpha
    while True:
        pyramid.append((current_premultiplied_color, current_alpha))
        H, W = current_alpha.shape[:2]
        if min(H, W) == 1: break
        current_premultiplied_color = cv2.resize(current_premultiplied_color, (int(W / dk), int(H / dk)), interpolation=cv2.INTER_AREA)
        current_alpha_resized = cv2.resize(current_alpha, (int(W / dk), int(H / dk)), interpolation=cv2.INTER_AREA)
        current_alpha = current_alpha_resized[:, :, None] if current_alpha_resized.ndim == 2 else current_alpha_resized
    return pyramid[::-1]

def pad_rgb(np_rgba_hwc_uint8): 
    np_rgba_hwc = np_rgba_hwc_uint8.astype(np.float32) / 255.0
    color = np_rgba_hwc[..., :3]
    alpha = np_rgba_hwc[..., 3:4]
    pyramid = build_alpha_pyramid(color=color, alpha=alpha)
    top_c, top_a = pyramid[0]
    sum_top_a = np.sum(top_a, axis=(0, 1), keepdims=True).clip(1e-8, 1e32)
    fg = np.sum(top_c, axis=(0, 1), keepdims=True) / sum_top_a
    for layer_c, layer_a in pyramid:
        layer_h, layer_w, _ = layer_c.shape
        fg_resized = cv2.resize(fg, (layer_w, layer_h), interpolation=cv2.INTER_LINEAR)
        if fg_resized.ndim == 2 : fg_resized = fg_resized[:,:,np.newaxis]
        if fg_resized.shape[2] == 1 and layer_c.shape[2] == 3: fg_resized = np.repeat(fg_resized, 3, axis=2)
        fg = fg_resized 
        fg = layer_c + fg * (1.0 - layer_a)
    return (fg * 255.0).clip(0, 255).astype(np.uint8)

def dist_sample_deterministic(dist: DiagonalGaussianDistribution, perturbation: torch.Tensor): 
    x = dist.mean + dist.std * perturbation.to(dist.std.device, dtype=dist.std.dtype)
    return x

class TransparentVAE(torch.nn.Module): 
    def __init__(self, sd_vae, dtype=torch.float16, encoder_file=None, decoder_file=None, alpha=300.0, latent_c=4):
        super().__init__()
        self.dtype = dtype
        self.sd_vae = sd_vae
        if self.sd_vae is not None:
            self.sd_vae.to(dtype=self.dtype)
            self.sd_vae.requires_grad_(False)
        self.encoder = LatentTransparencyOffsetEncoder(latent_c=latent_c)
        if encoder_file is not None and SAFETENSORS_AVAILABLE:
            try:
                temp = load_safetensors(encoder_file, device="cpu")
                self.encoder.load_state_dict(temp, strict=True)
                del temp
            except Exception as e: print(f"[TVAE] Failed to load encoder from {encoder_file}: {e}")
        self.encoder.to(dtype=self.dtype)
        self.alpha = alpha
        self.decoder = UNet1024(in_channels=3, out_channels=4, latent_c=latent_c)
        if decoder_file is not None and SAFETENSORS_AVAILABLE:
            try:
                temp = load_safetensors(decoder_file, device="cpu")
                self.decoder.load_state_dict(temp, strict=True)
                del temp
            except Exception as e: print(f"[TVAE] Failed to load decoder from {decoder_file}: {e}")
        self.decoder.to(dtype=self.dtype)
        self.latent_c = latent_c

    def sd_decode_sample(self, latent):
        if self.sd_vae is None: raise RuntimeError("sd_vae not initialized in TransparentVAE")
        return self.sd_vae.decode(latent).sample

    def decode(self, latent, aug=True):
        if self.sd_vae is None: raise RuntimeError("sd_vae not initialized for decode")
        origin_pixel = self.sd_decode_sample(latent.to(self.sd_vae.dtype))
        origin_pixel = (origin_pixel * 0.5 + 0.5).clamp(0,1)
        dec_device = next(self.decoder.parameters()).device 
        latent_for_decoder = latent.to(device=dec_device, dtype=self.dtype)
        origin_pixel_for_decoder = origin_pixel.to(device=dec_device, dtype=self.dtype)
        if not aug:
            y = self.decoder(origin_pixel_for_decoder, latent_for_decoder)
            return origin_pixel, y.clamp(0,1)
        list_y = []
        for i in range(int(latent.shape[0])):
            y_single = self.estimate_augmented(origin_pixel_for_decoder[i:i + 1], latent_for_decoder[i:i + 1])
            list_y.append(y_single)
        y = torch.cat(list_y, dim=0)
        return origin_pixel, y.clamp(0,1)

    def encode(self, img_rgba_01_bchw, img_rgb_m11_bchw, padded_img_rgb_01_bchw, use_offset=True):
        if self.sd_vae is None: raise RuntimeError("sd_vae not initialized for encode")
        a_bchw_01 = img_rgba_01_bchw[:, 3:4, :, :]
        vae_feed = img_rgb_m11_bchw.to(device=self.sd_vae.device, dtype=self.sd_vae.dtype)
        latent_dist = self.sd_vae.encode(vae_feed).latent_dist
        offset_feed_rgb = padded_img_rgb_01_bchw
        if offset_feed_rgb.shape[1] == 1 and a_bchw_01.shape[1] == 1:
             offset_feed_rgb = offset_feed_rgb.repeat(1,3,1,1)
        offset_feed = torch.cat([offset_feed_rgb, a_bchw_01], dim=1).to(device=next(self.encoder.parameters()).device, dtype=self.dtype)
        offset = self.encoder(offset_feed) * self.alpha
        latent_sampled = dist_sample_deterministic(dist=latent_dist, perturbation=offset) if use_offset else latent_dist.sample()
        latent = latent_sampled * self.sd_vae.config.scaling_factor 
        return latent

    @torch.no_grad()
    def estimate_augmented(self, pixel_b1hw, latent_b1hw):
        args = [[False, 0], [False, 1], [False, 2], [False, 3], [True, 0], [True, 1], [True, 2], [True, 3]]
        result = []
        dec_device = next(self.decoder.parameters()).device
        for flip, rok in args:
            feed_pixel = pixel_b1hw.clone().to(dec_device)
            feed_latent = latent_b1hw.clone().to(dec_device)
            if flip:
                feed_pixel = torch.flip(feed_pixel, dims=(3,))
                feed_latent = torch.flip(feed_latent, dims=(3,))
            feed_pixel = torch.rot90(feed_pixel, k=rok, dims=(2, 3))
            feed_latent = torch.rot90(feed_latent, k=rok, dims=(2, 3))
            eps = self.decoder(feed_pixel, feed_latent) 
            eps = torch.rot90(eps, k=-rok, dims=(2, 3))
            if flip: eps = torch.flip(eps, dims=(3,))
            result.append(eps)
        result_stacked = torch.stack(result, dim=0)
        median = torch.median(result_stacked, dim=0).values
        return median
# --- END: TransparentVAE ---

# --- START: run_layerdiffuse function ---
def run_layerdiffuse(
    flux_model_id_or_path, vae_path_transparent, lora_path_layerdiffuse,
    clip_encoder_path, t5_encoder_path,
    prompt, neg_prompt, input_image_pil, width, height, num_inference_steps,
    guidance_scale, image_count, seed, lora_strength, img2img_strength
):
    print("--- [FLUX Script] Starting FLUX LayerDiffuse Generation (with TransparentVAE logic) ---")
    if not DIFFUSERS_AVAILABLE or not SAFETENSORS_AVAILABLE:
        err_img = Image.new("RGB", (width, height), color="maroon")
        return [(err_img, "Error: Core libraries (Diffusers/Safetensors) not available.")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    if seed is None or seed == -1 or seed == 0:
        effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
    else:
        effective_seed = int(seed)

    is_img2img = input_image_pil is not None
    mode_name = "Image-to-Image" if is_img2img else "Text-to-Image"
    print(f"  [FLUX Script] Mode: {mode_name}")

    base_output_dir = ""
    if is_img2img:
        base_output_dir = shared.opts.outdir_img2img_samples if hasattr(shared, 'opts') and hasattr(shared.opts, 'outdir_img2img_samples') and shared.opts.outdir_img2img_samples else os.path.join(paths.script_path, "outputs", "img2img-images")
    else:
        base_output_dir = shared.opts.outdir_txt2img_samples if hasattr(shared, 'opts') and hasattr(shared.opts, 'outdir_txt2img_samples') and shared.opts.outdir_txt2img_samples else os.path.join(paths.script_path, "outputs", "txt2img-images")
    final_save_dir = os.path.join(base_output_dir, "FluxZayn")
    try:
        os.makedirs(final_save_dir, exist_ok=True)
    except OSError as e_mkdir:
        print(f"  [FLUX Script] ERROR creating save directory {final_save_dir}: {e_mkdir}. Images will not be saved.")
        final_save_dir = None

    def create_error_image(message):
        err_img = Image.new("RGB", (width, height), color="red")
        print(f"  [FLUX Script] Error Image Created: {message}")
        return [(err_img, f"Error: {message}")]

    pipe = None
    trans_vae_instance = None
    try:
        pipeline_class = FluxImg2ImgPipeline if is_img2img else FluxPipeline
        is_local_dir = flux_model_id_or_path and os.path.isdir(flux_model_id_or_path)
        if is_local_dir: resolved_model_path = flux_model_id_or_path
        elif flux_model_id_or_path and not os.path.exists(flux_model_id_or_path) : resolved_model_path = flux_model_id_or_path
        else: resolved_model_path = "black-forest-labs/FLUX.1-schnell"
        print(f"  [FLUX Script] Loading base FLUX from: {resolved_model_path}")

        pipe = pipeline_class.from_pretrained(resolved_model_path, torch_dtype=dtype)

        if FLUX_SCHNELL_SCHEDULER_AVAILABLE and "schnell" in resolved_model_path.lower():
            print("  [FLUX Script] Schnell model detected. Swapping to FluxSchnellScheduler for optimal performance.")
            pipe.scheduler = FluxSchnellScheduler.from_config(pipe.scheduler.config)
        elif "schnell" in resolved_model_path.lower():
            print("  [FLUX Script] WARNING: 'schnell' model used, but FluxSchnellScheduler was not found. Performance may not be optimal.")

        if lora_path_layerdiffuse and os.path.isfile(lora_path_layerdiffuse):
            pipe.load_lora_weights(lora_path_layerdiffuse, lora_scale=float(lora_strength))

        if vae_path_transparent and os.path.isfile(vae_path_transparent) and pipe.vae is not None:
            vae_latent_channels = pipe.vae.config.latent_channels
            trans_vae_instance = TransparentVAE(sd_vae=pipe.vae, dtype=dtype, latent_c=vae_latent_channels)
            tvae_state_dict = load_safetensors(vae_path_transparent, device="cpu")
            trans_vae_instance.load_state_dict(tvae_state_dict, strict=False)
            trans_vae_instance.eval()

        pipe.to(device)
        if trans_vae_instance: trans_vae_instance.to(device)
        pipe.enable_model_cpu_offload()

        all_output_images_with_labels = []
        for i in range(image_count):
            current_seed = effective_seed + i
            image_generator = torch.Generator(device=device).manual_seed(current_seed)
            print(f"  [FLUX Script] Image {i+1}/{image_count}, Seed: {current_seed}")
            
            metadata = PngInfo()
            params_line = f"Steps: {num_inference_steps}, CFG scale: {guidance_scale}, Seed: {current_seed}, Size: {width}x{height}"
            params_line += f", Scheduler: {pipe.scheduler.__class__.__name__}"
            
            # MODIFIED: Fix for the f-string SyntaxError
            model_name = os.path.basename(resolved_model_path.rstrip('/\\'))
            params_line += f", Model: {model_name}"

            if lora_path_layerdiffuse and os.path.isfile(lora_path_layerdiffuse):
                 params_line += f", LoRA: {os.path.basename(lora_path_layerdiffuse)}, LoRA Strength: {lora_strength}"
            if vae_path_transparent and os.path.isfile(vae_path_transparent):
                 params_line += f", Transparent VAE: {os.path.basename(vae_path_transparent)}"
            if is_img2img:
                params_line += f", Denoising strength: {img2img_strength}"

            full_metadata_string = f"{prompt}\nNegative prompt: {neg_prompt}\n{params_line}"
            metadata.add_text("parameters", full_metadata_string)

            call_params = {"prompt": prompt or " ", "width": width, "height": height, "num_inference_steps": num_inference_steps, "guidance_scale": guidance_scale, "generator": image_generator}
            if not is_img2img and neg_prompt and neg_prompt.strip():
                call_params["negative_prompt"] = neg_prompt

            if is_img2img:
                if trans_vae_instance and input_image_pil:
                    input_pil_rgba = input_image_pil.convert("RGBA")
                    img_rgba_01_bchw = transforms.ToTensor()(input_pil_rgba).unsqueeze(0).to(device=device, dtype=dtype)
                    img_rgb_prem_01 = img_rgba_01_bchw[:, :3] * img_rgba_01_bchw[:, 3:4]
                    img_rgb_m11_bchw = (img_rgb_prem_01 * 2.0 - 1.0)
                    rgba_np_hwc_uint8 = (img_rgba_01_bchw.squeeze(0).permute(1,2,0) * 255).byte().cpu().numpy()
                    padded_rgb_np_hwc_uint8 = pad_rgb(rgba_np_hwc_uint8)
                    padded_img_rgb_01_bchw = transforms.ToTensor()(Image.fromarray(padded_rgb_np_hwc_uint8)).unsqueeze(0).to(device=device, dtype=dtype)
                    with torch.no_grad():
                        initial_latents = trans_vae_instance.encode(img_rgba_01_bchw, img_rgb_m11_bchw, padded_img_rgb_01_bchw, use_offset=True)
                    call_params["latents"] = initial_latents
                    call_params["image"] = input_image_pil.convert("RGB")
                elif input_image_pil:
                    call_params["image"] = input_image_pil.convert("RGB")
                else: return create_error_image("Img2Img missing input.")
                call_params["strength"] = img2img_strength
            
            call_params["output_type"] = "latent"
            if is_img2img and "negative_prompt" in call_params: del call_params["negative_prompt"]

            output_latents = pipe(**call_params).images
            unpacked_latents = pipe._unpack_latents(output_latents, height, width, pipe.vae_scale_factor)
            denormalized_latents = (unpacked_latents / pipe.vae.config.scaling_factor)
            if hasattr(pipe.vae.config, "shift_factor") and pipe.vae.config.shift_factor != 0:
                 denormalized_latents += pipe.vae.config.shift_factor

            generated_pil_images = []
            label_prefix = ""
            if trans_vae_instance:
                with torch.no_grad():
                    _, decoded_rgba_01 = trans_vae_instance.decode(denormalized_latents.to(next(trans_vae_instance.decoder.parameters()).device, trans_vae_instance.dtype), aug=True)
                for single_rgba_t in decoded_rgba_01:
                    generated_pil_images.append(transforms.ToPILImage()(single_rgba_t.cpu().float().clamp(0,1)))
                label_prefix = "Layered"
            else:
                with torch.no_grad():
                    decoded_rgb_m11 = pipe.vae.decode(denormalized_latents.to(pipe.vae.dtype)).sample
                decoded_rgb_01 = (decoded_rgb_m11 / 2 + 0.5).clamp(0, 1)
                for single_rgb_t in decoded_rgb_01:
                    generated_pil_images.append(transforms.ToPILImage()(single_rgb_t.cpu()))
                label_prefix = "Composite"
            
            if final_save_dir:
                for idx, pil_image in enumerate(generated_pil_images):
                    filename_suffix = f"_{idx}" if len(generated_pil_images) > 1 else ""
                    filename = f"FluxZayn_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{current_seed}{filename_suffix}.png"
                    save_path = os.path.join(final_save_dir, filename)
                    pil_image.save(save_path, pnginfo=metadata)
                    print(f"  [FLUX Script] Saved image with metadata to: {save_path}")

            all_output_images_with_labels.extend([(img, f"{label_prefix} (Seed {current_seed})") for img in generated_pil_images])

        if not all_output_images_with_labels: return create_error_image("No images produced.")
        return all_output_images_with_labels
    except Exception as e:
        full_error = traceback.format_exc()
        print(f"!!! [FLUX Script] CRITICAL ERROR: {e}\n{full_error}")
        return create_error_image(f"Runtime: {e}")
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        import gc; gc.collect()
# --- END: run_layerdiffuse function ---

# --- START: UI Code ---
def get_default_model_path_for_script(model_type_dir, filename): 
    if hasattr(paths, 'models_path') and paths.models_path: 
        potential_path = os.path.join(paths.models_path, model_type_dir, filename)
        if os.path.exists(potential_path):
            return potential_path
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    possible_forge_root = os.path.dirname(os.path.dirname(script_dir))
    potential_path_forge_style = os.path.join(possible_forge_root, "models", model_type_dir, filename)
    if os.path.exists(potential_path_forge_style):
        return potential_path_forge_style
    if hasattr(paths, 'script_path') and paths.script_path:
         potential_path_from_script_path = os.path.join(paths.script_path, "models", model_type_dir, filename)
         if os.path.exists(potential_path_from_script_path):
             return potential_path_from_script_path
    potential_path_cwd = os.path.join(os.getcwd(), "models", model_type_dir, filename)
    if os.path.exists(potential_path_cwd):
        return potential_path_cwd
    return ""

def run_layerdiffuse_with_status_update_for_script( 
    p_flux_model_id_or_path, p_vae_path_transparent, p_lora_path_layerdiffuse, 
    p_clip_encoder_path, p_t5_encoder_path,
    p_prompt, p_neg_prompt, p_input_image_pil, 
    p_width, p_height, p_num_inference_steps, p_guidance_scale,
    p_image_count, p_seed, p_lora_strength, p_img2img_strength):
    
    yield [], "Processing... (FLUX LayerDiffuse Script). Check console for progress."
    
    try:
        results = run_layerdiffuse(
            p_flux_model_id_or_path, p_vae_path_transparent, p_lora_path_layerdiffuse,
            p_clip_encoder_path, p_t5_encoder_path,
            p_prompt, p_neg_prompt, p_input_image_pil,
            p_width, p_height, p_num_inference_steps, p_guidance_scale,
            p_image_count, p_seed, p_lora_strength, p_img2img_strength
        )
        
        if results and isinstance(results, list) and len(results) > 0 and \
           isinstance(results[0], tuple) and len(results[0]) == 2 and \
           isinstance(results[0][0], Image.Image) and "Error:" in results[0][1]:
            yield results, results[0][1]
        elif not results:
             err_img = Image.new("RGB", (p_width, p_height), color="orange")
             yield [(err_img, "Warning: No results returned from pipeline.")], "Warning: No results returned."
        else:
             yield results, f"Generation complete. {len(results)} image(s)/layer(s) created."
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"[FLUX Script] UI level error during generation call: {e}\n{tb_str}")
        error_img = Image.new("RGB", (p_width, p_height), color="purple") 
        error_message_for_status = f"Error: {str(e)}. Check console for details."
        yield [(error_img, f"UI Error: {str(e)}")], error_message_for_status

DEFAULT_TRANSPARENT_VAE_NAME_SCRIPT = "TransparentVAE.safetensors"
DEFAULT_LAYERDIFFUSE_MODEL_DIR_SCRIPT = "LayerDiffuse" 

def create_flux_layerdiffuse_tab_internal_script():
    print("[FLUX LayerDiffuse Script] Creating UI tab (internal_script function).")
    default_tvae_path = get_default_model_path_for_script(DEFAULT_LAYERDIFFUSE_MODEL_DIR_SCRIPT, DEFAULT_TRANSPARENT_VAE_NAME_SCRIPT)

    with gr.Blocks(analytics_enabled=False) as flux_ld_interface_script: 
        gr.HTML("<div style='text-align: center; margin-bottom: 20px;'><h1>FluxZayn LayerDiffusion by DarkForce </h1></div>")

        with gr.Row():
            with gr.Column(scale=2): 
                with gr.Tabs():
                    with gr.TabItem("Models & Paths"):
                        with gr.Group():
                            gr.Markdown("### FLUX Model")
                            flux_model_id_or_path_ui = gr.Textbox( 
                                label="FLUX Model Directory or HuggingFace ID", 
                                placeholder="e.g., black-forest-labs/FLUX.1-schnell or /path/to/flux_model_dir",
                                value="black-forest-labs/FLUX.1-schnell", 
                                info="Path to local FLUX model directory (not single file) or a HuggingFace model ID."
                            )
                        with gr.Group():
                            gr.Markdown("### Layer Generation Assets")
                            vae_path_transparent_ui = gr.Textbox( 
                                label="TransparentVAE Weights Path (.safetensors or .pth)", 
                                placeholder=f"e.g., models/{DEFAULT_LAYERDIFFUSE_MODEL_DIR_SCRIPT}/{DEFAULT_TRANSPARENT_VAE_NAME_SCRIPT}",
                                value=default_tvae_path, 
                                info=f"Path to TransparentVAE weights. Expected in models/{DEFAULT_LAYERDIFFUSE_MODEL_DIR_SCRIPT}/ by default."
                            )
                            lora_path_layerdiffuse_ui = gr.Textbox( 
                                label="Layer LoRA Path (.safetensors)", 
                                placeholder="e.g., /path/to/your/layer_lora.safetensors",
                                value="",
                                info="Path to the LoRA file specifically for layer separation with FLUX."
                            )
                            lora_strength_ui = gr.Slider(
                                label="Layer LoRA Strength",
                                minimum=0.0, maximum=2.0, step=0.05, value=1.0,
                                info="Strength of the applied Layer LoRA."
                            )
                        with gr.Group():
                            gr.Markdown("### Optional: Custom Text Encoders")
                            clip_encoder_path_ui = gr.Textbox(
                                label="Custom CLIP Text Encoder Weights (.safetensors)",
                                placeholder="(Optional) /path/to/clip_weights.safetensors", value="",
                                info="Overrides CLIP weights from the base FLUX model."
                            )
                            t5_encoder_path_ui = gr.Textbox(
                                label="Custom T5 Text Encoder Weights (.safetensors)",
                                placeholder="(Optional) /path/to/t5_weights.safetensors", value="",
                                info="Overrides T5 weights from the base FLUX model."
                            )
                    with gr.TabItem("Text to Image & Settings"):
                        prompt_ui = gr.Textbox(label="Prompt", lines=3, placeholder="A vibrant illustration of...")
                        neg_prompt_ui = gr.Textbox(label="Negative Prompt", lines=2, placeholder="ugly, blurry, watermark...")
                        with gr.Row():
                            width_ui = gr.Slider(label="Width", minimum=256, maximum=2048, step=64, value=1024)
                            height_ui = gr.Slider(label="Height", minimum=256, maximum=2048, step=64, value=1024)
                        with gr.Row():
                            num_inference_steps_ui = gr.Slider(label="Inference Steps", minimum=1, maximum=100, step=1, value=20)
                            guidance_scale_ui = gr.Slider(label="CFG Scale", minimum=0.0, maximum=20.0, step=1.0, value=4.0)
                        with gr.Row():
                            image_count_ui = gr.Slider(label="Number of Images", minimum=1, maximum=16, step=1, value=1)
                            seed_ui = gr.Number(label="Seed", value=-1, precision=0, info="-1 or 0 for random seed.")
                    with gr.TabItem("Image-to-Image"):
                        input_image_pil_ui = gr.Image(label="Input Image (for Img2Img, RGBA recommended)", type="pil", image_mode="RGBA", height=300) 
                        img2img_strength_ui = gr.Slider(
                            label="Denoising Strength (for Img2Img)",
                            minimum=0.0, maximum=1.0, step=0.01, value=0.70,
                            info="Controls how much the input image influences the output."
                        )
                generate_button_ui = gr.Button("Generate Images", variant="primary", elem_id="flux_ld_generate_button_script")
                status_textbox_ui = gr.Textbox(label="Status", value="Ready.", interactive=False, lines=1, max_lines=3)
            with gr.Column(scale=3): 
                output_gallery_ui = gr.Gallery(
                    label="Generated Images & Layers", show_label=True, elem_id="flux_ld_gallery_output_script",
                    columns=1, height="auto", object_fit="contain", preview=True 
                )
                gr.Markdown("### Notes & Tips")
                gr.Markdown(
                    "- **Base FLUX Model:** Default is `FLUX.1-schnell`. It's faster but may require different settings for best results."
                    "- **Layer Separation:** Requires correct TransparentVAE weights and a compatible Layer LoRA."
                    "- **Performance:** Img2Img is broken, so don't use."
                    "- **Touch Grass:** It will help with your render times."
                )
        
        inputs_list_ui = [
            flux_model_id_or_path_ui, vae_path_transparent_ui, lora_path_layerdiffuse_ui, 
            clip_encoder_path_ui, t5_encoder_path_ui,
            prompt_ui, neg_prompt_ui, input_image_pil_ui, 
            width_ui, height_ui, num_inference_steps_ui, guidance_scale_ui,
            image_count_ui, seed_ui, lora_strength_ui, img2img_strength_ui
        ]
        generate_button_ui.click(
            fn=run_layerdiffuse_with_status_update_for_script, 
            inputs=inputs_list_ui,
            outputs=[output_gallery_ui, status_textbox_ui],
            show_progress="full" 
        )
    return (flux_ld_interface_script, "FLUX LayerDiffuse", "flux_layerdiffuse_tab_script") 
# --- END: UI Code ---

# --- START: Tab Registration Logic ---
def on_ui_tabs_for_flux_script(): 
    print("[FLUX LayerDiffuse Script] on_ui_tabs_for_flux_script callback invoked.")
    try:
        if not DIFFUSERS_AVAILABLE or not SAFETENSORS_AVAILABLE: 
            with gr.Blocks() as error_block:
                gr.HTML("<div style='color:red;text-align:center;padding:20px;'><h1>FLUX LayerDiffuse Error</h1><p>Core libraries (Diffusers/Safetensors) are missing. Please check console and ensure `install.py` ran correctly and Forge was restarted.</p></div>")
            return [(error_block, "FLUX LD (Lib Error)", "flux_ld_lib_error_tab")]
        tab_tuple = create_flux_layerdiffuse_tab_internal_script()
        print(f"[FLUX LayerDiffuse Script] Tab created successfully: '{tab_tuple[1]}'")
        return [tab_tuple]
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"[FLUX LayerDiffuse Script] CRITICAL ERROR creating tab: {e}{error_details}")
        escaped_error = html.escape(str(e)); escaped_traceback = html.escape(error_details).replace("\n", "<br>")
        with gr.Blocks(analytics_enabled=False) as error_tab_block:
            gr.HTML(f"<div style='text-align:center;color:red;padding:20px;'><h2>Error Loading FLUX LayerDiffuse Tab</h2><p><strong>Error:</strong> {escaped_error}</p><details><summary>Traceback</summary><pre style='max-height:200px;overflow:auto;text-align:left;background:#f0f0f0;padding:10px;'>{escaped_traceback}</pre></details></div>")
        return [(error_tab_block, "FLUX LD (UI Error)", "flux_layerdiffuse_ui_error_tab")]

if 'script_callbacks' in globals():
    script_callbacks.on_ui_tabs(on_ui_tabs_for_flux_script) 
    print("[FLUX LayerDiffuse Script] Main script loaded. Tab registration attempted.")
else:
    print("[FLUX LayerDiffuse Script] Main script loaded, but 'script_callbacks' not found. Skipping tab registration.")
# --- END: Tab Registration Logic ---
