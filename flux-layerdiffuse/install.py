
import launch
import os

print("[FLUX LayerDiffuse] Checking and installing requirements (from install.py)...")

requirements = {
    "diffusers": "diffusers>=0.27.0", 
    "transformers": "transformers>=4.38.0",
    "safetensors": "safetensors>=0.4.0",
    "accelerate": "accelerate>=0.25.0",
    "opencv-python-headless": "opencv-python-headless" # For cv2 used in TransparentVAE
}

for package_name, requirement_str in requirements.items():
    already_installed = False
    try:
        __import__(package_name)
        already_installed = True 
    except ImportError:
        pass 

    if not already_installed:
        print(f"[FLUX LayerDiffuse] Attempting to install: {requirement_str}")
        launch.run_pip(f"install --upgrade {requirement_str}", f"FLUX LayerDiffuse: {package_name}")

print("[FLUX LayerDiffuse] Requirement check/installation process complete (from install.py).")

try:
    from modules import paths as forge_paths # Attempt to use Forge's paths module
    models_path_to_use = None
    if hasattr(forge_paths, 'models_path') and forge_paths.models_path:
        models_path_to_use = forge_paths.models_path
    elif hasattr(launch, 'paths_internal') and launch.paths_internal and hasattr(launch.paths_internal, 'models_path'):
        models_path_to_use = launch.paths_internal.models_path
    
    if models_path_to_use:
        layerdiffuse_models_dir = os.path.join(models_path_to_use, "LayerDiffuse") 
        if not os.path.exists(layerdiffuse_models_dir):
            try:
                os.makedirs(layerdiffuse_models_dir, exist_ok=True)
                print(f"[FLUX LayerDiffuse] Created directory: {layerdiffuse_models_dir} (from install.py)")
            except Exception as e_mkdir:
                print(f"[FLUX LayerDiffuse] Error creating directory {layerdiffuse_models_dir}: {e_mkdir} (from install.py)")
    else:
        print("[FLUX LayerDiffuse] Could not determine Forge's models_path automatically (from install.py).")
        print("    Please ensure 'models/LayerDiffuse/' exists in your Forge installation for TransparentVAE.")
except Exception as e_dir_create:
    print(f"[FLUX LayerDiffuse] Error during model directory check in install.py: {e_dir_create}")

print("[FLUX LayerDiffuse] install.py finished.")
