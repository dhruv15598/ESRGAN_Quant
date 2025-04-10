import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import argparse
import os
import time
import glob
from esrgan_model import RRDBNet
import torch.quantization


# --- Helper stuff ---

def load_weights(checkpoint_file, model, device):
    """Loads weights into the model."""
    print(f"=> Loading weights for a model instance from: {checkpoint_file}")
    # Load checkpoint onto the specified device directly
    try:
        # Try loading with weights_only=True first for safety
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
    except:
        # If it fails (e.g., older format), load without weights_only
        print("  Warning: Loading with weights_only=True failed. Attempting full load.")
        checkpoint = torch.load(checkpoint_file, map_location=device)

    model_state_dict = model.state_dict()

    # Handle potential 'state_dict' key in checkpoint
    if isinstance(checkpoint, dict) and (
            'state_dict' in checkpoint or 'params_ema' in checkpoint):  # Check for common keys
        ckpt_state_dict = checkpoint.get('state_dict', checkpoint.get('params_ema', {}))
    elif isinstance(checkpoint, dict):
        ckpt_state_dict = checkpoint  # Assume checkpoint IS the state dict if no known wrapper keys
    else:
        # Handle case where checkpoint might be just the tensor dict directly (less common)
        print("  Warning: Checkpoint seems to be just the state_dict itself.")
        ckpt_state_dict = checkpoint

    if not ckpt_state_dict:
        raise ValueError(f"Could not extract state_dict from checkpoint file: {checkpoint_file}")

    # Filter keys (handle potential 'module.' prefix from DataParallel)
    state_dict = {
        k.replace('module.', ''): v for k, v in ckpt_state_dict.items()
        if
        k.replace('module.', '') in model_state_dict and v.size() == model_state_dict[k.replace('module.', '')].size()
    }

    if not state_dict:
        raise ValueError(f"No matching keys found between checkpoint {checkpoint_file} and model state_dict.")

    loaded_keys = state_dict.keys()
    ignored_keys = [k for k in model_state_dict if k not in loaded_keys]
    if ignored_keys:
        print(f"  Warning: The following keys were missing/ignored in {checkpoint_file}: {ignored_keys}")

    model_state_dict.update(state_dict)
    model.load_state_dict(model_state_dict)
    print(f"  Successfully loaded {len(loaded_keys)} parameter tensors.")
    return model

def save_tensor_as_image(tensor, filename):
    """Saves a CHW tensor (range [0, 1]) as an image file."""
    tensor = tensor.squeeze(0).cpu().detach()
    tensor.clamp_(0, 1)
    img = transforms.ToPILImage()(tensor)
    img.save(filename)
    print(f"Saved image to {filename}")


def save_std_heatmap(std_tensor, filename, colormap='viridis'):  # Changed default colormap
    """Saves a HW standard deviation tensor as a colored heatmap image."""
    std_map = std_tensor.squeeze().cpu().detach()
    s_min, s_max = std_map.min(), std_map.max()
    if (s_max - s_min) < 1e-8:
        std_norm = torch.zeros_like(std_map)
    else:
        std_norm = (std_map - s_min) / (s_max - s_min)
    std_map_np = std_norm.numpy()
    try:
        colored_std = getattr(plt.cm, colormap)(std_map_np)
    except AttributeError:
        print(f"Warning: Colormap '{colormap}' not found. Using 'viridis'.")
        colored_std = plt.cm.viridis(std_map_np)

    colored_std = np.nan_to_num(colored_std)
    colored_std_uint8 = (colored_std[..., :3] * 255).astype(np.uint8)
    stdmap_pil = Image.fromarray(colored_std_uint8)
    stdmap_pil.save(filename)
    print(f"Saved heatmap to {filename}")


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ESRGAN Super-Resolution with Ensemble Uncertainty')
    parser.add_argument('--input', type=str, required=True, help='Path to the low-resolution input image')
    # --- Argument for the directory containing ensemble model subdirectories ---
    parser.add_argument('--weights_dir', type=str, default='mySRGAN/model_weights_ensemble',
                        help='Path to the directory containing ensemble model folders (model1, model2, ...)')
    parser.add_argument('--output_sr', type=str, default='output_ensemble_sr_ptdq.png',
                        help='Path to save the super-resolved image')
    parser.add_argument('--output_uq', type=str, default='output_ensemble_uq_heatmap_ptdq.png',
                        help='Path to save the uncertainty heatmap')
    parser.add_argument('--crop_size', type=int, default=None,
                        help='Optional size to center crop the input image (e.g., 200)')
    parser.add_argument('--cpu', action='store_true', help='Force use CPU even if CUDA is available')
    parser.add_argument('--quantize', action='store_true', help='Apply Post-Training Dynamic Quantization (forces CPU)')

    args = parser.parse_args()

    # Setup device
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # --- Find ensemble model weight files ---
    model_weight_files = sorted(glob.glob(os.path.join(args.weights_dir, 'model*', '*.pth.tar')))
    if not model_weight_files:
        raise FileNotFoundError(f"No '.pth.tar' weight files found in subdirectories under {args.weights_dir}")
    num_ensemble_models = len(model_weight_files)
    print(f"Found {num_ensemble_models} ensemble model weight files:")
    for wf in model_weight_files:
        print(f"  - {wf}")

    # --- Instantiate and load ensemble models ---
    ensemble_models = []
    for i, weight_file in enumerate(model_weight_files):
        print(f"\nLoading Ensemble Member {i + 1}/{num_ensemble_models}")
        # 1. Instantiate the FP32 model
        fp32_model = RRDBNet(in_channels=3, out_channels=3, channels=64,
                             growth_channels=32, upscale_factor=2,
                             residual_beta=0.2)

        # 2. Load weights into the FP32 model (use initial device)
        fp32_model = load_weights(weight_file, fp32_model, device)
        fp32_model.eval()

        # 3. Apply Quantization if enabled
        if args.quantize:
            print(f"  Applying Dynamic Quantization to Member {i + 1}...")
            # Ensure model is on CPU for dynamic quantization step
            fp32_model.to('cpu')
            layers_to_quantize = {nn.Conv2d}
            quantized_model = torch.quantization.quantize_dynamic(
                model=fp32_model,
                qconfig_spec=layers_to_quantize,
                dtype=torch.qint8
            )
            # The model to add to the list is the quantized one
            inference_model = quantized_model
            print(f"  Dynamic Quantization applied for Member {i + 1}.")
        else:
            # Otherwise, the model to add is the original FP32 one
            inference_model = fp32_model

        # 4. Ensure the final model is on the correct device and add to list
        inference_model.to(device)  # Move to CPU if quantizing, or original device if not
        ensemble_models.append(inference_model)

    # Load and preprocess the input image
    input_image = Image.open(args.input).convert("RGB")
    transform_list = []
    if args.crop_size and args.crop_size > 0:
        print(f"Applying center crop of size ({args.crop_size}, {args.crop_size})")
        transform_list.append(transforms.CenterCrop((args.crop_size, args.crop_size)))
    transform_list.append(transforms.ToTensor())
    preprocess = transforms.Compose(transform_list)
    lr_tensor = preprocess(input_image).unsqueeze(0).to(device)
    print(f"\nInput tensor shape: {lr_tensor.shape}")

    # --- Perform Ensemble Inference with Precise Timing ---
    print(f"Running inference for {num_ensemble_models} ensemble members...")
    sr_outputs = []
    total_model_call_time = 0.0  # Accumulator for model call duration
    overall_start_time = time.time()  # Keep track of overall loop time

    with torch.no_grad():
        for i, model in enumerate(ensemble_models):
            print(f"  Inferencing with model {i + 1}/{num_ensemble_models}...")
            # --- Time only the model call ---
            iter_start_time = time.time()
            sr_out = model(lr_tensor)
            iter_end_time = time.time()
            total_model_call_time += (iter_end_time - iter_start_time)
            # -------------------------------
            # Move result to CPU after timing
            sr_outputs.append(sr_out.cpu())

    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time

    # --- Updated Timing Report ---
    if num_ensemble_models > 0:
        avg_model_call_time_per_member = total_model_call_time / num_ensemble_models
        avg_overall_time_per_member = overall_duration / num_ensemble_models  # Previous calculation method
        print(f"\nInference Timing Report:")
        print(f"  Total time for loop execution: {overall_duration:.4f} seconds.")
        print(f"  Total accumulated model call time: {total_model_call_time:.4f} seconds.")
        # --- Report per member averages ---
        print(f"  Average overall time per member: {avg_overall_time_per_member:.4f} seconds.")
        print(f"  Average model call time per member: {avg_model_call_time_per_member:.4f} seconds.")
    else:
        print(f"\nInference Timing Report:")
        print(f"  Total time for loop execution: {overall_duration:.4f} seconds. (No members run)")
        print(f"  Total accumulated model call time: {total_model_call_time:.4f} seconds.")

    # --- Stack results and calculate mean and standard deviation ---
    # ... (Aggregation remains the same) ...
    stacked_sr = torch.stack(sr_outputs, dim=0)
    mean_sr = torch.mean(stacked_sr, dim=0)
    std_sr = torch.std(stacked_sr, dim=0)
    uncertainty_map = torch.mean(std_sr, dim=1)

    # --- Save Outputs (Unchanged, maybe uncomment) ---
    save_tensor_as_image(mean_sr, args.output_sr)
    save_std_heatmap(uncertainty_map, args.output_uq, colormap='viridis')

    print("\nProcessing complete.")