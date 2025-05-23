import torch
import torch.nn as nn
import torch.profiler
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import argparse
import os
import time
# Import quantization module
import torch.quantization

# Import your model definition
from esrgan_dropout import DRRRDBNet


# --- Helper Functions (Mostly Unchanged) ---

def load_weights(checkpoint_file, model, device):
    """Loads weights into the model."""
    print("=> Loading weights from:", checkpoint_file)
    try:
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
    except:
        print("  Warning: Loading with weights_only=True failed. Attempting full load.")
        checkpoint = torch.load(checkpoint_file, map_location=device)

    model_state_dict = model.state_dict()

    if isinstance(checkpoint, dict) and ('state_dict' in checkpoint or 'params_ema' in checkpoint):
        ckpt_state_dict = checkpoint.get('state_dict', checkpoint.get('params_ema', {}))
    elif isinstance(checkpoint, dict):
        ckpt_state_dict = checkpoint
    else:
        print("  Warning: Checkpoint seems to be just the state_dict itself.")
        ckpt_state_dict = checkpoint

    if not ckpt_state_dict:
        raise ValueError(f"Could not extract state_dict from checkpoint file: {checkpoint_file}")

    state_dict = {
        k.replace('module.', ''): v for k, v in ckpt_state_dict.items()
        if
        k.replace('module.', '') in model_state_dict and v.size() == model_state_dict[k.replace('module.', '')].size()
    }

    if not state_dict:
        raise ValueError("No matching keys found between checkpoint and model state_dict.")

    loaded_keys = state_dict.keys()
    ignored_keys = [k for k in model_state_dict if k not in loaded_keys]
    if ignored_keys:
        print(f"  Warning: The following keys were missing/ignored: {ignored_keys}")

    model_state_dict.update(state_dict)
    model.load_state_dict(model_state_dict)
    print(f"Successfully loaded {len(loaded_keys)} parameter tensors.")
    return model


def enable_dropout(model):
    """Keeps dropout layers active during inference."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            # Ensure dropout is explicitly set to training mode
            # This is necessary even if the main model is in eval mode
            m.train()


def save_tensor_as_image(tensor, filename):
    """Saves a CHW tensor (range [0, 1]) as an image file."""
    tensor = tensor.squeeze(0).cpu().detach()
    tensor.clamp_(0, 1)
    img = transforms.ToPILImage()(tensor)
    img.save(filename)
    print(f"Saved image to {filename}")


def save_std_heatmap(std_tensor, filename, colormap='viridis'):  # Changed default
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
    parser = argparse.ArgumentParser(
        description='ESRGAN Super-Resolution with MC Dropout Uncertainty (Dynamic Quantization)')
    parser.add_argument('--input', type=str, required=True, help='Path to the low-resolution input image')
    parser.add_argument('--weights', type=str, default='gen174.pth.tar', help='Path to the generator weights file')
    # Modified default output filenames
    parser.add_argument('--output_sr', type=str, default='output_dropout_sr_gpu.png',
                        help='Path to save the super-resolved image')
    parser.add_argument('--output_uq', type=str, default='output_dropout_uq_heatmap_gpu.png',
                        help='Path to save the uncertainty heatmap')
    parser.add_argument('--mc_passes', type=int, default=2, help='Number of Monte Carlo dropout passes')
    parser.add_argument('--crop_size', type=int, default=None,
                        help='Optional size to center crop the input image (e.g., 200)')
    parser.add_argument('--cpu', action='store_true', help='Force use CPU even if CUDA is available')
    parser.add_argument('--gpu', action='store_true', help='Attempt execution on GPU if available (overrides --cpu)')
    # Add quantization argument
    parser.add_argument('--quantize', action='store_true', help='Apply Post-Training Dynamic Quantization (forces CPU)')
    args = parser.parse_args()

    # --- Device Setup ---
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("Warning: --gpu flag specified, but CUDA is not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif args.cpu:  # Check if --cpu flag forces CPU
        device = torch.device("cpu")
    else:  # Default: Use CUDA if available, else CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Instantiate and Load Model ---
    # Instantiate the FP32 model first
    fp32_model = DRRRDBNet(in_channels=3, out_channels=3, channels=64,
                           growth_channels=32, upscale_factor=2,  # Should result in 4x total upscale
                           residual_beta=0.2)

    # Load the pre-trained weights into the FP32 model
    # Note: Load weights BEFORE quantizing
    fp32_model = load_weights(args.weights, fp32_model, device)  # Load to target device initially
    fp32_model.eval()  # Set to eval mode initially

    # --- Apply Dynamic Quantization (if enabled) ---
    if args.quantize:
        print("Applying Post-Training Dynamic Quantization...")
        #fp32_model.to('cpu')
        # Define layers to quantize dynamically (usually Linear, sometimes Conv2d).
        # Let's try targeting Conv2d as they are dominant in ESRGAN generator.
        layers_to_quantize = {nn.Conv2d}  # Or potentially {nn.Linear, nn.Conv2d} if Linear layers were present

        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model=fp32_model,
            qconfig_spec=layers_to_quantize,  # Specify which layer types to target
            dtype=torch.qint8  # Target data type
        )
        print("Dynamic Quantization applied.")
        # The model used for inference is now the quantized one
        model = quantized_model
        model.to(device)

    else:
        # Use the original FP32 model if quantization is not enabled
        model = fp32_model
        model.to(device)  # Ensure it's on the correct device

    # --- Load and Preprocess Input Image ---
    input_image = Image.open(args.input).convert("RGB")
    transform_list = []

    if args.crop_size and args.crop_size > 0:
        print(f"Applying center crop of size ({args.crop_size}, {args.crop_size})")
        transform_list.append(transforms.CenterCrop((args.crop_size, args.crop_size)))
    transform_list.append(transforms.ToTensor())
    preprocess = transforms.Compose(transform_list)
    # Ensure input tensor is on the same device as the model
    lr_tensor = preprocess(input_image).unsqueeze(0).to(device)
    print(f"Input tensor shape: {lr_tensor.shape}")

    # --- Perform Monte Carlo Dropout Inference ---
    print(f"Running {args.mc_passes} MC Dropout passes...")
    sr_passes = []
    total_model_call_time = 0.0  # Accumulator for model call duration
    overall_start_time = time.time()  # Keep track of overall loop time

    # Make sure model is in eval mode before loop
    model.eval() # Already done

    for i in range(args.mc_passes):
        enable_dropout(model)  # Ensure dropout layers are active for this pass
        with torch.no_grad():
            # Time only the model call
            iter_start_time = time.time()
            sr_out = model(lr_tensor)  # Use the (potentially quantized) model
            iter_end_time = time.time()
            total_model_call_time += (iter_end_time - iter_start_time)
            # --- Timing ends ---

        # Move result to CPU after timing
        sr_passes.append(sr_out.cpu())
        if (i + 1) % 5 == 0:
            print(f"  Completed pass {i + 1}/{args.mc_passes}")

    # --- Loop finished ---
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time

    # --- Updated Timing Report ---
    if args.mc_passes > 0:
        avg_model_call_time_per_pass = total_model_call_time / args.mc_passes
        avg_overall_time_per_pass = overall_duration / args.mc_passes
        print(f"\nInference Timing Report:")
        print(f"  Total time for loop execution: {overall_duration:.4f} seconds.")
        print(f"  Total accumulated model call time: {total_model_call_time:.4f} seconds.")
        print(f"  Average overall time per pass: {avg_overall_time_per_pass:.4f} seconds.")
        print(f"  Average model call time per pass: {avg_model_call_time_per_pass:.4f} seconds.")
    else:
        print(f"\nInference Timing Report:")
        print(f"  Total time for loop execution: {overall_duration:.4f} seconds. (No passes run)")
        print(f"  Total accumulated model call time: {total_model_call_time:.4f} seconds.")

    # --- Aggregate Results ---
    stacked_sr = torch.stack(sr_passes, dim=0)
    mean_sr = torch.mean(stacked_sr, dim=0)
    std_sr = torch.std(stacked_sr, dim=0)
    uncertainty_map = torch.mean(std_sr, dim=1)

    # --- Save Outputs ---
    save_tensor_as_image(mean_sr, args.output_sr)
    save_std_heatmap(uncertainty_map, args.output_uq, colormap='viridis')

    print("\nProcessing complete.")
