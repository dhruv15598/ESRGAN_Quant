import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import argparse
import os
import time

# Import your model definition
from esrgan_dropout import DRRRDBNet

# --- Helper Functions ---

def load_weights(checkpoint_file, model, device):
    """Loads weights into the model."""
    print("=> Loading weights from:", checkpoint_file)
    # Load checkpoint onto the specified device directly
    # Use weights_only=True for security if the checkpoint only contains weights
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
    model_state_dict = model.state_dict()

    # Handle potential 'state_dict' key in checkpoint
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        ckpt_state_dict = checkpoint["state_dict"]
    else:
        ckpt_state_dict = checkpoint # Assume checkpoint IS the state dict

    # Filter keys (handle potential 'module.' prefix from DataParallel)
    state_dict = {
        k.replace('module.', ''): v for k, v in ckpt_state_dict.items()
        if k.replace('module.', '') in model_state_dict and v.size() == model_state_dict[k.replace('module.', '')].size()
    }

    if not state_dict:
        raise ValueError("No matching keys found between checkpoint and model state_dict.")

    loaded_keys = state_dict.keys()
    ignored_keys = [k for k in model_state_dict if k not in loaded_keys]
    if ignored_keys:
        print(f"Warning: The following keys were missing/ignored: {ignored_keys}")

    model_state_dict.update(state_dict)
    model.load_state_dict(model_state_dict)
    print(f"Successfully loaded {len(loaded_keys)} parameter tensors.")
    return model

def enable_dropout(model):
    """Keeps dropout layers active during inference."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def save_tensor_as_image(tensor, filename):
    """Saves a CHW tensor (range [0, 1]) as an image file."""
    # Ensure tensor is on CPU and detach from graph
    tensor = tensor.squeeze(0).cpu().detach()
    # Clamp values just in case
    tensor.clamp_(0, 1)
    # Convert to PIL Image
    img = transforms.ToPILImage()(tensor)
    # Save the image
    img.save(filename)
    print(f"Saved image to {filename}")

def save_std_heatmap(std_tensor, filename, colormap='jet'):
    """Saves a HW standard deviation tensor as a colored heatmap image."""
    # Ensure tensor is on CPU and detach
    std_map = std_tensor.squeeze().cpu().detach() # Remove channel/batch dims if present

    # Normalize the standard deviation map to [0, 1] for colormap
    s_min, s_max = std_map.min(), std_map.max()
    if (s_max - s_min) < 1e-8: # Avoid division by zero for constant std dev
        std_norm = torch.zeros_like(std_map)
    else:
        std_norm = (std_map - s_min) / (s_max - s_min)

    # Convert to numpy and apply colormap
    std_map_np = std_norm.numpy()
    colored_std = getattr(plt.cm, colormap)(std_map_np) # Use plt.cm.<colormap_name>
    # Handle potential NaN/Inf in colormap output
    colored_std = np.nan_to_num(colored_std)
    # Convert to uint8 image (matplotlib output is RGBA float [0,1])
    colored_std_uint8 = (colored_std[..., :3] * 255).astype(np.uint8)
    stdmap_pil = Image.fromarray(colored_std_uint8)

    # Save the image
    stdmap_pil.save(filename)
    print(f"Saved heatmap to {filename}")


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ESRGAN Super-Resolution with MC Dropout Uncertainty')
    parser.add_argument('--input', type=str, required=True, help='Path to the low-resolution input image')
    parser.add_argument('--weights', type=str, default='gen174.pth.tar', help='Path to the generator weights file')
    parser.add_argument('--output_sr', type=str, default='output_sr.png', help='Path to save the super-resolved image')
    parser.add_argument('--output_uq', type=str, default='output_uq_heatmap.png', help='Path to save the uncertainty heatmap')
    parser.add_argument('--mc_passes', type=int, default=2, help='Number of Monte Carlo dropout passes')
    parser.add_argument('--crop_size', type=int, default=None, help='Optional size to center crop the input image (e.g., 200)')
    parser.add_argument('--cpu', action='store_true', help='Force use CPU even if CUDA is available')

    args = parser.parse_args()

    # Setup device
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Instantiate the model
    # Use the same parameters as in the original app.py/training
    model = DRRRDBNet(in_channels=3, out_channels=3, channels=64,
                      growth_channels=32, upscale_factor=2, # Should result in 4x total upscale
                      residual_beta=0.2)

    # Load the pre-trained weights
    model = load_weights(args.weights, model, device)
    model.to(device)
    # We set model.eval() before the loop, but enable_dropout() overrides it for dropout layers

    # Load and preprocess the input image
    input_image = Image.open(args.input).convert("RGB")

    # Define transforms (add crop if specified)
    transform_list = []
    if args.crop_size and args.crop_size > 0:
        print(f"Applying center crop of size ({args.crop_size}, {args.crop_size})")
        transform_list.append(transforms.CenterCrop((args.crop_size, args.crop_size)))
    transform_list.append(transforms.ToTensor()) # Scales to [0, 1]
    # transform_list.append(transforms.Normalize((0,0,0), (1,1,1))) # Usually not needed if ToTensor is used

    preprocess = transforms.Compose(transform_list)
    lr_tensor = preprocess(input_image).unsqueeze(0).to(device) # Add batch dim and move to device
    print(f"Input tensor shape: {lr_tensor.shape}")

    # Perform Monte Carlo Dropout inference
    print(f"Running {args.mc_passes} MC Dropout passes...")
    start_time = time.time()
    sr_passes = []
    model.eval() # Set to eval mode for BatchNorm etc.
    for i in range(args.mc_passes):
        enable_dropout(model) # Activate dropout layers
        with torch.no_grad():
            sr_out = model(lr_tensor)
        # Move result to CPU to avoid accumulating GPU memory if running many passes
        sr_passes.append(sr_out.cpu())
        if (i + 1) % 5 == 0:
             print(f"  Completed pass {i+1}/{args.mc_passes}")

    end_time = time.time()
    # Calculate inference time per pass for comparison
    time_per_pass = (end_time - start_time) / args.mc_passes
    print(f"Inference finished. Average time per pass: {time_per_pass:.4f} seconds."
          f" Total time elapsed {(end_time - start_time):.4f} seconds")

    # Stack results and calculate mean and standard deviation
    # Ensure stacking happens on CPU
    stacked_sr = torch.stack(sr_passes, dim=0) # Shape: (mc_passes, B, C, H, W) B=1 here

    mean_sr = torch.mean(stacked_sr, dim=0) # Shape: (B, C, H, W)
    std_sr = torch.std(stacked_sr, dim=0)   # Shape: (B, C, H, W)

    # Calculate uncertainty map (mean std across channels)
    uncertainty_map = torch.mean(std_sr, dim=1) # Shape: (B, H, W) B=1 here

    # Save outputs
    #save_tensor_as_image(mean_sr, args.output_sr)
    #save_std_heatmap(uncertainty_map, args.output_uq)

    print("Processing complete.")