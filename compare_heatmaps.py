import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from scipy.stats import pearsonr

# --- Helper: Convert PIL Image to Grayscale NumPy array (0-1 range) ---
def image_to_gray_numpy(img_pil):
    """Converts PIL image to grayscale NumPy array float (0-1)."""
    if img_pil.mode == 'LA' or img_pil.mode == 'RGBA': # Handle alpha channel if present
        img_pil = img_pil.convert('RGB')
    img_gray_pil = img_pil.convert('L') # Convert to grayscale
    img_np = np.array(img_gray_pil).astype(np.float32) / 255.0 # Convert to float32 [0, 1]
    return img_np

# --- Main Execution ---
if __name__ == "__main__":
    # Define default paths based on the highlighted files
    default_path1 = "output_ensemble_uq_heatmap_gpu.png"
    default_path2 = "output_ensemble_uq_heatmap_gpu_DPTQ.png"

    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description='Objectively compare two heatmap images.')
    parser.add_argument('--img1', type=str, default=default_path1,
                        help=f'Path to the first heatmap image (Default: {default_path1})')
    parser.add_argument('--title1', type=str, default='Heatmap (FP32 GPU)',
                        help='Reference name for the first image')
    parser.add_argument('--img2', type=str, default=default_path2,
                        help=f'Path to the second heatmap image (Default: {default_path2})')
    parser.add_argument('--title2', type=str, default='Heatmap (DPTQ GPU)',
                        help='Reference name for the second image')
    parser.add_argument('--plot_diff', action='store_true',
                        help='Show a plot of the absolute difference image.')


    args = parser.parse_args()

    # --- Load Images ---
    try:
        print(f"Loading image 1: {args.img1}")
        img1_pil = Image.open(args.img1)
        print(f"Loading image 2: {args.img2}")
        img2_pil = Image.open(args.img2)
    except FileNotFoundError as e:
        print(f"\nError: Could not load image file.")
        print(f"Details: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred while loading images: {e}")
        exit()

    # --- Preprocess Images ---
    print("Converting images to grayscale NumPy arrays [0, 1]...")
    img1_np = image_to_gray_numpy(img1_pil)
    img2_np = image_to_gray_numpy(img2_pil)

    # --- Check Shapes ---
    if img1_np.shape != img2_np.shape:
        print("\nError: Image shapes do not match!")
        print(f"  Shape 1: {img1_np.shape}")
        print(f"  Shape 2: {img2_np.shape}")
        exit()

    print(f"Image shape: {img1_np.shape}")

    # --- Calculate Metrics ---
    print("\nCalculating comparison metrics...")

    # 1. Mean Absolute Error (MAE)
    # Average absolute difference per pixel. Range [0, 1]. Lower is more similar.
    mae = np.mean(np.abs(img1_np - img2_np))

    # 2. Mean Squared Error (MSE)
    # Average squared difference per pixel. Range [0, 1]. Lower is more similar. Penalizes large errors.
    # Uses scikit-image's implementation for consistency.
    mse_val = mse(img1_np, img2_np)

    # 3. Peak Signal-to-Noise Ratio (PSNR)
    # Based on MSE. Higher is better (more similar). Measured in dB.
    # data_range is max possible pixel value (1.0 since we normalized).
    # Handle potential division by zero if images are identical (MSE=0).
    if mse_val == 0:
        psnr_val = float('inf')
    else:
        psnr_val = psnr(img1_np, img2_np, data_range=1.0)

    # 4. Structural Similarity Index Measure (SSIM)
    # Measures similarity based on structure, contrast, luminance. Range [-1, 1] or [0, 1]. Higher is better.
    # data_range is max possible pixel value. Use win_size appropriate for image size (must be odd >=3).
    win_size = min(7, min(img1_np.shape[0], img1_np.shape[1])) # Ensure window fits
    if win_size % 2 == 0: win_size -= 1 # Make it odd
    if win_size < 3:
         print("Warning: Image too small for default SSIM window size. SSIM might be unreliable.")
         ssim_val = float('nan') # Cannot calculate
    else:
         ssim_val = ssim(img1_np, img2_np, data_range=1.0, win_size=win_size)

    # 5. Pearson Correlation Coefficient
    # Measures linear correlation between pixel values. Range [-1, 1]. Higher positive value means more similar patterns.
    corr_val, _ = pearsonr(img1_np.ravel(), img2_np.ravel()) # Flatten arrays for correlation

    # 6. Difference in Mean Intensity
    mean1 = np.mean(img1_np)
    mean2 = np.mean(img2_np)
    mean_diff = mean1 - mean2

    # 7. Difference in Standard Deviation (Contrast)
    std1 = np.std(img1_np)
    std2 = np.std(img2_np)
    std_diff = std1 - std2


    # --- Print Results ---
    print("\n--- Objective Comparison Results ---")
    print(f"Comparing '{args.title1}' vs '{args.title2}'")
    print("-" * 30)
    print(f"Mean Absolute Error (MAE):  {mae:.6f}  (Lower is better, range [0, 1])")
    print(f"Mean Squared Error (MSE):   {mse_val:.6f}  (Lower is better, range [0, 1])")
    print(f"Peak Signal-to-Noise (PSNR):{psnr_val:.2f} dB (Higher is better, Inf is identical)")
    print(f"Structural Similarity (SSIM):{ssim_val:.4f}  (Higher is better, range [0, 1] or [-1, 1])") # Note: skimage implementation often [0,1] for images
    print(f"Pearson Correlation Coeff:  {corr_val:.4f}  (Higher is better, range [-1, 1])")
    print(f"Mean Intensity Difference:  {mean_diff:+.4f}  ({args.title1} mean: {mean1:.4f}, {args.title2} mean: {mean2:.4f})")
    print(f"Std Deviation Difference:   {std_diff:+.4f}  ({args.title1} std: {std1:.4f}, {args.title2} std: {std2:.4f})")
    print("-" * 30)

    # --- Plot Difference Image (Optional) ---
    if args.plot_diff:
        print("Generating difference plot...")
        diff_img = np.abs(img1_np - img2_np)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        im0 = axes[0].imshow(img1_pil) # Show original color version for context
        axes[0].set_title(args.title1)
        axes[0].axis('off')

        im1 = axes[1].imshow(img2_pil) # Show original color version
        axes[1].set_title(args.title2)
        axes[1].axis('off')

        im2 = axes[2].imshow(diff_img, cmap='viridis', vmin=0, vmax=np.max(diff_img)) # Show difference map
        axes[2].set_title('Absolute Difference')
        axes[2].axis('off')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04) # Add colorbar for difference scale

        plt.tight_layout(pad=1.5)
        fig.suptitle('Objective Heatmap Comparison', fontsize=16)
        plt.show()

    print("\nComparison complete.")