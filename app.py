import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from model import DRRRDBNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

gen2 = DRRRDBNet(3, 3, 64, 32, 2, 0.2).to(device)

def load_weights(checkpoint_file, model):
    print("=> Loading weights from:", checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model_state_dict = model.state_dict()
    state_dict = {
        k: v for k, v in checkpoint["state_dict"].items()
        if k in model_state_dict and v.size() == model_state_dict[k].size()
    }
    model_state_dict.update(state_dict)
    model.load_state_dict(model_state_dict)
    print("Successfully loaded the pretrained model weights")
    return model

gen2 = load_weights("gen174.pth.tar", gen2)
gen2.eval()

def enable_dropout(model):
    """Keeps dropout layers active during inference."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def mcd_superres_crop(image, mc_passes=5):
    """
    1) Random crop input (200x200)
    2) Upscale that cropped patch 4× with bicubic so users can compare visually
    3) Run multiple forward passes (MCD) on the cropped patch
    4) Return: (cropped_image_4x, mean_SR, std_heatmap)
    """
    if image is None:
        return None, None, None

    # A) Random crop 200x200
    transform_crop = transforms.Compose([
        transforms.RandomCrop((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0,0,0), (1,1,1))
    ])
    lr_tensor = transform_crop(image)  # shape: (3, 200, 200)

    # Convert the cropped tensor to a PIL image
    cropped_pil = transforms.ToPILImage()(lr_tensor.clone().clamp_(0,1))

    # B) Upscale the cropped patch 4× using bicubic
    w, h = cropped_pil.size
    cropped_pil_4x = cropped_pil.resize((w*4, h*4), Image.BICUBIC)

    # Move the cropped tensor to device for SR
    lr_tensor = lr_tensor.unsqueeze(0).to(device)

    # C) Monte Carlo Dropout: multiple passes
    sr_passes = []
    for _ in range(mc_passes):
        gen2.eval()               # keep BN in eval mode
        enable_dropout(gen2)      # re-enable dropout layers
        with torch.no_grad():
            sr_out = gen2(lr_tensor)  
        sr_passes.append(sr_out)

    # Stack across passes -> (mc_passes, 1, 3, H, W)
    stacked = torch.stack(sr_passes, dim=0)

    # Mean & std across 'mc_passes' dimension
    mean_batch = torch.mean(stacked, dim=0) 
    std_batch  = torch.std(stacked, dim=0)

    # Convert mean SR to PIL
    mean_batch = mean_batch.squeeze(0).clamp_(0,1)  # shape (3, H, W)
    mean_pil = transforms.ToPILImage()(mean_batch.cpu())

    # D) Build a STD heatmap (collapsing across channels)
    std_map = torch.mean(std_batch, dim=1)  # shape: (H, W)
    s_min, s_max = std_map.min(), std_map.max()
    if (s_max - s_min) < 1e-8:
        std_norm = std_map.clone()
    else:
        std_norm = (std_map - s_min) / (s_max - s_min)

    # Convert std map to a color image via matplotlib's 'jet' colormap
    std_map_np = std_norm.squeeze().cpu().numpy()
    colored_std = plt.cm.jet(std_map_np)  # shape: (H, W, 4)
    colored_std = (colored_std[..., :3] * 255).astype(np.uint8)
    stdmap_pil = Image.fromarray(colored_std)

    # Return the 4× upscaled crop, the mean SR output, and the STD heatmap
    return cropped_pil_4x, mean_pil, stdmap_pil

# Create the Blocks layout
with gr.Blocks() as demo:
    gr.Markdown("# Uncertainty Estimation for Super Resolution using ESRGAN")
    
    # Create a row for the examples and the main interface components
    with gr.Row():
        # Create an examples component that only takes the image input.
        # Note: Each example is a list containing just the image path.
        examples = gr.Examples(
            examples=[["example1.jpg"], ["example2.jpg"],["example3.jpg"]],
            inputs=[gr.Image(type="pil", label="Upload an image")]
        )
    

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload an image")
            slider_input = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="MC Dropout Passes")
            submit_btn = gr.Button("Submit")
        with gr.Column():
            output1 = gr.Image(type="pil", label="1) Random Crop 4x Upscaled using bicubic interpolation")
            output2 = gr.Image(type="pil", label="2) Super-Resolved (Mean)")
            output3 = gr.Image(type="pil", label="3) STD Heatmap")
    
    submit_btn.click(
        fn=mcd_superres_crop,
        inputs=[image_input, slider_input],
        outputs=[output1, output2, output3]
    )

if __name__ == "__main__":
    demo.launch()
