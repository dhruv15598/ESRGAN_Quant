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
    cropped_pil = F.to_pil_image(lr_tensor.clone().clamp_(0,1))

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
    mean_pil = F.to_pil_image(mean_batch.cpu())

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


demo = gr.Interface(
    fn=mcd_superres_crop,
    inputs=[gr.Image(type="pil", label="Upload an image"), gr.Slider(minimum=1, maximum=20, value=5 ,step=1, label="MC Dropout Passes")],
    outputs=[
        gr.Image(type="pil", label="1) Random Crop 4x Upscaled using bicubic interpolation"),
        gr.Image(type="pil", label="2) Super-Resolved (Mean)"),
        gr.Image(type="pil", label="3) STD Heatmap")
    ],
    title="Uncertainity Estimation for Super Resolution using ESRGAN.",
    description = """
                    This is the demo for our paper: <b>Uncertainity Estimation for Super Resolution using ESRGAN.</b><br/>
                    Authors: Dr. Matias Valdenegro Toro, Dr. Marco Zullich, & Maniraj Sai.<br/>
                    Presented at the 2025 VISAPP Conference.<br/><br/>
                    <b>Usage</b>: Upload an image (or use one of the examples below) and click "Submit."
                """
    ,
    examples=[
        ["example1.jpg", 5],
        ["example2.jpg", 10],
        ["example3.jpg", 15]
    ]
    ,
    
    article =           """
                        <h3>About this Demo</h3>
                        <p>
                        This demo showcases an enhanced ESRGAN approach for image super-resolution. First, we take a 256×256 crop from the uploaded image to reduce computational load. Then, we apply a 4x upscale using our ESRGAN model, which has been modified to incorporate dropout layers. Through multiple forward passes (Monte Carlo Dropout), the demo not only produces a high-resolution output but also estimates pixelwise uncertainty. By visualizing these uncertainties in a color-coded heatmap, you can see which regions of the image the model is less confident about—an important insight for understanding model performance and reliability.
                        </p>
                        <h4>Citation</h4>
                        <pre>
                        @inproceedings{your_paper_2024,
                        title={Uncertainity Estimation for Super Resolution using ESRGAN.},
                        author={Matias Valdenegro Toro, Dr. Marco Zullich and Maniraj Sai Adapa},
                        booktitle={VISAPP Conference 2025},
                        year={2025}
                        }
                        </pre>
                        """
)

if __name__ == "__main__":
    demo.launch()
