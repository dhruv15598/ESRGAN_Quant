from model import RRDBNet
import gradio as gr
from PIL import Image
import torch
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_weights(checkpoint_file, model):
    print("=> Loading weights")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model_state_dict = model.state_dict()
    state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                  k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
    model_state_dict.update(state_dict)
    model.load_state_dict(model_state_dict)
    print("Successfully loaded the pretrained model weights")
    return model


def superres(image):
    gen2 = RRDBNet(3, 3, 64, 32, 2, 0.2).to(device)
    gen2 = load_weights("gen200.pth.tar", gen2)
    gen2.eval()
    transform4 = transforms.Compose([
        transforms.RandomCrop((200,200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
    ])
    lr_image = transform4(image)
    with torch.no_grad():
        lr_image = lr_image.to(device)
        lr_image = lr_image.unsqueeze(0)
        sr_image2 = gen2(lr_image)
        sr_image2 = sr_image2.cpu().squeeze(0).numpy()
    return sr_image2


demo = gr.Interface(
    superres,
    gr.Image(type="pil"),
    "image",
    flagging_options=["blurry", "incorrect", "other"]
)

if __name__ == "__main__":
    demo.launch()
