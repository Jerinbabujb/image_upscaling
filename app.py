import sys
import os
import types

# --- 1. CRITICAL COMPATIBILITY PATCH ---
# Fixes 'ModuleNotFoundError: No module named torchvision.transforms.functional_tensor'
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        from torchvision.transforms import functional as F
        # Create a dummy module to satisfy older library imports
        import types
        sys.modules['torchvision.transforms.functional_tensor'] = types.ModuleType('functional_tensor')
        sys.modules['torchvision.transforms.functional_tensor'].rgb_to_grayscale = F.rgb_to_grayscale
    except ImportError:
        pass

import torch
import gradio as gr
import cv2
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# --- 2. HARDWARE DETECTION ---
# Railway uses CPU by default. This ensures the app doesn't crash looking for CUDA.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Deployment Mode: Running on {device} ---")

# --- 3. MODEL INITIALIZATION ---
def load_upscaler():
    # Architecture settings for RealESRGAN_x4plus
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    model_path = os.path.join('experiments/pretrained_models', 'RealESRGAN_x4plus.pth')
    
    # Download weights if they don't exist in the Railway container
    if not os.path.exists(model_path):
        os.makedirs('experiments/pretrained_models', exist_ok=True)
        print("Downloading weights...")
        # Using -L to follow redirects from GitHub
        os.system(f'curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -o {model_path}')

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=200,         # Lowered to 200 for better CPU/RAM stability on Railway
        tile_pad=10,
        pre_pad=0,
        half=False,       # Must be False for CPU deployment
        device=device
    )
    return upsampler

# Load the model into memory
upsampler = load_upscaler()

# --- 4. PREDICTION LOGIC ---
def predict(image):
    if image is None:
        return None
    
    # Convert PIL Image to BGR for OpenCV
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    try:
        # outscale=4 matches our model
        output, _ = upsampler.enhance(img, outscale=4)
        
        # Convert back to RGB for Gradio
        res = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return Image.fromarray(res)
    except Exception as e:
        print(f"Error: {e}")
        return None

# --- 5. GRADIO UI ---
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Photo"),
    outputs=gr.Image(type="pil", label="4x Upscaled Result"),
    title="🚀 AI Image Upscaler",
    description="Powered by Real-ESRGAN. Processing takes 1-2 mins on CPU.",
    flagging_mode="never"
)

# --- 6. RAILWAY PORT BINDING ---
if __name__ == "__main__":
    # Railway passes the port as an environment variable
    server_port = int(os.environ.get("PORT", 7860))
    # server_name="0.0.0.0" is required for the cloud proxy to find the app
    demo.launch(server_name="0.0.0.0", server_port=server_port)
