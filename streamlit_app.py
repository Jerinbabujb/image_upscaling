import sys
import os
import subprocess

# --- 1. COMPATIBILITY PATCH ---
# This fixes the 'functional_tensor' error in the Cloud environment
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        import torchvision.transforms.functional as F
        sys.modules['torchvision.transforms.functional_tensor'] = F
    except ImportError:
        pass

import streamlit as st
from PIL import Image

st.set_page_config(page_title="AI Image Upscaler", layout="centered")
st.title("🚀 4x Image Upscaler")

# --- 2. AUTOMATIC SETUP ---
# Clone the logic if it's missing from the Streamlit container
if not os.path.exists("inference_realesrgan.py"):
    with st.spinner("Initializing AI Engine..."):
        subprocess.run(["git", "clone", "https://github.com/xinntao/Real-ESRGAN.git", "realsr_repo"])
        # Move files to root so the app can find them
        os.system("cp -r realsr_repo/* .")
        os.system("rm -rf realsr_repo")

# Ensure model weights are downloaded
MODEL_PATH = "experiments/pretrained_models/RealESRGAN_x4plus.pth"
if not os.path.exists(MODEL_PATH):
    os.makedirs("experiments/pretrained_models", exist_ok=True)
    subprocess.run(["wget", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", "-P", "experiments/pretrained_models/"])

# --- 3. UI LOGIC ---
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Original Image", use_container_width=True)
    
    if st.button("Upscale 4x"):
        with st.spinner("Processing... 1-2 minutes on CPU."):
            # Save upload
            with open("temp_input.png", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Create results folder
            if not os.path.exists("results"):
                os.makedirs("results")

            # Run inference (added --tile to save memory)
            cmd = [
                "python", "inference_realesrgan.py",
                "-n", "RealESRGAN_x4plus",
                "-i", "temp_input.png",
                "-o", "results",
                "--outscale", "4",
                "--tile", "400"
            ]
            subprocess.run(cmd)
            
            output_path = "results/temp_input_out.png"
            
            if os.path.exists(output_path):
                st.success("Done!")
                st.image(output_path, caption="Upscaled Image", use_container_width=True)
                
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="Download High-Res Image",
                        data=file,
                        file_name="upscaled_image.png",
                        mime="image/png"
                    )
            else:
                st.error("Upscale failed. The server might have run out of memory.")
