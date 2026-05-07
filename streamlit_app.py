import streamlit as st
import os
import subprocess
from PIL import Image

st.set_page_config(page_title="AI Image Upscaler", layout="centered")
st.title("🚀 4x Image Upscaler")
st.write("Using Real-ESRGAN to enhance your photos.")

# 1. Setup Environment (Download model if missing)
MODEL_PATH = "experiments/pretrained_models/RealESRGAN_x4plus.pth"
if not os.path.exists(MODEL_PATH):
    st.info("Downloading AI model... please wait.")
    os.makedirs("experiments/pretrained_models", exist_ok=True)
    subprocess.run(["wget", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", "-P", "experiments/pretrained_models/"])

# 2. File Uploader
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Original Image", use_container_width=True)
    
    if st.button("Upscale 4x"):
        with st.spinner("Processing... This takes about 1-2 mins on free servers."):
            # Save upload
            with open("temp_input.png", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Run inference
            cmd = [
                "python", "inference_realesrgan.py",
                "-n", "RealESRGAN_x4plus",
                "-i", "temp_input.png",
                "-o", "results",
                "--outscale", "4",
                "--tile", "400"
            ]
            subprocess.run(cmd)
            
            # Output path
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
                st.error("Upscale failed. Check logs.")
