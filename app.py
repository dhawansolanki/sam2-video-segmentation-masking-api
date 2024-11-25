from flask import Flask, request, jsonify
import os
import subprocess
import requests
import uuid
import torch
import numpy as np
import cv2
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor

# Initialize Flask app
app = Flask(__name__)

# Set up CUDA or MPS device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model
sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# Function to download file
def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
    else:
        raise Exception(f"Failed to download video. HTTP Status code: {response.status_code}")

# Function to process video frames
def process_video(filename):
    command = [
        "ffmpeg",
        "-i", filename,
        "-q:v", "2",  # Set the quality of the images
        "-start_number", "0",  # Start numbering from 0
        "magicroll/%03d.jpg"  # Save images as 000.jpg, 001.jpg, etc.
    ]
    subprocess.run(command, check=True)

# Flask route to process the video
@app.route('/process_video', methods=['POST'])
def process_video_api():
    try:
        # Get URL from request data
        url = request.json['url']
        filename = f"{uuid.uuid4().hex}.mp4"
        download_file(url, filename)

        # Process the downloaded video
        process_video(filename)

        # Return success response
        return jsonify({"status": "success", "message": "Video processed successfully!"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
