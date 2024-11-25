# Use the official NVIDIA PyTorch image with CUDA support
FROM nvcr.io/nvidia/pytorch:22.12-py3

# Set environment variables to use GPU and CUDA efficiently
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    git \
    && apt-get clean

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy your Flask app and necessary files to the container
COPY . /app

WORKDIR /app

# Expose the Flask app port
EXPOSE 5000

# Run the Flask app
CMD ["python3", "app.py"]
