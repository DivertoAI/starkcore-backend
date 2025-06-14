FROM python:3.10-slim

# Avoid interactive prompts or broken locales
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for torch + diffusers
RUN apt-get update && apt-get install -y \
    git curl libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire app into the container
COPY . .

# Must use handler.py (or runpod_handler.py) that includes `start({"handler": ...})`
CMD ["python", "-u", "handler.py"]