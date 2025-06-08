FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system-level dependencies if needed
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Set environment variable to force non-interactive mode if needed
ENV PYTHONUNBUFFERED=1

# Start the RunPod handler
CMD ["python", "runpod_handler.py"]