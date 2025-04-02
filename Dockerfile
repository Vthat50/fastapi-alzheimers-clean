

# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies and update CA certs
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git curl unzip gcc libc-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip certifi && \
    pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port
EXPOSE 8000

# Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
