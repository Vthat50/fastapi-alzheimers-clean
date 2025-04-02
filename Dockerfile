# Use a Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies & CA certs
RUN apt-get update && apt-get install -y \
    curl ca-certificates gcc git libc-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install certifi manually first to ensure trusted cert store
RUN python -m ensurepip --upgrade && \
    python -m pip install --upgrade pip setuptools wheel certifi

# Set environment to use certifi’s bundled CA
ENV SSL_CERT_FILE=/usr/local/lib/python3.9/site-packages/certifi/cacert.pem

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

