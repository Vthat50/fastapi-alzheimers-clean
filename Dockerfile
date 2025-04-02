# Use a more robust base image
FROM python:3.10-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl unzip gcc libc-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install certifi to fix SSL issues
RUN python -m pip install --upgrade pip setuptools wheel certifi

# Set environment to use certifi's CA bundle
ENV SSL_CERT_FILE=/usr/local/lib/python3.10/site-packages/certifi/cacert.pem

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY . .

# Expose port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


