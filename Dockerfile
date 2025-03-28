# syntax=docker/dockerfile:1

FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Install system packages
RUN apt-get update && \
    apt-get install -y unzip curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install awscli

# Copy the rest of the app
COPY . .

# Download and extract model from S3
RUN python3 -c "\
import os, zipfile\n\
os.system('aws s3 cp s3://fastapi-app-bucket-varsh/roberta_final_checkpoint.zip roberta_final_checkpoint.zip')\n\
assert zipfile.is_zipfile('roberta_final_checkpoint.zip'), 'Downloaded file is not a valid zip file'\n\
with zipfile.ZipFile('roberta_final_checkpoint.zip', 'r') as zip_ref:\n    zip_ref.extractall('roberta_final_checkpoint')\n"

# Expose port
ENV PORT=8000
EXPOSE $PORT

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
