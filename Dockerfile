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
    pip install awscli boto3

# Copy the rest of the app
COPY . .

# Create model directory to be populated at runtime
RUN mkdir -p roberta_final_checkpoint

# Expose port
ENV PORT=8000
EXPOSE $PORT

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
