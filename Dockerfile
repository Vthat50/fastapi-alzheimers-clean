# syntax=docker/dockerfile:1

FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Install system packages
RUN apt-get update && \
    apt-get install -y unzip curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code into container
COPY . .

# Set environment variable for FastAPI port
ENV PORT=8000
EXPOSE $PORT

# Start FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

