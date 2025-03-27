# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies and AWS CLI for S3 access
RUN apt-get update && apt-get install -y unzip curl && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install awscli

# Download and unzip model from S3
RUN curl "https://fastapi-app-bucket-varsh.s3.us-east-2.amazonaws.com/roberta_final_checkpoint.zip" -o model.zip && \
    unzip model.zip -d . && rm model.zip

# Copy everything else
COPY . .

# Expose port
EXPOSE 8000

# Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
