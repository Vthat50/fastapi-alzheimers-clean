FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl gcc build-essential ca-certificates \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && apt-get clean

# Upgrade pip and ensure certifi is installed
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel certifi \
    && pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



