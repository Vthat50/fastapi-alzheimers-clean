FROM python:3.10-slim

WORKDIR /app

# Fix CA certificates
RUN apt-get update && apt-get install -y \
    curl gcc build-essential ca-certificates \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && update-ca-certificates \
    && apt-get clean

# Ensure pip and certifi are up to date
RUN pip install --upgrade pip setuptools wheel certifi

# Force certifi usage
ENV SSL_CERT_FILE=$(python -m certifi)

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


