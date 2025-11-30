# ------------------------------
# Base Image
# ------------------------------
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------
# System dependencies
# ------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    wget \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------------------
# Working directory
# ------------------------------
WORKDIR /app

# ------------------------------
# Copy requirements
# ------------------------------
COPY requirements.txt /app/

# Upgrade pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# ------------------------------
# Install PaddlePaddle + PaddleOCR first
# ------------------------------
RUN pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu
RUN pip install paddleocr>=3.3.0

# ------------------------------
# Pre-download PaddleOCR official English models
# ------------------------------
RUN python -c "\
from paddleocr import PaddleOCR; \
PaddleOCR(lang='en', use_textline_orientation=True)"


# ------------------------------
# Install the rest of the dependencies
# ------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------
# Copy app code
# ------------------------------
COPY . /app/

# ------------------------------
# Expose port
# ------------------------------
EXPOSE 8000

# ------------------------------
# Use dynamic port for Render
# ------------------------------
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT