# -----------------------------------------
# Base Image
# -----------------------------------------
FROM python:3.10-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------
# System Dependencies (Tesseract + Poppler + Build tools + Paddle deps)
# -----------------------------------------
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------
# Create App Directory
# -----------------------------------------
WORKDIR /app

# -----------------------------------------
# Copy Requirements
# -----------------------------------------
COPY requirements.txt /app/

# Upgrade pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# -----------------------------------------
# Install PaddlePaddle + PaddleOCR first
# -----------------------------------------
RUN pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu
RUN pip install paddleocr>=3.3.0

# -----------------------------------------
# Install the rest of the Python dependencies
# -----------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------
# Copy Application Code
# -----------------------------------------
COPY . /app/

# -----------------------------------------
# Expose & Run
# -----------------------------------------
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
