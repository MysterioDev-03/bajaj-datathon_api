# ------------------------------
FROM python:3.10-slim
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu
RUN pip install paddleocr>=3.3.0
RUN pip install --no-cache-dir -r requirements.txt

# preload PaddleOCR
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='en', use_textline_orientation=True)"

COPY . .

EXPOSE 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000
