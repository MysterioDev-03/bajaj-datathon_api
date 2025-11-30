# -----------------------------------------
# Base Image
# -----------------------------------------
FROM python:3.10-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------
# System Dependencies (Tesseract + Poppler)
# -----------------------------------------
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    wget \
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

# -----------------------------------------
# Install Python Dependencies
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
