#!/bin/bash

# Sistem bağımlılıklarını yükle
apt-get update
apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# Python bağımlılıklarını yükle
pip install --no-cache-dir -r requirements.txt

# OpenCV'nin doğru yüklendiğini kontrol et
python -c "import cv2; print('OpenCV sürümü:', cv2.__version__)"

echo "Tüm bağımlılıklar başarıyla yüklendi!" 