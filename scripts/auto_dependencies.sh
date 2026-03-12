#!/bin/bash

# pipreqs ile kullanılan kütüphaneleri tespit et
echo "pipreqs ile kullanılan kütüphaneleri tespit ediliyor..."
pipreqs --force .

# pip-tools ile bağımlılıkları derle
echo "pip-tools ile bağımlılıklar derleniyor..."
pip-compile requirements.in

# Bağımlılıkları yükle
echo "Bağımlılıklar yükleniyor..."
pip install -r requirements.txt

# OpenCV'nin doğru yüklendiğini kontrol et
echo "OpenCV kontrol ediliyor..."
python -c "import cv2; print('OpenCV sürümü:', cv2.__version__)"

echo "Tüm bağımlılıklar başarıyla yüklendi!" 