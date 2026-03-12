#!/bin/bash

# Belirtilen dosya veya dizindeki Python dosyalarını formatla
format_python_files() {
    local target="$1"
    echo "Formatting $target..."
    
    # Import düzenini düzelt
    isort "$target"
    
    # PEP8 stil düzeltmeleri
    autopep8 --in-place --aggressive --aggressive "$target"
    
    # Black ile formatla
    black "$target"
    
    # Pylint ile kontrol et
    pylint "$target"
}

# Eğer argüman verilmişse o dosya/dizini, verilmemişse src/ dizinini formatla
if [ $# -eq 0 ]; then
    format_python_files "src/"
else
    format_python_files "$1"
fi 