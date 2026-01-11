#!/usr/bin/env bash
set -e

echo "🔧 Upgrading pip toolchain"
python -m pip install --upgrade pip setuptools wheel

echo "📦 Installing dependencies"
pip install -r requirements.txt

echo "🧠 Verifying numpy & faiss"
python - << 'EOF'
import numpy, faiss
print("numpy:", numpy.__version__)
print("faiss:", faiss.__version__)
EOF

echo "📁 Preparing folders"
mkdir -p logs
mkdir -p data

echo "🚀 Running index builder"
python build_index.py

echo "✅ Build completed successfully"
