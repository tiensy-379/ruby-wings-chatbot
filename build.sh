#!/usr/bin/env bash
set -e

echo "🔧 Upgrading pip toolchain"
python -m pip install --upgrade pip setuptools wheel

echo "📦 Installing dependencies"
pip install -r requirements.txt

echo "🧠 Verifying numpy & faiss"
python - << 'EOF'
import numpy
try:
    import faiss
    print("faiss:", faiss.__version__)
except Exception as e:
    print("faiss not available:", e)
print("numpy:", numpy.__version__)
EOF

echo "📁 Preparing folders"
mkdir -p logs
mkdir -p data

echo "⏭️  Skipping index build on Render (using prebuilt indexes from repo)"

echo "✅ Build completed successfully"
