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

if [ "$FAISS_ENABLED" = "true" ]; then
  echo "🚀 FAISS_ENABLED=true → Running index builder"
  python build_index.py
else
  echo "⏭️  FAISS_ENABLED=false → Skipping build_index"
fi

echo "✅ Build completed successfully"
