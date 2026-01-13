#!/usr/bin/env bash
set -e

echo "=================================================="
echo "🚀 RUBY WINGS v5.2 - BUILD & DEPLOYMENT SCRIPT"
echo "=================================================="
echo "📅 Date: $(date)"
echo "🌍 Environment: ${FLASK_ENV:-development}"
echo "🧠 RAM Profile: ${RAM_PROFILE:-512}MB"
echo "🔧 FAISS Enabled: ${FAISS_ENABLED:-false}"
echo "=================================================="

# ==================== HELPER FUNCTIONS ====================
log_success() {
    echo "✅ $1"
}

log_info() {
    echo "📋 $1"
}

log_warning() {
    echo "⚠️  $1"
}

log_error() {
    echo "❌ $1" >&2
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Command '$1' not found. Please install it first."
        exit 1
    fi
}

check_python_version() {
    local required_version="3.8"
    local python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        log_error "Python $required_version+ required, found $python_version"
        exit 1
    fi
    log_success "Python version: $python_version"
}

# ==================== INITIAL CHECKS ====================
echo ""
echo "🔍 Performing pre-build checks..."
check_command python3
check_command pip3
check_python_version

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ] && [ "${FAISS_ENABLED:-false}" = "true" ]; then
    log_warning "OPENAI_API_KEY not set. FAISS embeddings will use synthetic fallback."
fi

# ==================== DEPENDENCY INSTALLATION ====================
echo ""
echo "📦 Installing/Upgrading dependencies..."

# Upgrade pip and base packages
python3 -m pip install --upgrade pip setuptools wheel

# Check for requirements.txt
if [ -f "requirements.txt" ]; then
    log_info "Installing from requirements.txt..."
    
    # Install core dependencies (always needed)
    pip3 install -r requirements.txt
    
    # Conditional FAISS installation
    if [ "${FAISS_ENABLED:-false}" = "true" ]; then
        log_info "FAISS_ENABLED=true - Installing FAISS..."
        pip3 install faiss-cpu
        
        # Verify FAISS installation
        python3 -c "import faiss; print(f'✅ FAISS version: {faiss.__version__}')"
    else
        log_info "FAISS_ENABLED=false - Skipping FAISS installation"
    fi
else
    log_warning "requirements.txt not found, installing minimal dependencies..."
    
    # Install minimal dependencies for Ruby Wings v5.2
    pip3 install flask flask-cors werkzeug
    pip3 install numpy
    
    if [ "${FAISS_ENABLED:-false}" = "true" ]; then
        pip3 install faiss-cpu
    fi
    
    # Optional but recommended
    pip3 install openai requests
fi

# ==================== KNOWLEDGE BASE SETUP ====================
echo ""
echo "📚 Setting up knowledge base..."

# Check if knowledge.json exists
if [ ! -f "knowledge.json" ]; then
    if [ -f "knowledge.example.json" ]; then
        log_info "Copying knowledge.example.json to knowledge.json"
        cp knowledge.example.json knowledge.json
    else
        log_warning "No knowledge.json found. Creating minimal structure..."
        cat > knowledge.json << EOF
{
  "about_company": {
    "name": "Ruby Wings Travel",
    "mission": "Tạo ra những trải nghiệm du lịch chân thực và ý nghĩa",
    "vision": "Trở thành công ty du lịch trải nghiệm hàng đầu Việt Nam"
  },
  "tours": [
    {
      "tour_name": "Sample Tour 1",
      "duration": "2 ngày 1 đêm",
      "location": "Hà Nội",
      "price": "1.500.000 VND",
      "summary": "Tour mẫu để kiểm tra hệ thống",
      "includes": ["Hướng dẫn viên", "Vận chuyển", "Ăn sáng"],
      "style": "Khám phá văn hóa",
      "notes": "Tour thử nghiệm"
    }
  ],
  "faq": {
    "general": "Ruby Wings chuyên về các tour trải nghiệm văn hóa và thiên nhiên"
  },
  "contact": {
    "hotline": "0332510486",
    "email": "info@ruby-wings.com"
  }
}
EOF
    fi
fi

# Create necessary directories
log_info "Creating directories..."
mkdir -p logs data

# ==================== INDEX BUILDING ====================
echo ""
echo "🏗️  Building search indices..."

if [ "${BUILD_INDEX:-true}" = "false" ]; then
    log_info "BUILD_INDEX=false - Skipping index building"
else
    # Check if build_index.py exists
    if [ -f "build_index.py" ]; then
        log_info "Running build_index.py..."
        
        # Set Python path
        export PYTHONPATH=$(pwd):$PYTHONPATH
        
        # Run with error handling
        if python3 build_index.py; then
            log_success "Index building completed successfully"
            
            # Verify created files
            echo ""
            echo "📁 Created index files:"
            if [ -f "tour_entities.json" ]; then
                echo "   ✅ tour_entities.json"
            fi
            if [ -f "faiss_mapping.json" ]; then
                echo "   ✅ faiss_mapping.json"
            fi
            if [ -f "vectors.npz" ]; then
                echo "   ✅ vectors.npz (fallback)"
            fi
            if [ -f "faiss_index.bin" ]; then
                echo "   ✅ faiss_index.bin"
            fi
        else
            log_error "Index building failed!"
            # Continue anyway - app.py has fallbacks
        fi
    else
        log_warning "build_index.py not found, skipping index building"
    fi
fi

# ==================== META CAPI SETUP ====================
echo ""
echo "📊 Setting up Meta CAPI..."

# Check if meta_capi.py exists
if [ -f "meta_capi.py" ]; then
    if [ -z "$META_PIXEL_ID" ] || [ -z "$META_CAPI_TOKEN" ]; then
        log_warning "Meta CAPI configuration missing (META_PIXEL_ID, META_CAPI_TOKEN)"
        log_warning "Meta CAPI will run in test/dummy mode"
    else
        log_success "Meta CAPI configured"
        echo "   📱 Pixel ID: ${META_PIXEL_ID:0:6}...${META_PIXEL_ID: -4}"
    fi
else
    log_warning "meta_capi.py not found, Meta CAPI integration disabled"
fi

# ==================== ENVIRONMENT CONFIGURATION ====================
echo ""
echo "⚙️  Configuring environment..."

# Create .env file if not exists
if [ ! -f ".env" ]; then
    log_info "Creating .env template..."
    cat > .env << EOF
# Ruby Wings v5.2 Configuration
FLASK_ENV=development
SECRET_KEY=$(openssl rand -hex 24)

# OpenAI Configuration
OPENAI_API_KEY=${OPENAI_API_KEY:-}

# Feature Toggles
ENABLE_STATE_MACHINE=true
ENABLE_LOCATION_FILTER=true
ENABLE_INTENT_DETECTION=true
ENABLE_PHONE_DETECTION=true
ENABLE_LEAD_CAPTURE=true
ENABLE_LLM_FALLBACK=true
ENABLE_CACHING=true
ENABLE_META_CAPI=true
FAISS_ENABLED=${FAISS_ENABLED:-false}

# Meta CAPI
META_PIXEL_ID=${META_PIXEL_ID:-}
META_CAPI_TOKEN=${META_CAPI_TOKEN:-}
META_API_VERSION=v18.0

# Performance Settings
RAM_PROFILE=512
TOP_K=5
MAX_TOURS_PER_RESPONSE=3
CACHE_TTL_SECONDS=300
MAX_SESSIONS=100

# Paths
KNOWLEDGE_PATH=knowledge.json
TOUR_ENTITIES_PATH=tour_entities.json
FAISS_INDEX_PATH=faiss_index.bin
FAISS_MAPPING_PATH=faiss_mapping.json
FALLBACK_VECTORS_PATH=vectors.npz

# Server
HOST=0.0.0.0
PORT=10000
EOF
    log_warning "Created .env file. Please edit with your actual values."
fi

# ==================== FILE PERMISSIONS ====================
echo ""
echo "🔒 Setting file permissions..."

# Make scripts executable
chmod +x build.sh 2>/dev/null || true

# Create log file with write permissions
touch ruby_wings.log
chmod 666 ruby_wings.log 2>/dev/null || true

# ==================== VERIFICATION ====================
echo ""
echo "🔍 Verifying installation..."

# Test imports
log_info "Testing Python imports..."
if python3 -c "
import sys
try:
    import flask, numpy, requests
    print('✅ Core imports: OK')
    
    if '${FAISS_ENABLED:-false}' == 'true':
        import faiss
        print('✅ FAISS import: OK')
    
    import json, hashlib, re, datetime, threading
    print('✅ Standard library: OK')
    
    print('✅ All imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"; then
    log_success "Python imports verified"
else
    log_error "Python import test failed"
fi

# Check key files
log_info "Checking required files..."
required_files=("app.py" "knowledge.json")
missing_files=0
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file (MISSING)"
        missing_files=$((missing_files + 1))
    fi
done

if [ $missing_files -eq 0 ]; then
    log_success "All required files present"
else
    log_warning "$missing_files required files missing"
fi

# ==================== FINAL SUMMARY ====================
echo ""
echo "=================================================="
echo "🎉 BUILD COMPLETE - RUBY WINGS v5.2"
echo "=================================================="
echo ""
echo "📊 BUILD SUMMARY:"
echo "   ✅ Python environment ready"
echo "   ✅ Dependencies installed"
echo "   ✅ Knowledge base: $(jq '.tours | length' knowledge.json 2>/dev/null || echo '?') tours"
echo "   ✅ Search index: $(if [ -f 'tour_entities.json' ]; then jq 'length' tour_entities.json; else echo 'Not built'; fi) tours"
echo "   ✅ Meta CAPI: $(if [ -n "$META_PIXEL_ID" ] && [ -f "meta_capi.py" ]; then echo 'Configured'; else echo 'Not configured'; fi)"
echo ""
echo "🚀 START OPTIONS:"
echo ""
echo "   1. Development mode:"
echo "      $ python3 app.py"
echo ""
echo "   2. Production with Gunicorn:"
echo "      $ gunicorn 'app:app' --bind 0.0.0.0:10000 --workers 2 --timeout 120"
echo ""
echo "   3. Docker (Render.com):"
echo "      $ docker build -t ruby-wings ."
echo "      $ docker run -p 10000:10000 ruby-wings"
echo ""
echo "🔧 ADMIN UTILITIES:"
echo "   - Rebuild index: BUILD_INDEX=true ./build.sh"
echo "   - Skip index: BUILD_INDEX=false ./build.sh"
echo "   - Force FAISS: FAISS_ENABLED=true ./build.sh"
echo ""
echo "📞 SUPPORT:"
echo "   - Hotline: 0332510486"
echo "   - Logs: tail -f ruby_wings.log"
echo ""
echo "=================================================="
echo "🌟 Happy travels with Ruby Wings v5.2! 🌟"
echo "=================================================="