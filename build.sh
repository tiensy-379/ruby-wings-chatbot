#!/usr/bin/env bash
# Ruby Wings AI Chatbot v5.2.1 - Build & Deployment Script
# Compatible with: Render.com, Docker, Local development
# Last Updated: 2025-01-17

set -e  # Exit on error

echo "=================================================="
echo "🚀 RUBY WINGS v5.2.1 - BUILD & DEPLOYMENT SCRIPT"
echo "=================================================="
echo "📅 Date: $(date)"
echo "🌍 Environment: ${FLASK_ENV:-production}"
echo "🧠 RAM Profile: ${RAM_PROFILE:-512}MB"
echo "🔧 FAISS Enabled: ${FAISS_ENABLED:-false}"
echo "🐳 Platform: ${RENDER:+Render}${DOCKER:+Docker}${RENDER:+}${DOCKER:+}${RENDER}${DOCKER}"
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
    local required_version="3.9"
    local python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        log_error "Python $required_version+ required, found $python_version"
        exit 1
    fi
    log_success "Python version: $python_version"
}

# ==================== PLATFORM DETECTION ====================
detect_platform() {
    if [ -n "$RENDER" ]; then
        PLATFORM="render"
        log_info "Platform: Render.com"
    elif [ -n "$DOCKER" ] || [ -f "/.dockerenv" ]; then
        PLATFORM="docker"
        log_info "Platform: Docker"
    else
        PLATFORM="local"
        log_info "Platform: Local development"
    fi
    export PLATFORM
}

# ==================== INITIAL CHECKS ====================
echo ""
echo "🔍 Performing pre-build checks..."

detect_platform
check_command python3
check_command pip3
check_python_version

# Check for required files
required_files=("app.py" "requirements.txt")
missing_critical=0

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        log_error "Critical file missing: $file"
        missing_critical=$((missing_critical + 1))
    fi
done

if [ $missing_critical -gt 0 ]; then
    log_error "Cannot proceed without critical files"
    exit 1
fi

log_success "All critical files present"

# Check for important environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    log_warning "OPENAI_API_KEY not set. Chatbot will use fallback responses."
fi

if [ -z "$SECRET_KEY" ]; then
    log_warning "SECRET_KEY not set. Generating random key..."
    export SECRET_KEY=$(openssl rand -hex 32 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(32))")
fi

# ==================== DEPENDENCY INSTALLATION ====================
echo ""
echo "📦 Installing/Upgrading dependencies..."

# Upgrade pip and base packages
log_info "Upgrading pip, setuptools, wheel..."
python3 -m pip install --upgrade pip setuptools wheel --quiet

# Install from requirements.txt
log_info "Installing from requirements.txt..."
pip3 install -r requirements.txt --quiet

log_success "Core dependencies installed"

# Conditional FAISS installation
if [ "${FAISS_ENABLED:-false}" = "true" ] && [ "${RAM_PROFILE:-512}" -ge "2048" ]; then
    log_info "FAISS_ENABLED=true & RAM>=2GB - Installing FAISS..."
    
    if pip3 install faiss-cpu==1.7.4 --quiet; then
        # Verify FAISS installation
        python3 -c "import faiss; print(f'✅ FAISS version: {faiss.__version__}')" 2>/dev/null || log_warning "FAISS installed but import failed"
        log_success "FAISS installed successfully"
    else
        log_warning "FAISS installation failed, will use numpy fallback"
        export FAISS_ENABLED=false
    fi
else
    log_info "FAISS_ENABLED=false or RAM<2GB - Using numpy fallback"
fi

# ==================== KNOWLEDGE BASE SETUP ====================
echo ""
echo "📚 Setting up knowledge base..."

# Check if knowledge.json exists
if [ ! -f "knowledge.json" ]; then
    log_warning "knowledge.json not found!"
    
    # Try to find example or backup
    if [ -f "knowledge.example.json" ]; then
        log_info "Copying knowledge.example.json to knowledge.json"
        cp knowledge.example.json knowledge.json
    elif [ -f "data/knowledge.json" ]; then
        log_info "Copying from data/knowledge.json"
        cp data/knowledge.json knowledge.json
    else
        log_warning "Creating minimal knowledge.json structure..."
        cat > knowledge.json << 'EOF'
{
  "about_company": {
    "overview": "Ruby Wings tổ chức du lịch trải nghiệm, retreat, thiền, hành trình chữa lành với lịch trình linh hoạt theo nhu cầu",
    "mission": "Ruby Wings mang sứ mệnh lan tỏa giá trị sống chuẩn mực – chân thành – có chiều sâu",
    "contact": {
      "hotline": "0332510486",
      "website": "www.rubywings.vn",
      "email": "info@rubywings.vn"
    }
  },
  "tours": [
    {
      "tour_name": "Sample Tour - Huế Heritage",
      "summary": "Khám phá di sản văn hóa Huế",
      "location": "Huế, Thừa Thiên Huế",
      "region": "Miền Trung",
      "duration": "2 ngày 1 đêm",
      "price": "1.500.000 - 2.000.000 VNĐ/người",
      "includes": ["Xe du lịch", "Khách sạn 3*", "Bữa ăn", "Hướng dẫn viên"],
      "style": "Văn hóa - Lịch sử",
      "target_audience": ["Gia đình", "Nhóm bạn"]
    }
  ],
  "faqs": [
    {
      "question": "Làm sao để đặt tour?",
      "answer": "Bạn có thể đặt tour bằng cách gọi hotline 0332510486 hoặc để lại thông tin liên hệ."
    }
  ],
  "regions": {
    "mien_trung": ["Huế", "Quảng Trị", "Quảng Bình", "Đà Nẵng"]
  }
}
EOF
        log_warning "Created minimal knowledge.json - Please update with real data!"
    fi
else
    # Validate knowledge.json
    if python3 -c "import json; json.load(open('knowledge.json'))" 2>/dev/null; then
        tour_count=$(python3 -c "import json; print(len(json.load(open('knowledge.json')).get('tours', [])))" 2>/dev/null || echo "?")
        log_success "knowledge.json valid - $tour_count tours loaded"
    else
        log_error "knowledge.json is invalid JSON!"
        exit 1
    fi
fi

# Create necessary directories
log_info "Creating directories..."
mkdir -p logs data 2>/dev/null || true

# Copy knowledge.json to data/ if needed
if [ -f "knowledge.json" ] && [ ! -f "data/knowledge.json" ]; then
    cp knowledge.json data/knowledge.json 2>/dev/null || true
fi

# ==================== INDEX BUILDING ====================
echo ""
echo "🗂️  Building search indices..."

if [ "${BUILD_INDEX:-auto}" = "false" ]; then
    log_info "BUILD_INDEX=false - Skipping index building"
elif [ -f "build_index.py" ]; then
    log_info "Running build_index.py..."
    
    # Set Python path
    export PYTHONPATH=$(pwd):$PYTHONPATH
    
    # Run with error handling
    if python3 build_index.py 2>&1 | tee build_index.log; then
        log_success "Index building completed successfully"
        
        # Verify created files
        echo ""
        echo "📁 Created index files:"
        [ -f "tour_entities.json" ] && echo "   ✅ tour_entities.json" || echo "   ⚠️  tour_entities.json (missing)"
        [ -f "faiss_mapping.json" ] && echo "   ✅ faiss_mapping.json" || echo "   ⚠️  faiss_mapping.json (missing)"
        [ -f "vectors.npz" ] && echo "   ✅ vectors.npz (fallback)" || echo "   ⚠️  vectors.npz (missing)"
        
        if [ "${FAISS_ENABLED:-false}" = "true" ]; then
            [ -f "faiss_index.bin" ] && echo "   ✅ faiss_index.bin" || echo "   ⚠️  faiss_index.bin (missing)"
        fi
    else
        log_warning "Index building failed - app.py will use text search fallback"
        # Don't exit - app.py has fallbacks
    fi
else
    log_warning "build_index.py not found - skipping index building"
    log_info "App will use text search fallback"
fi

# ==================== META CAPI SETUP ====================
echo ""
echo "📊 Setting up Meta CAPI..."

if [ -f "meta_capi.py" ]; then
    if [ -n "$META_PIXEL_ID" ] && [ -n "$META_CAPI_TOKEN" ]; then
        log_success "Meta CAPI configured"
        echo "   📱 Pixel ID: ${META_PIXEL_ID:0:8}...${META_PIXEL_ID: -4}"
        echo "   🔑 Token: ${META_CAPI_TOKEN:0:8}...${META_CAPI_TOKEN: -4}"
        
        # Test Meta CAPI
        if python3 -c "from meta_capi import check_meta_capi_health; check_meta_capi_health()" 2>/dev/null; then
            log_success "Meta CAPI module loaded successfully"
        else
            log_warning "Meta CAPI module has issues but will continue"
        fi
    else
        log_warning "Meta CAPI credentials missing (META_PIXEL_ID, META_CAPI_TOKEN)"
        log_warning "Meta CAPI features will be disabled"
    fi
else
    log_warning "meta_capi.py not found - Meta CAPI disabled"
fi

# ==================== GOOGLE SHEETS SETUP ====================
echo ""
echo "📊 Setting up Google Sheets..."

if [ "${ENABLE_GOOGLE_SHEETS:-true}" = "true" ]; then
    if [ -n "$GOOGLE_SERVICE_ACCOUNT_JSON" ] && [ -n "$GOOGLE_SHEET_ID" ]; then
        log_success "Google Sheets configured"
        echo "   📄 Sheet ID: ${GOOGLE_SHEET_ID:0:8}...${GOOGLE_SHEET_ID: -4}"
        
        # Test credentials
        if python3 -c "import json; json.loads('$GOOGLE_SERVICE_ACCOUNT_JSON')" 2>/dev/null; then
            log_success "Google credentials JSON is valid"
        else
            log_warning "Google credentials JSON may be invalid"
        fi
    else
        log_warning "Google Sheets credentials missing"
        log_warning "Leads will be saved to fallback storage only"
    fi
else
    log_info "Google Sheets disabled (ENABLE_GOOGLE_SHEETS=false)"
fi

# ==================== ENVIRONMENT CONFIGURATION ====================
echo ""
echo "⚙️  Configuring environment..."

# Show current configuration
echo "📋 Current Configuration:"
echo "   FLASK_ENV: ${FLASK_ENV:-production}"
echo "   RAM_PROFILE: ${RAM_PROFILE:-512}MB"
echo "   FAISS_ENABLED: ${FAISS_ENABLED:-false}"
echo "   TOP_K: ${TOP_K:-5}"
echo "   GUNICORN_WORKERS: ${GUNICORN_WORKERS:-1}"
echo "   GUNICORN_THREADS: ${GUNICORN_THREADS:-2}"
echo "   CORS_ORIGINS: ${CORS_ORIGINS:-*}"

# Validate CORS configuration
if [ "$CORS_ORIGINS" = "*" ] && [ "${FLASK_ENV:-production}" = "production" ]; then
    log_warning "CORS_ORIGINS=* in production - consider restricting"
fi

# Create .env file if local development and doesn't exist
if [ "$PLATFORM" = "local" ] && [ ! -f ".env" ]; then
    log_info "Creating .env template for local development..."
    cat > .env << EOF
# Ruby Wings v5.2.1 Local Configuration
# DO NOT COMMIT THIS FILE TO GIT!

FLASK_ENV=development
DEBUG=true
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# OpenAI
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1

# Meta CAPI
META_PIXEL_ID=your-pixel-id
META_CAPI_TOKEN=your-token-here
ENABLE_META_CAPI_LEAD=true
ENABLE_META_CAPI_CALL=true

# Google Sheets
ENABLE_GOOGLE_SHEETS=true
GOOGLE_SERVICE_ACCOUNT_JSON={}
GOOGLE_SHEET_ID=your-sheet-id
GOOGLE_SHEET_NAME=RBW_Lead_Raw_Inbox

# Performance
RAM_PROFILE=512
FAISS_ENABLED=false
TOP_K=5

# Server
HOST=0.0.0.0
PORT=10000

# CORS (for local dev)
CORS_ORIGINS=*
EOF
    log_warning "Created .env file - PLEASE UPDATE WITH REAL VALUES!"
fi

# ==================== FILE PERMISSIONS ====================
echo ""
echo "🔒 Setting file permissions..."

# Make scripts executable
chmod +x build.sh 2>/dev/null || true
chmod +x *.sh 2>/dev/null || true

# Ensure log directory is writable
mkdir -p logs 2>/dev/null || true
touch logs/ruby_wings.log 2>/dev/null || true
chmod 666 logs/ruby_wings.log 2>/dev/null || true

# Ensure data directory is writable
mkdir -p data 2>/dev/null || true
chmod 755 data 2>/dev/null || true

log_success "File permissions set"

# ==================== MODULE VERIFICATION ====================
echo ""
echo "🔍 Verifying installation..."

log_info "Testing Python imports..."
if python3 << 'PYTHON_TEST'
import sys
try:
    # Core imports
    import flask
    import flask_cors
    import werkzeug
    print("✅ Flask: OK")
    
    import numpy
    print("✅ Numpy: OK")
    
    import openai
    print("✅ OpenAI: OK")
    
    import requests
    print("✅ Requests: OK")
    
    # Optional imports
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        print("✅ Google Sheets: OK")
    except ImportError:
        print("⚠️  Google Sheets: Not installed (optional)")
    
    # Check FAISS if enabled
    import os
    if os.getenv('FAISS_ENABLED', 'false').lower() == 'true':
        import faiss
        print(f"✅ FAISS: OK (version {faiss.__version__})")
    
    # Standard library
    import json, hashlib, re, datetime, threading, time
    print("✅ Standard library: OK")
    
    print("\n✅ All critical imports successful!")
    sys.exit(0)
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    sys.exit(1)
PYTHON_TEST
then
    log_success "Python imports verified"
else
    log_error "Python import test failed - check dependencies"
    exit 1
fi

# Check custom modules
echo ""
log_info "Checking custom modules..."
optional_modules=("entities.py" "meta_capi.py" "response_guard.py" "gunicorn_conf.py")
for module in "${optional_modules[@]}"; do
    if [ -f "$module" ]; then
        echo "   ✅ $module"
    else
        echo "   ⚠️  $module (optional, using fallback)"
    fi
done

# ==================== HEALTH CHECK ====================
echo ""
echo "🏥 Running health checks..."

# Test app.py syntax
if python3 -m py_compile app.py 2>/dev/null; then
    log_success "app.py syntax valid"
else
    log_error "app.py has syntax errors!"
    exit 1
fi

# Test knowledge.json validity
if [ -f "knowledge.json" ]; then
    if python3 -c "import json; json.load(open('knowledge.json'))" 2>/dev/null; then
        log_success "knowledge.json valid"
    else
        log_error "knowledge.json is invalid!"
        exit 1
    fi
fi

# ==================== FINAL SUMMARY ====================
echo ""
echo "=================================================="
echo "🎉 BUILD COMPLETE - RUBY WINGS v5.2.1"
echo "=================================================="
echo ""
echo "📊 BUILD SUMMARY:"
echo "   ✅ Platform: $PLATFORM"
echo "   ✅ Python: $(python3 --version | cut -d' ' -f2)"
echo "   ✅ Dependencies: Installed"
echo "   ✅ Knowledge base: $(python3 -c "import json; print(len(json.load(open('knowledge.json')).get('tours', [])))" 2>/dev/null || echo '?') tours"
echo "   ✅ Search index: $([ -f 'tour_entities.json' ] && echo 'Built' || echo 'Text fallback')"
echo "   ✅ Meta CAPI: $([ -n "$META_PIXEL_ID" ] && [ -f "meta_capi.py" ] && echo 'Configured' || echo 'Disabled')"
echo "   ✅ Google Sheets: $([ -n "$GOOGLE_SHEET_ID" ] && echo 'Configured' || echo 'Fallback only')"
echo ""

# Show start commands based on platform
if [ "$PLATFORM" = "render" ]; then
    echo "🚀 RENDER DEPLOYMENT:"
    echo "   Gunicorn will start automatically"
    echo "   Command: gunicorn app:app --config gunicorn_conf.py"
    
elif [ "$PLATFORM" = "docker" ]; then
    echo "🐳 DOCKER DEPLOYMENT:"
    echo "   Container will start automatically"
    echo "   Health check: curl http://localhost:10000/health"
    
else
    echo "🚀 START OPTIONS:"
    echo ""
    echo "   1. Development mode:"
    echo "      $ python3 app.py"
    echo ""
    echo "   2. Production with Gunicorn:"
    echo "      $ gunicorn app:app --config gunicorn_conf.py"
    echo "      or"
    echo "      $ gunicorn app:app --bind 0.0.0.0:10000 --workers 1 --threads 2 --timeout 60"
    echo ""
    echo "   3. Docker:"
    echo "      $ docker-compose up -d"
    echo ""
fi

echo "🔧 ADMIN UTILITIES:"
echo "   - Rebuild index: BUILD_INDEX=true ./build.sh"
echo "   - Skip index: BUILD_INDEX=false ./build.sh"
echo "   - Force FAISS: FAISS_ENABLED=true RAM_PROFILE=2048 ./build.sh"
echo "   - Health check: curl http://localhost:10000/health"
echo "   - Test chat: curl -X POST http://localhost:10000/api/chat -H 'Content-Type: application/json' -d '{\"message\":\"Xin chào\"}'"
echo ""
echo "📝 LOGS:"
echo "   - Application: logs/ruby_wings.log"
echo "   - Build: build_index.log (if index was built)"
echo "   - Real-time: tail -f logs/ruby_wings.log"
echo ""
echo "📞 SUPPORT:"
echo "   - Hotline: 0332510486"
echo "   - Email: info@rubywings.vn"
echo "   - Website: www.rubywings.vn"
echo ""
echo "=================================================="
echo "🌟 Ready to serve! Ruby Wings v5.2.1 🌟"
echo "=================================================="

# Return success
exit 0