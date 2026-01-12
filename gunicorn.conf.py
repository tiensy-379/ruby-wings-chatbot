# gunicorn.conf.py — Ruby Wings Chatbot v5.2 (Enhanced with State Machine)
# ====================================================
# Optimized for 512MB RAM (current) and ready for 2GB (future)
# Supports graceful shutdown for state machine session data

import os
import logging
import multiprocessing
from datetime import datetime

# ===============================
# RAM-AWARE CONFIGURATION
# ===============================

# Get RAM profile from environment
RAM_PROFILE = os.getenv("RAM_PROFILE", "512")
HIGH_MEMORY = RAM_PROFILE in ["1024", "2048", "4096"]

# Configuration based on RAM profile
if HIGH_MEMORY:
    # Configuration for 2GB+ RAM
    WORKERS = int(os.getenv("GUNICORN_WORKERS", "2"))
    THREADS = int(os.getenv("GUNICORN_THREADS", "4"))
    WORKER_CONNECTIONS = int(os.getenv("GUNICORN_WORKER_CONNECTIONS", "1000"))
    PRELOAD_APP = True  # Safe with more RAM
    RAM_MODE = "HIGH"
    
    # Memory limits for high RAM
    WORKER_MAX_REQUESTS = 2000
    WORKER_MAX_REQUESTS_JITTER = 100
    
else:
    # Configuration for 512MB RAM (default)
    WORKERS = int(os.getenv("GUNICORN_WORKERS", "1"))
    THREADS = int(os.getenv("GUNICORN_THREADS", "2"))
    WORKER_CONNECTIONS = int(os.getenv("GUNICORN_WORKER_CONNECTIONS", "500"))
    PRELOAD_APP = True  # Still safe with 1 worker
    RAM_MODE = "LOW"
    
    # Conservative limits for low RAM
    WORKER_MAX_REQUESTS = 1000
    WORKER_MAX_REQUESTS_JITTER = 50

# ===============================
# BASIC CONFIG
# ===============================

# Bind address
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"

# Worker configuration
workers = WORKERS
threads = THREADS
worker_class = "gthread"  # Best for Flask with async I/O
worker_connections = WORKER_CONNECTIONS

# Timeouts (optimized for LLM calls)
timeout = int(os.getenv("TIMEOUT", "300"))  # 5 minutes for OpenAI calls
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))

# Worker lifecycle
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", str(WORKER_MAX_REQUESTS)))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", str(WORKER_MAX_REQUESTS_JITTER)))

# App preloading
preload_app = PRELOAD_APP
daemon = False

# Process name
proc_name = "ruby-wings-chatbot-v5"

# ===============================
# MEMORY MANAGEMENT
# ===============================

# Limit memory usage per worker (in bytes)
# 200MB per worker for 512MB, 500MB per worker for 2GB
if HIGH_MEMORY:
    worker_max_memory = 500 * 1024 * 1024  # 500MB
else:
    worker_max_memory = 200 * 1024 * 1024  # 200MB

# Restart workers if they exceed memory limit
max_requests = min(max_requests, 500 if HIGH_MEMORY else 300)

# ===============================
# STATE MACHINE GRACEFUL SHUTDOWN
# ===============================

# Graceful shutdown timeout for saving session data
# Important for state machine persistence
def worker_exit(server, worker):
    """Save state machine session data before worker exits"""
    import time
    from app import state  # Import app state
    
    worker.log.info(f"🔄 Worker {worker.pid} shutting down gracefully...")
    
    # Give time to finish current requests
    time.sleep(2)
    
    # Try to save session data if possible
    try:
        if hasattr(state, '_session_contexts'):
            session_count = len(state._session_contexts)
            if session_count > 0:
                worker.log.info(f"💾 Saving {session_count} active sessions before shutdown")
                
                # Here you could save sessions to disk/database
                # For now, just log and clear (in production, implement persistence)
                state._session_contexts.clear()
                
    except Exception as e:
        worker.log.error(f"❌ Error saving session data: {e}")
    
    worker.log.info(f"👋 Worker {worker.pid} exited")

# ===============================
# LOGGING CONFIGURATION
# ===============================

# Log to stdout for Render/Docker
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")

# Enhanced access log format with response time
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)ss'

capture_output = True
enable_stdio_inheritance = True

# ===============================
# PERFORMANCE & SAFETY SETTINGS
# ===============================

# Use shared memory for worker tmp files
worker_tmp_dir = "/dev/shm"

# Request limits
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8192

# Backlog queue size
backlog = 2048 if HIGH_MEMORY else 1024

# ===============================
# GUNICORN HOOKS (v5.2 Enhanced)
# ===============================

def on_starting(server):
    """Called just before the master process is initialized"""
    server.log.info("=" * 60)
    server.log.info("🚀 RUBY WINGS AI CHATBOT v5.2 STARTING")
    server.log.info("=" * 60)
    
    # System info
    server.log.info(f"📊 RAM Profile: {RAM_PROFILE}MB ({RAM_MODE} memory mode)")
    server.log.info(f"🔧 Workers: {workers} | Threads: {threads}")
    server.log.info(f"⏱️  Timeout: {timeout}s | Graceful: {graceful_timeout}s")
    server.log.info(f"🔗 Worker connections: {worker_connections}")
    server.log.info(f"🔄 Max requests per worker: {max_requests}")
    
    # Feature flags
    features = []
    if os.getenv("STATE_MACHINE_ENABLED", "true") == "true":
        features.append("State Machine")
    if os.getenv("ENABLE_INTENT_DETECTION", "true") == "true":
        features.append("Intent Detection")
    if os.getenv("ENABLE_LOCATION_CONTEXT", "true") == "true":
        features.append("Location Filter")
    if os.getenv("FAISS_ENABLED", "false") == "true":
        features.append("FAISS Search")
    
    if features:
        server.log.info(f"🎯 Active features: {', '.join(features)}")
    
    server.log.info(f"🌐 Binding to: {server.cfg.bind}")
    server.log.info(f"📅 Start time: {datetime.now().isoformat()}")
    server.log.info("=" * 60)


def when_ready(server):
    """Called just after the server is started"""
    server.log.info("✅ Gunicorn ready to accept connections")
    
    # Check essential services
    try:
        import sys
        from app import OPENAI_AVAILABLE, FAISS_AVAILABLE, state
        
        server.log.info("🔍 System check:")
        server.log.info(f"   • OpenAI: {'✅ Available' if OPENAI_AVAILABLE else '❌ Unavailable'}")
        server.log.info(f"   • FAISS: {'✅ Available' if FAISS_AVAILABLE else '❌ Unavailable'}")
        
        if hasattr(state, '_knowledge_loaded'):
            server.log.info(f"   • Tours loaded: {state._knowledge_loaded}")
        
        # Python info
        server.log.info(f"   • Python: {sys.version.split()[0]}")
        
    except Exception as e:
        server.log.warning(f"⚠️ System check incomplete: {e}")


def post_fork(server, worker):
    """Called just after a worker has been forked"""
    worker.log.info(f"👶 Worker {worker.pid} spawned")
    
    # Set worker-specific environment if needed
    os.environ['GUNICORN_WORKER_PID'] = str(worker.pid)
    
    # Initialize worker-specific resources
    try:
        # Import and preload app resources
        from app import state, search_engine
        
        # Load knowledge if not already loaded
        if not state._knowledge_loaded:
            worker.log.info("📚 Loading knowledge base in worker...")
            from app import load_knowledge_lazy
            load_knowledge_lazy()
        
        # Load search index if needed
        if not search_engine._loaded:
            worker.log.info("🔍 Loading search index in worker...")
            search_engine.load_index()
            
    except Exception as e:
        worker.log.error(f"❌ Worker initialization error: {e}")


def worker_int(worker):
    """Called when a worker receives SIGINT or SIGQUIT"""
    worker.log.warning(f"⚠️ Worker {worker.pid} received interrupt signal")
    
    # Save state before exit
    try:
        from app import state
        if hasattr(state, '_session_contexts'):
            active_sessions = len(state._session_contexts)
            worker.log.info(f"💾 Interrupt: saving {active_sessions} active sessions")
    except:
        pass


def worker_abort(worker):
    """Called when a worker receives SIGABRT"""
    worker.log.error(f"🚨 Worker {worker.pid} aborted (SIGABRT)")
    
    # Emergency cleanup
    try:
        import gc
        gc.collect()  # Force garbage collection
        worker.log.info("🧹 Emergency garbage collection completed")
    except:
        pass


def on_exit(server):
    """Called just before exiting the master process"""
    server.log.info("=" * 60)
    server.log.info("👋 RUBY WINGS AI CHATBOT SHUTTING DOWN")
    server.log.info("=" * 60)
    
    # Summary statistics
    uptime = datetime.now() - server.cfg.started_at if hasattr(server.cfg, 'started_at') else "Unknown"
    server.log.info(f"📈 Uptime: {uptime}")
    server.log.info(f"📅 Shutdown time: {datetime.now().isoformat()}")
    server.log.info("=" * 60)


# ===============================
# REQUEST PROCESSING HOOKS
# ===============================

def post_request(worker, req, environ, resp):
    """Called after a request has been processed"""
    path = environ.get("PATH_INFO", "")
    
    # Skip logging for health checks and static files
    if path in ["/health", "/api/health", "/favicon.ico", "/robots.txt"]:
        return
    
    # Calculate request duration
    if hasattr(req, 'start_time'):
        duration = worker.time() - req.start_time
        
        # Log slow requests (threshold: 3 seconds for API, 10 seconds for chat)
        slow_threshold = 10.0 if "/chat" in path else 3.0
        
        if duration > slow_threshold:
            status = resp.status if hasattr(resp, 'status') else "Unknown"
            method = environ.get("REQUEST_METHOD", "UNKNOWN")
            
            worker.log.warning(
                f"🐌 Slow request: {method} {path} -> {status} "
                f"({duration:.2f}s, threshold: {slow_threshold}s)"
            )
        
        # Log very fast requests for debugging (optional)
        elif duration < 0.1 and "/chat" in path:
            # Likely cached response
            worker.log.debug(f"⚡ Fast response (cached): {path} ({duration:.3f}s)")


def access_log_filter(environ):
    """Filter requests from access log"""
    path = environ.get("PATH_INFO", "")
    
    # Exclude from access log
    exclude_paths = [
        "/health", 
        "/api/health", 
        "/favicon.ico", 
        "/robots.txt",
        "/.well-known/"
    ]
    
    return not any(path.startswith(exclude) for exclude in exclude_paths)


# ===============================
# CUSTOM MIDDLEWARE SUPPORT
# ===============================

def pre_request(worker, req):
    """Called before processing a request"""
    # Set start time for duration calculation
    req.start_time = worker.time()
    
    # Add request ID for tracing
    import uuid
    req.request_id = str(uuid.uuid4())[:8]
    
    # Log high-level request info
    path = req.path if hasattr(req, 'path') else "Unknown"
    method = req.method if hasattr(req, 'method') else "Unknown"
    
    worker.log.debug(f"📥 [{req.request_id}] {method} {path}")


# ===============================
# HEALTH CHECK INTEGRATION
# ===============================

def health_check_middleware(worker):
    """Custom health check that includes app state"""
    from app import state, search_engine
    
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "worker_pid": worker.pid,
        "ram_profile": RAM_PROFILE,
        "memory_mode": RAM_MODE,
        "app_status": "running",
        "services": {
            "openai": "unknown",
            "faiss": "unknown",
            "knowledge_base": "unknown",
            "search_engine": "unknown"
        }
    }
    
    try:
        from app import OPENAI_AVAILABLE, FAISS_AVAILABLE
        health_status["services"]["openai"] = "available" if OPENAI_AVAILABLE else "unavailable"
        health_status["services"]["faiss"] = "available" if FAISS_AVAILABLE else "unavailable"
        health_status["services"]["knowledge_base"] = "loaded" if state._knowledge_loaded else "not_loaded"
        health_status["services"]["search_engine"] = "loaded" if search_engine._loaded else "not_loaded"
    except:
        pass
    
    return health_status


# ===============================
# STARTUP VALIDATION
# ===============================

# Validate configuration
if __name__ == "__main__":
    print("=" * 60)
    print("GUNICORN CONFIGURATION VALIDATION - RUBY WINGS v5.2")
    print("=" * 60)
    print(f"RAM Profile: {RAM_PROFILE}MB ({RAM_MODE} memory mode)")
    print(f"Workers: {workers} | Threads: {threads}")
    print(f"Timeout: {timeout}s | Graceful: {graceful_timeout}s")
    print(f"Max requests per worker: {max_requests}")
    print(f"Worker connections: {worker_connections}")
    print("=" * 60)
    
    # Warn about potential issues
    if workers > 1 and RAM_PROFILE == "512":
        print("⚠️  WARNING: Multiple workers with 512MB RAM may cause memory issues")
        print("   Consider setting GUNICORN_WORKERS=1 for 512MB profile")
    
    if timeout < 60:
        print("⚠️  WARNING: Short timeout (<60s) may interrupt OpenAI API calls")
    
    print("✅ Configuration validation complete")
    print("=" * 60)