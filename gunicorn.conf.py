# gunicorn.conf.py — Ruby Wings Chatbot v5.2 (Enhanced with State Machine)
# ====================================================
# Optimized for 512MB RAM (current) and ready for 2GB+ (future)
# Supports graceful shutdown for state machine session data
# ĐỒNG BỘ: .env.example.ini, app.py v5.2, entities.py v5.2

import os
import sys
import logging
import multiprocessing
from datetime import datetime

# ===============================
# RAM-AWARE CONFIGURATION
# ===============================

# Get RAM profile from environment (ĐỒNG BỘ VỚI .ENV)
RAM_PROFILE = os.getenv("RAM_PROFILE", "512")
IS_LOW_RAM = os.getenv("IS_LOW_RAM", "true").lower() == "true"
HIGH_MEMORY = RAM_PROFILE in ["1024", "2048", "4096"] and not IS_LOW_RAM

# Configuration based on RAM profile
if HIGH_MEMORY:
    # Configuration for 2GB+ RAM
    WORKERS = int(os.getenv("WORKERS", "2"))
    THREADS = int(os.getenv("THREADS", "4"))
    WORKER_CONNECTIONS = 1000
    PRELOAD_APP = True  # Safe with more RAM
    RAM_MODE = "HIGH"
    
    # Memory limits for high RAM
    WORKER_MAX_REQUESTS = 2000
    WORKER_MAX_REQUESTS_JITTER = 100
    
else:
    # Configuration for 512MB RAM (default)
    WORKERS = int(os.getenv("WORKERS", "1"))
    THREADS = int(os.getenv("THREADS", "2"))
    WORKER_CONNECTIONS = 500
    PRELOAD_APP = True  # Still safe with 1 worker
    RAM_MODE = "LOW"
    
    # Conservative limits for low RAM
    WORKER_MAX_REQUESTS = 1000
    WORKER_MAX_REQUESTS_JITTER = 50

# ===============================
# BASIC CONFIG (ĐỒNG BỘ VỚI .ENV)
# ===============================

# Bind address
HOST = os.getenv("HOST", "0.0.0.0")
PORT = os.getenv("PORT", "10000")
bind = f"{HOST}:{PORT}"

# Worker configuration
workers = WORKERS
threads = THREADS
worker_class = "gthread"  # Best for Flask with async I/O
worker_connections = WORKER_CONNECTIONS

# Timeouts (optimized for LLM calls) - ĐỒNG BỘ VỚI .ENV
timeout = int(os.getenv("TIMEOUT", "30"))  # Default 30s from .env
graceful_timeout = 30  # Fixed for graceful shutdown
keepalive = 5

# Worker lifecycle
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", str(WORKER_MAX_REQUESTS)))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", str(WORKER_MAX_REQUESTS_JITTER)))

# App preloading
preload_app = PRELOAD_APP
daemon = False

# Process name
proc_name = "ruby-wings-chatbot-v5.2"

# ===============================
# MEMORY MANAGEMENT
# ===============================

# Limit memory usage per worker (in bytes)
if HIGH_MEMORY:
    worker_max_memory = 500 * 1024 * 1024  # 500MB
else:
    worker_max_memory = 200 * 1024 * 1024  # 200MB

# ===============================
# STATE MACHINE GRACEFUL SHUTDOWN
# ===============================

def worker_exit(server, worker):
    """
    Save state machine session data before worker exits.
    Important for state machine persistence.
    """
    worker.log.info(f"🔄 Worker {worker.pid} shutting down gracefully...")
    
    import time
    time.sleep(1)  # Give time to finish current requests
    
    # Try to save session data if possible
    try:
        # Import app components safely
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Try to access app state
        try:
            from app import app
            
            # Check if app has session contexts
            if hasattr(app, 'session_contexts'):
                session_count = len(app.session_contexts)
                if session_count > 0:
                    worker.log.info(f"💾 {session_count} active sessions detected")
                    
                    # In production, implement persistence here
                    # For now, just log
                    worker.log.info(f"💾 Session data would be saved here (implement persistence)")
                    
            elif hasattr(app, 'config') and 'SESSION_CONTEXTS' in app.config:
                session_count = len(app.config['SESSION_CONTEXTS'])
                if session_count > 0:
                    worker.log.info(f"💾 {session_count} active sessions in config")
                    
        except ImportError:
            worker.log.debug("App not yet loaded, skipping session save")
        except Exception as e:
            worker.log.warning(f"Could not access app state: {e}")
            
    except Exception as e:
        worker.log.error(f"❌ Error during graceful shutdown: {e}")
    
    worker.log.info(f"👋 Worker {worker.pid} exited cleanly")

# ===============================
# LOGGING CONFIGURATION
# ===============================

# Log to stdout for Render/Docker
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info").lower()

# Enhanced access log format with response time
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)ss'

capture_output = True
enable_stdio_inheritance = True

# ===============================
# PERFORMANCE & SAFETY SETTINGS
# ===============================

# Use shared memory for worker tmp files (Linux only)
try:
    if os.path.exists("/dev/shm"):
        worker_tmp_dir = "/dev/shm"
    else:
        worker_tmp_dir = None
except:
    worker_tmp_dir = None

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
    
    # Feature flags (ĐỒNG BỘ VỚI .ENV)
    features = []
    
    if os.getenv("STATE_MACHINE_ENABLED", "true").lower() == "true":
        features.append("State Machine")
    
    if os.getenv("ENABLE_INTENT_DETECTION", "true").lower() == "true":
        features.append("Intent Detection")
    
    if os.getenv("ENABLE_PHONE_DETECTION", "true").lower() == "true":
        features.append("Phone Detection")
    
    if os.getenv("ENABLE_LOCATION_CONTEXT", "true").lower() == "true":
        features.append("Location Filter")
    
    if os.getenv("FAISS_ENABLED", "true").lower() == "true":
        features.append("FAISS Search")
    
    if os.getenv("ENABLE_CACHING", "true").lower() == "true":
        features.append("Response Caching")
    
    if os.getenv("ENABLE_LEAD_CAPTURE", "true").lower() == "true":
        features.append("Lead Capture")
    
    if os.getenv("ENABLE_META_CAPI", "false").lower() == "true":
        features.append("Meta CAPI")
    
    if features:
        server.log.info(f"🎯 Active features: {', '.join(features)}")
    
    server.log.info(f"🌐 Binding to: {bind}")
    server.log.info(f"📅 Start time: {datetime.now().isoformat()}")
    server.log.info("=" * 60)
    
    # Store start time
    server.cfg.started_at = datetime.now()


def when_ready(server):
    """Called just after the server is started"""
    server.log.info("✅ Gunicorn ready to accept connections")
    
    # Check essential services
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        server.log.info("🔍 System check:")
        
        # Check Python version
        server.log.info(f"   • Python: {sys.version.split()[0]}")
        
        # Try to check app status (may not be loaded yet in preload mode)
        try:
            from app import OPENAI_AVAILABLE, FAISS_AVAILABLE
            server.log.info(f"   • OpenAI: {'✅ Available' if OPENAI_AVAILABLE else '❌ Unavailable'}")
            server.log.info(f"   • FAISS: {'✅ Available' if FAISS_AVAILABLE else '❌ Unavailable (using fallback)'}")
        except ImportError:
            server.log.debug("   • App not yet loaded, skipping service check")
        
        # Check critical files
        critical_files = [
            "knowledge.json",
            "entities.py",
            "meta_capi.py",
            "response_guard.py"
        ]
        
        missing_files = []
        for file in critical_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            server.log.warning(f"   ⚠️ Missing files: {', '.join(missing_files)}")
        else:
            server.log.info(f"   • Critical files: ✅ All present")
        
    except Exception as e:
        server.log.warning(f"⚠️ System check incomplete: {e}")


def post_fork(server, worker):
    """Called just after a worker has been forked"""
    worker.log.info(f"👶 Worker {worker.pid} spawned (Thread {worker.age})")
    
    # Set worker-specific environment
    os.environ['GUNICORN_WORKER_PID'] = str(worker.pid)
    
    # Initialize worker-specific resources
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Import app to trigger initialization
        try:
            from app import app, search_engine
            
            worker.log.info(f"📱 App loaded in worker {worker.pid}")
            
            # Check if search engine is loaded
            if hasattr(search_engine, '_loaded') and not search_engine._loaded:
                worker.log.info("🔍 Loading search index in worker...")
                try:
                    search_engine.load_index()
                    worker.log.info("✅ Search index loaded")
                except Exception as e:
                    worker.log.error(f"❌ Failed to load search index: {e}")
            
        except ImportError as e:
            worker.log.error(f"❌ Failed to import app: {e}")
        except Exception as e:
            worker.log.error(f"❌ Worker initialization error: {e}")
            
    except Exception as e:
        worker.log.error(f"❌ Critical worker initialization error: {e}")


def worker_int(worker):
    """Called when a worker receives SIGINT or SIGQUIT"""
    worker.log.warning(f"⚠️ Worker {worker.pid} received interrupt signal")
    
    # Try to save state before exit
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from app import app
        if hasattr(app, 'session_contexts'):
            active_sessions = len(app.session_contexts)
            if active_sessions > 0:
                worker.log.info(f"💾 Interrupt: {active_sessions} active sessions")
                
    except Exception:
        pass  # Silently fail if app not available


def worker_abort(worker):
    """Called when a worker receives SIGABRT"""
    worker.log.error(f"🚨 Worker {worker.pid} aborted (SIGABRT)")
    
    # Emergency cleanup
    try:
        import gc
        gc.collect()  # Force garbage collection
        worker.log.info("🧹 Emergency garbage collection completed")
    except Exception as e:
        worker.log.error(f"Failed to run garbage collection: {e}")


def on_exit(server):
    """Called just before exiting the master process"""
    server.log.info("=" * 60)
    server.log.info("👋 RUBY WINGS AI CHATBOT SHUTTING DOWN")
    server.log.info("=" * 60)
    
    # Summary statistics
    try:
        if hasattr(server.cfg, 'started_at'):
            uptime = datetime.now() - server.cfg.started_at
            server.log.info(f"📈 Uptime: {uptime}")
        else:
            server.log.info(f"📈 Uptime: Unknown")
    except Exception:
        pass
    
    server.log.info(f"📅 Shutdown time: {datetime.now().isoformat()}")
    server.log.info("=" * 60)


# ===============================
# REQUEST PROCESSING HOOKS
# ===============================

def post_request(worker, req, environ, resp):
    """Called after a request has been processed"""
    try:
        path = environ.get("PATH_INFO", "")
        
        # Skip logging for health checks and static files
        skip_paths = ["/health", "/api/health", "/favicon.ico", "/robots.txt", "/.well-known/"]
        if any(path.startswith(skip) for skip in skip_paths):
            return
        
        # Calculate request duration if available
        if hasattr(req, 'start_time'):
            try:
                duration = worker.time() - req.start_time
                
                # Log slow requests
                # Threshold: 10s for chat, 5s for other API calls
                if "/chat" in path or "/api/chat" in path:
                    slow_threshold = 10.0
                elif "/api/" in path:
                    slow_threshold = 5.0
                else:
                    slow_threshold = 3.0
                
                if duration > slow_threshold:
                    status = getattr(resp, 'status', 'Unknown')
                    method = environ.get("REQUEST_METHOD", "UNKNOWN")
                    
                    worker.log.warning(
                        f"🐌 Slow request: {method} {path} -> {status} "
                        f"({duration:.2f}s, threshold: {slow_threshold}s)"
                    )
                
                # Log very fast cached responses (optional debug)
                elif duration < 0.05 and "/chat" in path:
                    worker.log.debug(f"⚡ Fast response (likely cached): {path} ({duration:.3f}s)")
                    
            except Exception as e:
                worker.log.debug(f"Error calculating request duration: {e}")
                
    except Exception as e:
        # Don't let post_request hook crash the worker
        worker.log.debug(f"Error in post_request hook: {e}")


def pre_request(worker, req):
    """Called before processing a request"""
    try:
        # Set start time for duration calculation
        req.start_time = worker.time()
        
        # Add request ID for tracing
        import uuid
        req.request_id = str(uuid.uuid4())[:8]
        
        # Log request (debug level)
        path = getattr(req, 'path', 'Unknown')
        method = getattr(req, 'method', 'Unknown')
        
        worker.log.debug(f"📥 [{req.request_id}] {method} {path}")
        
    except Exception as e:
        # Don't let pre_request hook crash the worker
        worker.log.debug(f"Error in pre_request hook: {e}")


# ===============================
# HEALTH CHECK SUPPORT
# ===============================

def get_health_status(worker):
    """
    Get health status of worker.
    This is called by the /health endpoint in app.py
    """
    health = {
        "timestamp": datetime.now().isoformat(),
        "worker_pid": worker.pid,
        "ram_profile": RAM_PROFILE,
        "memory_mode": RAM_MODE,
        "status": "healthy"
    }
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from app import OPENAI_AVAILABLE, FAISS_AVAILABLE
        
        health["services"] = {
            "openai": "available" if OPENAI_AVAILABLE else "unavailable",
            "faiss": "available" if FAISS_AVAILABLE else "fallback"
        }
        
    except Exception as e:
        health["services"] = {"error": str(e)}
    
    return health


# ===============================
# STARTUP VALIDATION
# ===============================

# Validate configuration at import time
def validate_config():
    """Validate gunicorn configuration"""
    issues = []
    
    # Check RAM configuration
    if workers > 1 and RAM_PROFILE == "512":
        issues.append("⚠️ Multiple workers with 512MB RAM may cause OOM errors")
    
    # Check timeout
    if timeout < 30:
        issues.append("⚠️ Timeout <30s may interrupt OpenAI API calls")
    
    # Check worker limits
    if max_requests < 100:
        issues.append("⚠️ Very low max_requests may cause frequent worker restarts")
    
    # Check preload_app with multiple workers
    if preload_app and workers > 1 and RAM_PROFILE == "512":
        issues.append("ℹ️ preload_app=True with multiple workers on low RAM - monitor memory")
    
    return issues

# Run validation
if __name__ == "__main__":
    print("=" * 60)
    print("GUNICORN CONFIGURATION VALIDATION - RUBY WINGS v5.2")
    print("=" * 60)
    print(f"🔧 Configuration:")
    print(f"   • RAM Profile: {RAM_PROFILE}MB ({RAM_MODE} memory mode)")
    print(f"   • Workers: {workers} | Threads: {threads}")
    print(f"   • Bind: {bind}")
    print(f"   • Timeout: {timeout}s | Graceful: {graceful_timeout}s")
    print(f"   • Max requests per worker: {max_requests}")
    print(f"   • Worker connections: {worker_connections}")
    print(f"   • Preload app: {preload_app}")
    print(f"   • Log level: {loglevel}")
    print("=" * 60)
    
    # Check for issues
    issues = validate_config()
    
    if issues:
        print("\n⚠️  Configuration Warnings:")
        for issue in issues:
            print(f"   {issue}")
        print()
    
    # Check environment
    print("🌍 Environment:")
    env_vars = [
        "PORT", "HOST", "WORKERS", "THREADS", "TIMEOUT",
        "RAM_PROFILE", "IS_LOW_RAM",
        "STATE_MACHINE_ENABLED", "ENABLE_INTENT_DETECTION",
        "FAISS_ENABLED", "ENABLE_CACHING"
    ]
    
    for var in env_vars:
        value = os.getenv(var, "(not set)")
        print(f"   • {var}: {value}")
    
    print("=" * 60)
    print("✅ Configuration validation complete")
    print("=" * 60)
else:
    # Run validation on import and log issues
    issues = validate_config()
    if issues:
        logger = logging.getLogger("gunicorn.error")
        for issue in issues:
            logger.warning(issue)