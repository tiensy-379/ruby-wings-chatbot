# gunicorn.conf.py
import os
import multiprocessing

# ===== WORKER CONFIGURATION =====
# Number of worker processes
# For 512MB RAM, use 1 worker
workers = int(os.getenv("GUNICORN_WORKERS", "1"))

# Number of threads per worker
threads = int(os.getenv("GUNICORN_THREADS", "2"))

# Worker type (sync for CPU-bound, gevent for I/O-bound)
worker_class = "sync"  # Use sync for Flask, or "gevent" if installed

# Maximum concurrent requests per worker
worker_connections = int(os.getenv("GUNICORN_WORKER_CONNECTIONS", "1000"))

# ===== TIMEOUT CONFIGURATION =====
# Worker timeout (seconds) - separate from app TIMEOUT
timeout = int(os.getenv("GUNICORN_TIMEOUT", "300"))

# Graceful timeout for worker shutdown
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))

# Keep-alive timeout
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))

# ===== NETWORK CONFIGURATION =====
# Bind address (Render provides PORT)
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"

# Backlog (maximum queued connections)
backlog = int(os.getenv("GUNICORN_BACKLOG", "2048"))

# ===== LOGGING CONFIGURATION =====
# Log level
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")

# Access log - '-' means log to stdout
accesslog = "-"

# Error log - '-' means log to stderr
errorlog = "-"

# Access log format
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)s'

# ===== PROCESS NAMING =====
proc_name = "ruby-wings-chatbot"

# ===== SECURITY & LIMITS =====
# Maximum requests per worker before restart (prevents memory leaks)
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "1000"))

# Jitter to add to max_requests (random restart)
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "50"))

# Limit request line size
limit_request_line = int(os.getenv("GUNICORN_LIMIT_REQUEST_LINE", "4094"))

# Limit request fields size
limit_request_fields = int(os.getenv("GUNICORN_LIMIT_REQUEST_FIELDS", "100"))

# Limit request field size
limit_request_field_size = int(os.getenv("GUNICORN_LIMIT_REQUEST_FIELD_SIZE", "8190"))

# ===== PERFORMANCE TUNING =====
# Preload app (reduces memory usage)
preload_app = os.getenv("GUNICORN_PRELOAD_APP", "false").lower() == "true"

# Daemon mode (don't use in Render/container)
daemon = False

# PID file (optional)
pidfile = os.getenv("GUNICORN_PIDFILE", None)

# User/Group (for Linux systems, not applicable in Render)
user = os.getenv("GUNICORN_USER", None)
group = os.getenv("GUNICORN_GROUP", None)

# ===== SSL/TLS (if using custom domain with SSL) =====
# Only enable if you're terminating SSL at Gunicorn (not typical on Render)
# keyfile = os.getenv("GUNICORN_KEYFILE", None)
# certfile = os.getenv("GUNICORN_CERTFILE", None)

# ===== DEBUGGING =====
# Reload on code changes (development only)
reload = os.getenv("GUNICORN_RELOAD", "false").lower() == "true"

# Spew traceback on error
spew = os.getenv("GUNICORN_SPEW", "false").lower() == "true"

# ===== HOOKS (optional) =====
def on_starting(server):
    """Run when Gunicorn starts"""
    server.log.info("🚀 Starting Ruby Wings Chatbot v4.0...")

def on_reload(server):
    """Run when Gunicorn reloads"""
    server.log.info("🔄 Reloading Ruby Wings Chatbot...")

def when_ready(server):
    """Run when workers are ready"""
    server.log.info(f"✅ Gunicorn ready with {workers} worker(s) and {threads} thread(s) per worker")

def on_exit(server):
    """Run when Gunicorn exits"""
    server.log.info("👋 Shutting down Ruby Wings Chatbot...")

# ===== CUSTOM SETTINGS FOR FLASK/CHATBOT =====
# Environment variables for the app
raw_env = [
    f"FLASK_ENV={os.getenv('FLASK_ENV', 'production')}",
    f"PYTHONPATH=/opt/render/project/src",
]

# Post-fork function to initialize app in each worker
def post_fork(server, worker):
    """Initialize app after worker fork"""
    server.log.info(f"Worker {worker.pid} forked")

# Pre-fork function to run before worker fork
def pre_fork(server, worker):
    """Run before worker fork"""
    pass

# Pre-exec function to run before exec
def pre_exec(server):
    """Run before exec"""
    server.log.info("Forked child, re-executing.")

# Worker exit function
def worker_exit(server, worker):
    """Run when worker exits"""
    server.log.info(f"Worker {worker.pid} exited")

# ===== RENDER-SPECIFIC OPTIMIZATIONS =====
# Render runs in container, so optimize for that
if os.getenv("RENDER", None):
    # On Render, we're behind a load balancer that handles keep-alive
    keepalive = 2
    
    # Render uses 1:1 routing, so we can use sync workers
    worker_class = "sync"
    
    # Preload app to save memory (workers share memory)
    preload_app = True
    
    # Log Render instance info
    instance_id = os.getenv("RENDER_INSTANCE_ID", "unknown")
    raw_env.append(f"RENDER_INSTANCE_ID={instance_id}")