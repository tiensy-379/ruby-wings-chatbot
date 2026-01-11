# gunicorn.conf.py - Optimized for 512MB RAM
import os
import multiprocessing

# ===== WORKER CONFIGURATION =====
# Workers: 1 for 512MB RAM (sử dụng 1 worker + nhiều threads)
workers = int(os.getenv("GUNICORN_WORKERS", "1"))

# Threads per worker: 2-4 cho I/O bound app
threads = int(os.getenv("GUNICORN_THREADS", "2"))

# Worker type: sync cho Flask
worker_class = "sync"

# Worker connections
worker_connections = int(os.getenv("GUNICORN_WORKER_CONNECTIONS", "1000"))

# ===== TIMEOUT CONFIGURATION =====
# Worker timeout (seconds)
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))

# Graceful shutdown timeout
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))

# Keep-alive
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "2"))

# ===== NETWORK =====
# Bind address
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"

# Backlog size
backlog = int(os.getenv("GUNICORN_BACKLOG", "2048"))

# ===== LOGGING =====
# Log level
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")

# Access log to stdout
accesslog = "-"

# Error log to stderr
errorlog = "-"

# Access log format (Render-friendly)
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)ss'

# ===== PROCESS MANAGEMENT =====
# Process name
proc_name = "ruby-wings-chatbot-v4"

# Max requests before worker restart (prevents memory leaks)
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "1000"))

# Jitter to prevent all workers restarting at once
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "50"))

# ===== SECURITY LIMITS =====
# Limit request line
limit_request_line = 4094

# Limit number of request headers
limit_request_fields = 100

# Limit size of request headers
limit_request_field_size = 8190

# ===== PERFORMANCE =====
# Preload app to reduce memory usage
preload_app = os.getenv("GUNICORN_PRELOAD_APP", "true").lower() == "true"

# Don't daemonize in Render
daemon = False

# ===== RENDER-SPECIFIC =====
# Detect Render environment
is_render = os.getenv("RENDER", "")

if is_render:
    # Render uses HTTP/1.1 reverse proxy
    proxy_protocol = False
    
    # Render handles SSL termination
    forwarded_allow_ips = "*"
    
    # Log Render instance info
    instance_id = os.getenv("RENDER_INSTANCE_ID", "unknown")
    service_id = os.getenv("RENDER_SERVICE_ID", "unknown")
    
    # Add to environment
    raw_env = [
        f"RENDER_INSTANCE_ID={instance_id}",
        f"RENDER_SERVICE_ID={service_id}",
        f"RENDER=1",
    ]

# ===== HOOKS =====
def on_starting(server):
    """Called just before the master process is initialized"""
    server.log.info("🚀 Ruby Wings Chatbot v4.0 starting...")
    server.log.info(f"  Workers: {workers}")
    server.log.info(f"  Threads per worker: {threads}")
    server.log.info(f"  Timeout: {timeout}s")

def when_ready(server):
    """Called when the worker is ready to serve"""
    server.log.info(f"✅ Gunicorn ready on {bind}")

def worker_int(worker):
    """Called when a worker receives SIGINT or SIGQUIT"""
    worker.log.info("Worker interrupted")

def worker_abort(worker):
    """Called when a worker receives SIGABRT"""
    worker.log.info("Worker aborted")

def pre_fork(server, worker):
    """Called just before forking the worker subprocess"""
    pass

def post_fork(server, worker):
    """Called just after forking the worker subprocess"""
    server.log.info(f"Worker {worker.pid} forked")

def worker_exit(server, worker):
    """Called when a worker exits"""
    server.log.info(f"Worker {worker.pid} exited")

def on_exit(server):
    """Called when Gunicorn exits"""
    server.log.info("👋 Ruby Wings Chatbot shutting down")

# ===== ADVANCED TUNING =====
# Maximum number of pending connections
# backlog = min(2048, (64 * workers))

# Disable access logging for health checks (optional)
def access_log_filter(environ, response):
    """Filter out health check requests from access log"""
    path = environ.get('PATH_INFO', '')
    if path == '/api/health':
        return False
    return True

# Enable if needed
# access_log_filter_func = access_log_filter

# Worker tmp directory
worker_tmp_dir = "/dev/shm"

# Capture output
capture_output = True

# Enable stdout logging
enable_stdio_inheritance = True

# PID file (not used in Render)
pidfile = None

# User/Group (Render manages this)
user = None
group = None

# ===== THREAD POOL =====
# Thread pool for sync workers (Gunicorn 20+)
threads = threads

# ===== SSL (Render handles SSL) =====
# keyfile = None
# certfile = None
# ssl_version = 2
# cert_reqs = 0
# ca_certs = None
# suppress_ragged_eofs = True
# do_handshake_on_connect = False