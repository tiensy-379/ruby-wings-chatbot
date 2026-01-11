# gunicorn.conf.py - Optimized for Ruby Wings Chatbot v4.0 on Render (512MB)
import os
import sys
import logging

# ===== CONFIGURATION =====
# Bind address
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"

# Worker configuration for 512MB RAM
workers = int(os.getenv('GUNICORN_WORKERS', '1'))
threads = int(os.getenv('GUNICORN_THREADS', '2'))
worker_class = 'sync'  # Use sync for Flask apps

# Timeout settings
timeout = int(os.getenv('TIMEOUT', '120'))
graceful_timeout = int(os.getenv('GUNICORN_GRACEFUL_TIMEOUT', '30'))
keepalive = int(os.getenv('GUNICORN_KEEPALIVE', '5'))

# Performance settings
max_requests = int(os.getenv('GUNICORN_MAX_REQUESTS', '1000'))
max_requests_jitter = int(os.getenv('GUNICORN_MAX_REQUESTS_JITTER', '50'))
worker_connections = int(os.getenv('GUNICORN_WORKER_CONNECTIONS', '1000'))

# Preload app to reduce memory usage
preload_app = True

# ===== LOGGING =====
# Log to stdout/stderr for Render
accesslog = "-"
errorlog = "-"
loglevel = os.getenv('GUNICORN_LOG_LEVEL', 'info')

# Access log format
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# ===== SECURITY =====
# Request limits
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# ===== OPTIMIZATION =====
# Disable daemon mode for Render
daemon = False

# Process name
proc_name = "ruby-wings-chatbot-v4"

# Capture output
capture_output = True
enable_stdio_inheritance = True

# Worker temp directory
worker_tmp_dir = "/dev/shm"

# ===== RENDER-SPECIFIC =====
# Detect if running on Render
is_render = os.getenv('RENDER', False)

if is_render:
    # Render-specific optimizations
    backlog = 2048
    proxy_protocol = True
    forwarded_allow_ips = '*'
    
    # Log Render info
    instance_id = os.getenv('RENDER_INSTANCE_ID', 'unknown')
    service_id = os.getenv('RENDER_SERVICE_ID', 'unknown')
    
    # Add environment variables
    raw_env = [
        f'RENDER_INSTANCE_ID={instance_id}',
        f'RENDER_SERVICE_ID={service_id}',
    ]

# ===== HOOKS =====
def on_starting(server):
    """Called when the server starts"""
    server.log.info("🚀 Ruby Wings Chatbot v4.0 starting...")
    server.log.info(f"  Config: {workers} worker(s), {threads} thread(s)")
    server.log.info(f"  Timeout: {timeout}s, Graceful: {graceful_timeout}s")
    
    # Log memory info
    ram_profile = os.getenv('RAM_PROFILE', '512')
    server.log.info(f"  RAM Profile: {ram_profile}MB")

def when_ready(server):
    """Called when workers are ready"""
    server.log.info(f"✅ Gunicorn ready on {bind}")
    server.log.info(f"  Workers: {len(server.workers)}")
    
    # Log active upgrades
    upgrades = [k for k in os.environ if k.startswith('UPGRADE_') and os.environ.get(k) == 'true']
    if upgrades:
        server.log.info(f"  Active upgrades: {len(upgrades)}")

def worker_int(worker):
    """Worker received SIGINT or SIGQUIT"""
    worker.log.info("Worker interrupted")

def worker_abort(worker):
    """Worker received SIGABRT"""
    worker.log.info("Worker aborted")

def pre_fork(server, worker):
    """Called before forking worker"""
    pass

def post_fork(server, worker):
    """Called after forking worker"""
    worker.log.info(f"Worker {worker.pid} started")

def worker_exit(server, worker):
    """Worker exiting"""
    worker.log.info(f"Worker {worker.pid} exited")

def on_exit(server):
    """Server exiting"""
    server.log.info("👋 Ruby Wings Chatbot shutting down")
    server.log.info("Thank you for using Ruby Wings!")

# ===== CUSTOM MIDDLEWARE =====
def post_request(worker, req, environ, resp):
    """Called after each request"""
    # Skip logging for health checks
    if environ.get('PATH_INFO') == '/api/health':
        return
    
    # Log slow requests (> 5 seconds)
    if hasattr(req, 'start_time'):
        duration = worker.time() - req.start_time
        if duration > 5.0:
            worker.log.warning(f"Slow request: {environ.get('PATH_INFO')} took {duration:.2f}s")

# ===== REQUEST FILTER =====
def access_log_filter(environ):
    """Filter requests from access log"""
    # Skip health checks
    if environ.get('PATH_INFO') == '/api/health':
        return False
    
    # Skip favicon requests
    if environ.get('PATH_INFO') == '/favicon.ico':
        return False
    
    return True