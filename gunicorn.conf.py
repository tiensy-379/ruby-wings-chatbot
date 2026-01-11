# gunicorn.conf.py — Ruby Wings Chatbot v4.0 (Render / Gunicorn 21.x / 512MB)

import os
import logging
import multiprocessing

# ===============================
# BASIC CONFIG
# ===============================

bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"

workers = int(os.getenv("GUNICORN_WORKERS", "1"))
threads = int(os.getenv("GUNICORN_THREADS", "2"))

# gthread is best for Flask on Render
worker_class = "gthread"

timeout = int(os.getenv("TIMEOUT", "300"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))

max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "50"))

preload_app = True
daemon = False

proc_name = "ruby-wings-chatbot-v4"

# ===============================
# LOGGING
# ===============================

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")

access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
)

capture_output = True
enable_stdio_inheritance = True

# ===============================
# RENDER SAFE SETTINGS
# ===============================

worker_tmp_dir = "/dev/shm"
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# ===============================
# HOOKS (Gunicorn 21.x SAFE)
# ===============================

def on_starting(server):
    server.log.info("🚀 Ruby Wings Chatbot v4.0 starting")
    server.log.info(f"Workers: {workers}, Threads: {threads}")
    server.log.info(f"Timeout: {timeout}s, Graceful: {graceful_timeout}s")

    ram = os.getenv("RAM_PROFILE", "512")
    server.log.info(f"RAM profile: {ram}MB")


def when_ready(server):
    server.log.info("✅ Gunicorn ready")
    server.log.info(f"Listening at: {server.cfg.bind}")

    upgrades = [k for k in os.environ if k.startswith("UPGRADE_") and os.getenv(k) == "true"]
    if upgrades:
        server.log.info(f"Active upgrades: {len(upgrades)}")


def post_fork(server, worker):
    worker.log.info(f"Worker spawned (pid={worker.pid})")


def worker_exit(server, worker):
    worker.log.info(f"Worker exited (pid={worker.pid})")


def worker_int(worker):
    worker.log.warning(f"Worker interrupted (pid={worker.pid})")


def worker_abort(worker):
    worker.log.error(f"Worker aborted (pid={worker.pid})")


def on_exit(server):
    server.log.info("👋 Ruby Wings Chatbot shutting down")


# ===============================
# REQUEST HOOK
# ===============================

def post_request(worker, req, environ, resp):
    path = environ.get("PATH_INFO", "")
    if path == "/api/health":
        return

    if hasattr(req, "start_time"):
        duration = worker.time() - req.start_time
        if duration > 5:
            worker.log.warning(f"Slow request {path}: {duration:.2f}s")


def access_log_filter(environ):
    path = environ.get("PATH_INFO", "")
    if path in ("/api/health", "/favicon.ico"):
        return False
    return True
