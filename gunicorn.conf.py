# Gunicorn configuration optimized for 512MB–2GB with FAISS / numpy fallback

bind = "0.0.0.0:10000"

# Workers & threads
workers = 1
threads = 2

# IMPORTANT: do NOT preload app when using FAISS / large numpy arrays
preload_app = False

# Timeouts
timeout = 300
graceful_timeout = 30
keepalive = 5

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Prevent memory bloat from long-lived workers
max_requests = 1000
max_requests_jitter = 50
