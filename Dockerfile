# Ruby Wings Chatbot v5.2.1 - Production Dockerfile
# Multi-stage build for optimal image size and security

# ==================== BUILD STAGE ====================
FROM python:3.11-slim AS builder

# Build arguments
ARG RAM_PROFILE=512
ARG PYTHON_VERSION=3.11

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    # Install FAISS only if RAM_PROFILE >= 2048
    if [ "$RAM_PROFILE" -ge "2048" ]; then \
        pip install faiss-cpu==1.7.4; \
    fi

# ==================== RUNTIME STAGE ====================
FROM python:3.11-slim

# Build arguments
ARG RAM_PROFILE=512
ARG BUILD_DATE
ARG VCS_REF

# Metadata
LABEL maintainer="Ruby Wings AI Team" \
      version="5.2.1" \
      description="Ruby Wings AI Chatbot - Production Ready" \
      ram_profile="${RAM_PROFILE}MB" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.title="Ruby Wings Chatbot" \
      org.opencontainers.image.version="5.2.1"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    RAM_PROFILE=${RAM_PROFILE} \
    FLASK_APP=app.py \
    FLASK_ENV=production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r chatbot && useradd -r -g chatbot chatbot

# Create application directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application files
COPY --chown=chatbot:chatbot app.py .
COPY --chown=chatbot:chatbot entities.py .
COPY --chown=chatbot:chatbot meta_capi.py .
COPY --chown=chatbot:chatbot response_guard.py .
COPY --chown=chatbot:chatbot gunicorn_conf.py .
COPY --chown=chatbot:chatbot build_index.py .

# Create necessary directories
RUN mkdir -p /app/data /app/logs && \
    chown -R chatbot:chatbot /app

# Copy knowledge base (will be overridden by volume mount)
COPY --chown=chatbot:chatbot knowledge.json /app/data/knowledge.json

# Switch to non-root user
USER chatbot

# Expose port
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

# Set working directory for runtime
WORKDIR /app

# Start command
CMD ["gunicorn", "app:app", "--config", "gunicorn_conf.py"]