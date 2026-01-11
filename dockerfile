# LinguaSync Stage 1 - Production Dockerfile
# Multi-stage build for optimized container size

# ============================================================================
# Stage 1: Base Image with System Dependencies
# ============================================================================
FROM python:3.12-slim as base

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
# - build-essential: Needed for compiling Python packages
# - curl: For health checks
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# ============================================================================
# Stage 2: Dependencies Installation
# ============================================================================
FROM base as dependencies

# Copy requirements file
COPY requirements-stage1.txt .

# Install Python dependencies
# This is done in a separate stage to leverage Docker layer caching
RUN pip install --no-cache-dir -r requirements-stage1.txt

# ============================================================================
# Optional: Install MeCab for Better Japanese Tokenization
# Uncomment the following lines if you want MeCab in production
# Note: This adds ~50MB to the image size but improves tokenization quality
# ============================================================================
RUN apt-get update && apt-get install -y \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    && pip install mecab-python3 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Stage 3: Runtime Image
# ============================================================================
FROM base as runtime

# Copy installed dependencies from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
# Only copy what's needed for the API to run
COPY subtitle_processor_v2.py .
COPY rag_engine_v2.py .
COPY learning_generator.py .
COPY api_v2.py .

# Copy .env file if it exists (for local testing)
# In production, use environment variables instead
COPY .env ./

# Create directories for data
RUN mkdir -p data/subtitles data/processed
RUN mkdir -p chroma_db_v2

# Expose the port that the API runs on
EXPOSE 8000

# Health check - Docker will check if the container is healthy
# This hits the /health endpoint every 30 seconds
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command to run the API
# In production, this can be overridden
CMD ["uvicorn", "api_v2:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================================
# Build Instructions:
# ============================================================================
#
# 1. Build the image:
#    docker build -t linguasync-api:latest .
#
# 2. Run locally for testing:
#    docker run -p 8000:8000 \
#      -e OPENAI_API_KEY=your-key-here \
#      linguasync-api:latest
#
# 3. Test the container:
#    curl http://localhost:8000/health
#
# 4. For development with volume mounts:
#    docker run -p 8000:8000 \
#      -e OPENAI_API_KEY=your-key-here \
#      -v $(pwd)/data:/app/data \
#      -v $(pwd)/chroma_db_v2:/app/chroma_db_v2 \
#      linguasync-api:latest
#
# ============================================================================