# --- Builder Stage ---
FROM python:3.12-slim AS builder

# Copy the uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set build-time environment variables to limit resources
ENV UV_COMPILE_BYTECODE=1
ENV UV_JOBS=1 
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install system build dependencies
# We need these to build any C-based dependencies (like 'av' if it's not binary-only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    build-essential \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libswresample-dev \
    libswscale-dev \
    libavutil-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests
COPY pyproject.toml uv.lock ./

# Sync dependencies into a virtualenv
# Using --no-cache and UV_JOBS=1 to prevent t3.micro crashes
RUN uv sync --frozen --no-install-project --no-dev --no-cache

# --- Runtime Stage ---
FROM python:3.12-slim

WORKDIR /app

# Install runtime shared libraries
# We use the -dev package names as they are reliable meta-packages that pull in the correct versions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libswresample-dev \
    libswscale-dev \
    libavutil-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtualenv and app from builder
COPY --from=builder /app/.venv /app/.venv
COPY app/ ./app/

# Set environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=True
ENV VERIFY_TOKEN="my_secure_verify_token_123"

EXPOSE 8000

# Run the FastAPI application using the venv directly
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
