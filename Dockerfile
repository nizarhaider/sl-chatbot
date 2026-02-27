# Use the official lightweight Python image
FROM python:3.12-slim

# Copy the uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies for aiortc
RUN apt-get update && apt-get install -y \
    pkg-config \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libswresample-dev \
    libswscale-dev \
    libavutil-dev \
    && rm -rf /var/lib/apt/lists/*

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED=True

# Enable bytecode compilation for faster app startup
ENV UV_COMPILE_BYTECODE=1

# Set the working directory to /app
WORKDIR /app

# Copy project manifesting files First (for caching)
COPY pyproject.toml uv.lock ./

# Install dependencies using the lockfile
RUN uv sync --frozen --no-install-project --no-dev

# Copy the application code
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Set a default environment variable
ENV VERIFY_TOKEN="my_secure_verify_token_123"

# Run the FastAPI application
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
