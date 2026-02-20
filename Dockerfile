# Use the official lightweight Python image
FROM python:3.12-slim

# Copy the uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED=True

# Enable bytecode compilation for faster app startup
ENV UV_COMPILE_BYTECODE=1

# Set the working directory to /app
WORKDIR /app

# Copy project manifesting files First (for caching)
COPY pyproject.toml uv.lock ./

# Install dependencies using the lockfile (excluding the project itself to be copied later)
RUN uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application code
COPY main.py ./

# Expose port (uvicorn default is 8000)
EXPOSE 8000

# Set a default environment variable (override in your EC2 environment)
ENV VERIFY_TOKEN="my_secure_verify_token_123"

# Run the FastAPI application using the uv-managed environment
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
