# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Set the working directory to /app
WORKDIR /app

# Copy local code to the container image.
COPY requirements.txt ./

# Install dependencies required by the application
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY main.py ./

# Expose port (uvicorn default is 8000)
EXPOSE 8000

# Set a default environment variable (override in your EC2 environment)
ENV VERIFY_TOKEN="my_secure_verify_token_123"

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
