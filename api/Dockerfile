# Multi-stage build for Moroccan Income Prediction API
# Stage 1: Build dependencies
FROM python:3.9-slim-bullseye AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime image
FROM python:3.9-slim-bullseye

# Set metadata labels
LABEL maintainer="Moroccan Income Prediction Team" \
      description="API for predicting annual income of Moroccan individuals" \
      version="1.0.0" \
      org.opencontainers.image.source="https://github.com/moroccan-income-prediction" \
      org.opencontainers.image.licenses="MIT"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PORT="8000" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Default API key for development - should be overridden in production
    API_KEY="development_key" \
    # Set timezone
    TZ="Africa/Casablanca"

# Create logs directory with proper permissions
RUN mkdir -p /app/logs && \
    chown -R appuser:appuser /app

# Copy only necessary files
COPY --chown=appuser:appuser api/ /app/api/
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser models/ /app/models/

# Create directory for health check script
RUN mkdir -p /app/healthcheck

# Create a more comprehensive health check script
RUN echo '#!/bin/sh\n\
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)\n\
if [ "$response" = "200" ]; then\n\
  exit 0\n\
else\n\
  echo "Health check failed with status: $response"\n\
  exit 1\n\
fi' > /app/healthcheck/check.sh && \
    chmod +x /app/healthcheck/check.sh

# Install curl for health check and other essential tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Expose the port
EXPOSE 8000

# Create a startup script with proper signal handling
RUN echo '#!/bin/sh\n\
echo "Starting Moroccan Income Prediction API..."\n\
echo "API will be available at http://0.0.0.0:8000"\n\
echo "API documentation available at http://0.0.0.0:8000/docs"\n\
\n\
# Trap SIGTERM and SIGINT\n\
trap "echo Received SIGTERM/SIGINT, shutting down gracefully...; exit 0" TERM INT\n\
\n\
# Start uvicorn with specified parameters\n\
exec uvicorn api.api:app --host 0.0.0.0 --port 8000 --proxy-headers --workers 4\n\
' > /app/start.sh && \
    chmod +x /app/start.sh && \
    chown appuser:appuser /app/start.sh

# Switch to non-root user
USER appuser

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /app/healthcheck/check.sh

# Command to run the API with proper signal handling
CMD ["/app/start.sh"]