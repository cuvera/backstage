# Multi-stage build for Python FastAPI app
FROM python:3.11-alpine AS builder

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies to a specific directory
RUN pip install --user -r requirements.txt

# Production stage
FROM python:3.11-alpine AS production

RUN apk add --no-cache \
    curl \
    ffmpeg \
    mediainfo \
    ca-certificates

# Create non-root user (Alpine syntax)
RUN addgroup -S appuser && adduser -S -G appuser appuser

# Create data directory with proper permissions
RUN mkdir -p /data && \
    chown -R appuser:appuser /data && \
    chmod 1777 /data

WORKDIR /app
# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY app ./app

# Set PATH to include user packages
ENV PATH="/home/appuser/.local/bin:$PATH"

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]