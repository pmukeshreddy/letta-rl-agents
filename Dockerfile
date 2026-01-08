FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

# Copy source
COPY src/ src/
COPY skills/ skills/
COPY ui/ ui/
COPY scripts/ scripts/

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 8000 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# Default command
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
