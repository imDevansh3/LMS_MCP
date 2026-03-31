FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY mcp_server.py .

# Default port (can be overridden by environment variables at runtime)
ENV MCP_PORT=8001

# Expose MCP server port (reads from environment)
EXPOSE ${MCP_PORT}

# Health check - uses Python socket to check if port is listening
# More reliable than curl for SSE endpoints
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(5); s.connect(('localhost', int('${MCP_PORT:-8001}'))) or exit(1)" || exit 1

# Run the MCP server
CMD ["python", "mcp_server.py"]
