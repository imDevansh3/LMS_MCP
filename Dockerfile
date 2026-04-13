FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for psycopg2
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server file
COPY mcp_server.py .

# Expose SSE port
EXPOSE 8001

# Environment variables (override these at runtime)
ENV MCP_PORT=8001
ENV DATABASE_URL=""
ENV AZURE_OPENAI_ENDPOINT=""
ENV AZURE_OPENAI_API_KEY=""
ENV AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini"
ENV AZURE_OPENAI_API_VERSION="2025-01-01-preview"
ENV FALKORDB_HOST="localhost"
ENV FALKORDB_PORT="6379"
ENV FALKORDB_GRAPH="curriculum"

# Run the MCP server
CMD ["python", "mcp_server.py"]
