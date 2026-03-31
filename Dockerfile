FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server file
COPY mcp_server.py .

# Expose SSE port
EXPOSE 8001

# Run the MCP server
CMD ["python", "mcp_server.py"]
