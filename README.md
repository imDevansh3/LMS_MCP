# Neulearn MCP Server

FastMCP-based Model Context Protocol server providing tools for the Neulearn LMS agent system.

## Features

- **54 MCP Tools** across 5 categories:
  - 12 Episodic Memory Tools
  - 16 Semantic Profile Tools
  - 10 Semantic Query Tools
  - 8 Mentor Chatbot Tools
  - 8 Viva Agent Tools

- **SSE Transport**: Server-Sent Events for real-time communication
- **PostgreSQL Integration**: Direct database access for all LMS data
- **Multi-Agent Support**: Tools designed for episodic, semantic, mentor, and viva agents

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL database (LMS schema)
- Required packages: `fastmcp`, `psycopg2`, `python-dotenv`, `uvicorn`

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env`:
```bash
DATABASE_URL=postgresql://user:password@localhost:5432/LMS
MCP_PORT=8001
```

3. Run the server:
```bash
python mcp_server.py
```

The server will start on `http://0.0.0.0:8001` with SSE endpoint at `/sse`.

## Docker Deployment

Build and run using Docker:

```bash
docker build -t neulearn-mcp-server .
docker run -p 8001:8001 -e DATABASE_URL="postgresql://..." neulearn-mcp-server
```

## Tool Categories

### Episodic Memory Tools (12)
- Session start/end tracking
- Exercise attempt/pass/fail recording
- Course completion tracking
- Semantic rebuild triggers

### Semantic Profile Tools (16)
- Profile creation and updates
- Struggle/strength tracking
- Learning pattern analysis
- Confidence evolution

### Semantic Query Tools (10)
- Profile retrieval
- Topic performance queries
- Skill correlation analysis
- Learning trajectory tracking

### Mentor Chatbot Tools (8)
- User context detection
- Semantic summaries
- Topic performance analysis
- Chat history management
- Exercise context retrieval
- Course progress tracking

### Viva Agent Tools (8)
- Capstone details and requirements
- Code review retrieval
- Test results analysis
- Viva session management
- Question/response recording
- Session completion

## API Usage

Connect via MCP client (e.g., `langchain_mcp_adapters`):

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "neulearn": {"url": "http://localhost:8001/sse", "transport": "sse"}
})

tools = await client.get_tools()
result = await tools[0].ainvoke({"user_id": "usr-001"})
```

## Architecture

- **FastMCP Framework**: Simplified MCP server creation
- **PostgreSQL Connection Pool**: Efficient database access
- **JSON Responses**: All tools return structured JSON
- **Error Handling**: Comprehensive logging and error responses

## Health Check

```bash
curl http://localhost:8001/sse
```

## License

MIT
