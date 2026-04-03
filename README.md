# Neulearn MCP Server

Complete Model Context Protocol (MCP) server for the Neulearn Learning Management System.

## Features

- **21 MCP Tools** across 4 categories:
  - **Episodic Memory** (9 tools): Process learning events into structured episodes
  - **Semantic Profiles** (5 tools): Build and update learner intelligence profiles
  - **Mentor Chatbot** (7 tools): Context-aware teaching assistance
  - **VIVA Agent** (7 tools): Capstone examination support

- **AI-Powered**: Uses Azure OpenAI for intelligent episode analysis and insights
- **Production Ready**: Docker support, connection pooling, structured logging
- **SSE Transport**: Server-Sent Events for real-time MCP communication

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials

# Run server
python mcp_server.py
```

Server will be available at: `http://localhost:8001/sse`

### Docker Deployment

```bash
# Build image
docker build -t neulearn-mcp:latest .

# Run container
docker run -p 8001:8001 \
  -e DATABASE_URL="postgresql://user:pass@host:5432/dbname" \
  -e AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/" \
  -e AZURE_OPENAI_API_KEY="your-key" \
  neulearn-mcp:latest
```

### Docker Compose

```bash
docker-compose up -d
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | Required |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Required |
| `AZURE_OPENAI_DEPLOYMENT` | Model deployment name | `gpt-4o-mini` |
| `AZURE_OPENAI_API_VERSION` | API version | `2025-01-01-preview` |
| `MCP_PORT` | Server port | `8001` |
| `MCP_HOST` | Server host | `0.0.0.0` |

## MCP Tools

### Episodic Memory Tools
- `process_self_assessment` - Process onboarding self-assessment
- `process_pre_assessment` - Process MCQ test results
- `process_pathway_assigned` - Record pathway assignment
- `process_course_activity` - Process study session
- `process_mentor_chat` - Process chat session
- `process_course_completed` - Aggregate course completion
- `process_capstone_code_review` - Analyze code review
- `process_capstone_test_run` - Cluster test failures
- `process_capstone_viva` - Score viva examination

### Semantic Profile Tools
- `build_initial_semantic_profile` - Create initial learner profile
- `update_profile_course_completed` - Update after course completion
- `update_profile_pathway_assigned` - Update for new pathway
- `update_profile_capstone_viva` - Update after viva (pass/fail)
- `update_profile_manual` - Periodic pattern analysis

### Mentor Chatbot Tools
- `get_student_context` - Fetch complete student snapshot
- `get_topic_content` - Retrieve topic/concept content
- `get_exercise_context` - Get exercise details and attempts
- `search_similar_questions` - Find previous similar questions
- `log_mentor_chat_turn` - Record chat conversation turn
- `get_prerequisite_gaps` - Identify missing prerequisites
- `suggest_learning_path` - AI-powered next-step recommendations

### VIVA Agent Tools
- `get_capstone_details` - Get capstone requirements
- `get_capstone_review` - Retrieve code review results
- `get_capstone_test` - Get test execution results
- `start_viva` - Initialize viva session
- `record_viva_turn` - Log viva conversation
- `get_viva_session` - Retrieve session state
- `complete_viva` - Finalize with pass/fail verdict

## Architecture

```
┌─────────────────────────────────────────────┐
│          MCP Server (FastMCP)               │
│                                             │
│  ┌────────────┐  ┌──────────────┐         │
│  │ Episodic   │  │  Semantic    │         │
│  │ Tools (9)  │  │  Tools (5)   │         │
│  └────────────┘  └──────────────┘         │
│                                             │
│  ┌────────────┐  ┌──────────────┐         │
│  │  Mentor    │  │    VIVA      │         │
│  │ Tools (7)  │  │  Tools (7)   │         │
│  └────────────┘  └──────────────┘         │
│                                             │
└─────────────────┬───────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
   ┌────▼────┐      ┌───────▼──────┐
   │ PostgreSQL│      │ Azure OpenAI │
   │ Database  │      │   (GPT-4o)   │
   └───────────┘      └──────────────┘
```

## Database Requirements

The server requires these PostgreSQL tables:
- `episodic_episodes` (partitioned by user_id)
- `semantic_profile_versions`
- `users`, `sessions`, `capstones`
- `raw_viva_turns`, `raw_mentor_chat_turns`
- `modules` (for topic/skill metadata)

## Development

### Testing
```bash
# Test episodic tools
python test_episodic.py

# Test semantic tools
python test_semantic.py

# Test VIVA tools
python viva_test.py
```

### Logs
Server logs structured output with timestamps:
```
2026-04-03 12:00:00 - neulearn.mcp - INFO - Configuration loaded successfully
2026-04-03 12:00:00 - neulearn.mcp - INFO - Database pool initialized (min=2, max=10)
```

## License

Proprietary - Neulearn Team

## Support

For issues or questions, contact the Neulearn development team.
