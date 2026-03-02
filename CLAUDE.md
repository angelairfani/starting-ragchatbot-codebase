# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Prerequisites

Install `uv` (Python package manager) if not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Running the Application

```bash
# Install dependencies
uv sync

# Start the server (from repo root)
./run.sh

# Or manually
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app serves on `http://localhost:8000`. The frontend is served as static files by FastAPI — no separate frontend build step.

Requires a `.env` file in the repo root:
```
ANTHROPIC_API_KEY=sk-ant-...
```

## Architecture

This is a single-process full-stack app: FastAPI serves both the API and the static frontend from `../frontend`.

**Key architectural decisions:**

- **Two-call Claude pattern**: Every query makes two Anthropic API calls. The first call gives Claude the `search_course_content` tool and lets it decide whether to search. If it calls the tool, a second API call is made with the search results injected as `tool_result` messages. If not (general knowledge questions), the first response is returned directly.

- **Tool-based retrieval**: The RAG retrieval is not pre-fetched before calling Claude. Instead, Claude itself decides what to search for and with what parameters (query, course_name, lesson_number). This means the search query sent to ChromaDB is Claude's own reformulation, not the raw user input.

- **ChromaDB collections**: Two separate collections — `course_catalog` (one doc per course, used for fuzzy course name resolution) and `course_content` (chunked lesson text, used for semantic search). Course name filtering works by first doing a semantic search in `course_catalog` to resolve a partial name to a full title, then filtering `course_content` by that title.

- **Startup document loading**: `app.py` loads all `.txt/.pdf/.docx` files from `../docs` on startup. Already-indexed courses (matched by title) are skipped. The ChromaDB data persists in `backend/chroma_db/` between restarts.

- **Session history**: Conversation history is stored in-memory in `SessionManager` (not persisted). Only the last `MAX_HISTORY=2` exchanges are kept, appended to the system prompt as a string.

## Backend Module Responsibilities

| File | Responsibility |
|------|---------------|
| `app.py` | FastAPI routes, startup doc loading, static file serving |
| `rag_system.py` | Orchestrator — wires all components together |
| `ai_generator.py` | Anthropic API calls, tool execution loop |
| `search_tools.py` | Tool definition + execution, source tracking |
| `vector_store.py` | ChromaDB interface, course name resolution, search |
| `document_processor.py` | File parsing, lesson extraction, text chunking |
| `session_manager.py` | In-memory conversation history per session |
| `config.py` | All tunable parameters (chunk size, model, max results, etc.) |
| `models.py` | Pydantic models: `Course`, `Lesson`, `CourseChunk` |

## Course Document Format

Files in `docs/` must follow this structure for the parser to extract lessons correctly:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <lesson title>
Lesson Link: <url>
<lesson content...>

Lesson 1: <lesson title>
...
```

## Key Configuration (backend/config.py)

- `ANTHROPIC_MODEL`: `claude-sonnet-4-20250514`
- `EMBEDDING_MODEL`: `all-MiniLM-L6-v2`
- `CHUNK_SIZE`: 800 chars, `CHUNK_OVERLAP`: 100 chars
- `MAX_RESULTS`: 5 search results returned to Claude
- `MAX_HISTORY`: 2 conversation exchanges retained
- `CHROMA_PATH`: `./chroma_db` (relative to `backend/`)
