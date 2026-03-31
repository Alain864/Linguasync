# LinguaSync Stage 0

LinguaSync is a prototype Japanese-learning app that recommends anime episodes based on JLPT level and generates study material from subtitle data. The current implementation uses a Streamlit frontend, a FastAPI backend, LangGraph orchestration, OpenAI for generation, and an S3-backed FAISS vector store for retrieval.

## Architecture

```text
User Query
  -> Streamlit UI (frontend/app.py)
  -> FastAPI API (backend/api.py)
  -> LangGraph Orchestrator (backend/orchestration/langgraph_orchestrator.py)
  -> RAG Engine (backend/rag/engine.py)
  -> S3 + FAISS + OpenAI
```

## Project Structure

```text
linguasync-stage0/
├── README.md
├── .env
├── .env.example
├── docker-compose.yml
├── backend/
│   ├── api.py
│   ├── generation/
│   │   └── learning_generator.py
│   ├── orchestration/
│   │   └── langgraph_orchestrator.py
│   └── rag/
│       └── engine.py
├── data/
│   ├── processed/
│   │   └── episodes.json
│   └── subtitles/
├── docker/
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
├── frontend/
│   └── app.py
├── pipeline/
│   ├── add_new_episodes.py
│   └── subtitle_processor.py
├── requirements/
│   └── backend.txt
├── scripts/
│   └── debug_index.py
└── tests/
```

## Main Features

- Recommend anime episodes by JLPT level (`N5` to `N1`) or `All Levels`
- Detect level hints directly from user queries
- Search by anime title or free-text preference
- Generate recommendation text, vocabulary help, grammar notes, cultural notes, and pre-watch prep
- Store and search embeddings with FAISS while persisting index artifacts in S3

## Environment Variables

Create a `.env` file in the project root. You can start from `.env.example`.

Required values:

```env
OPENAI_API_KEY=your_openai_api_key
S3_BUCKET_NAME=your_s3_bucket_name
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
```

Optional:

```env
AWS_SESSION_TOKEN=your_aws_session_token
```

## Local Development

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements/backend.txt
```

### 2. Start the backend

```bash
uvicorn backend.api:app --reload --port 8000
```

### 3. Start the frontend

```bash
streamlit run frontend/app.py
```

### 4. Open the app

- Frontend: `http://localhost:8501`
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

## Docker Compose

The compose setup builds:

- Backend from `docker/Dockerfile.backend`
- Frontend from `docker/Dockerfile.frontend`

Run:

```bash
docker compose up --build
```

Stop:

```bash
docker compose down
```

Open:

- Frontend: `http://localhost:8501`
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

## API Endpoints

- `GET /health` - service and storage health
- `GET /stats` - content library statistics
- `GET /anime` - available anime series
- `GET /anime/{anime_name}` - episodes for one series
- `GET /levels` - JLPT level metadata
- `GET /search` - flexible search endpoint
- `POST /recommend` - recommendation workflow
- `POST /learning-package` - full learning package for one episode

## Data Pipeline

- `pipeline/subtitle_processor.py` processes subtitle files and writes structured episode data to `data/processed/episodes.json`
- `backend/rag/engine.py` creates embeddings, builds a FAISS index, and syncs index artifacts to S3
- `pipeline/add_new_episodes.py` updates an existing S3-backed index with additional episodes
- `scripts/debug_index.py` helps inspect the stored index and metadata

## Utility Commands

Process subtitle data:

```bash
python -m pipeline.subtitle_processor
```

Add newly uploaded episodes to the S3-backed index:

```bash
python -m pipeline.add_new_episodes
```

Inspect the vector store:

```bash
python -m scripts.debug_index
```

## Current Dataset

The checked-in processed dataset is stored in `data/processed/episodes.json`. It contains anime episode metadata, subtitle timing, vocabulary extraction, and JLPT-level estimates for the indexed episodes currently used by the app.

## Troubleshooting

Problem: API container starts but `/health` fails
- Check that `OPENAI_API_KEY`, `S3_BUCKET_NAME`, and `AWS_REGION` are set
- Confirm the configured AWS credentials can access the S3 bucket

Problem: Frontend loads but cannot reach backend
- Verify the `api` service is healthy
- Confirm the frontend is using `API_BASE_URL=http://api:8000` inside Docker

Problem: No recommendations are returned
- Confirm the S3 bucket contains the FAISS index and metadata files
- Check logs from the `api` service for retrieval or OpenAI errors
