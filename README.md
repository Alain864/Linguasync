# LinguaSync Stage 0 - Local Prototype

A working RAG system that helps language learners find appropriate Japanese anime content (Chainsaw Man, Jujutsu Kaisen) based on their level, with vocabulary lists and grammar explanations. Includes LangGraph orchestration and supports both local ChromaDB and AWS OpenSearch deployments.

## 📁 Project Structure

```
linguasync-stage0/
├── README.md                          # This file
├── requirements-stage2.txt            # Python dependencies
├── .env                               # API keys (create this)
├── subtitle_processor_v3.py           # Parse and analyze subtitles
├── rag_engine_v3.py                   # Vector storage and retrieval
├── learning_generator_v2.py           # LLM content generation
├── api_v3.py                          # FastAPI backend
├── app_v2.py                          # Streamlit frontend
├── langgraph_orchestrator.py          # LangGraph orchestration
├── docker-compose.yml                 # Docker Compose for local dev
├── Dockerfile.stage2                  # Docker for Stage 2
├── deploy-stage_docker.sh             # AWS deployment script
├── data/
│   ├── processed_episodes_v3.json     # Processed episode data
│   └── subtitles/                    # SRT files
│       ├── Chainsaw Man/             # Sample anime subtitles
│       └── Jujutsu Kaisen/
└── chroma_db_v2/                      # Vector database
```

## 🚀 Quick Start

### Option 1: Local Development (No Docker)

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements-stage2.txt
```

### 2. Set Up API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Add Sample Subtitles

The project includes sample subtitles for Chainsaw Man and Jujutsu Kaisen in `data/subtitles/`.

### 4. Process Subtitles (One-time)

```bash
# This analyzes subtitles and creates the vector database
python subtitle_processor_v3.py
```

### 5. Start the Backend

```bash
# In one terminal
uvicorn api_v3:app --reload --port 8000
```

### 6. Start the Frontend

```bash
# In another terminal
streamlit run app_v2.py
```

Visit `http://localhost:8501` to use LinguaSync!

### Option 2: Docker Compose (Recommended)

```bash
# 1. Create .env file with OPENAI_API_KEY
# 2. Run Docker Compose
docker-compose up

# Access:
# - API: http://localhost:8000
# - Frontend: http://localhost:8501
# - API Docs: http://localhost:8000/docs
```

## 🧪 Testing the System

1. **Process Sample Data**: Run `python subtitle_processor_v3.py` to analyze subtitles
2. **Query Recommendations**: Use the Streamlit interface to ask for content recommendations
3. **Test Queries**:
   - "I'm N4 level, recommend something engaging"
   - "Find content with simple dialogue"
   - "Show me vocabulary from Chainsaw Man episode 1"

## 📝 Sample Subtitle Format

Expected SRT format:

```
1
00:00:01,000 --> 00:00:04,000
これは日本語の字幕です。

2
00:00:05,000 --> 00:00:08,000
次の行です。
```

## 🔧 What's Working in Stage 0

✅ Subtitle parsing (Japanese SRT files)
✅ Basic Japanese text analysis
✅ Vector embeddings with ChromaDB
✅ Content recommendation by level
✅ Vocabulary extraction
✅ Grammar pattern detection
✅ Simple web interface
✅ LangGraph orchestration (Stage 2)
✅ AWS S3 integration (S3 versions)
✅ OpenSearch support (Stage 2)

## 🚫 What's NOT in Stage 0

❌ User authentication
❌ Progress tracking
❌ Audio analysis
❌ Multi-language support (beyond Japanese)
❌ Production deployment (available in Stage 2)

## 📊 Architecture Overview

```
User Query → Streamlit UI → FastAPI → LangGraph Orchestrator → RAG Engine → OpenAI
                                          ↓
                                      ChromaDB / OpenSearch
```

## 🐛 Troubleshooting

**Problem**: "ModuleNotFoundError: No module named 'MeCab'"
- **Solution**: MeCab requires system installation. For Stage 0, we use simple character-based analysis.

**Problem**: "ChromaDB connection error"
- **Solution**: Delete `chroma_db_v2/` folder and re-run `python subtitle_processor_v3.py`

**Problem**: "No content found"
- **Solution**: Ensure subtitle files are in `data/subtitles/` and you've run the processor

## 🎓 Next Steps

- **Stage 1**: Docker deployment with persistent storage
- **Stage 2**: AWS deployment with OpenSearch and LangGraph (partially implemented)
- **Stage 3**: Content library scaling
- **Stage 4**: User profiles & progress tracking
- **Stage 5**: Multi-language support

## 🚀 Deployment

For AWS ECS deployment, see `deploy-stage_docker.sh` and the ECS task definitions.