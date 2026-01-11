# LinguaSync Stage 0 - Local Prototype

## 🎯 What This Does

A working RAG system that helps language learners find appropriate Japanese content (anime/dramas) based on their level, with vocabulary lists and grammar explanations.

## 📁 Project Structure

```
linguasync-stage0/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .env                       # API keys (create this)
├── subtitle_processor.py      # Parse and analyze subtitles
├── rag_engine.py             # Vector storage and retrieval
├── learning_generator.py     # LLM content generation
├── api.py                    # FastAPI backend
├── app.py                    # Streamlit frontend
├── data/
│   └── subtitles/            # Place your SRT files here
└── chroma_db/                # Auto-created vector database
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Add Sample Subtitles

Place Japanese subtitle files (.srt format) in `data/subtitles/`:

```
data/subtitles/
├── attack_on_titan_s01e01.srt
├── your_name_movie.srt
└── steins_gate_s01e01.srt
```

### 4. Process Subtitles (One-time)

```bash
# This analyzes subtitles and creates the vector database
python subtitle_processor.py
```

### 5. Start the Backend

```bash
# In one terminal
uvicorn api:app --reload --port 8000
```

### 6. Start the Frontend

```bash
# In another terminal
streamlit run app.py
```

Visit `http://localhost:8501` to use LinguaSync!

## 🧪 Testing the System

1. **Process Sample Data**: Run `subtitle_processor.py` to analyze subtitles
2. **Query Recommendations**: Use the Streamlit interface to ask for content recommendations
3. **Test Queries**:
   - "I'm N4 level, recommend something engaging"
   - "Find content with simple dialogue"
   - "Show me vocabulary from Attack on Titan episode 1"

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

## 🚫 What's NOT in Stage 0

❌ AWS deployment
❌ User authentication
❌ Progress tracking
❌ Audio analysis
❌ Multi-language support
❌ Advanced orchestration

## 📊 Architecture Overview

```
User Query → Streamlit UI → FastAPI → RAG Engine → OpenAI
                                          ↓
                                      ChromaDB
```

## 🐛 Troubleshooting

**Problem**: "ModuleNotFoundError: No module named 'MeCab'"
- **Solution**: MeCab requires system installation. For Stage 0, we use simple character-based analysis.

**Problem**: "ChromaDB connection error"
- **Solution**: Delete `chroma_db/` folder and re-run `subtitle_processor.py`

**Problem**: "No content found"
- **Solution**: Ensure subtitle files are in `data/subtitles/` and you've run the processor

## 🎓 Next Steps

Once Stage 0 is validated:
- Stage 1: AWS deployment with OpenSearch
- Stage 2: LangGraph orchestration
- Stage 3: Content library scaling
- Stage 4: User profiles & progress
- Stage 5: Spanish language support


withouth docker
streamlit run app.py
uvicorn api:app --reload --port 8000

mit docker-compose
Access:
#      - API: http://localhost:8000
#      - Frontend: http://localhost:8501
#      - API Docs: http://localhost:8000/docs