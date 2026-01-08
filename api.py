"""
FastAPI Backend for LinguaSync

This module:
1. Provides REST API endpoints
2. Orchestrates RAG retrieval and LLM generation
3. Handles user queries
4. Returns structured learning content
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import os

from rag_engine import RAGEngine
from learning_generator import LearningGenerator

# Initialize FastAPI app
app = FastAPI(
    title="LinguaSync API",
    description="AI-powered language learning content matcher",
    version="0.1.0 (Stage 0)"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For Stage 0 only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
rag_engine = None
learning_generator = None

def get_rag_engine():
    """Lazy initialization of RAG engine"""
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine()
    return rag_engine

def get_learning_generator():
    """Lazy initialization of learning generator"""
    global learning_generator
    if learning_generator is None:
        learning_generator = LearningGenerator()
    return learning_generator


# ============================================================================
# Request/Response Models
# ============================================================================

class RecommendationRequest(BaseModel):
    """Request model for content recommendations"""
    user_level: str  # N5, N4, N3, N2, N1
    query: Optional[str] = ""  # Optional search query
    n_results: Optional[int] = 3  # Number of results

class LearningPackageRequest(BaseModel):
    """Request model for complete learning package"""
    episode_id: str
    user_level: str


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "LinguaSync API",
        "version": "0.1.0",
        "stage": "0 - Local Prototype"
    }

@app.get("/stats")
async def get_stats():
    """
    Get statistics about indexed content
    
    Returns collection statistics and available content info
    """
    try:
        rag = get_rag_engine()
        stats = rag.get_collection_stats()
        
        return {
            "success": True,
            "stats": stats,
            "message": "Content library statistics"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.post("/recommend")
async def recommend_content(request: RecommendationRequest):
    """
    Recommend content based on user level and preferences
    
    This endpoint:
    1. Uses RAG to find matching content
    2. Generates personalized recommendation with LLM
    3. Returns episode suggestions with reasoning
    """
    try:
        print(f"📥 Recommendation request: Level={request.user_level}, Query='{request.query}'")
        
        # Get RAG engine and search for matching episodes
        rag = get_rag_engine()
        episodes = rag.search_episodes_by_level(
            level=request.user_level,
            query=request.query,
            n_results=request.n_results
        )
        
        if not episodes:
            return {
                "success": False,
                "message": f"No content found for {request.user_level} level. Try adding more subtitles!"
            }
        
        # Generate personalized recommendation
        generator = get_learning_generator()
        recommendation_text = generator.generate_recommendation(
            episodes=episodes,
            user_level=request.user_level,
            user_query=request.query
        )
        
        print(f"✅ Found {len(episodes)} matching episodes")
        
        return {
            "success": True,
            "recommendation": recommendation_text,
            "episodes": episodes,
            "user_level": request.user_level
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")

@app.post("/learning-package")
async def get_learning_package(request: LearningPackageRequest):
    """
    Generate a complete learning package for an episode
    
    This endpoint:
    1. Retrieves episode data from RAG
    2. Finds vocabulary examples
    3. Generates vocabulary list, grammar notes, and cultural context
    4. Returns comprehensive learning materials
    """
    try:
        print(f"📥 Learning package request: {request.episode_id}")
        
        rag = get_rag_engine()
        generator = get_learning_generator()
        
        # Find the episode
        results = rag.collection.get(
            ids=[f"episode_{request.episode_id}"],
            include=['metadatas']
        )
        
        if not results['ids']:
            raise HTTPException(status_code=404, detail=f"Episode not found: {request.episode_id}")
        
        episode = results['metadatas'][0]
        
        # Get vocabulary examples from the episode
        examples = rag.find_vocabulary_examples(
            episode_id=request.episode_id,
            n_examples=10
        )
        
        # Generate complete learning package
        package = generator.generate_complete_learning_package(
            episode={
                'episode_id': request.episode_id,
                'title': episode['title'],
                'level': episode['level'],
                'total_lines': episode['total_lines'],
                'vocab_count': episode['vocab_count']
            },
            examples=examples,
            user_level=request.user_level
        )
        
        print(f"✅ Learning package generated for {episode['title']}")
        
        return {
            "success": True,
            "package": package
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating learning package: {str(e)}")

@app.get("/levels")
async def get_levels():
    """
    Get available JLPT levels
    
    Returns list of supported levels for the UI
    """
    return {
        "success": True,
        "levels": [
            {"code": "N5", "name": "N5 - Beginner", "description": "Basic Japanese"},
            {"code": "N4", "name": "N4 - Elementary", "description": "Elementary conversations"},
            {"code": "N3", "name": "N3 - Intermediate", "description": "Everyday Japanese"},
            {"code": "N2", "name": "N2 - Upper Intermediate", "description": "News and articles"},
            {"code": "N1", "name": "N1 - Advanced", "description": "Native-level content"}
        ]
    }


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("="*60)
    print("🚀 LinguaSync API Starting...")
    print("="*60)
    
    # Check for required files
    if not os.path.exists("chroma_db"):
        print("⚠️  Warning: Vector database not found")
        print("   Run 'python rag_engine.py' to initialize")
    
    if not os.path.exists("data/processed_episodes.json"):
        print("⚠️  Warning: Processed episodes not found")
        print("   Run 'python subtitle_processor.py' first")
    
    print("\n✅ API Ready!")
    print("📍 Docs: http://localhost:8000/docs")
    print("="*60)


# ============================================================================
# Run with: uvicorn api:app --reload --port 8000
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)