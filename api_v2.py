"""
FastAPI Backend V2 for LinguaSync - Stage 1

New features over Stage 0:
1. Anime library browsing
2. Season/episode navigation
3. Enhanced search with anime filter
4. Richer episode metadata
5. Better error handling and validation
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import os

from rag_engine_v2 import RAGEngineV2
from learning_generator import LearningGenerator

# Initialize FastAPI app
app = FastAPI(
    title="LinguaSync API V2",
    description="AI-powered Japanese learning content matcher with anime library",
    version="2.0.0 (Stage 1)"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (lazy loading)
rag_engine = None
learning_generator = None

def get_rag_engine():
    """Lazy initialization of RAG engine V2"""
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngineV2()
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
    user_level: str = Field(..., description="JLPT level (N5, N4, N3, N2, N1)")
    query: Optional[str] = Field("", description="Optional search query")
    anime_filter: Optional[str] = Field(None, description="Filter by anime name")
    n_results: Optional[int] = Field(3, description="Number of results", ge=1, le=10)

class LearningPackageRequest(BaseModel):
    """Request model for complete learning package"""
    episode_id: str = Field(..., description="Episode ID")
    user_level: str = Field(..., description="User's JLPT level")

class AnimeSearchRequest(BaseModel):
    """Request model for anime search"""
    anime_name: str = Field(..., description="Anime name to search")
    level: Optional[str] = Field(None, description="Optional JLPT level filter")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check and API information"""
    return {
        "status": "online",
        "service": "LinguaSync API V2",
        "version": "2.0.0",
        "stage": "1 - Production Ready Core",
        "features": [
            "Anime library browsing",
            "Season/episode navigation",
            "Enhanced content search",
            "Learning package generation"
        ]
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        rag = get_rag_engine()
        stats = rag.get_collection_stats()
        
        return {
            "status": "healthy",
            "database": "connected",
            "episodes_indexed": stats.get('episode_count', 0),
            "anime_count": stats.get('anime_count', 0)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/stats")
async def get_stats():
    """
    Get comprehensive statistics about the content library
    
    Returns detailed information about anime series, episodes, and content levels
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

@app.get("/anime")
async def list_anime():
    """
    List all available anime series
    
    Returns a list of all anime in the library with episode counts and levels
    """
    try:
        rag = get_rag_engine()
        stats = rag.get_collection_stats()
        
        anime_list = stats.get('anime_list', [])
        
        # Sort by name
        anime_list.sort(key=lambda x: x['name'])
        
        return {
            "success": True,
            "anime": anime_list,
            "total_count": len(anime_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing anime: {str(e)}")

@app.get("/anime/{anime_name}")
async def get_anime_episodes(
    anime_name: str,
    level: Optional[str] = Query(None, description="Filter by JLPT level")
):
    """
    Get all episodes of a specific anime
    
    Args:
        anime_name: Name of the anime (URL encoded)
        level: Optional JLPT level filter
        
    Returns:
        List of episodes for that anime, sorted by season and episode number
    """
    try:
        rag = get_rag_engine()
        
        # Decode and search for anime
        episodes = rag.search_by_anime(anime_name, level=level)
        
        if not episodes:
            raise HTTPException(
                status_code=404, 
                detail=f"No episodes found for anime: {anime_name}"
            )
        
        return {
            "success": True,
            "anime_name": anime_name,
            "episodes": episodes,
            "total_episodes": len(episodes)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching anime: {str(e)}")

@app.post("/recommend")
async def recommend_content(request: RecommendationRequest):
    """
    Recommend content based on user level and preferences
    
    Enhanced in V2 to support anime filtering and better search
    """
    try:
        print(f"📥 Recommendation request: Level={request.user_level}, Query='{request.query}'")
        
        rag = get_rag_engine()
        
        # If anime filter is specified, search within that anime
        if request.anime_filter:
            episodes = rag.search_by_anime(request.anime_filter, level=request.user_level)
            
            # If no episodes match the level, get all episodes of that anime
            if not episodes:
                episodes = rag.search_by_anime(request.anime_filter)
        else:
            # General search by level and query
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
            episodes=episodes[:3],  # Top 3 for recommendation text
            user_level=request.user_level,
            user_query=request.query or ""
        )
        
        print(f"✅ Found {len(episodes)} matching episodes")
        
        return {
            "success": True,
            "recommendation": recommendation_text,
            "episodes": episodes[:request.n_results],
            "user_level": request.user_level,
            "total_found": len(episodes)
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")

@app.post("/learning-package")
async def get_learning_package(request: LearningPackageRequest):
    """
    Generate a complete learning package for an episode
    
    Returns vocabulary, grammar notes, and cultural context
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
            raise HTTPException(
                status_code=404, 
                detail=f"Episode not found: {request.episode_id}"
            )
        
        episode_metadata = results['metadatas'][0]
        
        # Get vocabulary examples from the episode
        examples = rag.find_vocabulary_examples(
            episode_id=request.episode_id,
            n_examples=10
        )
        
        # Generate complete learning package
        package = generator.generate_complete_learning_package(
            episode={
                'episode_id': request.episode_id,
                'anime_name': episode_metadata['anime_name'],
                'title': episode_metadata['title'],
                'level': episode_metadata['level'],
                'total_lines': episode_metadata['total_lines'],
                'vocab_count': episode_metadata['vocab_count']
            },
            examples=examples,
            user_level=request.user_level
        )
        
        # Add additional metadata
        package['season'] = episode_metadata.get('season')
        package['episode_number'] = episode_metadata.get('episode_number')
        package['duration_minutes'] = episode_metadata.get('duration', 0) // 60
        
        print(f"✅ Learning package generated for {episode_metadata['title']}")
        
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
    Get available JLPT levels with descriptions
    
    Returns list of supported levels for the UI
    """
    return {
        "success": True,
        "levels": [
            {
                "code": "N5",
                "name": "N5 - Beginner",
                "description": "Basic Japanese",
                "kanji_count": "~100 kanji",
                "vocab_count": "~800 words"
            },
            {
                "code": "N4",
                "name": "N4 - Elementary",
                "description": "Elementary conversations",
                "kanji_count": "~300 kanji",
                "vocab_count": "~1,500 words"
            },
            {
                "code": "N3",
                "name": "N3 - Intermediate",
                "description": "Everyday Japanese",
                "kanji_count": "~650 kanji",
                "vocab_count": "~3,000 words"
            },
            {
                "code": "N2",
                "name": "N2 - Upper Intermediate",
                "description": "News and articles",
                "kanji_count": "~1,000 kanji",
                "vocab_count": "~6,000 words"
            },
            {
                "code": "N1",
                "name": "N1 - Advanced",
                "description": "Native-level content",
                "kanji_count": "~2,000 kanji",
                "vocab_count": "~10,000 words"
            }
        ]
    }

@app.get("/search")
async def search_content(
    query: str = Query(..., description="Search query"),
    level: Optional[str] = Query(None, description="JLPT level filter"),
    anime: Optional[str] = Query(None, description="Anime name filter"),
    limit: int = Query(10, ge=1, le=50)
):
    """
    Flexible search endpoint for content discovery
    
    Supports semantic search across episodes with multiple filters
    """
    try:
        rag = get_rag_engine()
        
        # Build search based on filters
        if anime:
            episodes = rag.search_by_anime(anime, level=level)
        elif level:
            episodes = rag.search_episodes_by_level(
                level=level,
                query=query,
                n_results=limit
            )
        else:
            # General semantic search without level filter
            episodes = rag.search_episodes_by_level(
                level="N3",  # Default middle level
                query=query,
                n_results=limit
            )
        
        return {
            "success": True,
            "query": query,
            "filters": {
                "level": level,
                "anime": anime
            },
            "results": episodes,
            "total_found": len(episodes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("="*60)
    print("🚀 LinguaSync API V2 Starting...")
    print("="*60)
    
    # Check for required files
    if not os.path.exists("chroma_db_v2"):
        print("⚠️  Warning: Vector database not found")
        print("   Run 'python rag_engine_v2.py' to initialize")
    
    if not os.path.exists("data/processed_episodes_v2.json"):
        print("⚠️  Warning: Processed episodes not found")
        print("   Run 'python subtitle_processor_v2.py' first")
    
    print("\n✅ API V2 Ready!")
    print("📍 Docs: http://localhost:8000/docs")
    print("📍 Health: http://localhost:8000/health")
    print("="*60)


# ============================================================================
# Run with: uvicorn api_v2:app --reload --port 8000
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)