"""
api_v3.py - FastAPI Backend for Stage 2

New features:
- LangGraph orchestration for multi-step reasoning
- Enhanced learning packages
- Pre-watch preparation
- CloudWatch logging
- AWS service integration
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import os
import logging

from rag_engine_v3 import RAGEngineV3
from learning_generator_v2 import LearningGeneratorV2
from langgraph_orchestrator import LangGraphOrchestrator

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def setup_cloudwatch_logging():
    """Setup CloudWatch logging for API"""
    try:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.info("Logging initialized for API V3")
    except Exception as e:
        print(f"Logging setup error: {e}")

setup_cloudwatch_logging()

# Initialize FastAPI app
app = FastAPI(
    title="LinguaSync API V3",
    description="AI-powered Japanese learning with LangGraph orchestration",
    version="3.0.0 (Stage 2)"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (lazy loading)
rag_engine = None
learning_generator = None
orchestrator = None

def get_components():
    """Lazy initialization of all components"""
    global rag_engine, learning_generator, orchestrator
    
    if rag_engine is None:
        logger.info("🔧 Initializing RAG Engine V3...")
        rag_engine = RAGEngineV3()
    
    if learning_generator is None:
        logger.info("🔧 Initializing Learning Generator V2...")
        learning_generator = LearningGeneratorV2()
    
    if orchestrator is None:
        logger.info("🔧 Initializing LangGraph Orchestrator...")
        orchestrator = LangGraphOrchestrator(rag_engine, learning_generator)
    
    return rag_engine, learning_generator, orchestrator


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


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check and API information"""
    return {
        "status": "online",
        "service": "LinguaSync API V3",
        "version": "3.0.0",
        "stage": "2 - Enhanced Learning Features",
        "features": [
            "LangGraph multi-step reasoning",
            "Enhanced grammar explanations",
            "Cultural context generation",
            "Pre-watch vocabulary preparation",
            "Amazon OpenSearch Serverless",
            "CloudWatch logging"
        ]
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        rag, _, _ = get_components()
        stats = rag.get_collection_stats()
        
        return {
            "status": "healthy",
            "database": "connected",
            "episodes_indexed": stats.get('episode_count', 0),
            "anime_count": stats.get('anime_count', 0),
            "opensearch": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get comprehensive statistics about the content library"""
    try:
        rag, _, _ = get_components()
        stats = rag.get_collection_stats()
        
        return {
            "success": True,
            "stats": stats,
            "message": "Content library statistics"
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/anime")
async def list_anime():
    """List all available anime series"""
    try:
        rag, _, _ = get_components()
        stats = rag.get_collection_stats()
        
        anime_list = stats.get('anime_list', [])
        anime_list.sort(key=lambda x: x['name'])
        
        return {
            "success": True,
            "anime": anime_list,
            "total_count": len(anime_list)
        }
    except Exception as e:
        logger.error(f"Error listing anime: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing anime: {str(e)}")

@app.get("/anime/{anime_name}")
async def get_anime_episodes(
    anime_name: str,
    level: Optional[str] = Query(None, description="Filter by JLPT level")
):
    """Get all episodes of a specific anime"""
    try:
        rag, _, _ = get_components()
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
        logger.error(f"Error fetching anime: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching anime: {str(e)}")

@app.post("/recommend")
async def recommend_content(request: RecommendationRequest):
    """
    Recommend content using LangGraph orchestration
    
    NEW: Uses multi-step reasoning for better recommendations
    """
    try:
        logger.info(f"📥 Recommendation request: Level={request.user_level}, Query='{request.query}'")
        
        _, _, orchestrator_instance = get_components()
        
        # Execute LangGraph workflow
        workflow_result = orchestrator_instance.execute_recommendation_workflow(
            user_level=request.user_level,
            query=request.query,
            anime_filter=request.anime_filter,
            n_results=request.n_results
        )
        
        # Check for errors
        if workflow_result.get('errors'):
            logger.warning(f"Workflow errors: {workflow_result['errors']}")
        
        if not workflow_result.get('matched_episodes'):
            return {
                "success": False,
                "message": f"No content found for {request.user_level} level. Try different criteria!"
            }
        
        logger.info(f"✅ Found {len(workflow_result['matched_episodes'])} episodes")
        
        return {
            "success": True,
            "recommendation": workflow_result['recommendation_text'],
            "episodes": workflow_result['matched_episodes'][:request.n_results],
            "selected_episode": workflow_result.get('selected_episode'),
            "user_level": request.user_level,
            "total_found": len(workflow_result['matched_episodes']),
            "workflow_step": workflow_result.get('step', 'unknown')
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")

@app.post("/learning-package")
async def get_learning_package(request: LearningPackageRequest):
    """
    Generate a complete learning package using LangGraph orchestration
    
    NEW: Enhanced with pre-watch prep and better grammar explanations
    """
    try:
        logger.info(f"📥 Learning package request: {request.episode_id}")
        
        _, _, orchestrator_instance = get_components()
        
        # Execute learning package workflow
        workflow_result = orchestrator_instance.execute_learning_package_workflow(
            episode_id=request.episode_id,
            user_level=request.user_level
        )
        
        episode = workflow_result['selected_episode']
        
        # Build complete package
        package = {
            'episode_id': request.episode_id,
            'title': episode['title'],
            'anime_name': episode['anime_name'],
            'level': episode['level'],
            'user_level': request.user_level,
            'vocabulary': workflow_result['vocabulary_list'],
            'grammar': workflow_result['grammar_notes'],
            'cultural_notes': workflow_result['cultural_context'],
            'pre_watch_prep': workflow_result['pre_watch_prep'],
            'stats': {
                'total_lines': episode['total_lines'],
                'vocab_count': episode['vocab_count'],
                'level_match': episode['level'] == request.user_level
            },
            'workflow_step': workflow_result.get('step', 'unknown')
        }
        
        logger.info(f"✅ Learning package generated for {episode['title']}")
        
        return {
            "success": True,
            "package": package
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating learning package: {str(e)}")

@app.get("/levels")
async def get_levels():
    """Get available JLPT levels with descriptions"""
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
    """Flexible search endpoint for content discovery"""
    try:
        rag, _, _ = get_components()
        
        if anime:
            episodes = rag.search_by_anime(anime, level=level)
        elif level:
            episodes = rag.search_episodes_by_level(
                level=level,
                query=query,
                n_results=limit
            )
        else:
            episodes = rag.search_episodes_by_level(
                level="N3",
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
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("="*60)
    logger.info("🚀 LinguaSync API V3 Starting...")
    logger.info("="*60)
    
    # Check environment variables
    required_vars = ['OPENAI_API_KEY', 'OPENSEARCH_ENDPOINT', 'AWS_REGION']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"⚠️  Missing environment variables: {missing_vars}")
    
    # Initialize components
    try:
        get_components()
        logger.info("✅ All components initialized")
    except Exception as e:
        logger.error(f"❌ Component initialization failed: {e}")
    
    logger.info("\n✅ API V3 Ready!")
    logger.info("📚 Docs: http://localhost:8000/docs")
    logger.info("🔍 Health: http://localhost:8000/health")
    logger.info("="*60)


# ============================================================================
# Run with: uvicorn api_v3:app --reload --port 8000
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)