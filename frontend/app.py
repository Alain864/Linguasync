"""
Streamlit Frontend V2 for SemanticTutor - Stage 1

New features over Stage 0:
1. Anime library browser
2. Season/episode navigation
3. Filter by anime series
4. Enhanced episode cards with more metadata
5. Better organization and UX
"""

import streamlit as st
import requests
from typing import Dict, List, Optional

# ============================================================================
# Configuration
# ============================================================================

import os

# API Base URL - read from environment variable for Docker/AWS deployment
# In Docker Compose: API_BASE_URL=http://api:8000
# In local dev: API_BASE_URL=http://localhost:8000
API_BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:8000')

# Page configuration
st.set_page_config(
    page_title="SemanticTutor - Japanese Learning",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .version-badge {
        text-align: center;
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    .episode-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    .anime-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        background-color: #e8f4f8;
        font-size: 0.9rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# API Helper Functions
# ============================================================================

def check_api_health() -> bool:
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_api_stats() -> Optional[Dict]:
    """Get content library statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_anime_list() -> List[Dict]:
    """Get list of all anime in the library"""
    try:
        response = requests.get(f"{API_BASE_URL}/anime")
        if response.status_code == 200:
            return response.json()['anime']
        return []
    except:
        return []

def get_anime_episodes(anime_name: str, level: Optional[str] = None) -> List[Dict]:
    """Get all episodes of a specific anime"""
    try:
        url = f"{API_BASE_URL}/anime/{anime_name}"
        params = {}
        if level:
            params['level'] = level
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()['episodes']
        return []
    except:
        return []

def get_levels() -> List[Dict]:
    """Get available JLPT levels"""
    try:
        response = requests.get(f"{API_BASE_URL}/levels")
        if response.status_code == 200:
            return response.json()['levels']
        return []
    except:
        return []

def get_recommendations(level: str, query: str = "", anime_filter: str = None, n_results: int = 3) -> Optional[Dict]:
    """Get content recommendations"""
    try:
        payload = {
            "user_level": level,
            "query": query,
            "n_results": n_results
        }
        if anime_filter:
            payload["anime_filter"] = anime_filter
        
        response = requests.post(f"{API_BASE_URL}/recommend", json=payload)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None

def get_learning_package(episode_id: str, user_level: str) -> Optional[Dict]:
    """Get complete learning package for an episode"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/learning-package",
            json={
                "episode_id": episode_id,
                "user_level": user_level
            }
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error getting learning package: {e}")
        return None


# ============================================================================
# UI Components
# ============================================================================

def render_header():
    """Render main header"""
    st.markdown('<div class="main-header">🎌 SemanticTutor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Japanese Learning Content Matcher</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="version-badge">Stage 1 - Production Ready Core</div>',
        unsafe_allow_html=True
    )

def render_sidebar():
    """Render sidebar with stats and navigation"""
    st.sidebar.title("📖 Content Library")
    
    # API Health Check
    if check_api_health():
        st.sidebar.success("🟢 API Connected")
    else:
        st.sidebar.error("🔴 API Offline")
        st.sidebar.info("Run: uvicorn backend.api:app --reload")
        return
    
    # Statistics
    stats = get_api_stats()
    if stats and stats.get('success'):
        stats_data = stats['stats']
        
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Anime Series", stats_data.get('unique_anime', 0))
        col2.metric("Episodes", stats_data.get('episode_count', 0))
        
        st.sidebar.metric("Dialogue Lines", stats_data.get('line_count', 0))
        
        # Level distribution
        st.sidebar.subheader("📈 Level Distribution")
        level_dist = stats_data.get('level_distribution', {})
        for level in ['N5', 'N4', 'N3', 'N2', 'N1']:
            if level in level_dist:
                count = level_dist[level]
                st.sidebar.progress(count / max(level_dist.values()) if level_dist.values() else 0, 
                                   text=f"{level}: {count}")
        
        # Anime list
        st.sidebar.subheader("📚 Available Anime")
        anime_list = stats_data.get('anime_list', [])
        if anime_list:
            for anime in anime_list[:10]:  # Show first 10
                st.sidebar.text(f"• {anime['name']} ({anime['episodes']} eps)")
        
    else:
        st.sidebar.warning("⚠️ Content library not initialized")
    
    # Copyright footer at the bottom
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="text-align: center; font-size: 0.8rem; color: #888;">© 2026 Alain Cl</div>',
        unsafe_allow_html=True
    )

def render_episode_card(episode: Dict, user_level: str, show_anime_badge: bool = True):
    """Render an episode card with enhanced metadata"""
    
    with st.container():
        # Header with anime badge
        if show_anime_badge:
            st.markdown(
                f'<span class="anime-badge">{episode.get("anime_name", "Unknown")}</span>',
                unsafe_allow_html=True
            )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### {episode.get('title', 'Untitled')}")
            
            # Episode metadata
            metadata_parts = []
            if episode.get('season'):
                metadata_parts.append(f"S{episode['season']}")
            if episode.get('episode_number'):
                metadata_parts.append(f"E{episode['episode_number']}")
            if metadata_parts:
                st.caption(" • ".join(metadata_parts))
        
        with col2:
            # Level badge
            level = episode.get('level', 'N/A')
            level_color = {
                'N5': '🟢', 'N4': '🔵', 'N3': '🟡', 'N2': '🟠', 'N1': '🔴'
            }.get(level, '⚪')
            st.markdown(f"## {level_color} {level}")
        
        # Episode stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Lines", episode.get('total_lines', 0))
        col2.metric("Vocabulary", episode.get('vocab_count', 0))
        if episode.get('duration_minutes'):
            col3.metric("Duration", f"{episode['duration_minutes']}m")
        elif episode.get('relevance_score'):
            col3.metric("Match", f"{episode['relevance_score']:.1%}")
        
        # Action button
        if st.button(
            "📚 Get Learning Package", 
            key=f"learn_{episode.get('episode_id', 'unknown')}",
            use_container_width=True
        ):
            st.session_state.selected_episode = episode['episode_id']
            st.session_state.show_package = True
            st.rerun()

def render_learning_package(package_data: Dict):
    """Render complete learning package"""
    
    package = package_data['package']
    
    st.markdown("---")
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(f"📖 {package['anime_name']}")
        st.subheader(package['title'])
    with col2:
        if st.button("⬅️ Back"):
            st.session_state.show_package = False
            st.rerun()
    
    # Episode info
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Episode Level", package['level'])
    col2.metric("Your Level", package['user_level'])
    col3.metric("Total Lines", package['stats']['total_lines'])
    if package.get('duration_minutes'):
        col4.metric("Duration", f"{package['duration_minutes']}m")
    
    # Learning materials in tabs
    tab1, tab2, tab3 = st.tabs(["📝 Vocabulary", "📚 Grammar", "🎌 Cultural Context"])
    
    with tab1:
        st.markdown("### Essential Vocabulary")
        st.markdown(package['vocabulary'])
    
    with tab2:
        st.markdown("### Grammar Patterns")
        st.markdown(package['grammar'])
    
    with tab3:
        st.markdown("### Cultural Notes")
        st.markdown(package['cultural_notes'])

def render_anime_browser():
    """Render anime library browser"""
    st.subheader("📚 Anime Library Browser")
    
    anime_list = get_anime_list()
    
    if not anime_list:
        st.warning("No anime found in library. Add subtitle files and run the processors!")
        return
    
    # Select anime
    anime_names = [anime['name'] for anime in anime_list]
    selected_anime = st.selectbox(
        "Select an anime series",
        anime_names,
        key="anime_browser_select"
    )
    
    if selected_anime:
        # Get episodes for selected anime
        episodes = get_anime_episodes(selected_anime)
        
        if episodes:
            st.info(f"**{selected_anime}** has {len(episodes)} episodes")
            
            # Display episodes
            for episode in episodes:
                render_episode_card(
                    episode, 
                    st.session_state.get('user_level', 'N3'),
                    show_anime_badge=False  # Don't show badge since we're filtering by anime
                )
                st.markdown("---")
        else:
            st.warning(f"No episodes found for {selected_anime}")


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main application"""
    
    # Initialize session state
    if 'show_package' not in st.session_state:
        st.session_state.show_package = False
    if 'selected_episode' not in st.session_state:
        st.session_state.selected_episode = None
    if 'user_level' not in st.session_state:
        st.session_state.user_level = 'All Levels'
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'search'  # 'search' or 'browse'
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Check if showing learning package
    if st.session_state.show_package and st.session_state.selected_episode:
        with st.spinner("🎓 Generating your personalized learning package..."):
            # Use 'N3' as default if All Levels is selected
            level_for_package = st.session_state.user_level if st.session_state.user_level != 'All Levels' else 'N3'
            package = get_learning_package(
                st.session_state.selected_episode,
                level_for_package
            )
            
            if package and package.get('success'):
                render_learning_package(package)
            else:
                st.error("Failed to generate learning package")
                st.session_state.show_package = False
        
        return
    
    # Main content area
    st.markdown("---")
    
    # View mode selector
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🔍 Search Mode", use_container_width=True, 
                     type="primary" if st.session_state.view_mode == 'search' else "secondary"):
            st.session_state.view_mode = 'search'
            st.rerun()
    with col2:
        if st.button("📚 Browse Anime", use_container_width=True,
                     type="primary" if st.session_state.view_mode == 'browse' else "secondary"):
            st.session_state.view_mode = 'browse'
            st.rerun()
    
    st.markdown("---")
    
    # Render based on view mode
    if st.session_state.view_mode == 'browse':
        render_anime_browser()
    else:
        # Search Mode
        st.subheader("🎯 Find Your Perfect Content")
        
        # Search query (moved to top)
        search_query = st.text_input(
            "What are you interested in?",
            placeholder="e.g., 'action', 'daily conversations', 'mystery', 'N5 level anime'..."
        )
        
        # Options row - now below the search query
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            # Level selection with "All Levels" as default
            levels = get_levels()
            if levels:
                level_options = ["All Levels"] + [f"{l['code']} - {l['description']}" for l in levels]
                selected_level_display = st.selectbox(
                    "Your Japanese Level",
                    level_options,
                    index=0  # Default to "All Levels"
                )
                if selected_level_display == "All Levels":
                    selected_level = "All Levels"
                else:
                    selected_level = selected_level_display.split(" - ")[0]
                st.session_state.user_level = selected_level
            else:
                st.error("Cannot load levels from API")
                return
        
        with col2:
            # Anime filter (optional)
            anime_list = get_anime_list()
            anime_options = ["All Anime"] + [anime['name'] for anime in anime_list]
            selected_anime_filter = st.selectbox(
                "Filter by Anime",
                anime_options
            )
            anime_filter = None if selected_anime_filter == "All Anime" else selected_anime_filter
        
        with col3:
            # Number of results (1-3)
            n_results = st.selectbox(
                "Number of Results",
                [1, 2, 3],
                index=2  # Default to 3
            )
        
        # Search button
        if st.button("🔍 Find Content", type="primary", use_container_width=True):
            with st.spinner("🔍 Searching for perfect matches..."):
                results = get_recommendations(
                    level=selected_level,
                    query=search_query,
                    anime_filter=anime_filter,
                    n_results=n_results
                )
                
                if results and results.get('success'):
                    st.session_state.results = results
                else:
                    st.error("No content found. Make sure the API is running and content is indexed!")
                    return
        
        # Display results
        if 'results' in st.session_state:
            results = st.session_state.results
            
            st.markdown("---")
            st.subheader("🎬 Recommended Content")
            
            # AI Recommendation
            st.info(f"**🤖 AI Recommendation:**\n\n{results['recommendation']}")
            
            st.markdown("---")
            st.subheader(f"📺 Top {len(results['episodes'])} Episodes")
            
            # Display episodes
            for episode in results['episodes']:
                # Use the actual user level if specified, otherwise use episode level
                display_level = st.session_state.user_level if st.session_state.user_level != 'All Levels' else episode.get('level', 'N3')
                render_episode_card(episode, display_level)
                st.markdown("---")
        
        else:
            # Welcome message
            st.info("""
            **Welcome to SemanticTutor!**
            
            **How to use:**
            1. Enter what you're interested in (themes, genres, or difficulty level like "N5")
            2. Optionally select your Japanese level or filter by anime
            3. Click "Find Content" for AI recommendations
            4. Or switch to "Browse Anime" to explore library
            
            **You can also:**
            - 🎌 Browse content by anime series
            - 📺 See season and episode information
            - 🔍 Filter search by specific anime
            - 📊 Enhanced content statistics
            """)


# ============================================================================
# Run the app
# ============================================================================

if __name__ == "__main__":
    main()
