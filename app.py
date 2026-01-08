"""
Streamlit Frontend for LinguaSync

This module:
1. Provides user interface for content discovery
2. Displays recommendations and learning materials
3. Allows users to explore Japanese learning content
"""

import streamlit as st
import requests
from typing import Dict, List

# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="LinguaSync - Japanese Learning",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .episode-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    .stat-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #e8f4f8;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# API Helper Functions
# ============================================================================

def get_api_stats() -> Dict:
    """Get content library statistics from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_levels() -> List[Dict]:
    """Get available JLPT levels"""
    try:
        response = requests.get(f"{API_BASE_URL}/levels")
        if response.status_code == 200:
            return response.json()['levels']
        return []
    except:
        return []

def get_recommendations(level: str, query: str = "", n_results: int = 3) -> Dict:
    """Get content recommendations"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend",
            json={
                "user_level": level,
                "query": query,
                "n_results": n_results
            }
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None

def get_learning_package(episode_id: str, user_level: str) -> Dict:
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
    st.markdown('<div class="main-header">🎌 LinguaSync</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Japanese Learning Content Matcher</div>',
        unsafe_allow_html=True
    )

def render_stats_sidebar():
    """Render statistics in sidebar"""
    st.sidebar.title("📊 Content Library")
    
    stats = get_api_stats()
    if stats and stats.get('success'):
        stats_data = stats['stats']
        
        st.sidebar.metric("Total Episodes", f"~{stats_data['estimated_episodes']}")
        st.sidebar.metric("Indexed Lines", f"~{stats_data['estimated_lines']}")
        
        st.sidebar.subheader("Level Distribution")
        level_dist = stats_data.get('level_distribution', {})
        for level, count in sorted(level_dist.items()):
            if count > 0:
                st.sidebar.text(f"{level}: {count} items")
    else:
        st.sidebar.warning("⚠️ Content library not initialized")
        st.sidebar.info("Run the setup scripts first!")

def render_episode_card(episode: Dict, user_level: str):
    """Render an episode card with details"""
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(episode['title'])
            st.caption(f"Episode ID: {episode['episode_id']}")
        
        with col2:
            # Level badge
            level_color = {
                'N5': '🟢', 'N4': '🔵', 
                'N3': '🟡', 'N2': '🟠', 'N1': '🔴'
            }
            st.markdown(f"### {level_color.get(episode['level'], '⚪')} {episode['level']}")
        
        # Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Dialogue Lines", episode['total_lines'])
        col2.metric("Unique Vocabulary", episode['vocab_count'])
        col3.metric("Match Score", f"{episode['relevance_score']:.2f}")
        
        # Action button
        if st.button(f"📚 Get Learning Package", key=f"btn_{episode['episode_id']}"):
            st.session_state.selected_episode = episode['episode_id']
            st.session_state.show_package = True
            st.rerun()

def render_learning_package(package_data: Dict):
    """Render complete learning package"""
    
    package = package_data['package']
    
    st.markdown("---")
    st.title(f"📖 Learning Package: {package['title']}")
    
    # Episode info
    col1, col2, col3 = st.columns(3)
    col1.metric("Episode Level", package['level'])
    col2.metric("Your Level", package['user_level'])
    col3.metric("Total Lines", package['stats']['total_lines'])
    
    # Tabs for different sections
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
    
    # Back button
    if st.button("⬅️ Back to Recommendations"):
        st.session_state.show_package = False
        st.rerun()


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
    
    # Render header
    render_header()
    
    # Render sidebar stats
    render_stats_sidebar()
    
    # Check if showing learning package
    if st.session_state.show_package and st.session_state.selected_episode:
        # Show learning package view
        with st.spinner("🎓 Generating your personalized learning package..."):
            package = get_learning_package(
                st.session_state.selected_episode,
                st.session_state.user_level
            )
            
            if package and package.get('success'):
                render_learning_package(package)
            else:
                st.error("Failed to generate learning package")
                st.session_state.show_package = False
        
        return
    
    # Main content area
    st.markdown("---")
    
    # User input section
    st.subheader("🎯 Find Your Perfect Content")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Level selection
        levels = get_levels()
        if levels:
            level_options = [f"{l['code']} - {l['description']}" for l in levels]
            selected_level_display = st.selectbox(
                "Your Japanese Level",
                level_options,
                index=2  # Default to N3
            )
            selected_level = selected_level_display.split(" - ")[0]
            st.session_state.user_level = selected_level
        else:
            st.error("Cannot load levels from API")
            return
    
    with col2:
        # Search query
        search_query = st.text_input(
            "What are you interested in? (optional)",
            placeholder="e.g., 'action anime', 'daily conversations', 'mystery'..."
        )
    
    # Search button
    if st.button("🔍 Find Content", type="primary"):
        with st.spinner("🔍 Searching for perfect matches..."):
            results = get_recommendations(
                level=selected_level,
                query=search_query,
                n_results=3
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
        st.subheader("📺 Available Episodes")
        
        # Display episodes
        for episode in results['episodes']:
            render_episode_card(episode, selected_level)
            st.markdown("---")
    
    else:
        # Welcome message
        st.info("""
        👋 **Welcome to LinguaSync!**
        
        1. Select your Japanese level
        2. Optionally add what you're interested in
        3. Click "Find Content" to get personalized recommendations
        4. Get vocabulary lists, grammar notes, and cultural context!
        """)


# ============================================================================
# Run the app
# ============================================================================

if __name__ == "__main__":
    main()