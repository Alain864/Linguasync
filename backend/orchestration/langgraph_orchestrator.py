"""
backend/orchestration/langgraph_orchestrator.py - LangGraph Orchestrator for S3-based RAG

Updated to work with S3 vector storage instead of OpenSearch.
Enhanced with better level detection and episode matching logic.
"""

import os
import logging
import re
from typing import Dict, List, TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


class WorkflowState(TypedDict):
    """State that gets passed between workflow nodes"""
    # Input
    user_level: str
    query: str
    anime_filter: str
    episode_id: str
    n_results: int
    
    # Intermediate
    detected_level: Optional[str]  # Level detected from query
    matched_episodes: List[Dict]
    selected_episode: Dict
    vocabulary_examples: List[Dict]
    
    # Output
    recommendation_text: str
    vocabulary_list: str
    grammar_notes: str
    cultural_context: str
    pre_watch_prep: str
    
    # Workflow control
    step: str
    errors: List[str]


class LangGraphOrchestrator:
    """
    Multi-step reasoning orchestrator using LangGraph
    
    Works with S3-based RAG engine - same interface, different storage.
    
    Workflows:
    1. Content Recommendation: Detect Level → Find → Analyze → Recommend
    2. Learning Package: Fetch → Generate → Enhance
    """
    
    def __init__(self, rag_engine, learning_generator):
        """
        Initialize the orchestrator
        
        Args:
            rag_engine: RAGEngineS3 instance (S3-based)
            learning_generator: LearningGeneratorV2 instance
        """
        self.rag = rag_engine
        self.generator = learning_generator
        
        logger.info("✅ LangGraph Orchestrator initialized (S3 backend)")
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _extract_level_from_query(self, query: str) -> Optional[str]:
        """
        Extract JLPT level from user query
        
        Returns:
            Detected level (N5, N4, N3, N2, N1) or None
        """
        if not query:
            return None
        
        query_lower = query.lower()
        
        # Direct level mentions
        for level in ['n5', 'n4', 'n3', 'n2', 'n1']:
            if level in query_lower:
                return level.upper()
        
        # Common level descriptions
        level_mappings = {
            'beginner': 'N5',
            'basic': 'N5',
            'elementary': 'N4',
            'intermediate': 'N3',
            'upper intermediate': 'N2',
            'advanced': 'N1',
            'native': 'N1'
        }
        
        for keyword, level in level_mappings.items():
            if keyword in query_lower:
                return level
        
        return None
    
    def _determine_search_level(self, state: WorkflowState) -> str:
        """
        Determine which level to use for search
        
        Priority:
        1. Level detected from query
        2. User's selected level
        3. Default to N3
        """
        # First, try to extract level from query
        detected_level = self._extract_level_from_query(state.get('query', ''))
        
        if detected_level:
            logger.info(f"   🎯 Detected level from query: {detected_level}")
            return detected_level
        
        # Use user's selected level if not "All Levels"
        user_level = state.get('user_level', 'All Levels')
        if user_level != 'All Levels':
            logger.info(f"   👤 Using user selected level: {user_level}")
            return user_level
        
        # Default to N3 for broadest appeal
        logger.info(f"   🎲 No level specified, using default: N3")
        return 'N3'
    
    def _get_all_levels_episodes(self, query: str, anime_filter: Optional[str], n_results: int) -> List[Dict]:
        """
        Search across all levels when "All Levels" is selected
        """
        all_episodes = []
        
        if anime_filter:
            # Search within specific anime, no level filter
            episodes = self.rag.search_by_anime(anime_filter, level=None)
            all_episodes.extend(episodes[:n_results * 2])
        else:
            # Search across multiple levels
            for level in ['N5', 'N4', 'N3', 'N2', 'N1']:
                episodes = self.rag.search_episodes_by_level(
                    level=level,
                    query=query,
                    n_results=n_results
                )
                all_episodes.extend(episodes)
        
        # Remove duplicates and sort by relevance
        seen = set()
        unique_episodes = []
        for ep in all_episodes:
            if ep['episode_id'] not in seen:
                seen.add(ep['episode_id'])
                unique_episodes.append(ep)
        
        # Sort by relevance score
        unique_episodes.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return unique_episodes[:n_results * 3]
    
    # ========================================================================
    # Recommendation Workflow Nodes
    # ========================================================================
    
    def search_content_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 1: Search for matching content using S3 RAG
        Enhanced with level detection from query
        """
        logger.info(f"🔍 STEP 1: Searching for content...")
        
        try:
            user_level = state.get('user_level', 'All Levels')
            query = state.get('query', '')
            anime_filter = state.get('anime_filter')
            n_results = state.get('n_results', 10)
            
            # Detect level from query
            detected_level = self._extract_level_from_query(query)
            state['detected_level'] = detected_level
            
            # Determine search strategy
            if user_level == 'All Levels' and not detected_level:
                # Search across all levels
                logger.info("   📚 Searching across all levels...")
                episodes = self._get_all_levels_episodes(query, anime_filter, n_results)
            else:
                # Use specific level
                search_level = detected_level if detected_level else user_level
                
                if anime_filter:
                    # Search within specific anime
                    logger.info(f"   🎬 Searching {anime_filter} at {search_level} level...")
                    episodes = self.rag.search_by_anime(anime_filter, level=search_level)
                else:
                    # General search
                    logger.info(f"   🔍 Searching at {search_level} level...")
                    episodes = self.rag.search_episodes_by_level(
                        level=search_level,
                        query=query,
                        n_results=n_results * 2
                    )
            
            state['matched_episodes'] = episodes
            state['step'] = 'search_complete'
            
            logger.info(f"   ✅ Found {len(episodes)} matching episodes")
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            state['errors'] = state.get('errors', [])
            state['errors'].append(f"Search error: {str(e)}")
            state['matched_episodes'] = []
        
        return state
    
    def select_best_episode_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 2: Select the best episode from matches
        Enhanced to ensure the selected episode appears in results
        """
        logger.info("🎯 STEP 2: Selecting best episode...")
        
        episodes = state['matched_episodes']
        
        if not episodes:
            state['step'] = 'no_results'
            return state
        
        # Select best episode (highest relevance)
        best_episode = episodes[0]
        
        # Ensure this episode is in the top results
        # If user asked for specific anime/theme, prioritize that
        query = state.get('query', '').lower()
        
        # Check if query mentions specific anime name
        for ep in episodes[:5]:  # Check top 5
            anime_name = ep.get('anime_name', '').lower()
            # If query mentions this anime's name, prioritize it
            if anime_name and anime_name in query:
                best_episode = ep
                logger.info(f"   🎬 Prioritized {ep['anime_name']} (mentioned in query)")
                break
        
        state['selected_episode'] = best_episode
        state['step'] = 'episode_selected'
        
        logger.info(f"   ✅ Selected: {best_episode['anime_name']} - {best_episode['title']}")
        
        return state
    
    def generate_recommendation_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 3: Generate personalized recommendation text
        Uses the enhanced generator with scenario detection
        """
        logger.info("✍️  STEP 3: Generating recommendation...")
        
        try:
            # Determine effective level for recommendation context
            effective_level = state.get('detected_level') or state.get('user_level', 'N3')
            if effective_level == 'All Levels':
                effective_level = state['matched_episodes'][0].get('level', 'N3')
            
            recommendation = self.generator.generate_recommendation(
                episodes=state['matched_episodes'][:3],
                user_level=effective_level,
                user_query=state.get('query', '')
            )
            
            state['recommendation_text'] = recommendation
            state['step'] = 'recommendation_complete'
            
            logger.info("   ✅ Recommendation generated")
            
        except Exception as e:
            logger.error(f"❌ Recommendation generation failed: {e}")
            state['errors'] = state.get('errors', [])
            state['errors'].append(f"Generation error: {str(e)}")
            state['recommendation_text'] = "Unable to generate recommendation."
        
        return state
    
    # ========================================================================
    # Learning Package Workflow Nodes
    # ========================================================================
    
    def fetch_episode_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 1: Fetch episode details from S3
        Enhanced with direct lookup and better error handling
        """
        logger.info(f"📥 STEP 1: Fetching episode {state['episode_id']}...")
        
        try:
            episode_id = state['episode_id']
            
            # Try direct lookup first (more reliable)
            episode_doc = self.rag.vector_store.get_by_id(episode_id, doc_type='episode')
            
            if episode_doc:
                # Convert to episode format
                episode = {
                    'episode_id': episode_doc['episode_id'],
                    'anime_name': episode_doc['anime_name'],
                    'season': episode_doc.get('season'),
                    'episode_number': episode_doc['episode_number'],
                    'title': episode_doc['title'],
                    'level': episode_doc['level'],
                    'total_lines': episode_doc['total_lines'],
                    'vocab_count': episode_doc['vocab_count'],
                    'duration_minutes': episode_doc.get('duration', 0) // 60 if episode_doc.get('duration') else 0
                }
                
                state['selected_episode'] = episode
                state['step'] = 'episode_fetched'
                logger.info(f"   ✅ Found: {episode['title']}")
                
            else:
                # Fallback: search for the episode
                logger.info(f"   🔍 Direct lookup failed, trying search...")
                episodes = self.rag.search_episodes_by_level(
                    level=state.get('user_level', 'N3'),
                    query=episode_id,
                    n_results=50
                )
                
                # Find exact match
                episode = None
                for ep in episodes:
                    if ep['episode_id'] == episode_id:
                        episode = ep
                        break
                
                if not episode:
                    # Try partial match on title
                    for ep in episodes:
                        if episode_id.lower() in ep['title'].lower() or ep['title'].lower() in episode_id.lower():
                            episode = ep
                            logger.info(f"   ⚠️  Using partial match: {ep['title']}")
                            break
                
                if not episode:
                    raise ValueError(f"Episode not found: {episode_id}")
                
                state['selected_episode'] = episode
                state['step'] = 'episode_fetched'
                logger.info(f"   ✅ Found: {episode['title']}")
            
        except Exception as e:
            logger.error(f"❌ Episode fetch failed: {e}")
            state['errors'] = state.get('errors', [])
            state['errors'].append(f"Fetch error: {str(e)}")
        
        return state
    
    def fetch_examples_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 2: Fetch vocabulary examples from S3
        """
        logger.info("📝 STEP 2: Fetching vocabulary examples...")
        
        try:
            examples = self.rag.find_vocabulary_examples(
                episode_id=state['episode_id'],
                n_examples=15
            )
            
            state['vocabulary_examples'] = examples
            state['step'] = 'examples_fetched'
            
            logger.info(f"   Found {len(examples)} example sentences")
            
        except Exception as e:
            logger.error(f"❌ Examples fetch failed: {e}")
            state['errors'] = state.get('errors', [])
            state['errors'].append(f"Examples error: {str(e)}")
            state['vocabulary_examples'] = []
        
        return state
    
    def generate_vocabulary_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 3: Generate vocabulary list
        """
        logger.info("📚 STEP 3: Generating vocabulary list...")
        
        try:
            vocab_list = self.generator.generate_vocabulary_list(
                episode_examples=state['vocabulary_examples'],
                episode_title=state['selected_episode']['title']
            )
            
            state['vocabulary_list'] = vocab_list
            state['step'] = 'vocabulary_generated'
            
        except Exception as e:
            logger.error(f"❌ Vocabulary generation failed: {e}")
            state['errors'] = state.get('errors', [])
            state['errors'].append(f"Vocabulary error: {str(e)}")
            state['vocabulary_list'] = "Vocabulary list generation failed."
        
        return state
    
    def generate_grammar_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 4: Generate grammar notes
        """
        logger.info("📖 STEP 4: Generating grammar notes...")
        
        try:
            grammar = self.generator.generate_grammar_notes(
                episode_examples=state['vocabulary_examples'],
                episode_title=state['selected_episode']['title']
            )
            
            state['grammar_notes'] = grammar
            state['step'] = 'grammar_generated'
            
        except Exception as e:
            logger.error(f"❌ Grammar generation failed: {e}")
            state['errors'] = state.get('errors', [])
            state['errors'].append(f"Grammar error: {str(e)}")
            state['grammar_notes'] = "Grammar notes generation failed."
        
        return state
    
    def generate_cultural_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 5: Generate cultural context
        Enhanced to include episode examples for better language nuance analysis
        """
        logger.info("🎎 STEP 5: Generating cultural notes...")
        
        try:
            cultural = self.generator.generate_cultural_notes(
                episode_title=state['selected_episode']['title'],
                episode_level=state['selected_episode']['level'],
                episode_examples=state.get('vocabulary_examples', [])  # Pass actual dialogue
            )
            
            state['cultural_context'] = cultural
            state['step'] = 'cultural_generated'
            
        except Exception as e:
            logger.error(f"❌ Cultural notes failed: {e}")
            state['errors'] = state.get('errors', [])
            state['errors'].append(f"Cultural error: {str(e)}")
            state['cultural_context'] = "Cultural notes generation failed."
        
        return state
    
    def generate_prep_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 6: Generate pre-watch preparation
        """
        logger.info("🎯 STEP 6: Generating pre-watch prep...")
        
        try:
            prep = self.generator.generate_pre_watch_prep(
                episode_title=state['selected_episode']['title'],
                vocabulary_examples=state['vocabulary_examples'],
                user_level=state['user_level']
            )
            
            state['pre_watch_prep'] = prep
            state['step'] = 'learning_package_complete'
            
        except Exception as e:
            logger.error(f"❌ Prep generation failed: {e}")
            state['errors'] = state.get('errors', [])
            state['errors'].append(f"Prep error: {str(e)}")
            state['pre_watch_prep'] = "Pre-watch prep generation failed."
        
        return state
    
    # ========================================================================
    # Workflow Execution
    # ========================================================================
    
    def execute_recommendation_workflow(self,
                                         user_level: str,
                                         query: str = "",
                                         anime_filter: str = None,
                                         n_results: int = 3) -> Dict:
        """
        Execute the recommendation workflow using S3-based RAG
        
        Workflow: Detect Level → Search → Select → Recommend
        """
        logger.info("="*60)
        logger.info("🚀 Starting Recommendation Workflow (S3 Backend)")
        logger.info(f"   Level: {user_level}, Query: '{query}'")
        logger.info("="*60)
        
        # Build workflow graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("search", self.search_content_node)
        workflow.add_node("select", self.select_best_episode_node)
        workflow.add_node("recommend", self.generate_recommendation_node)
        
        # Define edges
        workflow.set_entry_point("search")
        workflow.add_edge("search", "select")
        workflow.add_edge("select", "recommend")
        workflow.add_edge("recommend", END)
        
        # Compile graph
        app = workflow.compile()
        
        # Initial state
        initial_state = {
            'user_level': user_level,
            'query': query,
            'anime_filter': anime_filter,
            'n_results': n_results,
            'detected_level': None,
            'matched_episodes': [],
            'selected_episode': {},
            'recommendation_text': "",
            'step': 'initialized',
            'errors': []
        }
        
        # Execute workflow
        try:
            result = app.invoke(initial_state)
            logger.info("="*60)
            logger.info("✅ Recommendation Workflow Complete")
            logger.info("="*60)
            return result
        
        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}")
            return {
                **initial_state,
                'errors': [str(e)],
                'step': 'workflow_failed'
            }
    
    def execute_learning_package_workflow(self,
                                           episode_id: str,
                                           user_level: str) -> Dict:
        """
        Execute the learning package workflow using S3-based RAG
        
        Workflow: Fetch → Examples → Vocab → Grammar → Cultural → Prep
        """
        logger.info("="*60)
        logger.info("🚀 Starting Learning Package Workflow (S3 Backend)")
        logger.info("="*60)
        
        # Build workflow graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("fetch_episode", self.fetch_episode_node)
        workflow.add_node("fetch_examples", self.fetch_examples_node)
        workflow.add_node("generate_vocabulary", self.generate_vocabulary_node)
        workflow.add_node("generate_grammar", self.generate_grammar_node)
        workflow.add_node("generate_cultural", self.generate_cultural_node)
        workflow.add_node("generate_prep", self.generate_prep_node)
        
        # Define edges
        workflow.set_entry_point("fetch_episode")
        workflow.add_edge("fetch_episode", "fetch_examples")
        workflow.add_edge("fetch_examples", "generate_vocabulary")
        workflow.add_edge("generate_vocabulary", "generate_grammar")
        workflow.add_edge("generate_grammar", "generate_cultural")
        workflow.add_edge("generate_cultural", "generate_prep")
        workflow.add_edge("generate_prep", END)
        
        # Compile graph
        app = workflow.compile()
        
        # Initial state
        initial_state = {
            'episode_id': episode_id,
            'user_level': user_level,
            'query': '',
            'anime_filter': None,
            'n_results': 0,
            'detected_level': None,
            'matched_episodes': [],
            'selected_episode': {},
            'vocabulary_examples': [],
            'recommendation_text': '',
            'vocabulary_list': '',
            'grammar_notes': '',
            'cultural_context': '',
            'pre_watch_prep': '',
            'step': 'initialized',
            'errors': []
        }
        
        # Execute workflow
        try:
            result = app.invoke(initial_state)
            logger.info("="*60)
            logger.info("✅ Learning Package Workflow Complete")
            logger.info("="*60)
            return result
        
        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}")
            return {
                **initial_state,
                'errors': [str(e)],
                'step': 'workflow_failed'
            }


def test_orchestrator():
    """Test the S3-based orchestrator"""
    from backend.rag.engine import RAGEngineS3
    from backend.generation.learning_generator import LearningGeneratorV2
    
    logger.info("="*60)
    logger.info("🧪 Testing S3-Based LangGraph Orchestrator")
    logger.info("="*60)
    
    # Initialize components
    rag = RAGEngineS3()
    generator = LearningGeneratorV2()
    orchestrator = LangGraphOrchestrator(rag, generator)
    
    # Test recommendation workflow with All Levels
    logger.info("\n📋 Test 1: All Levels Search")
    result = orchestrator.execute_recommendation_workflow(
        user_level="All Levels",
        query="space bounty hunters",
        n_results=3
    )
    
    logger.info(f"\nResult: {result.get('recommendation_text', 'No recommendation')}")
    logger.info(f"Episodes found: {len(result.get('matched_episodes', []))}")
    
    # Test with level in query
    logger.info("\n📋 Test 2: Level in Query")
    result = orchestrator.execute_recommendation_workflow(
        user_level="All Levels",
        query="relaxing anime about N5 level",
        n_results=3
    )
    
    logger.info(f"\nDetected level: {result.get('detected_level')}")
    logger.info(f"Result: {result.get('recommendation_text', 'No recommendation')}")
    
    logger.info("\n✅ Orchestrator test complete!")


if __name__ == "__main__":
    test_orchestrator()
