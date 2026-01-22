"""
langgraph_orchestrator_s3.py - LangGraph Orchestrator for S3-based RAG

Updated to work with S3 vector storage instead of OpenSearch.
API remains identical - just uses different storage backend.
"""

import os
import logging
from typing import Dict, List, TypedDict, Annotated
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
    1. Content Recommendation: Find → Analyze → Recommend
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
    # Recommendation Workflow Nodes
    # ========================================================================
    
    def search_content_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 1: Search for matching content using S3 RAG
        """
        logger.info(f"🔍 STEP 1: Searching for {state['user_level']} content...")
        
        try:
            if state.get('anime_filter'):
                # Search within specific anime
                episodes = self.rag.search_by_anime(
                    state['anime_filter'],
                    level=state['user_level']
                )
            else:
                # General search
                episodes = self.rag.search_episodes_by_level(
                    level=state['user_level'],
                    query=state.get('query', ''),
                    n_results=state.get('n_results', 10)
                )
            
            state['matched_episodes'] = episodes
            state['step'] = 'search_complete'
            
            logger.info(f"   Found {len(episodes)} matching episodes")
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            state['errors'] = state.get('errors', [])
            state['errors'].append(f"Search error: {str(e)}")
            state['matched_episodes'] = []
        
        return state
    
    def select_best_episode_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 2: Select the best episode from matches
        """
        logger.info("🎯 STEP 2: Selecting best episode...")
        
        episodes = state['matched_episodes']
        
        if not episodes:
            state['step'] = 'no_results'
            return state
        
        # Simple selection: highest relevance score
        # Could be enhanced with more complex logic
        best_episode = episodes[0]
        state['selected_episode'] = best_episode
        state['step'] = 'episode_selected'
        
        logger.info(f"   Selected: {best_episode['title']}")
        
        return state
    
    def generate_recommendation_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 3: Generate personalized recommendation text
        """
        logger.info("✍️  STEP 3: Generating recommendation...")
        
        try:
            recommendation = self.generator.generate_recommendation(
                episodes=state['matched_episodes'][:3],
                user_level=state['user_level'],
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
        """
        logger.info(f"📥 STEP 1: Fetching episode {state['episode_id']}...")
        
        try:
            # Search for the specific episode
            episodes = self.rag.search_episodes_by_level(
                level=state['user_level'],
                query=state['episode_id'],
                n_results=20
            )
            
            # Find exact match
            episode = None
            for ep in episodes:
                if ep['episode_id'] == state['episode_id']:
                    episode = ep
                    break
            
            if not episode:
                raise ValueError(f"Episode not found: {state['episode_id']}")
            
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
            vocab = self.generator.generate_vocabulary_list(
                episode_examples=state['vocabulary_examples'],
                episode_title=state['selected_episode']['title']
            )
            
            state['vocabulary_list'] = vocab
            state['step'] = 'vocabulary_generated'
            
        except Exception as e:
            logger.error(f"❌ Vocabulary generation failed: {e}")
            state['errors'] = state.get('errors', [])
            state['errors'].append(f"Vocabulary error: {str(e)}")
            state['vocabulary_list'] = "Vocabulary generation failed."
        
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
        """
        logger.info("🎎 STEP 5: Generating cultural notes...")
        
        try:
            cultural = self.generator.generate_cultural_notes(
                episode_title=state['selected_episode']['title'],
                episode_level=state['selected_episode']['level']
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
        
        Workflow: Search → Select → Recommend
        """
        logger.info("="*60)
        logger.info("🚀 Starting Recommendation Workflow (S3 Backend)")
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
    from rag_engine_s3 import RAGEngineS3
    from learning_generator_v2 import LearningGeneratorV2
    
    logger.info("="*60)
    logger.info("🧪 Testing S3-Based LangGraph Orchestrator")
    logger.info("="*60)
    
    # Initialize components
    rag = RAGEngineS3()
    generator = LearningGeneratorV2()
    orchestrator = LangGraphOrchestrator(rag, generator)
    
    # Test recommendation workflow
    logger.info("\n📋 Test 1: Recommendation Workflow")
    result = orchestrator.execute_recommendation_workflow(
        user_level="N3",
        query="action anime",
        n_results=3
    )
    
    logger.info(f"\nResult: {result.get('recommendation_text', 'No recommendation')}")
    logger.info(f"Episodes found: {len(result.get('matched_episodes', []))}")
    
    logger.info("\n✅ Orchestrator test complete!")


if __name__ == "__main__":
    test_orchestrator()