"""
langgraph_orchestrator.py - LangGraph Workflow for Stage 2

Implements multi-step reasoning for educational content generation:
1. Level Estimation
2. Content Matching
3. Vocabulary Extraction
4. Grammar Analysis
5. Cultural Context Generation
6. Response Assembly
"""

import logging
from typing import Dict, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

# Don't initialize OpenAI client at module level
# It will be initialized when needed


class WorkflowState(TypedDict):
    """State object passed between nodes in the workflow"""
    # Input
    user_level: str
    query: str
    anime_filter: str
    n_results: int
    
    # Intermediate states
    matched_episodes: List[Dict]
    selected_episode: Dict
    vocabulary_examples: List[Dict]
    
    # Outputs
    recommendation_text: str
    vocabulary_list: str
    grammar_notes: str
    cultural_context: str
    pre_watch_prep: str
    
    # Metadata
    errors: List[str]
    step: str


class LangGraphOrchestrator:
    """
    LangGraph orchestrator for educational content generation
    
    Manages the complete workflow from user query to learning package
    """
    
    def __init__(self, rag_engine, learning_generator):
        """
        Initialize orchestrator
        
        Args:
            rag_engine: RAGEngineV3 instance
            learning_generator: LearningGeneratorV2 instance
        """
        self.rag = rag_engine
        self.generator = learning_generator
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("match_content", self._match_content_node)
        workflow.add_node("select_best", self._select_best_episode_node)
        workflow.add_node("extract_examples", self._extract_examples_node)
        workflow.add_node("generate_vocabulary", self._generate_vocabulary_node)
        workflow.add_node("generate_grammar", self._generate_grammar_node)
        workflow.add_node("generate_culture", self._generate_culture_node)
        workflow.add_node("generate_prep", self._generate_prep_node)
        workflow.add_node("assemble_response", self._assemble_response_node)
        
        # Define edges (workflow flow)
        workflow.set_entry_point("match_content")
        workflow.add_edge("match_content", "select_best")
        workflow.add_edge("select_best", "extract_examples")
        workflow.add_edge("extract_examples", "generate_vocabulary")
        workflow.add_edge("generate_vocabulary", "generate_grammar")
        workflow.add_edge("generate_grammar", "generate_culture")
        workflow.add_edge("generate_culture", "generate_prep")
        workflow.add_edge("generate_prep", "assemble_response")
        workflow.add_edge("assemble_response", END)
        
        return workflow.compile()
    
    def _match_content_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 1: Match content based on user criteria
        """
        logger.info(f"🔍 Step 1: Matching content (Level: {state['user_level']})")
        
        try:
            if state.get('anime_filter'):
                episodes = self.rag.search_by_anime(
                    state['anime_filter'], 
                    level=state['user_level']
                )
            else:
                episodes = self.rag.search_episodes_by_level(
                    level=state['user_level'],
                    query=state.get('query', ''),
                    n_results=state.get('n_results', 5)
                )
            
            state['matched_episodes'] = episodes
            state['step'] = 'content_matched'
            logger.info(f"   ✅ Found {len(episodes)} matching episodes")
        
        except Exception as e:
            logger.error(f"   ❌ Error matching content: {e}")
            state['errors'] = state.get('errors', []) + [f"Content matching error: {e}"]
            state['matched_episodes'] = []
        
        return state
    
    def _select_best_episode_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 2: Select the best episode using LLM reasoning
        """
        logger.info("🎯 Step 2: Selecting best episode")
        
        if not state['matched_episodes']:
            logger.warning("   ⚠️  No episodes to select from")
            state['selected_episode'] = None
            return state
        
        try:
            # Use LLM to select and explain the best choice
            episodes_summary = "\n".join([
                f"{i+1}. {ep['title']} (Level: {ep['level']}, "
                f"{ep['total_lines']} lines, {ep['vocab_count']} words, "
                f"Score: {ep.get('relevance_score', 0)})"
                for i, ep in enumerate(state['matched_episodes'][:5])
            ])
            
            prompt = f"""You are an expert Japanese learning advisor. Select the BEST episode for this learner:

User Level: JLPT {state['user_level']}
User Query: {state.get('query', 'Looking for appropriate content')}

Available Episodes:
{episodes_summary}

Task: Select the single best episode (1-5) and explain why in 2-3 sentences.

Respond ONLY with valid JSON (no markdown, no code blocks):
{{"selected_index": 0, "reasoning": "your explanation here"}}"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            import json
            content = response.choices[0].message.content.strip()
            
            # Clean any potential markdown
            content = content.replace('```json', '').replace('```', '').strip()
            
            result = json.loads(content)
            selected_idx = result.get('selected_index', 0)
            
            # Ensure index is valid
            if selected_idx >= len(state['matched_episodes']):
                selected_idx = 0
            
            state['selected_episode'] = state['matched_episodes'][selected_idx]
            state['recommendation_text'] = result.get('reasoning', 'Selected based on level match.')
            state['step'] = 'episode_selected'
            
            logger.info(f"   ✅ Selected: {state['selected_episode']['title']}")
        
        except Exception as e:
            logger.error(f"   ❌ Error selecting episode: {e}")
            # Fallback: just pick the first one
            state['selected_episode'] = state['matched_episodes'][0]
            state['recommendation_text'] = f"Recommended {state['selected_episode']['title']} based on your {state['user_level']} level."
            state['errors'] = state.get('errors', []) + [f"Selection error: {e}"]
        
        return state
    
    def _extract_examples_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 3: Extract vocabulary examples from the episode
        """
        logger.info("📝 Step 3: Extracting vocabulary examples")
        
        if not state.get('selected_episode'):
            logger.warning("   ⚠️  No episode selected")
            state['vocabulary_examples'] = []
            return state
        
        try:
            examples = self.rag.find_vocabulary_examples(
                episode_id=state['selected_episode']['episode_id'],
                n_examples=15
            )
            
            state['vocabulary_examples'] = examples
            state['step'] = 'examples_extracted'
            logger.info(f"   ✅ Extracted {len(examples)} example lines")
        
        except Exception as e:
            logger.error(f"   ❌ Error extracting examples: {e}")
            state['vocabulary_examples'] = []
            state['errors'] = state.get('errors', []) + [f"Example extraction error: {e}"]
        
        return state
    
    def _generate_vocabulary_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 4: Generate curated vocabulary list
        """
        logger.info("📚 Step 4: Generating vocabulary list")
        
        try:
            vocab_list = self.generator.generate_vocabulary_list(
                episode_examples=state['vocabulary_examples'],
                episode_title=state['selected_episode']['title'],
                n_words=15
            )
            
            state['vocabulary_list'] = vocab_list
            state['step'] = 'vocabulary_generated'
            logger.info("   ✅ Vocabulary list generated")
        
        except Exception as e:
            logger.error(f"   ❌ Error generating vocabulary: {e}")
            state['vocabulary_list'] = "Vocabulary list unavailable"
            state['errors'] = state.get('errors', []) + [f"Vocabulary generation error: {e}"]
        
        return state
    
    def _generate_grammar_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 5: Generate grammar explanations in context
        """
        logger.info("📖 Step 5: Generating grammar notes")
        
        try:
            grammar_notes = self.generator.generate_grammar_notes(
                episode_examples=state['vocabulary_examples'],
                episode_title=state['selected_episode']['title']
            )
            
            state['grammar_notes'] = grammar_notes
            state['step'] = 'grammar_generated'
            logger.info("   ✅ Grammar notes generated")
        
        except Exception as e:
            logger.error(f"   ❌ Error generating grammar: {e}")
            state['grammar_notes'] = "Grammar notes unavailable"
            state['errors'] = state.get('errors', []) + [f"Grammar generation error: {e}"]
        
        return state
    
    def _generate_culture_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 6: Generate cultural context
        """
        logger.info("🎌 Step 6: Generating cultural context")
        
        try:
            cultural_notes = self.generator.generate_cultural_notes(
                episode_title=state['selected_episode']['title'],
                episode_level=state['selected_episode']['level']
            )
            
            state['cultural_context'] = cultural_notes
            state['step'] = 'culture_generated'
            logger.info("   ✅ Cultural context generated")
        
        except Exception as e:
            logger.error(f"   ❌ Error generating cultural context: {e}")
            state['cultural_context'] = "Cultural notes unavailable"
            state['errors'] = state.get('errors', []) + [f"Culture generation error: {e}"]
        
        return state
    
    def _generate_prep_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 7: Generate pre-watch preparation
        """
        logger.info("🎯 Step 7: Generating pre-watch preparation")
        
        try:
            prep_guide = self.generator.generate_pre_watch_prep(
                episode_title=state['selected_episode']['title'],
                vocabulary_examples=state['vocabulary_examples'][:10],
                user_level=state['user_level']
            )
            
            state['pre_watch_prep'] = prep_guide
            state['step'] = 'prep_generated'
            logger.info("   ✅ Pre-watch prep generated")
        
        except Exception as e:
            logger.error(f"   ❌ Error generating prep: {e}")
            state['pre_watch_prep'] = "Pre-watch preparation unavailable"
            state['errors'] = state.get('errors', []) + [f"Prep generation error: {e}"]
        
        return state
    
    def _assemble_response_node(self, state: WorkflowState) -> WorkflowState:
        """
        Node 8: Assemble final response
        """
        logger.info("🎁 Step 8: Assembling final response")
        
        state['step'] = 'complete'
        logger.info("   ✅ Workflow complete")
        
        return state
    
    def execute_recommendation_workflow(self, user_level: str, query: str = "", 
                                       anime_filter: str = None, n_results: int = 5) -> Dict:
        """
        Execute the complete recommendation workflow
        
        Args:
            user_level: JLPT level
            query: Search query
            anime_filter: Filter by anime name
            n_results: Number of results
            
        Returns:
            Complete workflow state with all generated content
        """
        logger.info("🚀 Starting LangGraph workflow")
        
        initial_state = WorkflowState(
            user_level=user_level,
            query=query,
            anime_filter=anime_filter,
            n_results=n_results,
            matched_episodes=[],
            selected_episode=None,
            vocabulary_examples=[],
            recommendation_text="",
            vocabulary_list="",
            grammar_notes="",
            cultural_context="",
            pre_watch_prep="",
            errors=[],
            step="init"
        )
        
        # Execute workflow
        final_state = self.workflow.invoke(initial_state)
        
        logger.info(f"✅ Workflow completed (Final step: {final_state['step']})")
        
        if final_state.get('errors'):
            logger.warning(f"⚠️  Errors encountered: {final_state['errors']}")
        
        return final_state
    
    def execute_learning_package_workflow(self, episode_id: str, user_level: str) -> Dict:
        """
        Execute workflow to generate complete learning package for specific episode
        
        Args:
            episode_id: Episode identifier (e.g., "steins_gate_e04")
            user_level: User's JLPT level
            
        Returns:
            Complete learning package
        """
        logger.info(f"🚀 Generating learning package for: {episode_id}")
        
        try:
            # Parse episode_id to get anime name and episode number
            # Format: "anime_name_eXX" or "anime_name_with_underscores_eXX"
            parts = episode_id.rsplit('_e', 1)
            
            if len(parts) != 2:
                raise ValueError(f"Invalid episode_id format: {episode_id}. Expected format: 'anime_name_eXX'")
            
            anime_slug = parts[0]
            episode_num_str = parts[1]
            
            try:
                episode_num = int(episode_num_str)
            except ValueError:
                raise ValueError(f"Invalid episode number in episode_id: {episode_id}")
            
            # Convert slug to anime name (e.g., "steins_gate" -> "Steins Gate")
            anime_name = anime_slug.replace('_', ' ').title()
            
            logger.info(f"   Searching for: {anime_name} Episode {episode_num}")
            
            # Search for all episodes of this anime
            episodes = self.rag.search_by_anime(anime_name)
            
            if not episodes:
                # Try without title case
                logger.warning(f"   No episodes found for '{anime_name}', trying alternate formats...")
                episodes = self.rag.search_by_anime(anime_slug.replace('_', ' '))
            
            if not episodes:
                raise ValueError(f"No episodes found for anime: {anime_name}")
            
            # Find the specific episode by episode number
            episode_data = None
            for ep in episodes:
                if ep.get('episode_number') == episode_num:
                    episode_data = ep
                    break
            
            if not episode_data:
                available_eps = [ep.get('episode_number') for ep in episodes]
                raise ValueError(f"Episode {episode_num} not found. Available episodes: {available_eps}")
            
            logger.info(f"   ✅ Found: {episode_data['title']}")
            
            # Get examples using the actual episode_id from database
            actual_episode_id = episode_data['episode_id']
            examples = self.rag.find_vocabulary_examples(actual_episode_id, n_examples=15)
            
            if not examples:
                logger.warning(f"   ⚠️  No vocabulary examples found for {actual_episode_id}")
                examples = []
            
            # Create state
            state = WorkflowState(
                user_level=user_level,
                query="",
                anime_filter=None,
                n_results=1,
                matched_episodes=[],
                selected_episode={
                    'episode_id': actual_episode_id,
                    'anime_name': episode_data['anime_name'],
                    'title': episode_data['title'],
                    'level': episode_data['level'],
                    'total_lines': episode_data.get('total_lines', 0),
                    'vocab_count': episode_data.get('vocab_count', 0)
                },
                vocabulary_examples=examples,
                recommendation_text="",
                vocabulary_list="",
                grammar_notes="",
                cultural_context="",
                pre_watch_prep="",
                errors=[],
                step="init"
            )
            
            # Generate all components
            state = self._generate_vocabulary_node(state)
            state = self._generate_grammar_node(state)
            state = self._generate_culture_node(state)
            state = self._generate_prep_node(state)
            state = self._assemble_response_node(state)
            
            logger.info("✅ Learning package generated")
            return state
        
        except Exception as e:
            logger.error(f"❌ Error generating learning package: {e}")
            raise


def test_orchestrator():
    """Test the LangGraph orchestrator"""
    from rag_engine_v3 import RAGEngineV3
    from learning_generator_v2 import LearningGeneratorV2
    
    logger.info("Testing LangGraph Orchestrator")
    
    rag = RAGEngineV3()
    generator = LearningGeneratorV2()
    orchestrator = LangGraphOrchestrator(rag, generator)
    
    # Test recommendation workflow
    result = orchestrator.execute_recommendation_workflow(
        user_level="N3",
        query="action anime",
        n_results=3
    )
    
    logger.info(f"Final state: {result['step']}")
    logger.info(f"Selected: {result.get('selected_episode', {}).get('title', 'None')}")
    logger.info(f"Errors: {result.get('errors', [])}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_orchestrator()