"""
LangGraph Orchestrator - Stage 2
Multi-step reasoning for enhanced learning recommendations
"""

from typing import Dict, List, TypedDict, Annotated
from langgraph.graph import Graph, StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import operator

class LearningState(TypedDict):
    """State for learning recommendation workflow"""
    user_level: str
    user_query: str
    episodes: List[Dict]
    selected_episode: Dict
    vocabulary: str
    grammar: str
    cultural_notes: str
    pre_watch_vocab: str
    recommendation: str
    errors: List[str]


class LangGraphOrchestrator:
    """
    Orchestrates multi-step learning content generation using LangGraph
    
    Flow:
    1. Analyze user query → Extract learning goals
    2. Select best episode → Match to goals
    3. Generate pre-watch vocabulary → Essential words to know
    4. Generate grammar notes → Contextual explanations
    5. Generate cultural context → Cultural understanding
    6. Create final recommendation → Tie everything together
    """
    
    def __init__(self):
        """Initialize orchestrator with LLM"""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7
        )
        
        # Build workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        workflow = StateGraph(LearningState)
        
        # Add nodes for each step
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("select_episode", self._select_episode)
        workflow.add_node("generate_pre_watch_vocab", self._generate_pre_watch_vocab)
        workflow.add_node("generate_grammar", self._generate_grammar)
        workflow.add_node("generate_cultural_notes", self._generate_cultural_notes)
        workflow.add_node("create_recommendation", self._create_recommendation)
        
        # Define flow
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "select_episode")
        workflow.add_edge("select_episode", "generate_pre_watch_vocab")
        workflow.add_edge("generate_pre_watch_vocab", "generate_grammar")
        workflow.add_edge("generate_grammar", "generate_cultural_notes")
        workflow.add_edge("generate_cultural_notes", "create_recommendation")
        workflow.add_edge("create_recommendation", END)
        
        return workflow.compile()
    
    def _analyze_query(self, state: LearningState) -> LearningState:
        """
        Step 1: Analyze user query to extract learning goals
        """
        prompt = ChatPromptTemplate.from_template("""
        Analyze this learner's request and extract their learning goals:
        
        Level: {level}
        Query: {query}
        
        What are they looking for? Consider:
        - Content type (action, romance, daily life, etc.)
        - Learning focus (vocabulary, grammar, listening practice)
        - Difficulty preference (challenging, comfortable, easy)
        
        Respond in 2-3 sentences summarizing their goals.
        """)
        
        response = self.llm.invoke(
            prompt.format(level=state['user_level'], query=state['user_query'])
        )
        
        state['learning_goals'] = response.content
        return state
    
    def _select_episode(self, state: LearningState) -> LearningState:
        """
        Step 2: Select the best episode from candidates
        """
        # Take top episode (already ranked by RAG)
        if state['episodes']:
            state['selected_episode'] = state['episodes'][0]
        else:
            state['errors'].append("No episodes available")
        
        return state
    
    def _generate_pre_watch_vocab(self, state: LearningState) -> LearningState:
        """
        Step 3: Generate pre-watch vocabulary - essential words to know BEFORE watching
        """
        episode = state['selected_episode']
        
        prompt = ChatPromptTemplate.from_template("""
        You are creating a pre-watch vocabulary guide for a language learner.
        
        Episode: {title}
        Anime: {anime}
        Learner Level: {level}
        
        Create a list of 10 ESSENTIAL words the learner should know BEFORE watching.
        These should be:
        - High-frequency words that appear multiple times
        - Critical for understanding the plot
        - Appropriate for {level} level
        
        Format:
        **Word** (romaji) - meaning
        _Why: Brief explanation of why this word is important_
        
        Focus on words they'll hear repeatedly in this episode.
        """)
        
        response = self.llm.invoke(prompt.format(
            title=episode.get('title', ''),
            anime=episode.get('anime_name', ''),
            level=state['user_level']
        ))
        
        state['pre_watch_vocab'] = response.content
        return state
    
    def _generate_grammar(self, state: LearningState) -> LearningState:
        """
        Step 4: Generate grammar explanations with real examples
        """
        episode = state['selected_episode']
        
        prompt = ChatPromptTemplate.from_template("""
        You are a Japanese grammar expert creating contextual explanations.
        
        Episode: {title}
        Level: {level}
        
        Identify 2-3 KEY grammar patterns that appear in Japanese anime at this level.
        For each pattern:
        
        1. **Pattern name** (e.g., ～ている form)
        2. What it means and when to use it
        3. Give a realistic example sentence from anime
        4. Explain what to listen for
        
        Make it practical - focus on patterns they'll actually hear in the episode.
        """)
        
        response = self.llm.invoke(prompt.format(
            title=episode.get('title', ''),
            level=state['user_level']
        ))
        
        state['grammar'] = response.content
        return state
    
    def _generate_cultural_notes(self, state: LearningState) -> LearningState:
        """
        Step 5: Generate cultural context notes
        """
        episode = state['selected_episode']
        
        prompt = ChatPromptTemplate.from_template("""
        You are a cultural consultant for Japanese language learners.
        
        Anime: {anime}
        Episode: {title}
        
        Provide 2-3 cultural insights that will help learners appreciate this content:
        
        Focus on:
        - Social norms or customs they'll see
        - Communication styles (formal/casual)
        - Cultural references they might miss
        - Context that aids comprehension
        
        Keep each insight to 1-2 sentences. Be specific and practical.
        Use emoji bullets (🎌, 🗣️, 💡) for visual interest.
        """)
        
        response = self.llm.invoke(prompt.format(
            anime=episode.get('anime_name', ''),
            title=episode.get('title', '')
        ))
        
        state['cultural_notes'] = response.content
        return state
    
    def _create_recommendation(self, state: LearningState) -> LearningState:
        """
        Step 6: Create final recommendation tying everything together
        """
        episode = state['selected_episode']
        
        prompt = ChatPromptTemplate.from_template("""
        Create a compelling recommendation for this episode.
        
        Episode: {title}
        Learner Level: {level}
        Learning Goals: {goals}
        
        Your recommendation should:
        1. Explain WHY this episode is perfect for their level and goals
        2. Set expectations (difficulty, what they'll learn)
        3. Be encouraging and specific
        4. Reference the pre-watch vocab and grammar points
        
        Keep it to 3-4 sentences. Be enthusiastic but realistic.
        """)
        
        response = self.llm.invoke(prompt.format(
            title=episode.get('title', ''),
            level=state['user_level'],
            goals=state.get('learning_goals', state['user_query'])
        ))
        
        state['recommendation'] = response.content
        return state
    
    def generate_learning_package(self, user_level: str, user_query: str, episodes: List[Dict]) -> Dict:
        """
        Generate complete learning package using LangGraph workflow
        
        Args:
            user_level: User's JLPT level
            user_query: User's search query
            episodes: List of candidate episodes from RAG
            
        Returns:
            Complete learning package with all components
        """
        # Initialize state
        initial_state = LearningState(
            user_level=user_level,
            user_query=user_query,
            episodes=episodes,
            selected_episode={},
            vocabulary="",
            grammar="",
            cultural_notes="",
            pre_watch_vocab="",
            recommendation="",
            errors=[]
        )
        
        # Run workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Return structured result
        return {
            'episode': final_state['selected_episode'],
            'recommendation': final_state['recommendation'],
            'pre_watch_vocabulary': final_state['pre_watch_vocab'],
            'grammar_notes': final_state['grammar'],
            'cultural_context': final_state['cultural_notes'],
            'errors': final_state.get('errors', [])
        }


def test_orchestrator():
    """Test the LangGraph orchestrator"""
    
    print("="*60)
    print("🧪 Testing LangGraph Orchestrator")
    print("="*60)
    
    orchestrator = LangGraphOrchestrator()
    
    # Test data
    test_episodes = [
        {
            'episode_id': 'cowboy_bebop_s01e01',
            'anime_name': 'Cowboy Bebop',
            'title': 'Cowboy Bebop - S01E01',
            'level': 'N4',
            'total_lines': 450,
            'vocab_count': 320
        }
    ]
    
    # Generate package
    package = orchestrator.generate_learning_package(
        user_level="N4",
        user_query="action anime with clear dialogue",
        episodes=test_episodes
    )
    
    print("\n✅ Generated Learning Package:")
    print(f"\nRecommendation:\n{package['recommendation']}")
    print(f"\nPre-Watch Vocab:\n{package['pre_watch_vocabulary']}")
    print(f"\nGrammar Notes:\n{package['grammar_notes']}")
    print(f"\nCultural Context:\n{package['cultural_context']}")


if __name__ == "__main__":
    test_orchestrator()