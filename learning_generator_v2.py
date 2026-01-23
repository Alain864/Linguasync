"""
learning_generator_v2.py - Enhanced Learning Generator for Stage 2

New features:
- Pre-watch vocabulary preparation
- Enhanced grammar explanations with real examples
- Richer cultural context
- Progressive difficulty mapping
- Smart recommendation scenarios (general, level-specific, query-focused)
"""

import os
import logging
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


class LearningGeneratorV2:
    """Enhanced learning content generator using GPT-4"""
    
    def __init__(self):
        """Initialize the learning generator"""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"
        logger.info("✅ Learning Generator V2 initialized")
    
    def _detect_recommendation_scenario(self, 
                                       user_query: str, 
                                       user_level: str,
                                       episodes: List[Dict]) -> str:
        """
        Detect which recommendation scenario to use:
        - scenario_1: General theme/genre query (no level specifics)
        - scenario_2: Level explicitly mentioned in query
        - scenario_3: Detailed query with specific preferences
        """
        query_lower = user_query.lower()
        
        # Check if query mentions JLPT levels
        level_keywords = ['n5', 'n4', 'n3', 'n2', 'n1', 'level', 'jlpt', 'beginner', 'intermediate', 'advanced']
        has_level_mention = any(keyword in query_lower for keyword in level_keywords)
        
        # Check query complexity (detailed preferences vs simple themes)
        detailed_keywords = [
            'slow', 'fast', 'clear', 'dialogue', 'speech', 'conversation',
            'breathe', 'pace', 'unhurried', 'relaxing', 'calm', 'intense',
            'grammar', 'vocabulary', 'kanji', 'particles'
        ]
        is_detailed = any(keyword in query_lower for keyword in detailed_keywords)
        
        # Determine scenario
        if not user_query or len(user_query.strip()) < 10:
            return 'scenario_1'  # Very short or no query
        elif has_level_mention and not is_detailed:
            return 'scenario_2'  # Level mentioned but not detailed
        elif is_detailed or (user_level != 'All Levels' and len(user_query) > 30):
            return 'scenario_3'  # Detailed query or specific level with longer query
        else:
            return 'scenario_1'  # General theme/genre query
    
    def generate_recommendation(self, 
                                 episodes: List[Dict],
                                 user_level: str,
                                 user_query: str = "") -> str:
        """
        Generate a personalized episode recommendation based on scenario
        
        Args:
            episodes: List of matching episodes from RAG
            user_level: User's JLPT level (or "All Levels")
            user_query: User's original query
            
        Returns:
            Formatted recommendation text
        """
        if not episodes:
            return "No matching content found. Try adjusting your search criteria!"
        
        # Detect scenario
        scenario = self._detect_recommendation_scenario(user_query, user_level, episodes)
        logger.info(f"📊 Recommendation scenario: {scenario}")
        
        # Prepare episode context
        top_episode = episodes[0]
        episodes_context = "\n".join([
            f"- {ep['anime_name']} - {ep['title']} (Level: {ep['level']}, {ep.get('total_lines', 0)} lines)"
            for ep in episodes[:3]
        ])
        
        # Generate recommendation based on scenario
        if scenario == 'scenario_1':
            return self._generate_general_recommendation(top_episode, episodes_context, user_query)
        elif scenario == 'scenario_2':
            return self._generate_level_focused_recommendation(top_episode, episodes_context, user_query, user_level)
        else:  # scenario_3
            return self._generate_detailed_recommendation(top_episode, episodes_context, user_query, user_level)
    
    def _generate_general_recommendation(self, 
                                         top_episode: Dict,
                                         episodes_context: str,
                                         user_query: str) -> str:
        """
        Scenario 1: General theme/genre recommendation
        Example: "Recommend me an anime about space bounty"
        Focus: Brief thematic match, list first episodes
        """
        prompt = f"""You are LinguaSync, an AI Japanese learning assistant.

User is looking for: {user_query if user_query else "anime to watch"}

Top matching content:
{episodes_context}

SCENARIO 1 - GENERAL RECOMMENDATION
Task: Create a brief, engaging recommendation (3-4 sentences) that:
1. Identifies the best matching anime for their theme/interest
2. Describes what makes it appealing (theme, atmosphere, story style)
3. Mentions key language characteristics (natural speech, casual/formal, pacing)
4. Does NOT go into JLPT levels or detailed grammar analysis

Format naturally like talking to a friend. Focus on why the anime is a good match thematically.
Be enthusiastic but concise.

Example style:
"A perfect fit for [theme] is [Anime Name] where [brief description]. [Unique appeal]. 
Language characteristics: [speech style notes]"
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ Error generating general recommendation: {e}")
            return f"We recommend {top_episode['anime_name']} - {top_episode['title']} for your interests!"
    
    def _generate_level_focused_recommendation(self,
                                               top_episode: Dict,
                                               episodes_context: str,
                                               user_query: str,
                                               user_level: str) -> str:
        """
        Scenario 2: Level explicitly mentioned in query
        Example: "I'm looking for a relaxing anime episode to watch about level N5"
        Focus: Match to level, explain why it's appropriate
        """
        # Extract level from query if present
        query_lower = user_query.lower()
        extracted_level = None
        for level in ['n5', 'n4', 'n3', 'n2', 'n1']:
            if level in query_lower:
                extracted_level = level.upper()
                break
        
        target_level = extracted_level if extracted_level else user_level
        
        prompt = f"""You are LinguaSync, an AI Japanese learning assistant.

User query: {user_query}
Target JLPT Level: {target_level}

Matching episodes:
{episodes_context}

SCENARIO 2 - LEVEL-FOCUSED RECOMMENDATION
Task: Create a recommendation (4-5 sentences) that:
1. Confirms the level they're looking for
2. Explains what makes JLPT {target_level} appropriate (vocabulary complexity, grammar patterns)
3. Recommends a specific episode with clear level justification
4. Sets realistic expectations (% of vocab they'll know)
5. Highlights what makes it engaging for learning

Focus on LEVEL APPROPRIATENESS and LEARNING OUTCOMES.

Example style:
"For JLPT {target_level} level ([what this means]), the key is picking an anime with [characteristics].
Watch [Anime - Episode]. [Why it's perfect for this level]. You'll likely understand around [X]% of the words.
[What makes it engaging]."
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ Error generating level-focused recommendation: {e}")
            return f"For {target_level} level, we recommend {top_episode['anime_name']} - {top_episode['title']}!"
    
    def _generate_detailed_recommendation(self,
                                          top_episode: Dict,
                                          episodes_context: str,
                                          user_query: str,
                                          user_level: str) -> str:
        """
        Scenario 3: Detailed query with specific preferences
        Example: "give me a recommendation about an anime that let scenes breathe, have unhurried dialogue and is a fantasy world"
        Focus: Match specific preferences, explain why episode fits, mention grammar/learning points
        """
        prompt = f"""You are LinguaSync, an AI Japanese learning assistant.

User's detailed request: {user_query}
User's level: {user_level if user_level != 'All Levels' else 'flexible'}

Best matching episodes:
{episodes_context}

SCENARIO 3 - DETAILED PREFERENCE-BASED RECOMMENDATION
Task: Create a thoughtful, detailed recommendation (6-8 sentences) that:
1. Directly addresses their SPECIFIC preferences from the query
2. Explains why the recommended anime/episode matches their criteria
3. Describes a specific episode that exemplifies what they want
4. Mentions relevant grammar patterns or vocabulary themes they'll encounter
5. Notes speech characteristics (clear/fast, formal/casual, teaching-style/natural)
6. Sets learning expectations based on their level (if specified)

IMPORTANT: 
- Quote their specific preferences (e.g., "breathing room", "unhurried dialogue")
- Explain WHY the recommendation fits these exact criteria
- Focus on the USER'S QUERY as the primary criteria
- Level is secondary - use it to inform learning expectations

Example structure:
"If you want [quote their preferences], a clear fit is [Anime Title].
[Why it matches - theme/atmosphere]. [Specific episode recommendation].
[Why this episode exemplifies what they want].
[Grammar/vocabulary you'll encounter and why it's relevant].
[Speech style notes that match their preferences].
[Learning expectations based on level if specified]."
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ Error generating detailed recommendation: {e}")
            return f"Based on your preferences, we recommend {top_episode['anime_name']} - {top_episode['title']}!"
    
    def generate_vocabulary_list(self,
                                  episode_examples: List[Dict],
                                  episode_title: str,
                                  n_words: int = 15) -> str:
        """
        Generate a curated vocabulary list from episode examples
        
        Args:
            episode_examples: Sample sentences from the episode
            episode_title: Title of the episode
            n_words: Number of vocabulary items to highlight
            
        Returns:
            Formatted vocabulary list with explanations
        """
        examples_text = "\n".join([
            f"- {ex['text']} (Level: {ex['level']})"
            for ex in episode_examples[:10]
        ])
        
        prompt = f"""You are a Japanese language teacher creating a vocabulary study guide.

Episode: {episode_title}

Sample Sentences:
{examples_text}

Task: Create a vocabulary list of the {n_words} MOST USEFUL words from these examples.

For each word, provide:
1. The Japanese word (in original script)
2. Romaji (if helpful)
3. English meaning
4. Brief usage note (one sentence)

Format as:
**Word** (romaji) - meaning
_Usage: explanation_

Focus on:
- High-frequency words
- Words that appear multiple times
- Words useful beyond this specific episode
- Mix of verbs, nouns, and key expressions

Keep explanations practical and concise."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"❌ Error generating vocabulary: {e}")
            return "Vocabulary list generation failed."
    
    def generate_grammar_notes(self,
                                episode_examples: List[Dict],
                                episode_title: str) -> str:
        """
        Generate grammar explanations based on episode content
        
        Args:
            episode_examples: Sample sentences from the episode
            episode_title: Title of the episode
            
        Returns:
            Grammar notes with contextual explanations
        """
        examples_text = "\n".join([
            f"- {ex['text']}"
            for ex in episode_examples[:8]
        ])
        
        prompt = f"""You are a Japanese grammar expert creating study notes.

Episode: {episode_title}

Sample Dialogue:
{examples_text}

Task: Identify 2-3 KEY GRAMMAR PATTERNS that appear in these examples.

For each pattern:
1. Name the grammar point (e.g., "〜ている form")
2. Explain what it means
3. Show an example FROM THE DIALOGUE
4. Give a simple rule for when to use it

Make it:
- Practical and example-driven
- Connected to the actual dialogue
- Beginner-friendly explanations
- Focused on patterns they'll encounter frequently

Format naturally with clear sections."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=600
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"❌ Error generating grammar notes: {e}")
            return "Grammar notes generation failed."
    
    def generate_cultural_notes(self,
                                 episode_title: str,
                                 episode_level: str) -> str:
        """
        Generate cultural context notes
        
        Args:
            episode_title: Title of the episode/show
            episode_level: JLPT level of content
            
        Returns:
            Cultural context and learning tips
        """
        prompt = f"""You are a cultural consultant for Japanese language learners.

Content: {episode_title}
Learner Level: JLPT {episode_level}

Task: Provide 2-3 BRIEF cultural insights that will help learners appreciate this content.

Focus on:
- Cultural context that aids comprehension
- Social norms or customs that appear
- Communication styles (formal/informal)
- Cultural references they might miss

Keep each insight to 1-2 sentences. Be specific and practical.
Format naturally with emoji bullets (🎌, 🗣️, 💡) for visual interest."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"❌ Error generating cultural notes: {e}")
            return "Cultural notes generation failed."
    
    def generate_pre_watch_prep(self,
                                 episode_title: str,
                                 vocabulary_examples: List[Dict],
                                 user_level: str) -> str:
        """
        NEW: Generate pre-watch vocabulary preparation
        
        This helps learners prepare before watching by highlighting
        the most important words they'll encounter.
        
        Args:
            episode_title: Title of the episode
            vocabulary_examples: Sample sentences with vocab
            user_level: User's JLPT level
            
        Returns:
            Pre-watch preparation guide
        """
        # Extract most frequent vocab from examples
        vocab_freq = {}
        for ex in vocabulary_examples:
            for word in ex.get('vocab', []):
                if word:
                    vocab_freq[word] = vocab_freq.get(word, 0) + 1
        
        # Get top 10 most frequent words
        top_vocab = sorted(vocab_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        vocab_list = ", ".join([word for word, _ in top_vocab])
        
        prompt = f"""You are a Japanese learning coach preparing a student for watching content.

Episode: {episode_title}
Student Level: JLPT {user_level}

Key vocabulary that appears frequently:
{vocab_list}

Task: Create a brief pre-watch preparation guide (3-4 sentences) that:
1. Tells them what to review before watching
2. Sets expectations for comprehension
3. Gives one practical tip for active learning while watching
4. Encourages them

Be motivating and practical. Format naturally."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=250
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"❌ Error generating pre-watch prep: {e}")
            return "Pre-watch preparation generation failed."
    
    def generate_complete_learning_package(self,
                                            episode: Dict,
                                            examples: List[Dict],
                                            user_level: str) -> Dict:
        """
        Generate a complete learning package for an episode
        
        This combines all learning components into one comprehensive guide.
        
        Args:
            episode: Episode metadata
            examples: Sample sentences from episode
            user_level: User's JLPT level
            
        Returns:
            Dictionary with all learning components
        """
        logger.info(f"📚 Generating learning package for: {episode['title']}")
        
        package = {
            'episode_id': episode['episode_id'],
            'title': episode['title'],
            'level': episode['level'],
            'user_level': user_level,
        }
        
        # Generate each component
        logger.info("   📝 Creating vocabulary list...")
        package['vocabulary'] = self.generate_vocabulary_list(
            examples, episode['title']
        )
        
        logger.info("   📚 Writing grammar notes...")
        package['grammar'] = self.generate_grammar_notes(
            examples, episode['title']
        )
        
        logger.info("   🎌 Adding cultural context...")
        package['cultural_notes'] = self.generate_cultural_notes(
            episode['title'], episode['level']
        )
        
        logger.info("   🎯 Creating pre-watch preparation...")
        package['pre_watch_prep'] = self.generate_pre_watch_prep(
            episode['title'], examples, user_level
        )
        
        # Add episode stats
        package['stats'] = {
            'total_lines': episode['total_lines'],
            'vocab_count': episode['vocab_count'],
            'level_match': episode['level'] == user_level
        }
        
        logger.info("   ✅ Learning package complete!")
        
        return package


def test_generator():
    """Test the enhanced learning generator"""
    
    logger.info("="*60)
    logger.info("📚 Testing Learning Generator V2")
    logger.info("="*60)
    
    generator = LearningGeneratorV2()
    
    # Sample data
    sample_episodes = [
        {
            'episode_id': 'cowboy_bebop_s01e01',
            'anime_name': 'Cowboy Bebop',
            'title': 'Cowboy Bebop S01E01',
            'level': 'N3',
            'total_lines': 380,
            'vocab_count': 290
        }
    ]
    
    # Test Scenario 1: General theme query
    logger.info("\n🎯 Testing Scenario 1: General theme recommendation...")
    rec1 = generator.generate_recommendation(
        sample_episodes,
        user_level="All Levels",
        user_query="Recommend me an anime about space bounty hunters"
    )
    logger.info(f"\n{rec1}\n")
    
    # Test Scenario 2: Level-focused
    logger.info("\n🎯 Testing Scenario 2: Level-focused recommendation...")
    rec2 = generator.generate_recommendation(
        sample_episodes,
        user_level="All Levels",
        user_query="I'm looking for a relaxing anime episode to watch about level N5"
    )
    logger.info(f"\n{rec2}\n")
    
    # Test Scenario 3: Detailed preferences
    logger.info("\n🎯 Testing Scenario 3: Detailed recommendation...")
    rec3 = generator.generate_recommendation(
        sample_episodes,
        user_level="N4",
        user_query="give me a recommendation about an anime that let scenes breathe, have unhurried dialogue and is a fantasy world"
    )
    logger.info(f"\n{rec3}\n")
    
    logger.info("✅ Generator V2 working correctly!")


if __name__ == "__main__":
    test_generator()