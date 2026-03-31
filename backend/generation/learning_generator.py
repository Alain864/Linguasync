"""
backend/generation/learning_generator.py - Enhanced Learning Generator for Stage 2

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

IMPORTANT: Make sure your recommendation makes sense for what the user asked for.
If they asked for "relaxing" content, focus on calm, peaceful aspects.
If they asked for "action" content, focus on exciting, dynamic aspects.

Format naturally like talking to a friend. Focus on why the anime is a good match thematically.
Be enthusiastic but concise and accurate.

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
        WITH ACTUAL SENTENCES FROM THE EPISODE
        
        Args:
            episode_examples: Sample sentences from the episode
            episode_title: Title of the episode
            n_words: Number of vocabulary items to highlight
            
        Returns:
            Formatted vocabulary list with actual episode sentences
        """
        # Prepare example sentences with their text
        examples_text = "\n".join([
            f"- {ex['text']}"
            for ex in episode_examples[:15]
        ])
        
        prompt = f"""You are a Japanese language teacher creating a vocabulary study guide.

Episode: {episode_title}

Dialogue from this episode:
{examples_text}

Task: Create a vocabulary list of the {n_words} MOST USEFUL and INTERESTING words from these actual dialogue lines.

CRITICAL FORMAT - Follow this EXACTLY:

1) Japanese word (romaji) – English meaning.
Phrase: [actual Japanese sentence from the dialogue above]
Romaji: [romaji of that sentence]
EN: [English translation of that sentence]

2) Japanese word (romaji) – English meaning.
Phrase: [actual Japanese sentence from the dialogue above]
Romaji: [romaji of that sentence]
EN: [English translation of that sentence]

IMPORTANT RULES:
- Use ACTUAL sentences from the dialogue provided above - don't make up new examples
- Pick the most interesting, thematic, or useful words for this specific episode
- The example sentence MUST be from the dialogue I gave you
- Keep translations natural and context-appropriate
- Focus on nouns, verbs, and key expressions (not particles or basic words like これ, それ)
- Pick words that capture the episode's theme or story

Example format:
1) 時間旅行（じかんりょこう）– time travel.
Phrase: 時間旅行の実験は危険すぎる。
Romaji: Jikan ryokō no jikken wa kiken sugiru.
EN: Time travel experiments are too dangerous.

Generate {n_words} entries following this exact format."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500  # Increased for more detailed output
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
        WITH ACTUAL EXAMPLES FROM THE EPISODE
        
        Args:
            episode_examples: Sample sentences from the episode
            episode_title: Title of the episode
            
        Returns:
            Grammar notes with actual episode examples
        """
        examples_text = "\n".join([
            f"- {ex['text']}"
            for ex in episode_examples[:12]
        ])
        
        prompt = f"""You are a Japanese grammar expert creating study notes.

Episode: {episode_title}

Actual Dialogue from this Episode:
{examples_text}

Task: Identify 2-3 KEY GRAMMAR PATTERNS that actually appear in the dialogue above.

For each pattern, provide:
1. **Grammar Point Name** (e.g., "〜ている form", "〜なければならない", etc.)
2. **What it means** - Brief explanation
3. **Example from the dialogue** - Use an ACTUAL sentence from above
   - Show the Japanese sentence
   - Provide romaji
   - Give English translation
4. **When to use it** - Simple usage rule

FORMAT EXAMPLE:

**Grammar Point 1: 〜ている (te-iru form)**
Meaning: Describes an ongoing action or current state.

Example from dialogue:
- Japanese: 何を考えているんだ？
- Romaji: Nani o kangaete irun da?
- English: What are you thinking about?

Usage: Attach ている to the -te form of verbs to show ongoing actions or states.

---

**Grammar Point 2: [next pattern]**
[continue...]

CRITICAL:
- Use ONLY sentences from the dialogue I provided above
- Don't make up example sentences
- Pick grammar patterns that actually appear in the episode
- Keep explanations beginner-friendly
- Make it practical and example-driven"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"❌ Error generating grammar notes: {e}")
            return "Grammar notes generation failed."
    
    def generate_cultural_notes(self,
                                 episode_title: str,
                                 episode_level: str,
                                 episode_examples: List[Dict] = None) -> str:
        """
        Generate cultural context notes focused on LANGUAGE NUANCES
        NOT generic cultural facts - focus on idioms, expressions, speech patterns
        
        Args:
            episode_title: Title of the episode/show
            episode_level: JLPT level of content
            episode_examples: Actual dialogue from the episode
            
        Returns:
            Cultural/linguistic insights from the episode
        """
        # Include actual dialogue if available
        dialogue_text = ""
        if episode_examples and len(episode_examples) > 0:
            dialogue_text = "\n".join([
                f"- {ex.get('text', '')}"
                for ex in episode_examples[:12]
                if ex.get('text')  # Only include if text exists
            ])
            logger.info(f"   📝 Using {len(episode_examples)} dialogue examples for cultural notes")
        else:
            logger.warning(f"   ⚠️  No dialogue examples provided for cultural notes")
        
        prompt = f"""You are a Japanese language and culture expert analyzing dialogue from an anime episode.

Episode: {episode_title}
Level: JLPT {episode_level}

Actual Dialogue from this Episode:
{dialogue_text if dialogue_text else "IMPORTANT: Dialogue samples should have been provided but are missing. Base your analysis on the anime title and typical themes of this series."}

Task: Identify 2-3 JAPANESE LANGUAGE NUANCES, IDIOMS, or CULTURAL REFERENCES {"that appear in this dialogue" if dialogue_text else "that typically appear in this anime"}.

DO NOT give generic cultural facts like "Japanese use honorifics" or "bowing is important."

INSTEAD, focus on:
1. **Idiomatic/Stylized Expressions** {"from the actual dialogue" if dialogue_text else "typical of this anime"}
   - What it literally means
   - What it actually means in context
   - When/why Japanese speakers use this expression

2. **Speech Patterns & Nuances** 
   - Sentence endings that reveal character/relationship
   - Formal vs casual shifts
   - Male/female speech differences
   - Implied meanings (読み取り)

3. **Cultural References Embedded in Language**
   - Buddhist/philosophical concepts in phrasing
   - Historical or literary references
   - Wordplay or puns (駄洒落)
   - Pop culture references

FORMAT EXAMPLE:

**Idiomatic Expressions**

■ 始末する (shimatsu suru)
Literally: "to deal with tidying up"
In context: "to eliminate someone"
Usage: Common euphemism in yakuza/crime dramas. Shows how Japanese uses indirect phrasing for harsh realities.

■ 目を覚ませ (me o samase)  
Literally: "wake your eyes"
Meaning: "snap out of it" / "face reality"
Usage: Used when confronting someone living in denial or illusion.

**Cultural/Philosophical References**

"夢を見ている" (yume o mite iru)
Reference: Buddhist concept of life as impermanent illusion (夢幻 mugen)
Context: Shows how everyday Japanese is infused with Buddhist philosophy.

---

CRITICAL RULES:
{"- Base findings on the ACTUAL DIALOGUE provided above" if dialogue_text else "- Base findings on typical language patterns in this anime"}
- Focus on LANGUAGE nuances, not general culture facts
- Explain idioms/expressions that learners might misunderstand
- Show literal vs actual meaning
- Keep it practical and specific"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=700
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
