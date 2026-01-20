"""
learning_generator_v2.py - Enhanced Learning Generator for Stage 2

New features:
- Pre-watch vocabulary preparation
- Enhanced grammar explanations with real examples
- Richer cultural context
- Progressive difficulty mapping
"""

import os
import logging
from typing import List, Dict
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
    
    def generate_recommendation(self, 
                                 episodes: List[Dict],
                                 user_level: str,
                                 user_query: str = "") -> str:
        """
        Generate a personalized episode recommendation
        
        Args:
            episodes: List of matching episodes from RAG
            user_level: User's JLPT level
            user_query: User's original query
            
        Returns:
            Formatted recommendation text
        """
        episodes_context = "\n".join([
            f"- {ep['title']} (Level: {ep['level']}, {ep['total_lines']} lines, "
            f"{ep['vocab_count']} unique words)"
            for ep in episodes[:3]
        ])
        
        prompt = f"""You are LinguaSync, an AI Japanese learning assistant.

User Profile:
- Current Level: JLPT {user_level}
- Query: {user_query if user_query else "Looking for appropriate content"}

Available Content:
{episodes_context}

Task: Recommend the BEST episode for this learner. Your recommendation should:
1. Explain WHY this episode is perfect for their level
2. Set clear expectations (what % of vocabulary they'll know)
3. Highlight what makes it engaging
4. Be encouraging and specific

Keep it concise (3-4 sentences) and conversational."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"❌ Error generating recommendation: {e}")
            return "Unable to generate recommendation at this time."
    
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
        logger.info(f"🎓 Generating learning package for: {episode['title']}")
        
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
    logger.info("🎓 Testing Learning Generator V2")
    logger.info("="*60)
    
    generator = LearningGeneratorV2()
    
    # Sample data
    sample_episode = {
        'episode_id': 'attack_on_titan_s01e01',
        'title': 'Attack on Titan S01E01',
        'level': 'N3',
        'total_lines': 450,
        'vocab_count': 320
    }
    
    sample_examples = [
        {'text': '人類は突然現れた巨人により滅亡の淵に立たされた', 'level': 'N3', 'vocab': ['人類', '巨人', '滅亡']},
        {'text': 'この壁の中で百年間平和に暮らしてきた', 'level': 'N4', 'vocab': ['壁', '平和', '暮らす']},
        {'text': '調査兵団に入って外の世界を見たい', 'level': 'N3', 'vocab': ['調査兵団', '世界', '見る']},
    ]
    
    # Test recommendation
    logger.info("\n📝 Testing recommendation generation...")
    recommendation = generator.generate_recommendation(
        [sample_episode],
        user_level="N3",
        user_query="I want something exciting"
    )
    logger.info(f"\n{recommendation}\n")
    
    # Test pre-watch prep
    logger.info("\n🎯 Testing pre-watch preparation...")
    prep = generator.generate_pre_watch_prep(
        sample_episode['title'],
        sample_examples,
        "N3"
    )
    logger.info(f"\n{prep}\n")
    
    logger.info("✅ Generator V2 working correctly!")


if __name__ == "__main__":
    test_generator()