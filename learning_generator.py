"""
Learning Generator - Creates educational content using LLM

This module:
1. Generates vocabulary lists with explanations
2. Creates grammar explanations in context
3. Provides cultural notes
4. Formats learning recommendations
"""

import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LearningGenerator:
    """Generates learning content using GPT-4"""
    
    def __init__(self):
        """Initialize the learning generator"""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"  # Cost-effective for Stage 0
    
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
        # Build context for LLM
        episodes_context = "\n".join([
            f"- {ep['title']} (Level: {ep['level']}, {ep['total_lines']} lines, "
            f"{ep['vocab_count']} unique words)"
            for ep in episodes[:3]  # Top 3 episodes
        ])
        
        prompt = f"""You are LinguaSync, an AI language learning assistant specializing in Japanese.

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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
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
        # Extract sample sentences
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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
        
        return response.choices[0].message.content
    
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
3. Show an example from the dialogue
4. Give a simple rule for when to use it

Make it:
- Practical and example-driven
- Connected to the actual dialogue
- Beginner-friendly explanations
- Focused on patterns they'll encounter frequently

Format naturally with clear sections."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600
        )
        
        return response.choices[0].message.content
    
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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400
        )
        
        return response.choices[0].message.content
    
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
        print(f"🎓 Generating learning package for: {episode['title']}")
        
        package = {
            'episode_id': episode['episode_id'],
            'title': episode['title'],
            'level': episode['level'],
            'user_level': user_level,
        }
        
        # Generate each component
        print("   📝 Creating vocabulary list...")
        package['vocabulary'] = self.generate_vocabulary_list(
            examples, episode['title']
        )
        
        print("   📚 Writing grammar notes...")
        package['grammar'] = self.generate_grammar_notes(
            examples, episode['title']
        )
        
        print("   🎌 Adding cultural context...")
        package['cultural_notes'] = self.generate_cultural_notes(
            episode['title'], episode['level']
        )
        
        # Add episode stats
        package['stats'] = {
            'total_lines': episode['total_lines'],
            'vocab_count': episode['vocab_count'],
            'level_match': episode['level'] == user_level
        }
        
        print("   ✅ Learning package complete!")
        
        return package


def test_generator():
    """Test the learning generator with sample data"""
    
    print("="*60)
    print("🎓 Testing Learning Generator")
    print("="*60)
    
    generator = LearningGenerator()
    
    # Sample episode data
    sample_episode = {
        'episode_id': 'attack_on_titan_s01e01',
        'title': 'Attack on Titan S01E01',
        'level': 'N3',
        'total_lines': 450,
        'vocab_count': 320
    }
    
    sample_examples = [
        {'text': '人類は突然現れた巨人により滅亡の淵に立たされた', 'level': 'N3'},
        {'text': 'この壁の中で百年間平和に暮らしてきた', 'level': 'N4'},
        {'text': '調査兵団に入って外の世界を見たい', 'level': 'N3'},
    ]
    
    # Test recommendation
    print("\n📍 Testing recommendation generation...")
    recommendation = generator.generate_recommendation(
        [sample_episode],
        user_level="N3",
        user_query="I want something exciting"
    )
    print(f"\n{recommendation}\n")
    
    print("✅ Generator working correctly!")


if __name__ == "__main__":
    test_generator()