"""
Subtitle Processor - Analyzes Japanese subtitle files

This module:
1. Parses SRT subtitle files
2. Analyzes Japanese text complexity
3. Estimates JLPT level
4. Prepares data for vector storage
"""

import os
import re
import pysrt
from pathlib import Path
from typing import List, Dict
import json

class SubtitleProcessor:
    """Processes Japanese subtitle files for language learning"""
    
    def __init__(self, subtitles_dir: str = "data/subtitles"):
        """
        Initialize the processor
        
        Args:
            subtitles_dir: Directory containing SRT files
        """
        self.subtitles_dir = Path(subtitles_dir)
        self.subtitles_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_srt(self, filepath: str) -> List[Dict]:
        """
        Parse an SRT subtitle file
        
        Args:
            filepath: Path to the SRT file
            
        Returns:
            List of subtitle entries with text, timestamps, and metadata
        """
        print(f"📖 Parsing: {filepath}")
        
        try:
            subs = pysrt.open(filepath, encoding='utf-8')
        except:
            # Try alternate encoding
            subs = pysrt.open(filepath, encoding='shift-jis')
        
        entries = []
        for sub in subs:
            # Extract Japanese text (remove markup)
            text = sub.text.strip()
            text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
            text = re.sub(r'\{[^}]+\}', '', text)  # Remove formatting codes
            
            if not text:
                continue
                
            entry = {
                'text': text,
                'start_time': str(sub.start),
                'end_time': str(sub.end),
                'index': sub.index
            }
            entries.append(entry)
            
        print(f"✅ Parsed {len(entries)} subtitle lines")
        return entries
    
    def estimate_jlpt_level(self, text: str) -> str:
        """
        Estimate JLPT level based on text characteristics
        
        This is a simplified heuristic for Stage 0. In production,
        you'd use a proper Japanese morphological analyzer like MeCab.
        
        Heuristics:
        - Hiragana/Katakana heavy = N5/N4
        - Some kanji, simple patterns = N4/N3
        - Complex kanji, formal language = N3/N2
        - Advanced kanji density = N2/N1
        
        Args:
            text: Japanese text to analyze
            
        Returns:
            Estimated JLPT level (N5, N4, N3, N2, N1)
        """
        # Count different character types
        hiragana = len(re.findall(r'[\u3040-\u309F]', text))
        katakana = len(re.findall(r'[\u30A0-\u30FF]', text))
        kanji = len(re.findall(r'[\u4E00-\u9FFF]', text))
        total_chars = len(text)
        
        if total_chars == 0:
            return "N5"
        
        kanji_ratio = kanji / total_chars
        
        # Simple heuristic
        if kanji_ratio < 0.15:
            return "N5"
        elif kanji_ratio < 0.25:
            return "N4"
        elif kanji_ratio < 0.35:
            return "N3"
        elif kanji_ratio < 0.45:
            return "N2"
        else:
            return "N1"
    
    def extract_vocabulary(self, text: str) -> List[str]:
        """
        Extract potential vocabulary words from text
        
        For Stage 0, we just split on common particles.
        In production, use MeCab for proper tokenization.
        
        Args:
            text: Japanese text
            
        Returns:
            List of vocabulary items
        """
        # Simple split on common particles (very basic)
        particles = ['は', 'が', 'を', 'に', 'へ', 'と', 'で', 'から', 'まで', 'の', 'や']
        
        # Split text
        words = [text]
        for particle in particles:
            new_words = []
            for word in words:
                new_words.extend(word.split(particle))
            words = new_words
        
        # Clean and filter
        vocab = [w.strip() for w in words if len(w.strip()) > 0]
        return vocab
    
    def analyze_subtitle_file(self, filepath: str) -> Dict:
        """
        Fully analyze a subtitle file
        
        Args:
            filepath: Path to SRT file
            
        Returns:
            Dictionary with episode metadata and analyzed content
        """
        # Parse subtitles
        entries = self.parse_srt(filepath)
        
        # Extract metadata from filename
        filename = Path(filepath).stem
        
        # Analyze each entry
        analyzed_entries = []
        all_vocab = []
        level_counts = {'N5': 0, 'N4': 0, 'N3': 0, 'N2': 0, 'N1': 0}
        
        for entry in entries:
            text = entry['text']
            level = self.estimate_jlpt_level(text)
            vocab = self.extract_vocabulary(text)
            
            level_counts[level] += 1
            all_vocab.extend(vocab)
            
            analyzed_entry = {
                **entry,
                'jlpt_level': level,
                'vocab': vocab,
                'char_count': len(text)
            }
            analyzed_entries.append(analyzed_entry)
        
        # Determine overall episode level (most common level)
        episode_level = max(level_counts, key=level_counts.get)
        
        # Calculate statistics
        total_lines = len(analyzed_entries)
        avg_chars = sum(e['char_count'] for e in analyzed_entries) / total_lines if total_lines > 0 else 0
        unique_vocab = list(set(all_vocab))
        
        metadata = {
            'episode_id': filename,
            'title': filename.replace('_', ' ').title(),
            'total_lines': total_lines,
            'episode_level': episode_level,
            'level_distribution': level_counts,
            'avg_chars_per_line': round(avg_chars, 1),
            'unique_vocab_count': len(unique_vocab),
            'entries': analyzed_entries
        }
        
        print(f"""
📊 Analysis Summary:
   Episode: {metadata['title']}
   Level: {episode_level}
   Lines: {total_lines}
   Unique Vocab: {len(unique_vocab)}
   Avg Characters: {metadata['avg_chars_per_line']}
        """)
        
        return metadata
    
    def process_all_subtitles(self) -> List[Dict]:
        """
        Process all SRT files in the subtitles directory
        
        Returns:
            List of analyzed episode metadata
        """
        srt_files = list(self.subtitles_dir.glob("*.srt"))
        
        if not srt_files:
            print(f"⚠️  No SRT files found in {self.subtitles_dir}")
            print("Please add some Japanese subtitle files (.srt) to get started!")
            return []
        
        print(f"🎬 Found {len(srt_files)} subtitle files")
        
        all_episodes = []
        for srt_file in srt_files:
            try:
                metadata = self.analyze_subtitle_file(str(srt_file))
                all_episodes.append(metadata)
            except Exception as e:
                print(f"❌ Error processing {srt_file.name}: {e}")
                continue
        
        return all_episodes


def main():
    """Main function to process subtitles and prepare for RAG system"""
    
    print("="*60)
    print("🎌 LinguaSync Subtitle Processor - Stage 0")
    print("="*60)
    
    # Initialize processor
    processor = SubtitleProcessor()
    
    # Process all subtitles
    episodes = processor.process_all_subtitles()
    
    if not episodes:
        print("\n⚠️  No episodes processed. Add SRT files to data/subtitles/")
        return
    
    # Save processed data
    output_file = "data/processed_episodes.json"
    os.makedirs("data", exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Processed {len(episodes)} episodes")
    print(f"💾 Saved to: {output_file}")
    print("\n🚀 Ready to initialize RAG engine!")
    print("   Next step: Run the RAG engine to create vector embeddings")


if __name__ == "__main__":
    main()