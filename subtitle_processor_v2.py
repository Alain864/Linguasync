"""
Enhanced Subtitle Processor - Stage 1

Improvements over Stage 0:
1. Smart filename parsing for anime/season/episode
2. Better Japanese tokenization (MeCab with fallback)
3. Improved JLPT level estimation using word frequency
4. Metadata validation and enrichment
5. Better error handling
"""

import os
import re
import pysrt
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from dataclasses import dataclass, asdict

# Try to import MeCab for proper Japanese tokenization
try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False
    print("⚠️  MeCab not available. Using fallback tokenization.")
    print("   Install with: pip install mecab-python3")


@dataclass
class EpisodeMetadata:
    """Structured metadata for an episode"""
    anime_name: str
    season: Optional[int]
    episode: int
    episode_id: str
    original_filename: str
    title: str  # Display title


class FilenameParser:
    """
    Parses subtitle filenames to extract metadata
    
    Supports common naming patterns:
    - attack_on_titan_s01e01.srt → Attack on Titan, S1E1
    - naruto_shippuden_s05e120.srt → Naruto Shippuden, S5E120
    - your_name_movie.srt → Your Name, Movie
    - death_note_ep23.srt → Death Note, E23
    - cowboy_bebop_01.srt → Cowboy Bebop, E01
    """
    
    @staticmethod
    def parse_filename(filename: str) -> EpisodeMetadata:
        """
        Parse filename to extract anime name, season, and episode
        
        Args:
            filename: Subtitle filename (e.g., "attack_on_titan_s01e05.srt")
            
        Returns:
            EpisodeMetadata object with parsed information
        """
        # Remove extension
        name_without_ext = Path(filename).stem
        
        # Pattern 1: Full format with season and episode (e.g., s01e05, S1E5)
        pattern_full = r'(.+?)_s(\d+)e(\d+)'
        match = re.search(pattern_full, name_without_ext, re.IGNORECASE)
        
        if match:
            anime_name = match.group(1).replace('_', ' ').title()
            season = int(match.group(2))
            episode = int(match.group(3))
            episode_id = f"{match.group(1)}_s{season:02d}e{episode:02d}"
            title = f"{anime_name} - S{season:02d}E{episode:02d}"
            
            return EpisodeMetadata(
                anime_name=anime_name,
                season=season,
                episode=episode,
                episode_id=episode_id,
                original_filename=filename,
                title=title
            )
        
        # Pattern 2: Episode only with "ep" prefix (e.g., death_note_ep23)
        pattern_ep = r'(.+?)_ep(\d+)'
        match = re.search(pattern_ep, name_without_ext, re.IGNORECASE)
        
        if match:
            anime_name = match.group(1).replace('_', ' ').title()
            episode = int(match.group(2))
            episode_id = f"{match.group(1)}_e{episode:02d}"
            title = f"{anime_name} - E{episode:02d}"
            
            return EpisodeMetadata(
                anime_name=anime_name,
                season=None,
                episode=episode,
                episode_id=episode_id,
                original_filename=filename,
                title=title
            )
        
        # Pattern 3: Just number at end (e.g., cowboy_bebop_01)
        pattern_num = r'(.+?)_(\d+)$'
        match = re.search(pattern_num, name_without_ext)
        
        if match:
            anime_name = match.group(1).replace('_', ' ').title()
            episode = int(match.group(2))
            episode_id = f"{match.group(1)}_e{episode:02d}"
            title = f"{anime_name} - E{episode:02d}"
            
            return EpisodeMetadata(
                anime_name=anime_name,
                season=None,
                episode=episode,
                episode_id=episode_id,
                original_filename=filename,
                title=title
            )
        
        # Pattern 4: Movie or no episode info
        anime_name = name_without_ext.replace('_', ' ').title()
        episode_id = name_without_ext
        title = anime_name
        
        return EpisodeMetadata(
            anime_name=anime_name,
            season=None,
            episode=1,  # Default to episode 1 for movies
            episode_id=episode_id,
            original_filename=filename,
            title=title
        )


class JapaneseTokenizer:
    """
    Tokenizes Japanese text using MeCab (if available) or fallback method
    
    MeCab provides accurate word segmentation and part-of-speech tagging.
    Fallback uses simple character-based heuristics.
    """
    
    def __init__(self):
        """Initialize tokenizer with MeCab if available"""
        self.mecab = None
        
        if MECAB_AVAILABLE:
            try:
                # Try to initialize MeCab
                self.mecab = MeCab.Tagger()
                print("✅ Using MeCab for Japanese tokenization")
            except Exception as e:
                print(f"⚠️  MeCab initialization failed: {e}")
                print("   Using fallback tokenization")
    
    def tokenize(self, text: str) -> List[Dict[str, str]]:
        """
        Tokenize Japanese text into words with metadata
        
        Args:
            text: Japanese text to tokenize
            
        Returns:
            List of tokens with properties (surface, base_form, pos)
        """
        if self.mecab:
            return self._tokenize_mecab(text)
        else:
            return self._tokenize_fallback(text)
    
    def _tokenize_mecab(self, text: str) -> List[Dict[str, str]]:
        """
        Tokenize using MeCab (accurate)
        
        MeCab output format: surface\tpos,pos1,...,base_form
        """
        tokens = []
        
        # Parse text with MeCab
        parsed = self.mecab.parse(text)
        
        for line in parsed.split('\n'):
            if line == 'EOS' or line == '':
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            surface = parts[0]  # The word as it appears
            features = parts[1].split(',')
            
            # Extract relevant features
            pos = features[0] if len(features) > 0 else 'Unknown'  # Part of speech
            base_form = features[6] if len(features) > 6 else surface  # Dictionary form
            
            tokens.append({
                'surface': surface,
                'base_form': base_form,
                'pos': pos,
                'length': len(surface)
            })
        
        return tokens
    
    def _tokenize_fallback(self, text: str) -> List[Dict[str, str]]:
        """
        Fallback tokenization using character types
        
        Not as accurate as MeCab, but works without dependencies.
        Groups characters by type (kanji, hiragana, katakana).
        """
        tokens = []
        current_token = ""
        current_type = None
        
        for char in text:
            # Determine character type
            if '\u4e00' <= char <= '\u9fff':  # Kanji
                char_type = 'kanji'
            elif '\u3040' <= char <= '\u309f':  # Hiragana
                char_type = 'hiragana'
            elif '\u30a0' <= char <= '\u30ff':  # Katakana
                char_type = 'katakana'
            else:
                char_type = 'other'
            
            # Group consecutive characters of same type
            if char_type == current_type or current_type is None:
                current_token += char
                current_type = char_type
            else:
                # Save previous token
                if current_token.strip():
                    tokens.append({
                        'surface': current_token,
                        'base_form': current_token,
                        'pos': current_type,
                        'length': len(current_token)
                    })
                current_token = char
                current_type = char_type
        
        # Add last token
        if current_token.strip():
            tokens.append({
                'surface': current_token,
                'base_form': current_token,
                'pos': current_type,
                'length': len(current_token)
            })
        
        return tokens


class JLPTLevelEstimator:
    """
    Estimates JLPT level based on text characteristics
    
    Uses multiple factors:
    1. Kanji density and complexity
    2. Word length distribution
    3. Particle usage patterns
    """
    
    @staticmethod
    def estimate_level(text: str, tokens: List[Dict]) -> str:
        """
        Estimate JLPT level for a text segment
        
        Args:
            text: Original Japanese text
            tokens: Tokenized words with metadata
            
        Returns:
            JLPT level (N5, N4, N3, N2, N1)
        """
        if not tokens or len(text) == 0:
            return "N5"
        
        # Factor 1: Kanji density
        kanji_chars = len(re.findall(r'[\u4E00-\u9FFF]', text))
        kanji_ratio = kanji_chars / len(text)
        
        # Factor 2: Average word length (longer = more complex)
        avg_word_length = sum(t['length'] for t in tokens) / len(tokens)
        
        # Factor 3: Unique kanji count (more unique kanji = higher level)
        unique_kanji = len(set(re.findall(r'[\u4E00-\u9FFF]', text)))
        
        # Scoring system
        score = 0
        
        # Kanji density scoring
        if kanji_ratio < 0.15:
            score += 1  # N5 range
        elif kanji_ratio < 0.25:
            score += 2  # N4 range
        elif kanji_ratio < 0.35:
            score += 3  # N3 range
        elif kanji_ratio < 0.45:
            score += 4  # N2 range
        else:
            score += 5  # N1 range
        
        # Word length scoring
        if avg_word_length < 2:
            score += 1
        elif avg_word_length < 3:
            score += 2
        elif avg_word_length < 4:
            score += 3
        else:
            score += 4
        
        # Unique kanji scoring
        if unique_kanji < 10:
            score += 1
        elif unique_kanji < 20:
            score += 2
        elif unique_kanji < 30:
            score += 3
        else:
            score += 4
        
        # Average the scores
        avg_score = score / 3
        
        # Map to JLPT levels
        if avg_score < 1.5:
            return "N5"
        elif avg_score < 2.5:
            return "N4"
        elif avg_score < 3.5:
            return "N3"
        elif avg_score < 4.5:
            return "N2"
        else:
            return "N1"


class SubtitleProcessorV2:
    """Enhanced subtitle processor for Stage 1"""
    
    def __init__(self, subtitles_dir: str = "data/subtitles"):
        """
        Initialize the enhanced processor
        
        Args:
            subtitles_dir: Directory containing SRT files
        """
        self.subtitles_dir = Path(subtitles_dir)
        self.subtitles_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = JapaneseTokenizer()
        self.filename_parser = FilenameParser()
        self.level_estimator = JLPTLevelEstimator()
    
    def parse_srt(self, filepath: str) -> List[Dict]:
        """
        Parse an SRT subtitle file
        
        Args:
            filepath: Path to the SRT file
            
        Returns:
            List of subtitle entries with text, timestamps, and metadata
        """
        print(f"📖 Parsing: {Path(filepath).name}")
        
        try:
            subs = pysrt.open(filepath, encoding='utf-8')
        except:
            try:
                subs = pysrt.open(filepath, encoding='shift-jis')
            except Exception as e:
                print(f"❌ Failed to parse {filepath}: {e}")
                return []
        
        entries = []
        for sub in subs:
            # Clean text
            text = sub.text.strip()
            text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
            text = re.sub(r'\{[^}]+\}', '', text)  # Remove formatting
            text = re.sub(r'\[.*?\]', '', text)  # Remove brackets
            
            if not text or len(text) < 2:
                continue
            
            entry = {
                'text': text,
                'start_time': str(sub.start),
                'end_time': str(sub.end),
                'index': sub.index,
                'duration_seconds': sub.duration.seconds
            }
            entries.append(entry)
        
        print(f"✅ Parsed {len(entries)} subtitle lines")
        return entries
    
    def analyze_subtitle_file(self, filepath: str) -> Dict:
        """
        Fully analyze a subtitle file with enhanced metadata
        
        Args:
            filepath: Path to SRT file
            
        Returns:
            Dictionary with rich episode metadata and analyzed content
        """
        # Parse filename for metadata
        filename = Path(filepath).name
        metadata = self.filename_parser.parse_filename(filename)
        
        print(f"\n📺 Analyzing: {metadata.title}")
        
        # Parse subtitles
        entries = self.parse_srt(filepath)
        
        if not entries:
            return None
        
        # Analyze each entry with tokenization
        analyzed_entries = []
        all_tokens = []
        level_counts = {'N5': 0, 'N4': 0, 'N3': 0, 'N2': 0, 'N1': 0}
        
        for entry in entries:
            text = entry['text']
            
            # Tokenize
            tokens = self.tokenizer.tokenize(text)
            
            # Estimate level
            level = self.level_estimator.estimate_level(text, tokens)
            level_counts[level] += 1
            
            # Extract vocabulary (base forms of content words)
            vocab = [
                t['base_form'] 
                for t in tokens 
                if t['length'] >= 2  # Skip single characters
            ]
            
            all_tokens.extend(tokens)
            
            analyzed_entry = {
                **entry,
                'jlpt_level': level,
                'vocab': vocab[:10],  # Top 10 words per line
                'char_count': len(text),
                'token_count': len(tokens)
            }
            analyzed_entries.append(analyzed_entry)
        
        # Determine overall episode level (weighted by line count)
        episode_level = max(level_counts, key=level_counts.get)
        
        # Calculate statistics
        total_lines = len(analyzed_entries)
        avg_chars = sum(e['char_count'] for e in analyzed_entries) / total_lines
        total_duration = sum(e['duration_seconds'] for e in analyzed_entries)
        
        # Get unique vocabulary
        all_vocab = []
        for e in analyzed_entries:
            all_vocab.extend(e['vocab'])
        unique_vocab = list(set(all_vocab))
        
        result = {
            # Metadata from filename
            'episode_id': metadata.episode_id,
            'anime_name': metadata.anime_name,
            'season': metadata.season,
            'episode': metadata.episode,
            'title': metadata.title,
            'original_filename': metadata.original_filename,
            
            # Content statistics
            'total_lines': total_lines,
            'episode_level': episode_level,
            'level_distribution': level_counts,
            'avg_chars_per_line': round(avg_chars, 1),
            'unique_vocab_count': len(unique_vocab),
            'total_duration_seconds': total_duration,
            'avg_line_duration': round(total_duration / total_lines, 1) if total_lines > 0 else 0,
            
            # Detailed entries
            'entries': analyzed_entries
        }
        
        print(f"""
📊 Analysis Summary:
   Anime: {metadata.anime_name}
   Episode: S{metadata.season or 0}E{metadata.episode}
   Level: {episode_level}
   Lines: {total_lines}
   Duration: {total_duration // 60}m {total_duration % 60}s
   Unique Vocab: {len(unique_vocab)}
   Avg Characters: {result['avg_chars_per_line']}
        """)
        
        return result
    
    def process_all_subtitles(self) -> List[Dict]:
        """
        Process all SRT files in the subtitles directory
        
        Returns:
            List of analyzed episode metadata
        """
        srt_files = list(self.subtitles_dir.glob("*.srt"))
        
        if not srt_files:
            print(f"⚠️  No SRT files found in {self.subtitles_dir}")
            print("Please add Japanese subtitle files (.srt) to get started!")
            return []
        
        print(f"🎬 Found {len(srt_files)} subtitle files")
        
        all_episodes = []
        for srt_file in sorted(srt_files):  # Sort for consistent ordering
            try:
                result = self.analyze_subtitle_file(str(srt_file))
                if result:
                    all_episodes.append(result)
            except Exception as e:
                print(f"❌ Error processing {srt_file.name}: {e}")
                continue
        
        return all_episodes


def main():
    """Main function to process subtitles"""
    
    print("="*60)
    print("🎌 LinguaSync Subtitle Processor V2 - Stage 1")
    print("="*60)
    
    # Initialize enhanced processor
    processor = SubtitleProcessorV2()
    
    # Process all subtitles
    episodes = processor.process_all_subtitles()
    
    if not episodes:
        print("\n⚠️  No episodes processed. Add SRT files to data/subtitles/")
        return
    
    # Save processed data
    output_file = "data/processed_episodes_v2.json"
    os.makedirs("data", exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Processed {len(episodes)} episodes")
    print(f"💾 Saved to: {output_file}")
    
    # Print summary by anime
    anime_summary = {}
    for ep in episodes:
        anime = ep['anime_name']
        if anime not in anime_summary:
            anime_summary[anime] = []
        anime_summary[anime].append(ep)
    
    print("\n📚 Content Library:")
    for anime, eps in anime_summary.items():
        print(f"   {anime}: {len(eps)} episodes")
    
    print("\n🚀 Next: Run rag_engine_v2.py to create vector embeddings")


if __name__ == "__main__":
    main()