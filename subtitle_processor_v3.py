"""
subtitle_processor_v3.py - Subtitle Processor for Stage 2

New features:
- Multi-folder structure: data/subtitles/Anime_Name/season_X/episode_Y.srt
- Organized processing by show/season/episode
- S3 upload support for processed data
- CloudWatch logging integration
"""

import os
import re
import pysrt
import json
import boto3
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Try to import MeCab for proper Japanese tokenization
try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False
    print("⚠️  MeCab not available. Using fallback tokenization.")

# Configure CloudWatch logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add CloudWatch handler if running on AWS
def setup_cloudwatch_logging():
    """Setup CloudWatch logging for AWS environment"""
    try:
        cloudwatch = boto3.client('logs', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        log_group = '/linguasync/subtitle-processor'
        log_stream = f'processing-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        
        # Create log group if it doesn't exist
        try:
            cloudwatch.create_log_group(logGroupName=log_group)
        except cloudwatch.exceptions.ResourceAlreadyExistsException:
            pass
        
        # Create log stream
        try:
            cloudwatch.create_log_stream(logGroupName=log_group, logStreamName=log_stream)
        except:
            pass
        
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        
        logger.info(f"CloudWatch logging initialized: {log_group}/{log_stream}")
    except Exception as e:
        # Fallback to console logging
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.warning(f"CloudWatch setup failed, using console logging: {e}")

setup_cloudwatch_logging()


@dataclass
class EpisodeMetadata:
    """Structured metadata for an episode"""
    anime_name: str
    season: Optional[int]
    episode: int
    episode_id: str
    original_filename: str
    title: str
    file_path: str  # Path within the folder structure


class FilenameParser:
    """
    Parses subtitle filenames and folder structure to extract metadata
    
    Supports folder structure:
    - data/subtitles/Attack_on_Titan/season_1/episode_01.srt
    - data/subtitles/Naruto_Shippuden/season_5/episode_120.srt
    """
    
    @staticmethod
    def parse_from_path(file_path: Path, base_dir: Path) -> EpisodeMetadata:
        """
        Parse file path to extract anime name, season, and episode
        
        Args:
            file_path: Full path to subtitle file
            base_dir: Base subtitles directory
            
        Returns:
            EpisodeMetadata object with parsed information
        """
        # Get relative path from base directory
        try:
            rel_path = file_path.relative_to(base_dir)
            parts = rel_path.parts
        except ValueError:
            # Fallback if not relative to base_dir
            parts = file_path.parts[-3:]  # Take last 3 parts
        
        # Extract anime name from folder
        if len(parts) >= 3:
            anime_name = parts[0].replace('_', ' ')
            season_folder = parts[1]
            episode_file = parts[2]
            
            # Parse season number
            season_match = re.search(r'season[_\s](\d+)', season_folder, re.IGNORECASE)
            season = int(season_match.group(1)) if season_match else None
            
            # Parse episode number
            episode_match = re.search(r'[eE](\d+)', episode_file, re.IGNORECASE)
            if episode_match:
                episode = int(episode_match.group(1))
            else:
                # Try numeric pattern
                num_match = re.search(r'(\d+)', episode_file)
                episode = int(num_match.group(1)) if num_match else 1
        
        elif len(parts) == 2:
            # No season folder: data/subtitles/Anime_Name/episode_01.srt
            anime_name = parts[0].replace('_', ' ')
            season = None
            episode_file = parts[1]
            
            episode_match = re.search(r'[eE](\d+)', episode_file, re.IGNORECASE)
            if episode_match:
                episode = int(episode_match.group(1))
            else:
                num_match = re.search(r'(\d+)', episode_file)
                episode = int(num_match.group(1)) if num_match else 1
        
        else:
            # Single file - use filename parsing
            anime_name = file_path.stem.replace('_', ' ')
            season = None
            episode = 1
        
        # Generate episode_id
        anime_id = anime_name.lower().replace(' ', '_')
        if season:
            episode_id = f"{anime_id}_s{season:02d}e{episode:02d}"
            title = f"{anime_name} - S{season:02d}E{episode:02d}"
        else:
            episode_id = f"{anime_id}_e{episode:02d}"
            title = f"{anime_name} - E{episode:02d}"
        
        return EpisodeMetadata(
            anime_name=anime_name,
            season=season,
            episode=episode,
            episode_id=episode_id,
            original_filename=file_path.name,
            title=title,
            file_path=str(rel_path) if len(parts) >= 2 else str(file_path.name)
        )


class JapaneseTokenizer:
    """Tokenizes Japanese text using MeCab (if available) or fallback method"""
    
    def __init__(self):
        self.mecab = None
        if MECAB_AVAILABLE:
            try:
                self.mecab = MeCab.Tagger()
                logger.info("✅ Using MeCab for Japanese tokenization")
            except Exception as e:
                logger.warning(f"⚠️  MeCab initialization failed: {e}")
    
    def tokenize(self, text: str) -> List[Dict[str, str]]:
        """Tokenize Japanese text into words with metadata"""
        if self.mecab:
            return self._tokenize_mecab(text)
        else:
            return self._tokenize_fallback(text)
    
    def _tokenize_mecab(self, text: str) -> List[Dict[str, str]]:
        """Tokenize using MeCab (accurate)"""
        tokens = []
        parsed = self.mecab.parse(text)
        
        for line in parsed.split('\n'):
            if line == 'EOS' or line == '':
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            surface = parts[0]
            features = parts[1].split(',')
            pos = features[0] if len(features) > 0 else 'Unknown'
            base_form = features[6] if len(features) > 6 else surface
            
            tokens.append({
                'surface': surface,
                'base_form': base_form,
                'pos': pos,
                'length': len(surface)
            })
        
        return tokens
    
    def _tokenize_fallback(self, text: str) -> List[Dict[str, str]]:
        """Fallback tokenization using character types"""
        tokens = []
        current_token = ""
        current_type = None
        
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                char_type = 'kanji'
            elif '\u3040' <= char <= '\u309f':
                char_type = 'hiragana'
            elif '\u30a0' <= char <= '\u30ff':
                char_type = 'katakana'
            else:
                char_type = 'other'
            
            if char_type == current_type or current_type is None:
                current_token += char
                current_type = char_type
            else:
                if current_token.strip():
                    tokens.append({
                        'surface': current_token,
                        'base_form': current_token,
                        'pos': current_type,
                        'length': len(current_token)
                    })
                current_token = char
                current_type = char_type
        
        if current_token.strip():
            tokens.append({
                'surface': current_token,
                'base_form': current_token,
                'pos': current_type,
                'length': len(current_token)
            })
        
        return tokens


class JLPTLevelEstimator:
    """Estimates JLPT level based on text characteristics"""
    
    @staticmethod
    def estimate_level(text: str, tokens: List[Dict]) -> str:
        """Estimate JLPT level for a text segment"""
        if not tokens or len(text) == 0:
            return "N5"
        
        kanji_chars = len(re.findall(r'[\u4E00-\u9FFF]', text))
        kanji_ratio = kanji_chars / len(text)
        avg_word_length = sum(t['length'] for t in tokens) / len(tokens)
        unique_kanji = len(set(re.findall(r'[\u4E00-\u9FFF]', text)))
        
        score = 0
        
        # Kanji density scoring
        if kanji_ratio < 0.15:
            score += 1
        elif kanji_ratio < 0.25:
            score += 2
        elif kanji_ratio < 0.35:
            score += 3
        elif kanji_ratio < 0.45:
            score += 4
        else:
            score += 5
        
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
        
        avg_score = score / 3
        
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


class S3Uploader:
    """Handles uploading processed data to S3"""
    
    def __init__(self, bucket_name: str = None):
        """Initialize S3 uploader"""
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET_NAME', 'linguasync-data')
        self.s3_client = None
        
        try:
            self.s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
            logger.info(f"✅ S3 client initialized for bucket: {self.bucket_name}")
        except Exception as e:
            logger.warning(f"⚠️  S3 client initialization failed: {e}")
    
    def upload_processed_episode(self, episode_data: Dict, anime_name: str, season: int = None, episode: int = 1):
        """Upload processed episode data to S3"""
        if not self.s3_client:
            logger.warning("S3 client not available, skipping upload")
            return False
        
        try:
            # Construct S3 key
            anime_key = anime_name.lower().replace(' ', '_')
            if season:
                s3_key = f"processed/{anime_key}/season_{season:02d}/episode_{episode:02d}.json"
            else:
                s3_key = f"processed/{anime_key}/episode_{episode:02d}.json"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json.dumps(episode_data, ensure_ascii=False, indent=2),
                ContentType='application/json'
            )
            
            logger.info(f"✅ Uploaded to S3: s3://{self.bucket_name}/{s3_key}")
            return True
        
        except Exception as e:
            logger.error(f"❌ S3 upload failed: {e}")
            return False


class SubtitleProcessorV3:
    """Enhanced subtitle processor for Stage 2"""
    
    def __init__(self, subtitles_dir: str = "data/subtitles"):
        """Initialize the processor with multi-folder support"""
        self.subtitles_dir = Path(subtitles_dir)
        self.tokenizer = JapaneseTokenizer()
        self.filename_parser = FilenameParser()
        self.level_estimator = JLPTLevelEstimator()
        self.s3_uploader = S3Uploader()
        
        logger.info(f"📂 Initialized SubtitleProcessorV3 with base dir: {self.subtitles_dir}")
    
    def find_all_subtitles(self) -> List[Path]:
        """
        Recursively find all SRT files in the subtitles directory
        
        Returns:
            List of paths to SRT files
        """
        if not self.subtitles_dir.exists():
            logger.warning(f"Subtitles directory not found: {self.subtitles_dir}")
            return []
        
        srt_files = list(self.subtitles_dir.rglob("*.srt"))
        logger.info(f"🔍 Found {len(srt_files)} subtitle files")
        return srt_files
    
    def parse_srt(self, filepath: Path) -> List[Dict]:
        """Parse an SRT subtitle file"""
        logger.info(f"📖 Parsing: {filepath.name}")
        
        try:
            subs = pysrt.open(str(filepath), encoding='utf-8')
        except:
            try:
                subs = pysrt.open(str(filepath), encoding='shift-jis')
            except Exception as e:
                logger.error(f"❌ Failed to parse {filepath}: {e}")
                return []
        
        entries = []
        for sub in subs:
            text = sub.text.strip()
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'\{[^}]+\}', '', text)
            text = re.sub(r'\[.*?\]', '', text)
            
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
        
        logger.info(f"✅ Parsed {len(entries)} subtitle lines")
        return entries
    
    def analyze_subtitle_file(self, filepath: Path) -> Optional[Dict]:
        """Fully analyze a subtitle file with metadata"""
        metadata = self.filename_parser.parse_from_path(filepath, self.subtitles_dir)
        
        logger.info(f"📺 Analyzing: {metadata.title}")
        
        entries = self.parse_srt(filepath)
        if not entries:
            return None
        
        analyzed_entries = []
        all_tokens = []
        level_counts = {'N5': 0, 'N4': 0, 'N3': 0, 'N2': 0, 'N1': 0}
        
        for entry in entries:
            text = entry['text']
            tokens = self.tokenizer.tokenize(text)
            level = self.level_estimator.estimate_level(text, tokens)
            level_counts[level] += 1
            
            vocab = [t['base_form'] for t in tokens if t['length'] >= 2]
            all_tokens.extend(tokens)
            
            analyzed_entry = {
                **entry,
                'jlpt_level': level,
                'vocab': vocab[:10],
                'char_count': len(text),
                'token_count': len(tokens)
            }
            analyzed_entries.append(analyzed_entry)
        
        episode_level = max(level_counts, key=level_counts.get)
        total_lines = len(analyzed_entries)
        avg_chars = sum(e['char_count'] for e in analyzed_entries) / total_lines
        total_duration = sum(e['duration_seconds'] for e in analyzed_entries)
        
        all_vocab = []
        for e in analyzed_entries:
            all_vocab.extend(e['vocab'])
        unique_vocab = list(set(all_vocab))
        
        result = {
            'episode_id': metadata.episode_id,
            'anime_name': metadata.anime_name,
            'season': metadata.season,
            'episode': metadata.episode,
            'title': metadata.title,
            'original_filename': metadata.original_filename,
            'file_path': metadata.file_path,
            'total_lines': total_lines,
            'episode_level': episode_level,
            'level_distribution': level_counts,
            'avg_chars_per_line': round(avg_chars, 1),
            'unique_vocab_count': len(unique_vocab),
            'total_duration_seconds': total_duration,
            'avg_line_duration': round(total_duration / total_lines, 1) if total_lines > 0 else 0,
            'entries': analyzed_entries,
            'processed_at': datetime.now().isoformat()
        }
        
        logger.info(f"""
📊 Analysis Summary:
   Anime: {metadata.anime_name}
   Episode: {metadata.title}
   Level: {episode_level}
   Lines: {total_lines}
   Duration: {total_duration // 60}m {total_duration % 60}s
   Unique Vocab: {len(unique_vocab)}
        """)
        
        # Upload to S3
        self.s3_uploader.upload_processed_episode(
            result, 
            metadata.anime_name, 
            metadata.season, 
            metadata.episode
        )
        
        return result
    
    def process_all_subtitles(self) -> List[Dict]:
        """Process all SRT files in the directory structure"""
        srt_files = self.find_all_subtitles()
        
        if not srt_files:
            logger.warning(f"⚠️  No SRT files found in {self.subtitles_dir}")
            return []
        
        logger.info(f"🎬 Processing {len(srt_files)} subtitle files")
        
        all_episodes = []
        for srt_file in sorted(srt_files):
            try:
                result = self.analyze_subtitle_file(srt_file)
                if result:
                    all_episodes.append(result)
            except Exception as e:
                logger.error(f"❌ Error processing {srt_file.name}: {e}")
                continue
        
        return all_episodes


def main():
    """Main function to process subtitles"""
    
    logger.info("="*60)
    logger.info("🎌 LinguaSync Subtitle Processor V3 - Stage 2")
    logger.info("="*60)
    
    processor = SubtitleProcessorV3()
    episodes = processor.process_all_subtitles()
    
    if not episodes:
        logger.warning("\n⚠️  No episodes processed. Check subtitle directory structure!")
        return
    
    # Save locally
    output_file = "data/processed_episodes_v3.json"
    os.makedirs("data", exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n✅ Processed {len(episodes)} episodes")
    logger.info(f"💾 Saved to: {output_file}")
    
    # Summary by anime
    anime_summary = {}
    for ep in episodes:
        anime = ep['anime_name']
        if anime not in anime_summary:
            anime_summary[anime] = []
        anime_summary[anime].append(ep)
    
    logger.info("\n📚 Content Library:")
    for anime, eps in anime_summary.items():
        logger.info(f"   {anime}: {len(eps)} episodes")
    
    logger.info("\n🚀 Next: Run rag_engine_v3.py to create vector embeddings")


if __name__ == "__main__":
    main()