"""
Subtitle Processor V3 - Stage 2 (Fixed)
- Processes subtitles from nested folders
- Handles MeCab gracefully
- Uploads to S3 organized by show/season/episode
"""

import os
import sys
import json
import boto3
from pathlib import Path
from typing import Dict, List, Optional
from subtitle_processor_v2 import (
    FilenameParser,
    JLPTLevelEstimator
)

# Handle MeCab import gracefully
MECAB_AVAILABLE = False
try:
    # Suppress MeCab's verbose error messages
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import MeCab
        MECAB_AVAILABLE = True
        print("✅ MeCab available - using proper Japanese tokenization")
except Exception:
    print("ℹ️  MeCab not available - using fallback tokenization (this is fine)")


class JapaneseTokenizerV3:
    """Japanese tokenizer with better MeCab error handling"""
    
    def __init__(self):
        self.mecab = None
        
        if MECAB_AVAILABLE:
            try:
                self.mecab = MeCab.Tagger()
                # Test it works
                self.mecab.parse("テスト")
            except Exception as e:
                # MeCab installed but can't initialize
                self.mecab = None
    
    def tokenize(self, text: str) -> List[Dict[str, str]]:
        """Tokenize Japanese text"""
        if self.mecab:
            return self._tokenize_mecab(text)
        else:
            return self._tokenize_fallback(text)
    
    def _tokenize_mecab(self, text: str) -> List[Dict[str, str]]:
        """MeCab tokenization"""
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
        """Fallback tokenization - group by character type"""
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


class SubtitleProcessorV3:
    """Processes subtitles from nested folders"""
    
    def __init__(self, subtitles_dir: str = "data/subtitles"):
        self.subtitles_dir = Path(subtitles_dir)
        self.tokenizer = JapaneseTokenizerV3()
        self.parser = FilenameParser()
        self.estimator = JLPTLevelEstimator()
    
    def find_all_srt_files(self) -> List[Path]:
        """
        Find all .srt files recursively in subdirectories
        
        Handles structures like:
        data/subtitles/Death Note/death_note_01.srt
        data/subtitles/Cowboy Bebop/cowboy_bebop_s01e01.srt
        """
        srt_files = []
        
        if not self.subtitles_dir.exists():
            print(f"❌ Directory not found: {self.subtitles_dir}")
            return []
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.subtitles_dir):
            for file in files:
                if file.endswith('.srt'):
                    srt_files.append(Path(root) / file)
        
        return sorted(srt_files)
    
    def analyze_subtitle_file(self, filepath: Path) -> Dict:
        """Analyze a subtitle file"""
        print(f"📖 Processing: {filepath.name}")
        
        # Use existing Stage 0/1 logic
        from subtitle_processor_v2 import SubtitleProcessorV2
        processor = SubtitleProcessorV2()
        
        try:
            metadata = processor.analyze_subtitle_file(str(filepath))
            return metadata
        except Exception as e:
            print(f"❌ Error processing {filepath.name}: {e}")
            return None


class S3SubtitleProcessorV3:
    """Processes subtitles and uploads to S3"""
    
    def __init__(self, bucket_name: str):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
        self.processor = SubtitleProcessorV3()
        self.parser = FilenameParser()
    
    def upload_raw_subtitle(self, local_path: Path) -> str:
        """Upload raw subtitle to S3 with organized structure"""
        filename = local_path.name
        metadata = self.parser.parse_filename(filename)
        
        # Create organized S3 path
        anime_slug = metadata.anime_name.lower().replace(' ', '_')
        season = f"s{metadata.season:02d}" if metadata.season else "s01"
        episode = f"e{metadata.episode:02d}"
        
        s3_key = f"raw/{anime_slug}/{season}/{episode}.srt"
        
        # Upload
        self.s3_client.upload_file(
            str(local_path),
            self.bucket_name,
            s3_key
        )
        
        print(f"  ✅ Uploaded: {s3_key}")
        return s3_key
    
    def process_and_store(self, s3_key: str, local_path: Path) -> Dict:
        """Process subtitle and upload results to S3"""
        # Process
        metadata = self.processor.analyze_subtitle_file(local_path)
        
        if not metadata:
            return None
        
        # Create processed S3 key
        processed_key = s3_key.replace('raw/', 'processed/').replace('.srt', '.json')
        
        # Upload processed data
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=processed_key,
            Body=json.dumps(metadata, ensure_ascii=False, indent=2),
            ContentType='application/json'
        )
        
        print(f"  ✅ Processed: {processed_key}")
        
        return metadata
    
    def process_all_local_files(self) -> List[Dict]:
        """
        Process all subtitle files from nested folders
        
        Handles:
        data/subtitles/Death Note/*.srt
        data/subtitles/Cowboy Bebop/*.srt
        """
        # Find all .srt files recursively
        srt_files = self.processor.find_all_srt_files()
        
        if not srt_files:
            print(f"\n⚠️  No .srt files found in {self.processor.subtitles_dir}")
            print("\n💡 Expected structure:")
            print("   data/subtitles/Death Note/death_note_01.srt")
            print("   data/subtitles/Cowboy Bebop/cowboy_bebop_s01e01.srt")
            return []
        
        print(f"\n📁 Found {len(srt_files)} subtitle files in subdirectories")
        
        # Show folder structure
        folders = set(f.parent.name for f in srt_files)
        print(f"📺 Anime folders: {', '.join(sorted(folders))}")
        
        all_metadata = []
        
        for srt_file in srt_files:
            try:
                # Upload raw file
                s3_key = self.upload_raw_subtitle(srt_file)
                
                # Process and store
                metadata = self.process_and_store(s3_key, srt_file)
                
                if metadata:
                    all_metadata.append(metadata)
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        return all_metadata
    
    def list_all_processed(self) -> List[str]:
        """List all processed files in S3"""
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix='processed/'
        )
        
        if 'Contents' not in response:
            return []
        
        return [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.json')]
    
    def get_processed_episode(self, s3_key: str) -> Dict:
        """Get processed episode from S3"""
        response = self.s3_client.get_object(
            Bucket=self.bucket_name,
            Key=s3_key
        )
        return json.loads(response['Body'].read())


def main():
    """Main processing function"""
    
    print("="*60)
    print("🎌 LinguaSync Subtitle Processor V3 - Stage 2")
    print("="*60)
    
    # Get bucket name
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    
    if not bucket_name:
        print("\n❌ Error: S3_BUCKET_NAME environment variable not set")
        print("   Set it with: export S3_BUCKET_NAME=your-bucket-name")
        return
    
    print(f"\n💾 S3 Bucket: {bucket_name}")
    
    # Initialize processor
    processor = S3SubtitleProcessorV3(bucket_name)
    
    # Process all files
    print("\n📤 Uploading and processing subtitle files...")
    all_metadata = processor.process_all_local_files()
    
    if not all_metadata:
        print("\n⚠️  No files were processed")
        return
    
    print(f"\n✅ Processed {len(all_metadata)} episodes")
    
    # Summary by anime
    anime_summary = {}
    for ep in all_metadata:
        anime = ep['anime_name']
        if anime not in anime_summary:
            anime_summary[anime] = 0
        anime_summary[anime] += 1
    
    print("\n📚 Content Library:")
    for anime, count in sorted(anime_summary.items()):
        print(f"   {anime}: {count} episodes")
    
    print(f"\n💾 Files organized in S3: s3://{bucket_name}/")
    print("   raw/ - Original subtitle files by anime/season/episode")
    print("   processed/ - Analyzed episode data")
    print("\n🚀 Next: Run index_to_opensearch.py to index in OpenSearch")


if __name__ == "__main__":
    main()