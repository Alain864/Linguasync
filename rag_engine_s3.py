"""
rag_engine_s3.py - S3-Based RAG Engine (OpenSearch Replacement)

Cost-effective alternative using:
- S3 for data storage
- FAISS for local vector similarity search
- JSON files for metadata
- No expensive cloud vector database needed

This maintains all functionality while dramatically reducing costs.
"""

import os
import json
import logging
import pickle
import boto3
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
import faiss
from collections import defaultdict

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def setup_logging():
    """Setup logging"""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.info("Logging initialized for S3 RAG Engine")

setup_logging()


class S3VectorStore:
    """
    S3-based vector storage using FAISS for similarity search
    
    Architecture:
    - Vectors stored in FAISS index (fast local search)
    - Metadata stored in JSON (lightweight, queryable)
    - Both synced to S3 for persistence
    - Local cache for performance
    """
    
    def __init__(self, bucket_name: str, prefix: str = "vectors/"):
        """
        Initialize S3 vector store
        
        Args:
            bucket_name: S3 bucket name
            prefix: S3 prefix for vector storage
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        
        # Local cache directory
        self.cache_dir = Path("/tmp/linguasync_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # FAISS index (will be loaded or created)
        self.dimension = 1536  # OpenAI embedding dimension
        self.index = None
        self.metadata = []  # List of metadata dicts, indexed by FAISS ID
        
        # Index files
        self.index_file = self.cache_dir / "faiss_index.bin"
        self.metadata_file = self.cache_dir / "metadata.json"
        
        # Load or initialize
        self._load_or_initialize()
        
        logger.info(f"✅ S3 Vector Store initialized: s3://{bucket_name}/{prefix}")
    
    def _load_or_initialize(self):
        """Load existing index from S3 or create new one"""
        try:
            # Try to download from S3
            logger.info("📥 Attempting to load index from S3...")
            
            # Download FAISS index
            index_s3_key = f"{self.prefix}faiss_index.bin"
            self.s3_client.download_file(
                self.bucket_name,
                index_s3_key,
                str(self.index_file)
            )
            
            # Download metadata
            metadata_s3_key = f"{self.prefix}metadata.json"
            self.s3_client.download_file(
                self.bucket_name,
                metadata_s3_key,
                str(self.metadata_file)
            )
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_file))
            
            # Load metadata
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            logger.info(f"✅ Loaded index with {len(self.metadata)} documents from S3")
            
        except Exception as e:
            logger.info(f"📝 Creating new index (no existing index found: {e})")
            # Create new FAISS index (using cosine similarity)
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine
            self.metadata = []
    
    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        """
        Add documents with their embeddings to the index
        
        Args:
            documents: List of document metadata
            embeddings: Corresponding embeddings
        """
        if not documents or not embeddings:
            return
        
        # Normalize embeddings for cosine similarity
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Add metadata
        self.metadata.extend(documents)
        
        logger.info(f"➕ Added {len(documents)} documents to index")
    
    def search(self, query_embedding: List[float], filters: Dict = None, k: int = 10) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query vector
            filters: Optional filters (type, level, anime_name, episode_id)
            k: Number of results
            
        Returns:
            List of matching documents with scores
        """
        if self.index.ntotal == 0:
            return []
        
        # Normalize query for cosine similarity
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # Search more results to allow for filtering
        search_k = min(k * 10, self.index.ntotal)
        scores, indices = self.index.search(query_array, search_k)
        
        # Collect results with filtering
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            doc = self.metadata[idx].copy()
            doc['score'] = float(score)
            
            # Apply filters
            if filters:
                if not self._match_filters(doc, filters):
                    continue
            
            results.append(doc)
            
            if len(results) >= k:
                break
        
        return results
    
    def _match_filters(self, doc: Dict, filters: Dict) -> bool:
        """Check if document matches filters"""
        for key, value in filters.items():
            if value is None:
                continue
            if key not in doc:
                return False
            if doc[key] != value:
                return False
        return True
    
    def save_to_s3(self):
        """Save index and metadata to S3"""
        try:
            # Save FAISS index locally
            faiss.write_index(self.index, str(self.index_file))
            
            # Save metadata locally
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            # Upload to S3
            index_s3_key = f"{self.prefix}faiss_index.bin"
            self.s3_client.upload_file(
                str(self.index_file),
                self.bucket_name,
                index_s3_key
            )
            
            metadata_s3_key = f"{self.prefix}metadata.json"
            self.s3_client.upload_file(
                str(self.metadata_file),
                self.bucket_name,
                metadata_s3_key
            )
            
            logger.info(f"💾 Saved index to S3: s3://{self.bucket_name}/{self.prefix}")
            
        except Exception as e:
            logger.error(f"❌ Error saving to S3: {e}")
            raise
    
    def get_by_id(self, doc_id: str, doc_type: str = None) -> Optional[Dict]:
        """Get document by ID"""
        for doc in self.metadata:
            if doc.get('episode_id') == doc_id or doc.get('anime_name') == doc_id:
                if doc_type is None or doc.get('type') == doc_type:
                    return doc
        return None
    
    def filter_by_type(self, doc_type: str) -> List[Dict]:
        """Get all documents of a specific type"""
        return [doc for doc in self.metadata if doc.get('type') == doc_type]
    
    def count_by_type(self, doc_type: str) -> int:
        """Count documents by type"""
        return sum(1 for doc in self.metadata if doc.get('type') == doc_type)


class RAGEngineS3:
    """
    S3-based RAG engine - drop-in replacement for OpenSearch version
    
    Key differences:
    - Uses FAISS instead of OpenSearch
    - Stores everything in S3
    - Local caching for performance
    - Much lower cost
    """
    
    def __init__(self):
        """Initialize the S3-based RAG engine"""
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'linguasync-data')
        self.s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        
        # Initialize S3 vector store
        self.vector_store = S3VectorStore(self.bucket_name, prefix="vectors/")
        
        logger.info("✅ RAG Engine S3 initialized")
    
    def create_embedding(self, text: str) -> List[float]:
        """Create vector embedding using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"❌ Embedding error: {e}")
            raise
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Create embeddings in batches (much faster!)
        
        OpenAI allows up to 2048 inputs per request.
        We use smaller batches to be safe.
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.openai_client.embeddings.create(
                    input=batch,
                    model="text-embedding-3-small"
                )
                # Extract embeddings in order
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                if (i + batch_size) % 500 == 0:
                    logger.info(f"   Processed {min(i + batch_size, len(texts))}/{len(texts)} embeddings...")
                
            except Exception as e:
                logger.error(f"❌ Batch embedding error at position {i}: {e}")
                # Fall back to individual embeddings for this batch
                for text in batch:
                    try:
                        emb = self.create_embedding(text)
                        all_embeddings.append(emb)
                    except:
                        # Use zero vector as fallback
                        all_embeddings.append([0.0] * 1536)
        
        return all_embeddings
    
    def load_episodes_from_s3(self, prefix: str = "processed/") -> List[Dict]:
        """
        Load episode data from S3
        
        Args:
            prefix: S3 prefix where episodes are stored
            
        Returns:
            List of episode dictionaries
        """
        episodes = []
        
        try:
            # List all JSON files in the processed folder
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                logger.warning(f"No files found in s3://{self.bucket_name}/{prefix}")
                return episodes
            
            for obj in response['Contents']:
                key = obj['Key']
                if not key.endswith('.json'):
                    continue
                
                # Download and parse JSON
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
                content = response['Body'].read().decode('utf-8')
                episode = json.loads(content)
                episodes.append(episode)
            
            logger.info(f"📥 Loaded {len(episodes)} episodes from S3")
            return episodes
            
        except Exception as e:
            logger.error(f"❌ Error loading from S3: {e}")
            return []
    
    def index_episodes(self, episodes: List[Dict]):
        """
        Index episodes into S3 vector store
        
        This creates three types of documents:
        1. Anime-level (one per series)
        2. Episode-level (one per episode)
        3. Line-level (dialogue lines) - BATCHED for speed
        """
        logger.info(f"🔨 Indexing {len(episodes)} episodes...")
        
        # Track anime series
        anime_dict = defaultdict(lambda: {
            'episodes': [],
            'episode_ids': [],
            'total_lines': 0,
            'levels': []
        })
        
        all_documents = []
        all_embeddings = []
        
        # First pass: Collect all texts that need embeddings
        logger.info("\n📝 Collecting texts for batch embedding...")
        episode_texts = []
        episode_docs = []
        line_texts = []
        line_docs = []
        
        # Process each episode
        for idx, episode in enumerate(episodes):
            episode_id = episode.get('episode_id')
            anime_name = episode.get('anime_name')
            
            # Get level from episode_level field
            episode_level = episode.get('episode_level', episode.get('level', 'N3'))
            
            if (idx + 1) % 20 == 0:
                logger.info(f"   Collected {idx + 1}/{len(episodes)} episodes...")
            
            # Track anime
            anime_dict[anime_name]['episodes'].append(episode)
            anime_dict[anime_name]['episode_ids'].append(episode_id)
            anime_dict[anime_name]['total_lines'] += episode.get('total_lines', len(episode.get('entries', [])))
            anime_dict[anime_name]['levels'].append(episode_level)
            
            # 1. Episode-level document
            episode_text = self._create_episode_summary(episode)
            
            # Get episode number and vocab count
            episode_number = episode.get('episode', episode.get('episode_number'))
            total_lines = episode.get('total_lines', len(episode.get('entries', [])))
            vocab_count = episode.get('unique_vocab_count', len(episode.get('unique_vocab', [])))
            duration = episode.get('total_duration_seconds', episode.get('duration_seconds', 0))
            
            episode_doc = {
                'type': 'episode',
                'episode_id': episode_id,
                'anime_name': anime_name,
                'season': episode.get('season'),
                'episode_number': episode_number,
                'title': episode.get('title'),
                'level': episode_level,
                'total_lines': total_lines,
                'vocab_count': vocab_count,
                'duration': duration,
                'document': episode_text
            }
            
            episode_texts.append(episode_text)
            episode_docs.append(episode_doc)
            
            # 2. Line-level documents (sample for vocabulary)
            entries = episode.get('entries', episode.get('lines', []))
            # Sample every Nth line to avoid too many documents
            sample_rate = max(1, len(entries) // 50)
            
            for entry in entries[::sample_rate]:
                line_text = entry.get('text', '')
                if not line_text or len(line_text) < 3:
                    continue
                
                # Get line level from jlpt_level field
                line_level = entry.get('jlpt_level', entry.get('level', episode_level))
                
                line_doc = {
                    'type': 'line',
                    'episode_id': episode_id,
                    'anime_name': anime_name,
                    'level': line_level,
                    'timestamp': entry.get('start_time', entry.get('timestamp', '')),
                    'text': line_text,
                    'vocab': ','.join([v for v in entry.get('vocab', []) if v and v != '*']),
                    'document': line_text
                }
                
                line_texts.append(line_text)
                line_docs.append(line_doc)
        
        # Second pass: Create embeddings in batches
        logger.info(f"\n🚀 Creating embeddings in batches...")
        logger.info(f"   Episodes: {len(episode_texts)}")
        logger.info(f"   Lines: {len(line_texts)}")
        logger.info(f"   Total: {len(episode_texts) + len(line_texts)}")
        
        # Batch embed episodes
        logger.info(f"\n📺 Embedding episodes...")
        episode_embeddings = self.create_embeddings_batch(episode_texts, batch_size=100)
        
        # Add episodes to index
        for doc, embedding in zip(episode_docs, episode_embeddings):
            all_documents.append(doc)
            all_embeddings.append(embedding)
        
        logger.info(f"   ✅ Added {len(episode_docs)} episodes")
        
        # Batch embed lines
        logger.info(f"\n💬 Embedding dialogue lines...")
        line_embeddings = self.create_embeddings_batch(line_texts, batch_size=100)
        
        # Add lines to index
        for doc, embedding in zip(line_docs, line_embeddings):
            all_documents.append(doc)
            all_embeddings.append(embedding)
        
        logger.info(f"   ✅ Added {len(line_docs)} lines")
        
        # Add all documents to vector store
        logger.info(f"\n💾 Adding {len(all_documents)} documents to vector store...")
        self.vector_store.add_documents(all_documents, all_embeddings)
        
        # 3. Anime-level documents
        logger.info(f"\n📚 Indexing {len(anime_dict)} anime series...")
        anime_texts = []
        anime_docs = []
        
        for anime_name, data in anime_dict.items():
            most_common_level = max(set(data['levels']), key=data['levels'].count)
            
            anime_text = f"{anime_name} anime series with {len(data['episodes'])} episodes"
            
            anime_doc = {
                'type': 'anime',
                'anime_name': anime_name,
                'episode_count': len(data['episodes']),
                'typical_level': most_common_level,
                'episode_ids': data['episode_ids'],
                'total_lines': data['total_lines'],
                'document': anime_text
            }
            
            anime_texts.append(anime_text)
            anime_docs.append(anime_doc)
        
        # Batch embed anime
        anime_embeddings = self.create_embeddings_batch(anime_texts, batch_size=10)
        self.vector_store.add_documents(anime_docs, anime_embeddings)
        
        # Save everything to S3
        logger.info("\n💾 Saving index to S3...")
        self.vector_store.save_to_s3()
        
        logger.info("\n✅ Indexing complete!")
        logger.info(f"📊 Total documents: {len(self.vector_store.metadata)}")
    
    def _create_episode_summary(self, episode: Dict) -> str:
        """Create searchable text summary for an episode"""
        # Get level from episode_level field (not 'level')
        level = episode.get('episode_level', episode.get('level', 'N3'))
        
        parts = [
            f"Anime: {episode.get('anime_name')}",
            f"Episode {episode.get('episode', episode.get('episode_number', 'Unknown'))}: {episode.get('title')}",
            f"JLPT Level: {level}",
            f"Contains {episode.get('total_lines', len(episode.get('entries', [])))} dialogue lines"
        ]
        
        # Add sample vocabulary from first few entries
        entries = episode.get('entries', [])
        vocab_set = set()
        for entry in entries[:10]:
            vocab_list = entry.get('vocab', [])
            for word in vocab_list:
                if word and word != '*':
                    vocab_set.add(word)
        
        if vocab_set:
            vocab_list = list(vocab_set)[:20]
            parts.append(f"Vocabulary: {', '.join(vocab_list)}")
        
        return " | ".join(parts)
    
    def search_episodes_by_level(self, level: str, query: str = "", n_results: int = 10) -> List[Dict]:
        """
        Search for episodes matching level and query
        
        Args:
            level: JLPT level (N5, N4, N3, N2, N1)
            query: Optional search query
            n_results: Number of results
            
        Returns:
            List of matching episodes
        """
        # Create search text
        search_text = f"Japanese {level} level content"
        if query:
            search_text = f"{query} {search_text}"
        
        # Create embedding
        query_embedding = self.create_embedding(search_text)
        
        # Search with flexibility (adjacent levels)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            filters={'type': 'episode'},
            k=n_results * 3
        )
        
        # Filter by level (allow adjacent levels)
        level_order = ['N5', 'N4', 'N3', 'N2', 'N1']
        target_idx = level_order.index(level)
        
        episodes = []
        for doc in results:
            episode_level = doc.get('level')
            if episode_level not in level_order:
                continue
            
            episode_idx = level_order.index(episode_level)
            
            # Allow same level or adjacent
            if abs(target_idx - episode_idx) <= 1:
                episodes.append({
                    'episode_id': doc['episode_id'],
                    'anime_name': doc['anime_name'],
                    'season': doc.get('season'),
                    'episode_number': doc['episode_number'],
                    'title': doc['title'],
                    'level': doc['level'],
                    'total_lines': doc['total_lines'],
                    'vocab_count': doc['vocab_count'],
                    'duration_minutes': doc['duration'] // 60,
                    'relevance_score': round(doc['score'], 3)
                })
        
        # Sort by relevance
        episodes.sort(key=lambda x: x['relevance_score'], reverse=True)
        return episodes[:n_results]
    
    def search_by_anime(self, anime_name: str, level: Optional[str] = None) -> List[Dict]:
        """
        Find all episodes of a specific anime
        
        Args:
            anime_name: Name of the anime
            level: Optional JLPT level filter
            
        Returns:
            List of episodes
        """
        # Get all episode documents
        all_episodes = self.vector_store.filter_by_type('episode')
        
        # Filter by anime name (case-insensitive, fuzzy)
        search_name = anime_name.lower().replace(' ', '').replace('_', '')
        matching_episodes = []
        
        for doc in all_episodes:
            db_name = doc['anime_name'].lower().replace(' ', '').replace('_', '')
            
            # Check if names match
            if search_name in db_name or db_name in search_name:
                # Apply level filter if specified
                if level and doc.get('level') != level:
                    continue
                
                matching_episodes.append({
                    'episode_id': doc['episode_id'],
                    'anime_name': doc['anime_name'],
                    'season': doc.get('season'),
                    'episode_number': doc['episode_number'],
                    'title': doc['title'],
                    'level': doc['level'],
                    'total_lines': doc.get('total_lines', 0),
                    'vocab_count': doc.get('vocab_count', 0)
                })
        
        # Sort by season and episode number
        matching_episodes.sort(key=lambda x: (x.get('season') or 0, x['episode_number']))
        
        logger.info(f"Found {len(matching_episodes)} episodes for {anime_name}")
        return matching_episodes
    
    def find_vocabulary_examples(self, episode_id: str, n_examples: int = 10) -> List[Dict]:
        """Find example dialogue lines from a specific episode"""
        search_text = "Japanese vocabulary example sentences"
        query_embedding = self.create_embedding(search_text)
        
        results = self.vector_store.search(
            query_embedding=query_embedding,
            filters={'type': 'line', 'episode_id': episode_id},
            k=n_examples * 2
        )
        
        examples = []
        for doc in results:
            examples.append({
                'text': doc['text'],
                'timestamp': doc.get('timestamp', ''),
                'level': doc.get('level', ''),
                'vocab': doc.get('vocab', '').split(',') if doc.get('vocab') else []
            })
        
        return examples[:n_examples]
    
    def get_collection_stats(self) -> Dict:
        """Get comprehensive statistics about the content library"""
        try:
            stats = {
                'total_items': len(self.vector_store.metadata),
                'anime_count': self.vector_store.count_by_type('anime'),
                'episode_count': self.vector_store.count_by_type('episode'),
                'line_count': self.vector_store.count_by_type('line'),
                'anime_list': []
            }
            
            # Get anime list
            anime_docs = self.vector_store.filter_by_type('anime')
            for doc in anime_docs:
                stats['anime_list'].append({
                    'name': doc['anime_name'],
                    'episodes': doc['episode_count'],
                    'level': doc['typical_level']
                })
            
            stats['unique_anime'] = len(stats['anime_list'])
            
            return stats
        
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {'error': str(e)}


def main():
    """Main function to initialize and index content"""
    
    logger.info("="*60)
    logger.info("🔍 LinguaSync RAG Engine S3 (OpenSearch Replacement)")
    logger.info("="*60)
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not found")
        return
    
    if not os.getenv("S3_BUCKET_NAME"):
        logger.error("❌ S3_BUCKET_NAME not found")
        return
    
    rag = RAGEngineS3()
    
    # Load episodes from S3
    episodes = rag.load_episodes_from_s3()
    
    if not episodes:
        logger.error("❌ No episodes found in S3")
        return
    
    # Index episodes
    rag.index_episodes(episodes)
    
    # Show statistics
    logger.info("\n" + "="*60)
    stats = rag.get_collection_stats()
    logger.info(f"📊 Content Library Statistics:")
    logger.info(f"   Total items: {stats['total_items']}")
    logger.info(f"   Anime series: {stats['unique_anime']}")
    logger.info(f"   Episodes: {stats['episode_count']}")
    logger.info(f"   Lines: {stats['line_count']}")
    
    if stats['anime_list']:
        logger.info(f"\n📚 Available Anime:")
        for anime in stats['anime_list']:
            logger.info(f"   - {anime['name']}: {anime['episodes']} episodes ({anime['level']})")
    
    logger.info("\n✅ S3 RAG Engine ready!")
    logger.info("💰 Cost savings: ~95% vs OpenSearch Serverless")


if __name__ == "__main__":
    main()