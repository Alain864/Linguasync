"""
rag_engine_v3.py - RAG Engine with Amazon OpenSearch Serverless

New features for Stage 2:
- Amazon OpenSearch Serverless for vector storage
- CloudWatch logging integration
- S3 integration for data retrieval
- Enhanced metadata for LangGraph workflows
"""

import os
import json
import logging
import boto3
from typing import List, Dict, Optional
from pathlib import Path
from openai import OpenAI
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def setup_cloudwatch_logging():
    """Setup CloudWatch logging"""
    try:
        log_group = '/linguasync/rag-engine'
        log_stream = f'indexing-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        
        logger.info(f"Logging initialized: {log_group}/{log_stream}")
    except Exception as e:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.warning(f"CloudWatch setup fallback: {e}")

setup_cloudwatch_logging()


class OpenSearchClient:
    """Client for Amazon OpenSearch Serverless"""
    
    def __init__(self, endpoint: str = None):
        """
        Initialize OpenSearch client
        
        Args:
            endpoint: OpenSearch Serverless endpoint URL
        """
        self.endpoint = endpoint or os.getenv('OPENSEARCH_ENDPOINT')
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.index_name = 'linguasync-episodes-v2'
        
        if not self.endpoint:
            raise ValueError("OPENSEARCH_ENDPOINT environment variable not set")
        
        # Setup AWS authentication
        credentials = boto3.Session().get_credentials()
        self.awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            self.region,
            'aoss',  # Amazon OpenSearch Serverless
            session_token=credentials.token
        )
        
        # Initialize OpenSearch client
        self.client = OpenSearch(
            hosts=[{'host': self.endpoint.replace('https://', ''), 'port': 443}],
            http_auth=self.awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300
        )
        
        logger.info(f"✅ OpenSearch client initialized: {self.endpoint}")
        
        # Create index if it doesn't exist
        self._create_index()
    
    def _create_index(self):
        """Create the vector index with proper mappings"""

        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 512
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 1536,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 512,
                                "m": 16
                            }
                        }
                    },
                    "document": {"type": "text"},
                    "type": {"type": "keyword"},
                    "episode_id": {"type": "keyword"},
                    "anime_name": {"type": "keyword"},
                    "season": {"type": "integer"},
                    "episode_number": {"type": "integer"},
                    "title": {"type": "text"},
                    "level": {"type": "keyword"},
                    "timestamp": {"type": "keyword"},
                    "text": {"type": "text"},
                    "vocab": {"type": "keyword"},
                    "total_lines": {"type": "integer"},
                    "vocab_count": {"type": "integer"},
                    "duration": {"type": "integer"},
                    "typical_level": {"type": "keyword"},
                    "episode_count": {"type": "integer"},
                    "episode_ids": {"type": "keyword"}
                }
            }
        }
        
        try:
            if not self.client.indices.exists(index=self.index_name):
                self.client.indices.create(index=self.index_name, body=index_body)
                logger.info(f"✨ Created index: {self.index_name}")
            else:
                logger.info(f"✅ Index already exists: {self.index_name}")
        except Exception as e:
            logger.error(f"❌ Error creating index: {e}")
            raise
    
    def index_document(self, doc_id: str, document: Dict):
        """Index a single document"""
        try:
            response = self.client.index(
            index=self.index_name,
            body=document
            )
            generated_id = response['_id']           # ← capture the auto-generated ID
            logger.debug(f"Indexed document with auto-ID: {generated_id}")
        except Exception as e:
            logger.error(f"❌ Error indexing document {doc_id}: {e}")
            raise
    
    def search(self, query_vector: List[float], filters: Dict = None, size: int = 10) -> Dict:
        """
        Search using vector similarity
        
        Args:
            query_vector: Embedding vector
            filters: Optional filters (type, level, anime_name, etc.)
            size: Number of results
            
        Returns:
            Search results
        """
        query = {
            "size": size,
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector,
                                    "k": size
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        # Add filters
        if filters:
            filter_clauses = []
            for key, value in filters.items():
                if value is not None:
                    filter_clauses.append({"term": {key: value}})
            
            if filter_clauses:
                query["query"]["bool"]["filter"] = filter_clauses
        
        try:
            response = self.client.search(
                index=self.index_name,
                body=query
            )
            return response
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return {"hits": {"hits": []}}


class RAGEngineV3:
    """Enhanced RAG engine with OpenSearch Serverless"""
    
    def __init__(self):
        """Initialize the RAG engine"""
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.opensearch = OpenSearchClient()
        self.s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'linguasync-data')
        
        logger.info("✅ RAG Engine V3 initialized")
    
    def create_embedding(self, text: str) -> List[float]:
        """Create vector embedding using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=1536
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"❌ Error creating embedding: {e}")
            raise
    
    def load_episodes_from_s3(self) -> List[Dict]:
        """
        Load processed episodes from S3
        
        Returns:
            List of episode metadata
        """
        episodes = []
        
        try:
            # List all processed JSON files in S3
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix='processed/')
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    if obj['Key'].endswith('.json'):
                        # Download and parse JSON
                        response = self.s3_client.get_object(
                            Bucket=self.bucket_name,
                            Key=obj['Key']
                        )
                        episode_data = json.loads(response['Body'].read())
                        episodes.append(episode_data)
            
            logger.info(f"📥 Loaded {len(episodes)} episodes from S3")
            return episodes
        
        except Exception as e:
            logger.error(f"❌ Error loading from S3: {e}")
            # Fallback to local file
            return self._load_episodes_local()
    
    def _load_episodes_local(self) -> List[Dict]:
        """Fallback: load from local file"""
        try:
            local_file = "data/processed_episodes_v3.json"
            if os.path.exists(local_file):
                with open(local_file, 'r', encoding='utf-8') as f:
                    episodes = json.load(f)
                logger.info(f"📥 Loaded {len(episodes)} episodes from local file")
                return episodes
        except Exception as e:
            logger.error(f"❌ Error loading local file: {e}")
        
        return []
    
    def index_episodes(self, episodes_data: List[Dict]):
        """
        Index episodes with rich metadata into OpenSearch
        
        Creates three types of documents:
        1. Episode-level: For content recommendations
        2. Anime-level: For series discovery
        3. Line-level: For vocabulary/grammar examples
        """
        logger.info(f"\n📄 Indexing {len(episodes_data)} episodes...")
        
        anime_series = {}
        indexed_count = 0
        
        for episode in episodes_data:
            episode_id = episode['episode_id']
            anime_name = episode['anime_name']
            
            # Track anime series
            if anime_name not in anime_series:
                anime_series[anime_name] = []
            anime_series[anime_name].append(episode)
            
            # 1. Episode-level document
            episode_summary = f"""
            Anime: {anime_name}
            Title: {episode['title']}
            Season: {episode.get('season', 'N/A')}
            Episode: {episode['episode']}
            JLPT Level: {episode['episode_level']}
            Duration: {episode['total_duration_seconds'] // 60} minutes
            Lines: {episode['total_lines']}
            Vocabulary: {episode['unique_vocab_count']} unique words
            Average line length: {episode['avg_chars_per_line']} characters
            Level distribution: {episode['level_distribution']}
            """
            
            embedding = self.create_embedding(episode_summary)
            
            episode_doc = {
                'embedding': embedding,
                'document': episode_summary,
                'type': 'episode',
                'episode_id': episode_id,
                'anime_name': anime_name,
                'season': episode.get('season'),
                'episode_number': episode['episode'],
                'title': episode['title'],
                'level': episode['episode_level'],
                'total_lines': episode['total_lines'],
                'vocab_count': episode['unique_vocab_count'],
                'duration': episode['total_duration_seconds']
            }
            
            self.opensearch.index_document(f"episode_{episode_id}", episode_doc)
            indexed_count += 1
            
            # 2. Line-level documents (sample)
            entries = episode['entries']
            sample_size = min(len(entries), 20)
            
            if len(entries) > sample_size:
                step = len(entries) // sample_size
                sampled_entries = [entries[i] for i in range(0, len(entries), step)][:sample_size]
            else:
                sampled_entries = entries
            
            for entry in sampled_entries:
                line_context = f"{anime_name} - {entry['text']}"
                line_embedding = self.create_embedding(line_context)
                
                line_doc = {
                    'embedding': line_embedding,
                    'document': line_context,
                    'type': 'line',
                    'episode_id': episode_id,
                    'anime_name': anime_name,
                    'title': episode['title'],
                    'level': entry['jlpt_level'],
                    'timestamp': entry['start_time'],
                    'text': entry['text'],
                    'vocab': ','.join(entry['vocab'][:5])
                }
                
                self.opensearch.index_document(
                    f"line_{episode_id}_{entry['index']}", 
                    line_doc
                )
                indexed_count += 1
            
            if indexed_count % 50 == 0:
                logger.info(f"   Indexed {indexed_count} documents...")
        
        # 3. Anime-level documents
        for anime_name, episodes in anime_series.items():
            total_episodes = len(episodes)
            levels = [ep['episode_level'] for ep in episodes]
            most_common_level = max(set(levels), key=levels.count)
            
            anime_summary = f"""
            Anime Series: {anime_name}
            Total Episodes: {total_episodes}
            Typical Level: {most_common_level}
            Available episodes: {', '.join([ep['title'] for ep in episodes[:5]])}
            """
            
            anime_embedding = self.create_embedding(anime_summary)
            episode_ids_str = ','.join([ep['episode_id'] for ep in episodes])
            
            anime_doc = {
                'embedding': anime_embedding,
                'document': anime_summary,
                'type': 'anime',
                'anime_name': anime_name,
                'episode_count': total_episodes,
                'typical_level': most_common_level,
                'episode_ids': episode_ids_str
            }
            
            anime_id = anime_name.lower().replace(' ', '_')
            self.opensearch.index_document(f"anime_{anime_id}", anime_doc)
            indexed_count += 1
        
        logger.info(f"✅ Indexed {indexed_count} documents successfully!")
        logger.info(f"   - {len(anime_series)} anime series")
        logger.info(f"   - {len(episodes_data)} episodes")
    
    def search_episodes_by_level(self, level: str, query: str = "", n_results: int = 5) -> List[Dict]:
        """Find episodes matching a JLPT level and optional query"""
        if query:
            search_text = f"{query} JLPT {level} Japanese learning content"
        else:
            search_text = f"Engaging Japanese anime content for JLPT {level} learners"
        
        query_embedding = self.create_embedding(search_text)
        
        results = self.opensearch.search(
            query_vector=query_embedding,
            filters={"type": "episode"},
            size=n_results * 3
        )
        
        episodes = []
        level_order = ['N5', 'N4', 'N3', 'N2', 'N1']
        target_idx = level_order.index(level)
        
        for hit in results['hits']['hits']:
            source = hit['_source']
            score = hit['_score']
            
            episode_level = source['level']
            episode_idx = level_order.index(episode_level)
            
            if abs(target_idx - episode_idx) <= 1:
                episodes.append({
                    'episode_id': source['episode_id'],
                    'anime_name': source['anime_name'],
                    'season': source.get('season'),
                    'episode_number': source['episode_number'],
                    'title': source['title'],
                    'level': source['level'],
                    'total_lines': source['total_lines'],
                    'vocab_count': source['vocab_count'],
                    'duration_minutes': source['duration'] // 60,
                    'relevance_score': round(score, 3)
                })
        
        episodes.sort(key=lambda x: x['relevance_score'], reverse=True)
        return episodes[:n_results]
    
    def search_by_anime(self, anime_name: str, level: Optional[str] = None) -> List[Dict]:
        """Find all episodes of a specific anime - FIXED to use proper text search"""
        try:
            # Use match_all with post-filter since anime_name is a keyword field
            # We need to get all episodes and filter in Python
            query = {
                "size": 500,  # Get more results to filter
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"type": "episode"}}
                        ]
                    }
                }
            }
            
            if level:
                query["query"]["bool"]["must"].append({"term": {"level": level}})
            
            results = self.opensearch.client.search(
                index=self.opensearch.index_name,
                body=query
            )
            
            # Filter episodes in Python (case-insensitive matching)
            episodes = []
            search_name = anime_name.lower().replace(' ', '').replace('_', '')
            
            for hit in results['hits']['hits']:
                source = hit['_source']
                db_name = source['anime_name'].lower().replace(' ', '').replace('_', '')
                
                # Check if the anime names match (fuzzy)
                if search_name in db_name or db_name in search_name:
                    episodes.append({
                        'episode_id': source['episode_id'],
                        'anime_name': source['anime_name'],
                        'season': source.get('season'),
                        'episode_number': source['episode_number'],
                        'title': source['title'],
                        'level': source['level'],
                        'total_lines': source.get('total_lines', 0),
                        'vocab_count': source.get('vocab_count', 0)
                    })
            
            episodes.sort(key=lambda x: (x.get('season') or 0, x['episode_number']))
            logger.info(f"Found {len(episodes)} episodes for {anime_name}")
            return episodes
        
        except Exception as e:
            logger.error(f"Search by anime failed: {e}")
            return []
    
    def find_vocabulary_examples(self, episode_id: str, n_examples: int = 10) -> List[Dict]:
        """Find example dialogue lines from a specific episode"""
        search_text = "Japanese vocabulary example sentences"
        query_embedding = self.create_embedding(search_text)
        
        results = self.opensearch.search(
            query_vector=query_embedding,
            filters={"type": "line", "episode_id": episode_id},
            size=n_examples * 2
        )
        
        examples = []
        for hit in results['hits']['hits']:
            source = hit['_source']
            examples.append({
                'text': source['text'],
                'timestamp': source['timestamp'],
                'level': source['level'],
                'vocab': source.get('vocab', '').split(',') if source.get('vocab') else []
            })
        
        return examples[:n_examples]
    
    def get_collection_stats(self) -> Dict:
        """Get comprehensive statistics about the content library"""
        try:
            # Count documents by type
            stats = {
                'total_items': 0,
                'anime_count': 0,
                'episode_count': 0,
                'line_count': 0,
                'level_distribution': {},
                'anime_list': []
            }
            
            # Get counts
            for doc_type in ['anime', 'episode', 'line']:
                count_query = {
                    "query": {"term": {"type": doc_type}}
                }
                result = self.opensearch.client.count(
                    index=self.opensearch.index_name,
                    body=count_query
                )
                count = result['count']
                
                if doc_type == 'anime':
                    stats['anime_count'] = count
                elif doc_type == 'episode':
                    stats['episode_count'] = count
                elif doc_type == 'line':
                    stats['line_count'] = count
                
                stats['total_items'] += count
            
            # Get anime list
            anime_results = self.opensearch.client.search(
                index=self.opensearch.index_name,
                body={
                    "size": 100,
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"type": "anime"}}
                            ]
                        }
                    }
                }
            )
            
            for hit in anime_results['hits']['hits']:
                source = hit['_source']
                stats['anime_list'].append({
                    'name': source['anime_name'],
                    'episodes': source['episode_count'],
                    'level': source['typical_level']
                })
            
            stats['unique_anime'] = len(stats['anime_list'])
            
            return stats
        
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {'error': str(e)}


def main():
    """Main function to initialize and index content"""
    
    logger.info("="*60)
    logger.info("🔍 LinguaSync RAG Engine V3 - Stage 2")
    logger.info("="*60)
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not found")
        return
    
    if not os.getenv("OPENSEARCH_ENDPOINT"):
        logger.error("❌ OPENSEARCH_ENDPOINT not found")
        return
    
    rag = RAGEngineV3()
    
    # Load episodes from S3 or local
    episodes = rag.load_episodes_from_s3()
    
    if not episodes:
        logger.error("❌ No episodes found")
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
    
    logger.info("\n✅ RAG Engine V3 ready!")
    logger.info("🚀 Next: Run api_v3.py with LangGraph orchestration")


if __name__ == "__main__":
    main()