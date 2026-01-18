"""
OpenSearch Vector Engine - Stage 2
Replaces ChromaDB with Amazon OpenSearch Serverless
"""

import os
import json
import boto3
from typing import List, Dict, Optional
from openai import OpenAI
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv
import os
load_dotenv()

class OpenSearchVectorEngine:
    """Vector storage using Amazon OpenSearch Serverless"""
    
    def __init__(self, opensearch_endpoint: str, index_name: str = "linguasync-episodes"):
        """
        Initialize OpenSearch engine
        
        Args:
            opensearch_endpoint: OpenSearch Serverless endpoint
            index_name: Name of the index
        """
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.index_name = index_name
        
        # AWS credentials for OpenSearch
        session = boto3.Session()
        credentials = session.get_credentials()
        
        # AWS4Auth for signing requests
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            session.region_name or 'us-east-1',
            'aoss',  # Amazon OpenSearch Serverless
            session_token=credentials.token
        )
        
        # Initialize OpenSearch client
        self.client = OpenSearch(
            hosts=[{'host': opensearch_endpoint.replace('https://', ''), 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30
        )
        
        # Create index if it doesn't exist
        self._ensure_index_exists()
    
    def _ensure_index_exists(self):
        """Create index with vector mapping if it doesn't exist"""
        
        if self.client.indices.exists(index=self.index_name):
            print(f"✅ Index '{self.index_name}' already exists")
            return
        
        # Index mapping with vector field
        mapping = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 512
                }
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": 1536,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "nmslib"
                        }
                    },
                    "type": {"type": "keyword"},
                    "episode_id": {"type": "keyword"},
                    "anime_name": {"type": "keyword"},
                    "season": {"type": "integer"},
                    "episode_number": {"type": "integer"},
                    "title": {"type": "text"},
                    "level": {"type": "keyword"},
                    "text": {"type": "text"},
                    "timestamp": {"type": "keyword"},
                    "vocab": {"type": "keyword"},
                    "total_lines": {"type": "integer"},
                    "vocab_count": {"type": "integer"},
                    "duration": {"type": "integer"}
                }
            }
        }
        
        self.client.indices.create(index=self.index_name, body=mapping)
        print(f"✨ Created index '{self.index_name}'")
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create vector embedding using OpenAI
        
        Args:
            text: Text to embed
            
        Returns:
            1536-dimensional vector
        """
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            dimensions=1536
        )
        return response.data[0].embedding
    
    def index_episode(self, episode_data: Dict):
        """
        Index a single episode with its lines
        
        Args:
            episode_data: Episode metadata from processor
        """
        episode_id = episode_data['episode_id']
        
        # Create episode-level document
        episode_summary = f"""
        Anime: {episode_data['anime_name']}
        Title: {episode_data['title']}
        Season: {episode_data.get('season', 'N/A')}
        Episode: {episode_data['episode']}
        Level: {episode_data['episode_level']}
        Duration: {episode_data['total_duration_seconds'] // 60} minutes
        Vocabulary: {episode_data['unique_vocab_count']} unique words
        """
        
        episode_vector = self.create_embedding(episode_summary)
        
        episode_doc = {
            "vector": episode_vector,
            "type": "episode",
            "episode_id": episode_id,
            "anime_name": episode_data['anime_name'],
            "season": episode_data.get('season'),
            "episode_number": episode_data['episode'],
            "title": episode_data['title'],
            "level": episode_data['episode_level'],
            "total_lines": episode_data['total_lines'],
            "vocab_count": episode_data['unique_vocab_count'],
            "duration": episode_data['total_duration_seconds']
        }
        
        # Index episode
        self.client.index(
            index=self.index_name,
            id=f"episode_{episode_id}",
            body=episode_doc
        )
        
        # Index sample lines (every 5th line)
        entries = episode_data['entries']
        for i, entry in enumerate(entries):
            if i % 5 == 0:
                line_vector = self.create_embedding(entry['text'])
                
                line_doc = {
                    "vector": line_vector,
                    "type": "line",
                    "episode_id": episode_id,
                    "anime_name": episode_data['anime_name'],
                    "title": episode_data['title'],
                    "level": entry['jlpt_level'],
                    "timestamp": entry['start_time'],
                    "text": entry['text'],
                    "vocab": ','.join(entry['vocab'][:5])
                }
                
                self.client.index(
                    index=self.index_name,
                    id=f"line_{episode_id}_{i}",
                    body=line_doc
                )
        
        print(f"✅ Indexed: {episode_data['title']}")
    
    def search_episodes_by_level(self, level: str, query: str = "", n_results: int = 5) -> List[Dict]:
        """
        Search episodes by JLPT level and optional query
        
        Args:
            level: JLPT level (N5, N4, N3, N2, N1)
            query: Optional search query
            n_results: Number of results
            
        Returns:
            List of matching episodes
        """
        # Create search query
        if query:
            search_text = f"{query} JLPT {level} Japanese learning content"
        else:
            search_text = f"Engaging Japanese anime content for JLPT {level} learners"
        
        query_vector = self.create_embedding(search_text)
        
        # KNN search
        search_body = {
            "size": n_results * 2,
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "vector": {
                                    "vector": query_vector,
                                    "k": n_results * 2
                                }
                            }
                        }
                    ],
                    "filter": [
                        {"term": {"type": "episode"}}
                    ]
                }
            }
        }
        
        response = self.client.search(index=self.index_name, body=search_body)
        
        # Filter by level (±1 level)
        level_order = ['N5', 'N4', 'N3', 'N2', 'N1']
        target_idx = level_order.index(level)
        
        episodes = []
        for hit in response['hits']['hits']:
            source = hit['_source']
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
                    'relevance_score': round(hit['_score'], 3)
                })
        
        return episodes[:n_results]
    
    def get_vocabulary_examples(self, episode_id: str, n_examples: int = 10) -> List[Dict]:
        """
        Get vocabulary example lines from episode
        
        Args:
            episode_id: Episode ID
            n_examples: Number of examples
            
        Returns:
            List of example lines
        """
        search_body = {
            "size": n_examples,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"type": "line"}},
                        {"term": {"episode_id": episode_id}}
                    ]
                }
            }
        }
        
        response = self.client.search(index=self.index_name, body=search_body)
        
        examples = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            examples.append({
                'text': source['text'],
                'timestamp': source['timestamp'],
                'level': source['level'],
                'vocab': source['vocab'].split(',') if source.get('vocab') else []
            })
        
        return examples
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        
        # Count by type
        episode_count = self.client.count(
            index=self.index_name,
            body={"query": {"term": {"type": "episode"}}}
        )['count']
        
        line_count = self.client.count(
            index=self.index_name,
            body={"query": {"term": {"type": "line"}}}
        )['count']
        
        # Get unique anime
        anime_agg = self.client.search(
            index=self.index_name,
            body={
                "size": 0,
                "query": {"term": {"type": "episode"}},
                "aggs": {
                    "unique_anime": {
                        "terms": {"field": "anime_name", "size": 100}
                    }
                }
            }
        )
        
        anime_list = []
        for bucket in anime_agg['aggregations']['unique_anime']['buckets']:
            anime_list.append({
                'name': bucket['key'],
                'episodes': bucket['doc_count']
            })
        
        return {
            'total_items': episode_count + line_count,
            'episode_count': episode_count,
            'line_count': line_count,
            'anime_count': len(anime_list),
            'anime_list': anime_list
        }


def main():
    """Test OpenSearch connection"""
    
    print("="*60)
    print("🔍 Testing OpenSearch Connection")
    print("="*60)
    
    endpoint = os.environ.get('OPENSEARCH_ENDPOINT')
    if not endpoint:
        print("❌ OPENSEARCH_ENDPOINT not set")
        return
    
    engine = OpenSearchVectorEngine(endpoint)
    
    # Test stats
    stats = engine.get_stats()
    print(f"\n📊 Stats: {stats}")


if __name__ == "__main__":
    main()