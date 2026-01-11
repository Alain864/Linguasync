"""
Enhanced RAG Engine - Stage 1

Improvements over Stage 0:
1. Richer metadata support (anime name, season, episode)
2. Better search capabilities
3. AWS-ready architecture (prepared for OpenSearch)
4. Improved error handling and validation
5. Better logging and monitoring
"""

import os
import json
from typing import List, Dict, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


class RAGEngineV2:
    """Enhanced vector storage and retrieval engine"""
    
    def __init__(self, 
                 db_path: str = "./chroma_db_v2",
                 collection_name: str = "japanese_episodes_v2"):
        """
        Initialize the enhanced RAG engine
        
        Args:
            db_path: Path to ChromaDB storage
            collection_name: Name of the vector collection
        """
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize ChromaDB with telemetry disabled
        self.chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            print(f"✅ Loaded existing collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={
                    "description": "Japanese learning content with rich metadata",
                    "version": "2.0",
                    "created_at": datetime.now().isoformat()
                }
            )
            print(f"✨ Created new collection: {collection_name}")
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create vector embedding using OpenAI
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding (1536 dimensions)
        """
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=1536  # Fixed dimension for consistency
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error creating embedding: {e}")
            raise
    
    def index_episodes(self, episodes_data: List[Dict]):
        """
        Index episodes with rich metadata into vector database
        
        Creates three types of vectors for flexible querying:
        1. Episode-level: For content recommendations
        2. Anime-level: For finding all episodes of a series
        3. Line-level: For vocabulary and grammar examples
        
        Args:
            episodes_data: List of processed episode metadata
        """
        print(f"\n🔄 Indexing {len(episodes_data)} episodes...")
        
        documents = []
        metadatas = []
        ids = []
        
        # Track anime series for anime-level indexing
        anime_series = {}
        
        for episode in episodes_data:
            episode_id = episode['episode_id']
            anime_name = episode['anime_name']
            
            # Track anime series
            if anime_name not in anime_series:
                anime_series[anime_name] = []
            anime_series[anime_name].append(episode)
            
            # 1. Episode-level vector - Rich summary for recommendations
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
            
            documents.append(episode_summary)
            metadatas.append({
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
            })
            ids.append(f"episode_{episode_id}")
            
            # 2. Line-level vectors - Sample key lines for vocabulary/grammar
            entries = episode['entries']
            
            # Sample intelligently: take lines from different parts of the episode
            sample_size = min(len(entries), 20)  # Up to 20 lines per episode
            if len(entries) > sample_size:
                # Sample evenly across the episode
                step = len(entries) // sample_size
                sampled_entries = [entries[i] for i in range(0, len(entries), step)][:sample_size]
            else:
                sampled_entries = entries
            
            for i, entry in enumerate(sampled_entries):
                line_text = entry['text']
                
                # Create richer line context
                line_context = f"{anime_name} - {entry['text']}"
                
                documents.append(line_context)
                metadatas.append({
                    'type': 'line',
                    'episode_id': episode_id,
                    'anime_name': anime_name,
                    'title': episode['title'],
                    'level': entry['jlpt_level'],
                    'timestamp': entry['start_time'],
                    'text': line_text,
                    'vocab': ','.join(entry['vocab'][:5])  # Top 5 vocab items
                })
                ids.append(f"line_{episode_id}_{entry['index']}")
        
        # 3. Anime-level vectors - Summary of entire series
        for anime_name, episodes in anime_series.items():
            # Create series summary
            total_episodes = len(episodes)
            levels = [ep['episode_level'] for ep in episodes]
            most_common_level = max(set(levels), key=levels.count)
            
            anime_summary = f"""
            Anime Series: {anime_name}
            Total Episodes: {total_episodes}
            Typical Level: {most_common_level}
            Available episodes: {', '.join([ep['title'] for ep in episodes[:5]])}
            """
            
            # ChromaDB only supports str, int, float, bool in metadata
            # Convert episode_ids list to comma-separated string
            episode_ids_str = ','.join([ep['episode_id'] for ep in episodes])
            
            documents.append(anime_summary)
            metadatas.append({
                'type': 'anime',
                'anime_name': anime_name,
                'episode_count': total_episodes,
                'typical_level': most_common_level,
                'episode_ids': episode_ids_str  # Store as comma-separated string
            })
            ids.append(f"anime_{anime_name.lower().replace(' ', '_')}")
        
        # Create embeddings in batches
        print(f"📊 Creating embeddings for {len(documents)} items...")
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size
            print(f"   Processing batch {batch_num}/{total_batches}...")
            
            # Create embeddings for batch
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
                dimensions=1536
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        # Add to ChromaDB
        print(f"💾 Storing vectors in database...")
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Indexed {len(documents)} items successfully!")
        print(f"   - {len([m for m in metadatas if m['type'] == 'anime'])} anime series")
        print(f"   - {len([m for m in metadatas if m['type'] == 'episode'])} episodes")
        print(f"   - {len([m for m in metadatas if m['type'] == 'line'])} dialogue lines")
    
    def search_episodes_by_level(self, 
                                  level: str, 
                                  query: str = "",
                                  n_results: int = 5) -> List[Dict]:
        """
        Find episodes matching a JLPT level and optional query
        
        Args:
            level: Target JLPT level (N5, N4, N3, N2, N1)
            query: Optional semantic search query
            n_results: Number of results to return
            
        Returns:
            List of matching episodes with metadata
        """
        # Build search query
        if query:
            search_text = f"{query} JLPT {level} Japanese learning content"
        else:
            search_text = f"Engaging Japanese anime content for JLPT {level} learners"
        
        # Create embedding
        query_embedding = self.create_embedding(search_text)
        
        # Search with filters
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 3,  # Get more for filtering
                where={"type": "episode"}
            )
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
        
        # Filter and rank by level match
        episodes = []
        level_order = ['N5', 'N4', 'N3', 'N2', 'N1']
        target_idx = level_order.index(level)
        
        for i, doc_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            # Check level compatibility (exact or ±1 level)
            episode_level = metadata['level']
            episode_idx = level_order.index(episode_level)
            
            if abs(target_idx - episode_idx) <= 1:
                episodes.append({
                    'episode_id': metadata['episode_id'],
                    'anime_name': metadata['anime_name'],
                    'season': metadata.get('season'),
                    'episode_number': metadata['episode_number'],
                    'title': metadata['title'],
                    'level': metadata['level'],
                    'total_lines': metadata['total_lines'],
                    'vocab_count': metadata['vocab_count'],
                    'duration_minutes': metadata['duration'] // 60,
                    'relevance_score': round(1 - distance, 3)
                })
        
        # Sort by relevance and return top N
        episodes.sort(key=lambda x: x['relevance_score'], reverse=True)
        return episodes[:n_results]
    
    def search_by_anime(self, anime_name: str, level: Optional[str] = None) -> List[Dict]:
        """
        Find all episodes of a specific anime
        
        Args:
            anime_name: Name of the anime
            level: Optional JLPT level filter
            
        Returns:
            List of episodes from that anime
        """
        search_text = f"{anime_name} anime episodes"
        query_embedding = self.create_embedding(search_text)
        
        # Build where filter
        where_filter = {"type": "episode", "anime_name": anime_name}
        if level:
            where_filter["level"] = level
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=50,  # Get many episodes
                where=where_filter
            )
        except:
            # Fallback: search without exact anime_name match
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=50,
                where={"type": "episode"}
            )
        
        episodes = []
        for i, doc_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            
            # Filter by anime name (case-insensitive partial match)
            if anime_name.lower() in metadata['anime_name'].lower():
                episodes.append({
                    'episode_id': metadata['episode_id'],
                    'anime_name': metadata['anime_name'],
                    'season': metadata.get('season'),
                    'episode_number': metadata['episode_number'],
                    'title': metadata['title'],
                    'level': metadata['level']
                })
        
        # Sort by season and episode number
        episodes.sort(key=lambda x: (x.get('season') or 0, x['episode_number']))
        return episodes
    
    def find_vocabulary_examples(self, 
                                  episode_id: str,
                                  n_examples: int = 10) -> List[Dict]:
        """
        Find example dialogue lines from a specific episode
        
        Args:
            episode_id: ID of the episode
            n_examples: Number of examples to return
            
        Returns:
            List of dialogue lines with metadata
        """
        search_text = "Japanese vocabulary example sentences"
        query_embedding = self.create_embedding(search_text)
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_examples * 2,
                where={
                    "type": "line",
                    "episode_id": episode_id
                }
            )
        except Exception as e:
            print(f"❌ Error finding examples: {e}")
            return []
        
        examples = []
        for i, doc_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            
            examples.append({
                'text': metadata['text'],
                'timestamp': metadata['timestamp'],
                'level': metadata['level'],
                'vocab': metadata['vocab'].split(',') if metadata.get('vocab') else []
            })
        
        return examples[:n_examples]
    
    def get_collection_stats(self) -> Dict:
        """
        Get comprehensive statistics about the content library
        
        Returns:
            Dictionary with detailed statistics
        """
        try:
            count = self.collection.count()
            
            # Get all metadata for analysis
            all_data = self.collection.get(
                limit=1000,  # Get up to 1000 items
                include=['metadatas']
            )
            
            stats = {
                'total_items': count,
                'anime_count': 0,
                'episode_count': 0,
                'line_count': 0,
                'level_distribution': {},
                'anime_list': []
            }
            
            anime_names = set()
            
            for metadata in all_data['metadatas']:
                item_type = metadata.get('type')
                
                if item_type == 'anime':
                    stats['anime_count'] += 1
                    anime_names.add(metadata['anime_name'])
                    stats['anime_list'].append({
                        'name': metadata['anime_name'],
                        'episodes': metadata['episode_count'],
                        'level': metadata['typical_level']
                    })
                elif item_type == 'episode':
                    stats['episode_count'] += 1
                    anime_names.add(metadata['anime_name'])
                elif item_type == 'line':
                    stats['line_count'] += 1
                
                # Track level distribution
                level = metadata.get('level', metadata.get('typical_level', 'Unknown'))
                stats['level_distribution'][level] = stats['level_distribution'].get(level, 0) + 1
            
            stats['unique_anime'] = len(anime_names)
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {'error': str(e)}


def main():
    """Main function to initialize and test the enhanced RAG engine"""
    
    print("="*60)
    print("🔍 LinguaSync RAG Engine V2 - Stage 1")
    print("="*60)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found")
        print("   Please create a .env file with your OpenAI API key")
        return
    
    # Initialize enhanced RAG engine
    rag = RAGEngineV2()
    
    # Load processed episodes (V2 format)
    episodes_file = "data/processed_episodes_v2.json"
    
    if not os.path.exists(episodes_file):
        print(f"❌ Error: {episodes_file} not found")
        print("   Please run subtitle_processor_v2.py first")
        return
    
    with open(episodes_file, 'r', encoding='utf-8') as f:
        episodes = json.load(f)
    
    # Index episodes
    rag.index_episodes(episodes)
    
    # Show comprehensive statistics
    print("\n" + "="*60)
    stats = rag.get_collection_stats()
    print(f"📊 Content Library Statistics:")
    print(f"   Total items in database: {stats['total_items']}")
    print(f"   Unique anime series: {stats['unique_anime']}")
    print(f"   Total episodes: {stats['episode_count']}")
    print(f"   Indexed dialogue lines: {stats['line_count']}")
    
    if stats['anime_list']:
        print(f"\n📚 Available Anime:")
        for anime in stats['anime_list']:
            print(f"   - {anime['name']}: {anime['episodes']} episodes ({anime['level']})")
    
    print(f"\n📈 Level Distribution:")
    for level in ['N5', 'N4', 'N3', 'N2', 'N1']:
        if level in stats['level_distribution']:
            print(f"   {level}: {stats['level_distribution'][level]} items")
    
    print("\n✅ RAG Engine V2 ready!")
    print("🚀 Next: Update API to use the new RAG engine")


if __name__ == "__main__":
    main()