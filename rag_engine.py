"""
RAG Engine - Vector storage and retrieval for LinguaSync

This module:
1. Creates embeddings using OpenAI
2. Stores vectors in ChromaDB (local)
3. Retrieves relevant content based on user queries
4. Matches content to learner level
"""

import os
import json
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGEngine:
    """Vector storage and retrieval engine for Japanese learning content"""
    
    def __init__(self, 
                 db_path: str = "./chroma_db",
                 collection_name: str = "japanese_episodes"):
        """
        Initialize the RAG engine
        
        Args:
            db_path: Path to ChromaDB storage
            collection_name: Name of the vector collection
        """
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize ChromaDB (local, persistent)
        # Disable telemetry to avoid warning messages
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
                metadata={"description": "Japanese learning content"}
            )
            print(f"✨ Created new collection: {collection_name}")
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create vector embedding using OpenAI
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding as list of floats
        """
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            dimensions=1536  # Explicitly set dimensions for consistency
        )
        return response.data[0].embedding
    
    def index_episodes(self, episodes_data: List[Dict]):
        """
        Index all episodes into the vector database
        
        This creates two types of vectors:
        1. Episode-level: For matching content to learner level
        2. Line-level: For finding specific vocabulary/grammar examples
        
        Args:
            episodes_data: List of processed episode metadata
        """
        print(f"\n🔄 Indexing {len(episodes_data)} episodes...")
        
        documents = []
        metadatas = []
        ids = []
        
        for episode in episodes_data:
            episode_id = episode['episode_id']
            
            # 1. Create episode-level vector
            # This summarizes the entire episode for recommendation
            episode_summary = f"""
            Title: {episode['title']}
            Level: {episode['episode_level']}
            Lines: {episode['total_lines']}
            Vocabulary: {episode['unique_vocab_count']} unique words
            Average line length: {episode['avg_chars_per_line']} characters
            """
            
            documents.append(episode_summary)
            metadatas.append({
                'type': 'episode',
                'episode_id': episode_id,
                'title': episode['title'],
                'level': episode['episode_level'],
                'total_lines': episode['total_lines'],
                'vocab_count': episode['unique_vocab_count']
            })
            ids.append(f"episode_{episode_id}")
            
            # 2. Create line-level vectors (sample key lines)
            # For Stage 0, we index every 5th line to keep it manageable
            entries = episode['entries']
            for i, entry in enumerate(entries):
                if i % 5 == 0:  # Sample every 5th line
                    line_text = entry['text']
                    
                    documents.append(line_text)
                    metadatas.append({
                        'type': 'line',
                        'episode_id': episode_id,
                        'title': episode['title'],
                        'level': entry['jlpt_level'],
                        'timestamp': entry['start_time'],
                        'vocab': ','.join(entry['vocab'][:5])  # First 5 vocab items
                    })
                    ids.append(f"line_{episode_id}_{i}")
        
        # Create embeddings in batches
        print(f"📊 Creating embeddings for {len(documents)} items...")
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            print(f"   Processing batch {i//batch_size + 1}...")
            
            # Create embeddings for batch
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
                dimensions=1536  # Explicitly set dimensions for consistency
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        # Add to ChromaDB
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Indexed {len(documents)} items successfully!")
    
    def search_episodes_by_level(self, 
                                  level: str, 
                                  query: str = "",
                                  n_results: int = 5) -> List[Dict]:
        """
        Find episodes matching a JLPT level
        
        Args:
            level: Target JLPT level (N5, N4, N3, N2, N1)
            query: Optional text query for semantic search
            n_results: Number of results to return
            
        Returns:
            List of matching episodes with metadata
        """
        # Create search query
        if query:
            search_text = f"{query} JLPT {level}"
        else:
            search_text = f"Engaging Japanese content for JLPT {level} learners"
        
        # Create embedding for search with consistent dimensions
        query_embedding = self.create_embedding(search_text)
        
        # Search in vector database
        results = self.collection.query(
            query_embeddings=[query_embedding],  # Use pre-computed embedding
            n_results=n_results * 2,  # Get more, filter later
            where={"type": "episode"}  # Only search episode-level vectors
        )
        
        episodes = []
        for i, doc_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            # Filter by level (exact match or adjacent levels)
            level_order = ['N5', 'N4', 'N3', 'N2', 'N1']
            target_idx = level_order.index(level)
            episode_level = metadata['level']
            episode_idx = level_order.index(episode_level)
            
            # Accept exact level or ±1 level
            if abs(target_idx - episode_idx) <= 1:
                episodes.append({
                    'episode_id': metadata['episode_id'],
                    'title': metadata['title'],
                    'level': metadata['level'],
                    'total_lines': metadata['total_lines'],
                    'vocab_count': metadata['vocab_count'],
                    'relevance_score': round(1 - distance, 3)  # Convert distance to similarity
                })
        
        # Return top N results
        return episodes[:n_results]
    
    def find_vocabulary_examples(self, 
                                  episode_id: str,
                                  n_examples: int = 10) -> List[Dict]:
        """
        Find example sentences from a specific episode
        
        Args:
            episode_id: ID of the episode
            n_examples: Number of examples to return
            
        Returns:
            List of subtitle lines with vocabulary
        """
        # Create embedding for search
        search_text = "Japanese vocabulary example sentences"
        query_embedding = self.create_embedding(search_text)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],  # Use pre-computed embedding
            n_results=n_examples * 2,
            where={
                "type": "line",
                "episode_id": episode_id
            }
        )
        
        examples = []
        for i, doc_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            document = results['documents'][0][i]
            
            examples.append({
                'text': document,
                'timestamp': metadata['timestamp'],
                'level': metadata['level'],
                'vocab': metadata['vocab'].split(',') if metadata.get('vocab') else []
            })
        
        return examples[:n_examples]
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the indexed content
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        
        # Get sample to analyze
        sample = self.collection.get(limit=100)
        
        level_dist = {}
        episode_count = 0
        line_count = 0
        
        for metadata in sample['metadatas']:
            if metadata['type'] == 'episode':
                episode_count += 1
            else:
                line_count += 1
            
            level = metadata.get('level', 'Unknown')
            level_dist[level] = level_dist.get(level, 0) + 1
        
        return {
            'total_items': count,
            'estimated_episodes': episode_count,
            'estimated_lines': line_count,
            'level_distribution': level_dist
        }


def main():
    """Main function to initialize and test the RAG engine"""
    
    print("="*60)
    print("🔍 LinguaSync RAG Engine - Stage 0")
    print("="*60)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("   Please create a .env file with your OpenAI API key")
        return
    
    # Initialize RAG engine
    rag = RAGEngine()
    
    # Load processed episodes
    episodes_file = "data/processed_episodes.json"
    
    if not os.path.exists(episodes_file):
        print(f"❌ Error: {episodes_file} not found")
        print("   Please run subtitle_processor.py first")
        return
    
    with open(episodes_file, 'r', encoding='utf-8') as f:
        episodes = json.load(f)
    
    # Index episodes
    rag.index_episodes(episodes)
    
    # Show statistics
    stats = rag.get_collection_stats()
    print(f"\n📊 Collection Statistics:")
    print(f"   Total items: {stats['total_items']}")
    print(f"   Episodes: ~{stats['estimated_episodes']}")
    print(f"   Lines: ~{stats['estimated_lines']}")
    
    print("\n✅ RAG Engine ready!")
    print("🚀 Next step: Start the API server (uvicorn api:app --reload)")


if __name__ == "__main__":
    main()