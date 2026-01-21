"""
Diagnostic Script for Learning Package Issue

This script helps identify why episode lookups are failing and provides fixes.
"""

import os
import sys
from pathlib import Path

# Add the uploads directory to the path to import modules
sys.path.insert(0, '/mnt/user-data/uploads')

import logging
from rag_engine_v3 import RAGEngineV3
from langgraph_orchestrator import LangGraphOrchestrator
from learning_generator_v2 import LearningGeneratorV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_episode_lookup():
    """Diagnose why episode lookups are failing"""
    
    print("=" * 80)
    print("🔍 LEARNING PACKAGE DIAGNOSTIC TOOL")
    print("=" * 80)
    
    try:
        # Initialize components
        print("\n1️⃣ Initializing RAG Engine...")
        rag = RAGEngineV3()
        
        # Get collection stats
        print("\n2️⃣ Checking collection statistics...")
        stats = rag.get_collection_stats()
        
        print(f"\n📊 Database Statistics:")
        print(f"   - Total items: {stats.get('total_items', 0)}")
        print(f"   - Anime count: {stats.get('anime_count', 0)}")
        print(f"   - Episode count: {stats.get('episode_count', 0)}")
        print(f"   - Line count: {stats.get('line_count', 0)}")
        
        if stats.get('anime_list'):
            print(f"\n📚 Available Anime:")
            for anime in stats['anime_list'][:10]:
                print(f"   - {anime['name']}: {anime['episodes']} episodes ({anime['level']})")
        
        # Test episode search
        print("\n3️⃣ Testing episode search...")
        
        # Try searching for Cowboy Bebop episodes
        test_anime = "Cowboy Bebop"
        episodes = rag.search_by_anime(test_anime)
        
        print(f"\n🔍 Search results for '{test_anime}':")
        print(f"   Found {len(episodes)} episodes")
        
        if episodes:
            print("\n📝 Episode IDs found:")
            for ep in episodes[:10]:
                print(f"   - {ep['episode_id']}")
                print(f"     Title: {ep['title']}")
                print(f"     Level: {ep['level']}")
        
            # Test the problematic episode_id
            problematic_id = "cowboy_bebop_e04"
            print(f"\n4️⃣ Checking if '{problematic_id}' exists...")
            
            # Direct search by episode_id
            search_query = f"episode {problematic_id}"
            query_embedding = rag.create_embedding(search_query)
            
            results = rag.opensearch.search(
                query_vector=query_embedding,
                filters={"type": "episode", "episode_id": problematic_id},
                size=1
            )
            
            if results['hits']['hits']:
                print(f"   ✅ Episode found in database!")
                episode = results['hits']['hits'][0]['_source']
                print(f"   - Title: {episode['title']}")
                print(f"   - Anime: {episode['anime_name']}")
                print(f"   - Level: {episode['level']}")
            else:
                print(f"   ❌ Episode NOT found in database")
                print(f"   This is the problem! The episode_id doesn't exist.")
                
                # Show available episode IDs for Cowboy Bebop
                print(f"\n   Available Cowboy Bebop episode IDs:")
                for ep in episodes:
                    print(f"   - {ep['episode_id']}")
        
        else:
            print(f"   ❌ No episodes found for {test_anime}")
        
        # Test vocabulary examples lookup
        if episodes:
            test_episode_id = episodes[0]['episode_id']
            print(f"\n5️⃣ Testing vocabulary examples for: {test_episode_id}")
            
            examples = rag.find_vocabulary_examples(test_episode_id, n_examples=5)
            print(f"   Found {len(examples)} example lines")
            
            if examples:
                print(f"\n   Sample line:")
                print(f"   - {examples[0]['text']}")
                print(f"   - Level: {examples[0]['level']}")
        
        print("\n" + "=" * 80)
        print("✅ DIAGNOSTIC COMPLETE")
        print("=" * 80)
        
        # Provide recommendations
        print("\n💡 RECOMMENDATIONS:")
        if not episodes:
            print("   1. Check if data is properly indexed in OpenSearch")
            print("   2. Verify OPENSEARCH_ENDPOINT is correct")
            print("   3. Re-run the indexing script: rag_engine_v3.py")
        elif results['hits']['hits']:
            print("   1. Episode exists - the API should work!")
            print("   2. Test the learning package endpoint directly")
        else:
            print("   1. The episode_id format may be incorrect")
            print("   2. Check the exact episode_ids in your database")
            print("   3. Update frontend to use correct episode_ids")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


def test_learning_package_generation():
    """Test the complete learning package workflow"""
    
    print("\n" + "=" * 80)
    print("🧪 TESTING LEARNING PACKAGE GENERATION")
    print("=" * 80)
    
    try:
        # Initialize components
        rag = RAGEngineV3()
        generator = LearningGeneratorV2()
        orchestrator = LangGraphOrchestrator(rag, generator)
        
        # Get first available episode
        stats = rag.get_collection_stats()
        if not stats.get('anime_list'):
            print("❌ No anime found in database")
            return
        
        # Search for episodes
        first_anime = stats['anime_list'][0]['name']
        episodes = rag.search_by_anime(first_anime)
        
        if not episodes:
            print(f"❌ No episodes found for {first_anime}")
            return
        
        test_episode_id = episodes[0]['episode_id']
        print(f"\n📝 Testing with episode: {test_episode_id}")
        print(f"   Title: {episodes[0]['title']}")
        print(f"   Level: {episodes[0]['level']}")
        
        # Generate learning package
        print("\n🚀 Generating learning package...")
        result = orchestrator.execute_learning_package_workflow(
            episode_id=test_episode_id,
            user_level="N3"
        )
        
        print("\n✅ Learning package generated successfully!")
        print(f"\n📦 Package Contents:")
        print(f"   - Vocabulary: {len(result.get('vocabulary_list', ''))} chars")
        print(f"   - Grammar: {len(result.get('grammar_notes', ''))} chars")
        print(f"   - Cultural: {len(result.get('cultural_context', ''))} chars")
        print(f"   - Prep: {len(result.get('pre_watch_prep', ''))} chars")
        
        if result.get('errors'):
            print(f"\n⚠️  Errors: {result['errors']}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


def show_episode_id_format():
    """Show the correct episode_id format"""
    
    print("\n" + "=" * 80)
    print("📋 EPISODE ID FORMAT GUIDE")
    print("=" * 80)
    
    try:
        rag = RAGEngineV3()
        
        # Get all anime
        stats = rag.get_collection_stats()
        
        print("\n📚 Episode ID formats by anime:")
        for anime in stats.get('anime_list', [])[:5]:
            episodes = rag.search_by_anime(anime['name'])
            if episodes:
                print(f"\n{anime['name']}:")
                for ep in episodes[:3]:
                    print(f"   - {ep['episode_id']}")
                if len(episodes) > 3:
                    print(f"   ... and {len(episodes) - 3} more")
    
    except Exception as e:
        print(f"❌ ERROR: {e}")


if __name__ == "__main__":
    # Run diagnostics
    diagnose_episode_lookup()
    
    # Test learning package
    test_learning_package_generation()
    
    # Show episode ID formats
    show_episode_id_format()