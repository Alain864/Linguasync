#!/usr/bin/env python3
"""
debug_index.py - Debug script to inspect S3 vector store contents

This helps diagnose why searches aren't returning results.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

# Set environment variables if not already set
os.environ.setdefault('S3_BUCKET_NAME', 'linguasync-data')
os.environ.setdefault('AWS_REGION', 'us-east-1')

from backend.rag.engine import RAGEngineS3

def debug_index():
    """Debug the vector store contents"""
    
    print("="*60)
    print("🔍 Debugging S3 Vector Store")
    print("="*60)
    
    # Initialize RAG engine
    print("\n1. Initializing RAG engine...")
    rag = RAGEngineS3()
    
    # Check total documents
    print(f"\n2. Total documents in index: {len(rag.vector_store.metadata)}")
    
    # Count by type
    print("\n3. Documents by type:")
    types = {}
    for doc in rag.vector_store.metadata:
        doc_type = doc.get('type', 'unknown')
        types[doc_type] = types.get(doc_type, 0) + 1
    
    for doc_type, count in sorted(types.items()):
        print(f"   - {doc_type}: {count}")
    
    # Count episodes by level
    print("\n4. Episodes by JLPT level:")
    levels = {}
    for doc in rag.vector_store.metadata:
        if doc.get('type') == 'episode':
            level = doc.get('level', 'unknown')
            levels[level] = levels.get(level, 0) + 1
    
    for level in ['N5', 'N4', 'N3', 'N2', 'N1', 'unknown']:
        count = levels.get(level, 0)
        print(f"   - {level}: {count} episodes")
    
    # Show sample episodes
    print("\n5. Sample episodes (first 5):")
    episode_count = 0
    for doc in rag.vector_store.metadata:
        if doc.get('type') == 'episode' and episode_count < 5:
            print(f"\n   Episode {episode_count + 1}:")
            print(f"      ID: {doc.get('episode_id', 'N/A')}")
            print(f"      Title: {doc.get('title', 'N/A')}")
            print(f"      Anime: {doc.get('anime_name', 'N/A')}")
            print(f"      Level: {doc.get('level', 'N/A')}")
            print(f"      Lines: {doc.get('total_lines', 0)}")
            print(f"      Vocab: {doc.get('vocab_count', 0)}")
            episode_count += 1
    
    # Show all unique levels found
    print("\n6. All unique levels in dataset:")
    all_levels = set()
    for doc in rag.vector_store.metadata:
        level = doc.get('level')
        if level:
            all_levels.add(level)
    print(f"   {sorted(all_levels)}")
    
    # Test a search
    print("\n7. Testing search for N3 content...")
    try:
        results = rag.search_episodes_by_level(level="N3", query="anime", n_results=5)
        print(f"   Found {len(results)} results")
        if results:
            print("\n   First result:")
            first = results[0]
            for key, value in first.items():
                print(f"      {key}: {value}")
        else:
            print("   ❌ No results found!")
            
            # Try without level filter
            print("\n8. Testing search without level filter...")
            # Get all episodes
            all_episodes = [doc for doc in rag.vector_store.metadata if doc.get('type') == 'episode']
            print(f"   Total episodes available: {len(all_episodes)}")
            
            if all_episodes:
                print("\n   Sample episode data structure:")
                sample = all_episodes[0]
                print(f"   Keys: {list(sample.keys())}")
                print(f"   Sample: {json.dumps(sample, indent=2, default=str)[:500]}...")
    
    except Exception as e:
        print(f"   ❌ Search failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Check anime list
    print("\n9. Available anime series:")
    anime_list = set()
    for doc in rag.vector_store.metadata:
        if doc.get('type') == 'episode':
            anime_name = doc.get('anime_name')
            if anime_name:
                anime_list.add(anime_name)
    
    for anime in sorted(anime_list):
        # Count episodes for this anime
        count = sum(1 for doc in rag.vector_store.metadata 
                   if doc.get('type') == 'episode' and doc.get('anime_name') == anime)
        print(f"   - {anime}: {count} episodes")
    
    print("\n" + "="*60)
    print("✅ Debug complete!")
    print("="*60)


if __name__ == "__main__":
    debug_index()
