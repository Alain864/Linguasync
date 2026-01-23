#!/usr/bin/env python3
"""
add_new_episodes.py - Safely add new episodes to existing S3 index

This script:
1. Downloads existing index from S3
2. Loads new episodes from S3
3. Adds only NEW episodes (skips duplicates)
4. Uploads updated index back to S3

This prevents losing your existing data!
"""

import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from rag_engine_s3 import RAGEngineS3

def add_new_episodes():
    """Add new episodes to existing index"""
    
    logger.info("="*60)
    logger.info("➕ Adding New Episodes to Existing Index")
    logger.info("="*60)
    
    # Initialize RAG engine (loads existing index from S3)
    logger.info("\n1. Loading existing index from S3...")
    rag = RAGEngineS3()
    
    # Get currently indexed episode IDs
    existing_episode_ids = set()
    for doc in rag.vector_store.metadata:
        if doc.get('type') == 'episode':
            existing_episode_ids.add(doc.get('episode_id'))
    
    logger.info(f"   Currently indexed: {len(existing_episode_ids)} episodes")
    
    # Load all episodes from S3 (including new ones)
    logger.info("\n2. Loading episodes from S3...")
    all_episodes = rag.load_episodes_from_s3()
    logger.info(f"   Found {len(all_episodes)} total episodes in S3")
    
    # Find new episodes
    new_episodes = []
    for episode in all_episodes:
        episode_id = episode.get('episode_id')
        if episode_id not in existing_episode_ids:
            new_episodes.append(episode)
    
    logger.info(f"\n3. Found {len(new_episodes)} NEW episodes to add")
    
    if len(new_episodes) == 0:
        logger.info("\n✅ No new episodes to add. Index is up to date!")
        return
    
    # Show what will be added
    logger.info("\n📺 New episodes:")
    anime_counts = {}
    for episode in new_episodes:
        anime_name = episode.get('anime_name')
        anime_counts[anime_name] = anime_counts.get(anime_name, 0) + 1
    
    for anime, count in sorted(anime_counts.items()):
        logger.info(f"   - {anime}: {count} episodes")
    
    # Ask for confirmation
    print("\n" + "="*60)
    response = input("Continue with indexing? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        logger.info("❌ Cancelled by user")
        return
    
    # Index new episodes only
    logger.info("\n4. Indexing new episodes...")
    rag.index_episodes(new_episodes)
    
    # Show final stats
    logger.info("\n" + "="*60)
    logger.info("📊 Final Statistics:")
    logger.info("="*60)
    
    stats = rag.get_collection_stats()
    logger.info(f"Total episodes: {stats['episode_count']}")
    logger.info(f"Total anime: {stats['unique_anime']}")
    logger.info(f"Total documents: {stats['total_items']}")
    
    # Level distribution
    logger.info("\n📈 Level Distribution:")
    levels = {}
    for doc in rag.vector_store.metadata:
        if doc.get('type') == 'episode':
            level = doc.get('level', 'unknown')
            levels[level] = levels.get(level, 0) + 1
    
    for level in ['N5', 'N4', 'N3', 'N2', 'N1']:
        count = levels.get(level, 0)
        if count > 0:
            logger.info(f"   {level}: {count} episodes")
    
    logger.info("\n" + "="*60)
    logger.info("✅ New Episodes Added Successfully!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Restart production API:")
    logger.info("   aws ecs update-service --cluster linguasync-cluster \\")
    logger.info("     --service linguasync-api-service-s3 --force-new-deployment")
    logger.info("\n2. Or restart local API:")
    logger.info("   uvicorn api_s3:app --reload --port 8000")


if __name__ == "__main__":
    add_new_episodes()