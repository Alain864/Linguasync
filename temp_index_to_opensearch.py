
"""Index processed S3 data into OpenSearch"""

import os
import boto3
from opensearch_engine import OpenSearchVectorEngine
from subtitle_processor_s3 import S3SubtitleProcessorV3
from dotenv import load_dotenv
load_dotenv()

def main():
    # Get config from environment
    bucket = os.environ['S3_BUCKET_NAME']
    endpoint = os.environ['OPENSEARCH_ENDPOINT']
    
    # Initialize
    s3_processor = S3SubtitleProcessorV3(bucket)
    opensearch = OpenSearchVectorEngine(endpoint)
    
    # Get all processed files
    processed_files = s3_processor.list_all_processed()
    print(f"Found {len(processed_files)} processed episodes")
    
    # Index each one
    for s3_key in processed_files:
        episode_data = s3_processor.get_processed_episode(s3_key)
        opensearch.index_episode(episode_data)
    
    # Show stats
    stats = opensearch.get_stats()
    print(f"\n📊 Indexed: {stats}")

if __name__ == "__main__":
    main()