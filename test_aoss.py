import os
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# Load environment variables
endpoint = os.environ.get('OPENSEARCH_ENDPOINT')
if not endpoint:
    raise ValueError("OPENSEARCH_ENDPOINT environment variable not set")

# Clean the endpoint (remove https:// if present)
endpoint = endpoint.replace("https://", "").rstrip("/")

region = 'us-east-1'          # ← change only if your collection is in another region
service = 'aoss'

# AWS credentials
credentials = boto3.Session().get_credentials()
if not credentials:
    raise ValueError("No AWS credentials found. Check ~/.aws/credentials or environment variables")

awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    service,
    session_token=credentials.token
)

print(f"Connecting to: {endpoint}")

# Create client
client = OpenSearch(
    hosts=[{'host': endpoint, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=30
)

print("Client created successfully")

# ────────────────────────────────────────────────
# Now test things that actually work on Serverless
# ────────────────────────────────────────────────

print("\nTesting basic connectivity and permissions...")

try:
    # This should work — checks if index exists (returns bool, no crash on 404)
    exists = client.indices.exists(index="linguasync-episodes")
    print(f"Index 'linguasync-episodes' exists: {exists}")
except Exception as e:
    print(f"Index exists check failed: {str(e)}")

try:
    # List all indices (very lightweight, supported in Serverless)
    indices_response = client.cat.indices(format="json")
    print("\nIndices in the collection:")
    if indices_response:
        for idx in indices_response:
            print(f"  - {idx['index']} (docs: {idx.get('docs.count', '0')})")
    else:
        print("  (no indices found)")
except Exception as e:
    print(f"Cat indices failed: {str(e)}")

print("\nTest finished.")