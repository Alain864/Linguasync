#!/bin/bash
# Setup Amazon OpenSearch Serverless for LinguaSync
# This script creates the collection with all required policies

set -e

# Configuration
COLLECTION_NAME="linguasync-episodes"
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="992382843355"

# Get your IAM role ARN (the ECS task role)
TASK_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/linguasyncTaskRolex"

# Also add your current user for testing
CURRENT_USER_ARN=$(aws sts get-caller-identity --query Arn --output text)

echo "======================================"
echo "OpenSearch Serverless Setup"
echo "======================================"
echo "Collection: ${COLLECTION_NAME}"
echo "Region: ${AWS_REGION}"
echo "Task Role: ${TASK_ROLE_ARN}"
echo "Current User: ${CURRENT_USER_ARN}"
echo ""

# 1. ENCRYPTION POLICY (Required)
echo "Step 1: Creating encryption policy..."
cat > encryption-policy.json <<EOF
{
  "Rules": [
    {
      "ResourceType": "collection",
      "Resource": [
        "collection/${COLLECTION_NAME}"
      ]
    }
  ],
  "AWSOwnedKey": true
}
EOF

aws opensearchserverless create-security-policy \
  --name linguasync-encryption \
  --type encryption \
  --policy file://encryption-policy.json \
  --region ${AWS_REGION} 2>/dev/null || echo "Encryption policy already exists"

echo "✅ Encryption policy created"

# 2. NETWORK POLICY (Required)
echo "Step 2: Creating network policy..."
cat > network-policy.json <<EOF
[
  {
    "Rules": [
      {
        "ResourceType": "collection",
        "Resource": [
          "collection/${COLLECTION_NAME}"
        ]
      },
      {
        "ResourceType": "dashboard",
        "Resource": [
          "collection/${COLLECTION_NAME}"
        ]
      }
    ],
    "AllowFromPublic": true
  }
]
EOF

aws opensearchserverless create-security-policy \
  --name linguasync-network \
  --type network \
  --policy file://network-policy.json \
  --region ${AWS_REGION} 2>/dev/null || echo "Network policy already exists"

echo "✅ Network policy created"

# 3. DATA ACCESS POLICY (Required)
echo "Step 3: Creating data access policy..."
cat > data-access-policy.json <<EOF
[
  {
    "Rules": [
      {
        "ResourceType": "index",
        "Resource": [
          "index/${COLLECTION_NAME}/*"
        ],
        "Permission": [
          "aoss:CreateIndex",
          "aoss:DeleteIndex",
          "aoss:UpdateIndex",
          "aoss:DescribeIndex",
          "aoss:ReadDocument",
          "aoss:WriteDocument"
        ]
      },
      {
        "ResourceType": "collection",
        "Resource": [
          "collection/${COLLECTION_NAME}"
        ],
        "Permission": [
          "aoss:CreateCollectionItems",
          "aoss:DeleteCollectionItems",
          "aoss:UpdateCollectionItems",
          "aoss:DescribeCollectionItems"
        ]
      }
    ],
    "Principal": [
      "${TASK_ROLE_ARN}",
      "${CURRENT_USER_ARN}"
    ]
  }
]
EOF

aws opensearchserverless create-access-policy \
  --name linguasync-access \
  --type data \
  --policy file://data-access-policy.json \
  --region ${AWS_REGION} 2>/dev/null || echo "Data access policy already exists"

echo "✅ Data access policy created"

# 4. CREATE COLLECTION
echo "Step 4: Creating OpenSearch Serverless collection..."
aws opensearchserverless create-collection \
  --name ${COLLECTION_NAME} \
  --type VECTORSEARCH \
  --region ${AWS_REGION} 2>/dev/null || echo "Collection already exists"

echo "✅ Collection creation initiated"

# Wait for collection to become active
echo "Step 5: Waiting for collection to become active..."
for i in {1..60}; do
  STATUS=$(aws opensearchserverless batch-get-collection \
    --names ${COLLECTION_NAME} \
    --region ${AWS_REGION} \
    --query 'collectionDetails[0].status' \
    --output text 2>/dev/null || echo "CREATING")
  
  if [ "$STATUS" = "ACTIVE" ]; then
    echo "✅ Collection is ACTIVE"
    break
  fi
  
  echo "   Status: $STATUS (waiting...)"
  sleep 10
  
  if [ $i -eq 60 ]; then
    echo "⚠️  Timeout waiting for collection. Check AWS Console."
    exit 1
  fi
done

# Get the collection endpoint
ENDPOINT=$(aws opensearchserverless batch-get-collection \
  --names ${COLLECTION_NAME} \
  --region ${AWS_REGION} \
  --query 'collectionDetails[0].collectionEndpoint' \
  --output text)

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo "Collection Name: ${COLLECTION_NAME}"
echo "Endpoint: ${ENDPOINT}"
echo ""
echo "Add this to your .env file:"
echo "OPENSEARCH_ENDPOINT=${ENDPOINT}"
echo ""
echo "Update your ECS task definition:"
echo "  \"OPENSEARCH_ENDPOINT\": \"${ENDPOINT}\""
echo "======================================"

# Cleanup temp files
rm -f encryption-policy.json network-policy.json data-access-policy.json

echo ""
echo "Next steps:"
echo "1. Add OPENSEARCH_ENDPOINT to your .env file"
echo "2. Update ecs-task-definition-stage2.json with the endpoint"
echo "3. Run: python rag_engine_v3.py to index data"