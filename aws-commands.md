# get secret
alaincl@Alains-MacBook-Air linguasync-stage0 % aws secretsmanager get-secret-value \
    --secret-id linguasync/\
    --region us-east-1 \
    --query 'SecretString' \
    --output text


# Get your VPC ID
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text)

# Create target group for Streamlit (port 8501)
aws elbv2 create-target-group \
    --name linguasync-frontend-tg \
    --protocol HTTP \
    --port 8501 \
    --vpc-id $VPC_ID \
    --target-type ip \
    --health-check-enabled \
    --health-check-path / \
    --health-check-interval-seconds 30 \
    --health-check-timeout-seconds 5 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 3 \
    --region us-east-1

# Save the target group ARN
FRONTEND_TG_ARN=$(aws elbv2 describe-target-groups --names linguasync-frontend-tg --query 'TargetGroups[0].TargetGroupArn' --output text)
echo "Frontend Target Group ARN: $FRONTEND_TG_ARN"

# Get ALB listener ARN
LISTENER_ARN=$(aws elbv2 describe-listeners \
    --load-balancer-arn $(aws elbv2 describe-load-balancers --names linguasync-alb --query 'LoadBalancers[0].LoadBalancerArn' --output text) \
    --region us-east-1 \
    --query 'Listeners[0].ListenerArn' \
    --output text)

# Add rule to forward traffic to frontend (port 8501 in URL path)
aws elbv2 create-rule \
    --listener-arn $LISTENER_ARN \
    --priority 10 \
    --conditions Field=path-pattern,Values='/app*' \
    --actions Type=forward,TargetGroupArn=$FRONTEND_TG_ARN \
    --region us-east-1

# Register task definition 
aws ecs register-task-definition \
    --cli-input-json file://task-definition-frontend.json \
    --region us-east-1

# Get subnets from API service
SUBNETS=$(aws ecs describe-services \
    --cluster linguasync-cluster \
    --services linguasync-api-service \
    --region us-east-1 \
    --query 'services[0].networkConfiguration.awsvpcConfiguration.subnets' \
    --output json | jq -r 'join(",")')

# Get security group from API service
SECURITY_GROUP=$(aws ecs describe-services \
    --cluster linguasync-cluster \
    --services linguasync-api-service \
    --region us-east-1 \
    --query 'services[0].networkConfiguration.awsvpcConfiguration.securityGroups[0]' \
    --output text)

echo "Subnets: $SUBNETS"
echo "Security Group: $SECURITY_GROUP"

# Allow ALB to access frontend on port 8501
aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP \
    --protocol tcp \
    --port 8501 \
    --source-group sg-0bc70f3597a9a5648 \
    --region us-east-1

# Create the service
aws ecs create-service \
    --cluster linguasync-cluster \
    --service-name linguasync-frontend-service \
    --task-definition linguasync-frontend:1 \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SECURITY_GROUP],assignPublicIp=ENABLED}" \
    --load-balancers "targetGroupArn=$FRONTEND_TG_ARN,containerName=linguasync-frontend,containerPort=8501" \
    --region us-east-1

# Test frontend through ALB
curl http://linguasync-alb-2085313274.us-east-1.elb.amazonaws.com/app/

# Test the recommend endpoint directly
curl -X POST http://linguasync-alb-2085313274.us-east-1.elb.amazonaws.com/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_level": "N4",
    "query": "simple conversation",
    "n_results": 3
  }'



docker buildx build --platform linux/amd64 -t linguasync-frontend .

docker build -f Dockerfile.frontend -t linguasync-frontend:latest .

docker buildx build --platform linux/amd64 -f Dockerfile.production -t linguasync-api:with-data --load .

docker build -f Dockerfile.production -t linguasync-nur-api .


docker tag linguasync-api:with-data 992382843355.dkr.ecr.us-east-1.amazonaws.com/linguasync-api-production:latest

docker buildx build --platform linux/amd64 -f Dockerfile.frontend -t linguasync-frontend:v3 --load .

docker tag linguasync-frontend:v3 992382843355.dkr.ecr.us-east-1.amazonaws.com/linguasync-frontend:v3


/////////////////////////////////////////////////////////////////////

ok it worked. Now continue with stage 2. Read carefully below is what I need you to create be straightforward and deliver what I am asking for and what is needed for it to work dont invent or overdo things.

Stage 2: Enhanced Learning Features 
Goal: Make recommendations truly educational 
Deliverables:
LangGraph orchestration for multi-step reasoning
Grammar explanations in context (find real usage examples)
Cultural context notes generation
Pre-watch vocabulary 

things to improve from stage 1.
currently the api backend data stored inside a container, Modify it to Use aws services.
make sure all data is processed, embedded.
In this stage will use Amazon OpenSearch Serverless.
CloudWatch Logs, they give me issues so I ommited them now we will use it.
organize files/database by show, season, episode. currently everything is processed and save in one place.



//////////////////////////////////////////////////////////////////////

# OpenSearch Serverless collection

```bash
# Set your region
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=992382843355

# Create encryption policy
aws opensearchserverless create-security-policy \
    --name linguasync-encryption \
    --type encryption \
    --policy '{
        "Rules": [
            {
                "ResourceType": "collection",
                "Resource": ["collection/linguasync-vectors"]
            }
        ],
        "AWSOwnedKey": true
    }' \
    --region $AWS_REGION

# Create network policy
aws opensearchserverless create-security-policy \
    --name linguasync-network \
    --type network \
    --policy '[{
        "Rules": [
            {
                "ResourceType": "collection",
                "Resource": ["collection/linguasync-vectors"]
            }
        ],
        "AllowFromPublic": true
    }]' \
    --region $AWS_REGION

aws opensearchserverless create-access-policy \
    --name linguasync-access \
    --type data \
    --policy '[
        {
            "Rules": [
                {
                    "ResourceType": "collection",
                    "Resource": ["collection/linguasync-episodes"],
                    "Permission": ["aoss:*"]
                },
                {
                    "ResourceType": "index",
                    "Resource": ["index/linguasync-episodes/*"],
                    "Permission": ["aoss:*"]
                }
            ],
            "Principal": ["arn:aws:iam::992382843355:role/linguasyncTaskRolex"]
        }
    ]' \
    --region $AWS_REGION

# Create collection
aws opensearchserverless create-collection \
    --name linguasync-vectors \
    --type VECTORSEARCH \
    --region $AWS_REGION

# Wait for creation (takes 5-10 minutes)
echo "Waiting for collection creation..."
sleep 600

# Get endpoint
OPENSEARCH_ENDPOINT=$(aws opensearchserverless batch-get-collection \
    --names linguasync-vectors \
    --region $AWS_REGION \
    --query 'collectionDetails[0].collectionEndpoint' \
    --output text)

echo "OpenSearch Endpoint: $OPENSEARCH_ENDPOINT"
# Save this - you'll need it!


## Step 2: Update IAM Role Permissions

```bash
# Add OpenSearch permissions to task role
aws iam put-role-policy \
    --role-name linguasyncTaskRolex \
    --policy-name OpenSearchAccess \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "aoss:APIAccessAll",
                    "aoss:List*",
                    "aoss:Get*",
                    "aoss:DescribeIndex",
                    "aoss:ReadDocument",
                    "aoss:WriteDocument"
                ],
                "Resource": "*"
            }
        ]
    }'


aws iam put-role-policy \
    --role-name linguasyncTaskRolex \
    --policy-name CloudWatchLogs \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "*"
            }
        ]
    }'


export S3_BUCKET_NAME=linguasync-subtitles-bkt 
export OPENSEARCH_ENDPOINT=https://x3qd4r5pd6t9p3vwaxt1.us-east-1.aoss.amazonaws.com