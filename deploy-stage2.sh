#!/bin/bash
# Deploy LinguaSync Stage 2 to AWS

set -e

echo "======================================"
echo "LinguaSync Stage 2 Deployment"
echo "======================================"

# Configuration
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="992382843355"
ECR_REPO="linguasync-api-stage2"
CLUSTER_NAME="linguasync-cluster"
SERVICE_NAME="linguasync-api-service-stage2"
TASK_FAMILY="linguasync-api-stage2"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color



docker build -t linguasync-api-stage2 .

docker build -t ${ECR_REPO}:latest -f Dockerfile.stage2 .




docker tag linguasync-api:with-data 992382843355.dkr.ecr.us-east-1.amazonaws.com/linguasync-api-production:latest



this backend stage2
docker buildx build --platform linux/amd64 -f Dockerfile.stage2 -t linguasync-api-stage2 --load .

docker tag linguasync-api-stage2:latest 992382843355.dkr.ecr.us-east-1.amazonaws.com/linguasync-api-stage2:latest

docker build -t linguasync-api-stage2:latest -f Dockerfile.stage2 .


docker tag linguasync-frontend:v3 992382843355.dkr.ecr.us-east-1.amazonaws.com/linguasync-frontend:v3



    aws ecr create-repository --repository-name linguasync-api-stage2 --region us-east-1 


    




echo -e "${YELLOW}Step 2: Tagging image...${NC}"
docker tag ${ECR_REPO}:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest
docker tag ${ECR_REPO}:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:$(git rev-parse --short HEAD)

echo -e "${YELLOW}Step 3: Logging into ECR...${NC}"
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com




echo -e "${YELLOW}Step 5: Pushing image to ECR...${NC}"
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:$(git rev-parse --short HEAD)

echo -e "${YELLOW}Step 6: Registering new task definition...${NC}"
TASK_DEFINITION_ARN=$(aws ecs register-task-definition \
    --cli-input-json file://ecs-task-definition-stage2.json \
    --region ${AWS_REGION} \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text)

echo -e "${GREEN}New task definition: ${TASK_DEFINITION_ARN}${NC}"

echo -e "${YELLOW}Step 7: Updating ECS service...${NC}"
aws ecs update-service \
    --cluster ${CLUSTER_NAME} \
    --service ${SERVICE_NAME} \
    --task-definition ${TASK_DEFINITION_ARN} \
    --region ${AWS_REGION} \
    --force-new-deployment

echo -e "${GREEN}Deployment initiated!${NC}"

echo -e "${YELLOW}Step 8: Waiting for service to stabilize...${NC}"
aws ecs wait services-stable \
    --cluster ${CLUSTER_NAME} \
    --services ${SERVICE_NAME} \
    --region ${AWS_REGION}

echo -e "${GREEN}======================================"
echo -e "Deployment Complete!"
echo -e "======================================${NC}"

# Show service status
echo -e "${YELLOW}Service Status:${NC}"
aws ecs describe-services \
    --cluster ${CLUSTER_NAME} \
    --services ${SERVICE_NAME} \
    --region ${AWS_REGION} \
    --query 'services[0].{Status:status,Running:runningCount,Desired:desiredCount,TaskDefinition:taskDefinition}' \
    --output table