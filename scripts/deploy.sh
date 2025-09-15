#!/bin/bash
# COSMOS 배포 스크립트

set -e

# 환경 변수
ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}
REGISTRY=${REGISTRY:-"your-registry.com"}

echo "🚀 Deploying COSMOS v${VERSION} to ${ENVIRONMENT}"

# 1. 도커 이미지 빌드
echo "📦 Building Docker image..."
docker build -t cosmos-emotion:${VERSION} .
docker tag cosmos-emotion:${VERSION} ${REGISTRY}/cosmos-emotion:${VERSION}

# 2. 이미지 푸시
echo "⬆️ Pushing to registry..."
docker push ${REGISTRY}/cosmos-emotion:${VERSION}

# 3. 데이터베이스 마이그레이션
echo "🗄️ Running migrations..."
python scripts/migrate.py --env ${ENVIRONMENT}

# 4. Kubernetes 배포
echo "☸️ Deploying to Kubernetes..."
kubectl apply -f deployment/kubernetes.yaml
kubectl set image deployment/cosmos-api cosmos=${REGISTRY}/cosmos-emotion:${VERSION}

# 5. 헬스체크 대기
echo "🏥 Waiting for health check..."
kubectl wait --for=condition=ready pod -l app=cosmos --timeout=300s

# 6. 스모크 테스트
echo "🧪 Running smoke tests..."
python scripts/smoke_test.py --url $(kubectl get service cosmos-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo "✅ Deployment complete!"