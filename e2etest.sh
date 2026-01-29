#!/bin/bash

# Exit on error
set -e

# Configuration
export RUNPOD_API_KEY=rpa_ZKZLYZ29PVCVGNVCDH50GWIWU6T4QA5NBDRUJFH1axwo14
export HF_TOKEN=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
DOCKER_USERNAME="pxxxe"

echo "=========================================="
echo "🚀 AZU.CX E2E Test Suite (Remote Build)"
echo "=========================================="

# Step 1: Pull Pre-built Images (built by GitHub Actions)
echo ""
echo "📥 Step 1: Pulling Pre-built Images from Docker Hub..."
echo "------------------------------------------"

echo "Pulling Core Image..."
docker pull ${DOCKER_USERNAME}/azu-core:latest
if [ $? -ne 0 ]; then
    echo "❌ Core pull failed! Make sure GitHub Actions built the image."
    exit 1
fi
echo "✅ Core image pulled successfully"

echo ""
echo "Pulling Worker Image..."
docker pull ${DOCKER_USERNAME}/azu-worker:latest
if [ $? -ne 0 ]; then
    echo "❌ Worker pull failed! Make sure GitHub Actions built the image."
    exit 1
fi
echo "✅ Worker image pulled successfully"

# Step 2: Run Infrastructure Test
echo ""
echo "🧪 Step 2: Running Infrastructure Test..."
echo "------------------------------------------"
echo "⚠️  Make sure to edit infra_test.py with:"
echo "    - Your Volume ID"
echo "    - Your Docker Hub username in CORE_IMG and WORKER_IMG"
echo ""
echo "Press Enter to continue or Ctrl+C to cancel..."
read

python3 infra_test.py

echo ""
echo "=========================================="
echo "✅ E2E Test Complete!"
echo "=========================================="
