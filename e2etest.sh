#!/bin/bash

# Exit on error
set -e

# Configuration
export RUNPOD_API_KEY=rpa_ZKZLYZ29PVCVGNVCDH50GWIWU6T4QA5NBDRUJFH1axwo14
export HF_TOKEN=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

echo "=========================================="
echo "🚀 AZU.CX E2E Test Suite"
echo "=========================================="
echo ""
echo "⚠️  Make sure GitHub Actions has built and pushed your images!"
echo "   Check: https://github.com/YOUR_USERNAME/azu.cx/actions"
echo ""
echo "⚠️  Make sure infra_test.py has correct config:"
echo "    - CORE_IMG = 'pxxxe/azu-core:latest'"
echo "    - WORKER_IMG = 'pxxxe/azu-worker:latest'"
echo "    - Your Volume ID"
echo ""
echo "Press Enter to continue or Ctrl+C to cancel..."
read

echo ""
echo "🚀 Launching RunPod instances..."
echo "   (RunPod will pull images directly from Docker Hub)"
echo ""

python3 infra_test.py

echo ""
echo "=========================================="
echo "✅ E2E Test Complete!"
echo "=========================================="
