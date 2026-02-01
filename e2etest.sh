#!/bin/bash

# Exit on error
set -e


echo "=========================================="
echo "🚀 AZU.CX E2E Test Suite"
echo "=========================================="

python3 infra_test.py

echo ""
echo "=========================================="
echo "✅ E2E Test Complete!"
echo "=========================================="
