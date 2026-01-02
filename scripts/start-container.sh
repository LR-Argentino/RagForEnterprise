#!/bin/bash

set -e

ENV=${1:-dev}

if [[ "$ENV" != "dev" && "$ENV" != "test" ]]; then
    echo "Usage: ./scripts/start.sh [dev|test]"
    echo ""
    echo "  dev  - SQLite backend (default)"
    echo "  test - CosmosDB backend"
    exit 1
fi

if [[ "$ENV" == "dev" ]]; then
    BACKEND="SQLite"
else
    BACKEND="CosmosDB"
fi

echo "=== RAGChat Server ==="
echo "Environment: $ENV"
echo "Backend: $BACKEND"
echo ""

echo "[1/2] Building Docker image..."
docker compose build

echo "[2/2] Starting server..."
APP_ENV=$ENV docker compose up
