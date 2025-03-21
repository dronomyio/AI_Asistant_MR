#!/bin/bash

# Check for required environment variables
if [ -z "$ANTHROPIC_API_KEY" ] || [ -z "$VOYAGE_API_KEY" ] || [ -z "$COHERE_API_KEY" ]; then
    echo "Error: Required API keys not set. Please set the following environment variables:"
    echo "  - ANTHROPIC_API_KEY"
    echo "  - VOYAGE_API_KEY"
    echo "  - COHERE_API_KEY"
    exit 1
fi

# Create directories
mkdir -p data
mkdir -p templates
mkdir -p static

# Check if Elasticsearch and Weaviate are running
if ! curl -s http://localhost:9200 > /dev/null || ! curl -s http://localhost:8080/v1/.well-known/ready > /dev/null; then
    echo "Elasticsearch or Weaviate not running. Starting containers..."
    docker-compose up -d weaviate elasticsearch
    echo "Waiting for services to start..."
    sleep 20
fi

# Check if data exists
if [ ! -f "data/modalai_docs.json" ]; then
    echo "No data found. Running full pipeline first..."
    ./docker-run.sh
    exit 0
fi

# Start the chat UI
echo "Starting Modal AI Documentation Chat UI..."
docker-compose up -d chat

# Wait for the chat service to be ready
echo "Waiting for the chat service to start..."
until $(curl --output /dev/null --silent --head --fail http://localhost:5000); do
    printf '.'
    sleep 2
done

echo "\n\nChat UI is ready! Open http://localhost:5000 in your browser."
echo "Press Ctrl+C to stop viewing logs."

# Show logs from the chat container
docker-compose logs -f chat