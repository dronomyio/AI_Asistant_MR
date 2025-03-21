#!/bin/bash

# Setup environment - make sure to add your API keys
if [ -z "$ANTHROPIC_API_KEY" ] || [ -z "$VOYAGE_API_KEY" ] || [ -z "$COHERE_API_KEY" ]; then
    echo "Please set your API keys first:"
    echo "export ANTHROPIC_API_KEY=your_key"
    echo "export VOYAGE_API_KEY=your_key"
    echo "export COHERE_API_KEY=your_key"
    exit 1
fi

# Create directories
mkdir -p data

# Check if Elasticsearch is running
if ! curl -s http://localhost:9200 > /dev/null; then
    echo "Elasticsearch not running. Starting container..."
    docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "xpack.security.enabled=false" elasticsearch:8.8.0
    echo "Waiting for Elasticsearch to start..."
    sleep 20
fi

# Run the pipeline
echo "Step 1: Scraping Modal AI documentation..."
python scrape_docs.py

echo "Step 2: Processing documents and creating chunks..."
python process_docs.py

echo "Step 3: Building contextual embeddings..."
python contextual_embeddings.py

echo "Step 4: Starting advanced retrieval system..."
python advanced_retrieval.py