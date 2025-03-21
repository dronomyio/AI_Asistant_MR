#!/bin/bash

# Check for required environment variables
if [ -z "$ANTHROPIC_API_KEY" ] || [ -z "$VOYAGE_API_KEY" ] || [ -z "$COHERE_API_KEY" ]; then
    echo "Error: Required API keys not set. Please set the following environment variables:"
    echo "  - ANTHROPIC_API_KEY"
    echo "  - VOYAGE_API_KEY"
    echo "  - COHERE_API_KEY"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# Start all services and wait for them to be ready
echo "Starting containers with docker-compose..."
docker-compose up -d

echo "Waiting for services to be ready..."
until $(curl --output /dev/null --silent --head --fail http://localhost:8080/v1/.well-known/ready); do
    printf '.'
    sleep 5
done
until $(curl --output /dev/null --silent --head --fail http://localhost:9200); do
    printf '.'
    sleep 5
done
echo "Services are ready!"

# Run the pipeline
echo "Running the Modal AI documentation retrieval pipeline..."

# 1. Scrape documents (if needed)
if [ ! -f "data/modalai_docs.json" ]; then
    echo "Step 1: Scraping Modal AI documentation..."
    docker-compose exec app python scrape_docs.py
else
    echo "Step 1: Using existing scraped documentation."
fi

# 2. Process documents
if [ ! -f "data/modalai_chunks.json" ]; then
    echo "Step 2: Processing documents and creating chunks..."
    docker-compose exec app python process_docs.py
else
    echo "Step 2: Using existing processed chunks."
fi

# 3. Build contextual embeddings with Weaviate
echo "Step 3: Building contextual embeddings and storing in Weaviate..."
docker-compose exec app python contextual_embeddings_weaviate.py

# 4. Start the search interface
echo "Step 4: Starting advanced retrieval system..."
docker-compose exec -it app python advanced_retrieval_weaviate.py

# Option to shut down containers
read -p "Would you like to shut down the containers? (y/n) " answer
if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
    echo "Shutting down containers..."
    docker-compose down
    echo "Containers have been stopped and removed."
else
    echo "Containers are still running. Use 'docker-compose down' to shut them down later."
fi