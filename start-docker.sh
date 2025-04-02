#!/bin/bash

echo "Starting Modal AI Documentation RAG System in Docker"

# Check if Docker is installed
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker and/or docker-compose is not installed."
    echo "Please install Docker Desktop from https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# Build and start the containers
docker-compose -f docker-compose-simple.yml up --build -d

echo ""
echo "Modal AI Documentation RAG System is starting up..."
echo "Please wait a moment for the system to initialize."
echo ""
echo "Chat interface will be available at: http://localhost:5678"
echo ""
echo "To view logs, run: docker-compose -f docker-compose-simple.yml logs -f"
echo "To stop the system, run: docker-compose -f docker-compose-simple.yml down"