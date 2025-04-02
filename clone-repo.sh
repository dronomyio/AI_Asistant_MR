#!/bin/bash

# Check for repository URL argument
if [ -z "$1" ]; then
    echo "Error: Please provide a repository URL."
    echo "Usage: $0 <repository_url> [local_name]"
    echo "Example: $0 https://github.com/modalai/documentation.git modal-docs"
    exit 1
fi

# Get repository URL from argument
REPO_URL="$1"

# Set local name (use second argument or derive from URL)
if [ -n "$2" ]; then
    LOCAL_NAME="$2"
else
    # Extract name from URL (remove .git extension if present)
    LOCAL_NAME=$(basename "$REPO_URL" .git)
fi

# Create repos directory if it doesn't exist
mkdir -p data/repos

# Clone the repository
echo "Cloning $REPO_URL into data/repos/$LOCAL_NAME..."
git clone "$REPO_URL" "data/repos/$LOCAL_NAME"

# Check if clone was successful
if [ $? -eq 0 ]; then
    echo "Repository cloned successfully!"
    echo ""
    echo "To process this repository and start the chat interface, run:"
    echo "./start-docker.sh"
else
    echo "Error: Failed to clone repository."
    exit 1
fi
