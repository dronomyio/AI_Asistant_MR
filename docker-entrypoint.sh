#!/bin/bash
set -e

echo "Starting Modal AI Documentation RAG System"

# Create sample repository if none exists
if [ -z "$(ls -A /app/data/repos)" ]; then
    echo "No repositories found in /app/data/repos, creating sample repository..."
    cd /app/data/repos
    
    # Create modal-docs directory
    mkdir -p modal-docs/voxl2/{hardware,software,flight,images}
    cd modal-docs
    
    # Create sample README
    cat > README.md << EOF
# Modal AI Drone Documentation

This repository contains documentation for Modal AI drone products, including VOXL 2.

## Products

- [VOXL 2](voxl2/README.md) - Our flagship flight controller
- [Flight Core](flight/README.md) - Flight control software

## About Modal AI

Modal AI specializes in developing autonomous drone and robotics technology, providing both hardware and software solutions for various applications including delivery, inspection, and surveying.
EOF

    # Create VOXL 2 readme
    cat > voxl2/README.md << EOF
# VOXL 2

VOXL 2 is Modal AI's flagship flight controller for drones and robots.

## Overview

VOXL 2 combines a heterogeneous compute architecture, automotive-grade sensors, and robust connectivity in a compact form factor designed for drones and robotics applications. It features a Qualcomm QRB5165 processor that provides powerful AI and computer vision capabilities.

## Sections

- [Hardware Setup](hardware/quickstart.md)
- [Software Setup](software/quickstart.md)
- [Flight Configuration](flight/configuration.md)

## Key Features

- QRB5165 Processor with integrated AI accelerator
- Multiple MIPI camera inputs
- Built-in IMU, barometer and compass sensors
- Advanced thermal management
- Compact form factor (75mm x 55mm x 15mm)
- Multiple connectivity options including WiFi, Bluetooth, and LTE
EOF

    # Create hardware quickstart
    cat > voxl2/hardware/quickstart.md << EOF
# VOXL 2 Hardware Quickstart

This guide will help you set up your VOXL 2 hardware.

## Connection Diagram

![VOXL 2 Hardware Connections](../images/voxl2_hardware.png)

## Required Components

- VOXL 2 Flight Controller
- Power Distribution Board
- 4S or 6S LiPo Battery
- Telemetry Radio (optional)
- GPS Module
- ESCs and Motors

## Connection Steps

1. Connect power to J1 connector
2. Connect telemetry radio to UART1
3. Connect GPS module to UART2
4. Connect ESCs to motor outputs M1-M8

## Power Requirements

VOXL 2 requires a stable power source between 6V and 18V. For most drone applications, a 4S or 6S LiPo battery is recommended.
EOF

    # Create a simple hardware image (just a placeholder text file)
    echo "Placeholder for hardware diagram" > voxl2/images/voxl2_hardware.txt
    
    # Initialize git repository
    git init
    git config --global user.email "example@example.com"
    git config --global user.name "Example User"
    git add .
    git commit -m "Initial commit with sample documentation"
    
    echo "Sample repository created successfully."
    cd /app
fi

# Check if hardware image exists and copy from app resources if not
if [ ! -f /app/data/repos/modal-docs/voxl2/images/voxl2_hardware.png ]; then
    echo "Copying hardware image to repository..."
    if [ -f /app/app/static/images/voxl2_hardware.png ]; then
        cp /app/app/static/images/voxl2_hardware.png /app/data/repos/modal-docs/voxl2/images/ || echo "Warning: Could not copy hardware image"
    else
        echo "Hardware image not found in source app/static/images"
    fi
fi

# Process repositories
echo "Processing repositories..."
python run.py process-repos --repos-dir=data/repos --chunks-file=repo_chunks.json --media-catalog=repo_media.json

# Create symbolic link for media
mkdir -p /app/app/static/media
ln -sfn /app/data/repos/modal-docs/voxl2/images /app/app/static/media/voxl2 || echo "Warning: Could not create symbolic link"

# Display environment info
echo "Environment:"
echo "- WEAVIATE_URL: $WEAVIATE_URL"
echo "- ELASTICSEARCH_URL: $ELASTICSEARCH_URL"
echo "- WEAVIATE_COLLECTION: $WEAVIATE_COLLECTION"
echo "- OPENAI_EMBEDDING_MODEL: $OPENAI_EMBEDDING_MODEL"
echo "- HOST: $HOST"
echo "- PORT: $PORT"

# Display API key status (without showing the actual keys)
echo "API Keys:"
[ -n "$ANTHROPIC_API_KEY" ] && echo "- ANTHROPIC_API_KEY: ✅" || echo "- ANTHROPIC_API_KEY: ❌"
[ -n "$VOYAGE_API_KEY" ] && echo "- VOYAGE_API_KEY: ✅" || echo "- VOYAGE_API_KEY: ❌"
[ -n "$COHERE_API_KEY" ] && echo "- COHERE_API_KEY: ✅" || echo "- COHERE_API_KEY: ❌"
[ -n "$OPENAI_API_KEY" ] && echo "- OPENAI_API_KEY: ✅" || echo "- OPENAI_API_KEY: ❌"

# Execute the command passed to docker run
echo "Starting application with command: $@"
cd /app
exec "$@"