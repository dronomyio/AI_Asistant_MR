# Modal AI Documentation Retrieval System

This project implements an advanced Retrieval Augmented Generation (RAG) system for the Modal AI documentation (https://docs.modalai.com) using contextual embeddings and hybrid search techniques.

## Features

- **Web Scraping**: Automatically downloads and processes the Modal AI documentation
- **Contextual Embeddings**: Uses Claude to generate context for each document chunk
- **Weaviate Vector Storage**: Stores embeddings in a powerful vector database
- **Hybrid Search**: Combines vector similarity and BM25 for better retrieval
- **Reranking**: Further improves search relevance with Cohere's reranking API
- **RAG Responses**: Generates helpful answers to user queries
- **Chat Interface**: User-friendly web UI for interacting with the system
- **Fully Dockerized**: Run everything with a single command

## Project Structure

```
modalai_docs_project/
├── app/                      # Chat web interface
│   ├── static/               # Static assets
│   ├── templates/            # HTML templates
│   ├── chat_server.py        # Flask + Socket.IO server
│   └── __init__.py
├── src/                      # Core modules
│   ├── db/                   # Database clients
│   │   ├── elasticsearch_client.py
│   │   ├── weaviate_client.py
│   │   └── __init__.py
│   ├── scraper/              # Web scraping
│   │   ├── modal_ai_scraper.py
│   │   └── __init__.py
│   ├── processor/            # Document processing
│   │   ├── document_processor.py
│   │   └── __init__.py
│   ├── embeddings/           # Contextual embeddings
│   │   ├── contextual_embeddings.py
│   │   └── __init__.py
│   ├── retrieval/            # Advanced retrieval
│   │   ├── advanced_retrieval.py
│   │   └── __init__.py
│   ├── utils/                # Utility functions
│   │   └── __init__.py
│   └── __init__.py
├── data/                     # Data storage directory
├── docker-compose.yml        # Docker services configuration
├── Dockerfile                # Container build configuration
├── run.py                    # CLI entry point
├── setup.py                  # Package definition
└── requirements.txt          # Dependencies
```

## Docker Setup

The entire system is containerized for easy deployment:

1. Set up environment variables:
   ```
   export ANTHROPIC_API_KEY="your_anthropic_key"
   export VOYAGE_API_KEY="your_voyage_key"
   export COHERE_API_KEY="your_cohere_key"
   ```

2. Run the Docker setup:
   ```
   docker-compose up -d
   ```

3. Execute the full pipeline:
   ```
   docker-compose exec app python run.py pipeline
   ```

4. Or run specific parts:
   ```
   # Scrape the documentation
   docker-compose exec app python run.py scrape
   
   # Process documents into chunks
   docker-compose exec app python run.py process
   
   # Generate contextual embeddings
   docker-compose exec app python run.py embed
   
   # Start the chat server
   docker-compose exec app python run.py chat
   ```

5. Access the chat interface at http://localhost:5005

## API Keys

This project requires three API keys:

1. **ANTHROPIC_API_KEY**: Used for Claude to generate contextual descriptions and RAG responses
   - Sign up at https://www.anthropic.com/
   
2. **VOYAGE_AI_KEY**: Used for high-quality embeddings
   - Sign up at https://www.voyageai.com/
   - You could modify the code to use other embedding providers if preferred

3. **COHERE_API_KEY**: Used for reranking search results
   - Sign up at https://cohere.com/
   
All three keys need to be set as environment variables before running the Docker containers.

## Troubleshooting

If you encounter port conflicts when starting the containers:

1. Check if any port is already in use:
   ```
   netstat -aln | grep -E ':(5005|8081|9200)'
   ```

2. Stop other Docker containers that might be using the ports:
   ```
   docker container ls
   docker stop [container-id]
   ```

3. Modify the port mappings in `docker-compose.yml` if needed.

## Local Development

For local development without Docker:

1. Set up environment variables:
   ```
   export ANTHROPIC_API_KEY="your_anthropic_key"
   export VOYAGE_API_KEY="your_voyage_key"
   export COHERE_API_KEY="your_cohere_key"
   ```

2. Install dependencies:
   ```
   pip install -e .
   ```

3. Start Weaviate and Elasticsearch:
   ```
   docker-compose up -d weaviate elasticsearch
   ```

4. Run the CLI:
   ```
   python run.py --help
   ```

## Component Architecture

- **Scraping Layer**: Retrieves documentation from the Modal AI website
- **Processing Layer**: Chunks documents and generates contextual embeddings
- **Storage Layer**: 
  - Weaviate: Stores vector embeddings for semantic search
  - Elasticsearch: Provides BM25 text search capabilities
- **Retrieval Layer**: 
  - Hybrid search combining results from Weaviate and Elasticsearch
  - Reranking for improved result ordering
- **Generation Layer**: Creates RAG responses using Claude
- **Interface Layer**: Web UI using Flask and Socket.IO for real-time communication

## Chat Interface

The chat interface provides:

- Real-time interaction with the Modal AI documentation
- Source citations for every answer
- Status indicators for system state
- Mobile-friendly responsive design

## References

- Based on techniques described in Anthropic's [Contextual Retrieval blog post](https://www.anthropic.com/news/contextual-retrieval)
- Implementation follows patterns from Anthropic's cookbook: https://github.com/anthropics/anthropic-cookbook

## Components Used

- **Claude 3**: For contextual enrichment and answer generation
- **Voyage AI**: For high-quality embeddings
- **Weaviate**: Vector database for semantic search
- **Elasticsearch**: For BM25 text search
- **Cohere**: For reranking search results
- **Flask & Socket.IO**: For the chat web interface# AI_Asistant_MR
