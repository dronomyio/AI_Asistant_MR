# Modal AI Documentation Retrieval System

This project implements an advanced Retrieval Augmented Generation (RAG) system for Modal AI drone documentation using contextual embeddings, hybrid search techniques, and multimodal content.

## Features

- **Git Repository Processing**: Automatically processes Git repositories for Modal AI documentation and code
  - **Markdown Processing**: Chunks markdown by section for meaningful context
  - **Code Processing**: Intelligently chunks code by function/class with context
  - **Multimedia Content**: Extracts and processes images, videos, and document files
- **Media Processing**: Extracts content from various media types
  - **OCR for Images**: Extracts text from diagrams, charts, and photos
  - **Media References**: Tracks and links media files to their source documents
- **Multimodal RAG**: Integrates multimedia content into the retrieval system
  - **Media-Aware Embeddings**: Includes processed media content in contextual embeddings
  - **Media References**: Includes relevant images and files in search results
  - **Visual Context**: RAG responses that reference relevant visual information
- **Contextual Embeddings**: Uses Claude to generate context for each document chunk
- **Chat Interface**: User-friendly web UI for interacting with the system

## Quickstart - Docker

The easiest way to get started is using Docker:

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/modalai-docs-project.git
   cd modalai-docs-project
   ```

2. (Optional) Clone a Modal AI repository:
   ```bash
   # Use the helper script to clone a Modal AI repo
   ./clone-repo.sh https://github.com/modalai/documentation.git
   ```

3. Start the Docker container:
   ```bash
   ./start-docker.sh
   ```

4. Access the chat interface at http://localhost:5678

### What Happens in Docker?

1. The Docker container starts and looks for repositories in the `data/repos` directory
2. If no repositories are found, it creates a sample repository with drone documentation
3. It processes all repositories, extracting content and media references
4. The chat server starts, providing a web interface to interact with the documentation

## Manual Setup (Without Docker)

If you prefer not to use Docker:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install flask flask-socketio python-dotenv
   ```

2. Prepare a repository:
   ```bash
   mkdir -p data/repos
   # Clone a repository or create a sample repo
   git clone https://github.com/modalai/documentation.git data/repos/modal-docs
   ```

3. Process the repository:
   ```bash
   python run.py process-repos
   ```

4. Create a symbolic link for media:
   ```bash
   mkdir -p app/static/media
   ln -s $(pwd)/data/repos/modal-docs/voxl2/images app/static/media/voxl2
   ```

5. Start the chat interface:
   ```bash
   python run.py chat
   ```

6. Access the chat interface at http://localhost:5678

## Adding Your Own Repositories

You can add your own repositories to the system:

1. Clone repositories into the `data/repos` directory:
   ```bash
   git clone <repo_url> data/repos/<repo_name>
   ```

2. Process the repositories:
   ```bash
   python run.py process-repos
   ```

3. Restart the chat server if it's already running.

## API Keys (Optional)

For full functionality with real embeddings and LLM responses, you'll need:

1. **ANTHROPIC_API_KEY**: For Claude contextual descriptions and responses
   - Sign up at https://www.anthropic.com/

2. **VOYAGE_AI_KEY**: For high-quality embeddings
   - Sign up at https://www.voyageai.com/

3. **COHERE_API_KEY**: For reranking search results
   - Sign up at https://cohere.com/

Set these as environment variables:
```bash
export ANTHROPIC_API_KEY="your_key"
export VOYAGE_API_KEY="your_key" 
export COHERE_API_KEY="your_key"
```

Or add them to a `.env` file:
```
ANTHROPIC_API_KEY=your_key
VOYAGE_API_KEY=your_key
COHERE_API_KEY=your_key
```

## Repository Processor Features

The repository processor includes several advanced features:

1. **Intelligent Chunking**:
   - Markdown files are chunked by section based on headings
   - Code files are chunked by function/class with surrounding context
   - Other files use standard chunking with overlap

2. **Media Extraction**:
   - Identifies and processes images referenced in markdown
   - Maintains links between documents and their media references

3. **Contextual Enrichment**:
   - Adds document titles and file paths for better context
   - Preserves code structure and comments

## Files and Directories

```
modalai_docs_project/
├── app/                      # Chat web interface
│   ├── static/               # Static assets
│   │   ├── images/           # Sample images
│   │   └── media/            # Symbolic links to repository media
│   ├── templates/            # HTML templates
│   └── chat_server.py        # Flask + Socket.IO server
├── src/                      # Core modules
│   ├── processor/            # Document and media processing
│   │   ├── document_processor.py
│   │   ├── media_processor.py
│   │   └── repo_processor.py # Git repository processor
│   ├── embeddings/           # Contextual embeddings
│   │   ├── contextual_embeddings.py
│   │   └── multimodal_embeddings.py
│   └── retrieval/            # Search and retrieval
│       └── advanced_retrieval.py
├── data/                     # Data storage
│   ├── repos/                # Git repositories
│   │   └── modal-docs/       # Modal AI documentation
│   ├── repo_chunks.json      # Processed repository chunks
│   └── repo_media.json       # Media catalog
├── docker-compose-simple.yml # Simplified Docker setup
├── docker-entrypoint.sh      # Docker startup script
├── Dockerfile                # Container configuration
├── run.py                    # Command-line interface
├── clone-repo.sh             # Helper for cloning repositories
└── start-docker.sh           # Docker startup script
```

## Troubleshooting

If you encounter issues:

1. **Chat server won't start**: Check if port 5678 is already in use

2. **Repository processing fails**: Ensure the repository is properly cloned and contains documentation

3. **Media not displaying**: Check if the symbolic links are properly set up in `app/static/media`

4. **Docker issues**: Make sure Docker and docker-compose are installed and running

## Development and Extensions

This project can be extended in several ways:

1. **Enhanced Media Processing**: Add support for more media types or improve extraction

2. **Better Embedding Generation**: Implement true multimodal embeddings with CLIP or similar models

3. **Improved Search**: Add faceted search or filtering by document type

4. **User Interface Enhancements**: Add media viewers or expand the chat capabilities

## References

- Based on Anthropic's [contextual embeddings and RAG techniques](https://www.anthropic.com/news/contextual-retrieval)
- Implementation inspired by [Anthropic's cookbook examples](https://github.com/anthropics/anthropic-cookbook)