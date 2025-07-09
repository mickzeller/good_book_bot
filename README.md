# The Good Book Bot

A Retrieval-Augmented Generation (RAG) demo for scripture search and question answering using FAISS vector store and HuggingFace embeddings.

## Features

- **Scripture Text Processing**: Loads and processes scripture text files
- **Vector Search**: Uses FAISS for efficient similarity search
- **Embeddings**: HuggingFace embeddings (all-MiniLM-L6-v2) for semantic search
- **Question Answering**: RAG pipeline for answering questions about scriptures
- **Jupyter Integration**: Interactive development environment

## Prerequisites

- Docker and Docker Compose
- Python 3.12 (if running locally)
- Poetry (if running locally)

## Quick Start with Docker

1. **Build and run the container:**
   ```bash
   docker-compose up --build
   ```

2. **Access Jupyter Lab:**
   Open your browser to `http://localhost:8888`

3. **Add your scripture data:**
   Place your `.txt` files in the `data/` directory

## Local Development

1. **Install dependencies:**
   ```bash
   poetry install
   ```

2. **Run Jupyter Lab:**
   ```bash
   poetry run jupyter lab
   ```

3. **Run the demo script:**
   ```bash
   poetry run python main.py
   ```

## Usage

The main functionality is in `main.py`:

```python
# Example usage
question = "What is the purpose of baptism?"
result = qa_chain.invoke({"query": question})
print(result["result"])
```

## Project Structure

```
├── data/                 # Scripture text files
├── main.py              # Main RAG demo script
├── pyproject.toml       # Poetry dependencies
├── docker-compose.yml   # Docker configuration
├── Dockerfile          # Container definition
└── README.md           # This file
```

## Dependencies

- **langchain**: RAG framework
- **langchain-community**: Community extensions
- **langchain-huggingface**: HuggingFace integrations
- **faiss-cpu**: Vector similarity search
- **sentence-transformers**: Text embeddings
- **tensorflow**: ML framework
- **pandas**: Data manipulation
- **jupyterlab**: Interactive development

## Docker Optimizations

The Docker setup includes several optimizations for faster builds:

- **Build caching**: Poetry and pip downloads are cached
- **Layer optimization**: Dependencies installed before code copy
- **Minimal base image**: Using python:3.12-slim
- **.dockerignore**: Excludes unnecessary files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational purposes.