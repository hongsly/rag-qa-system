# Docker Deployment Guide

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# 1. Make sure Ollama is running on host
ollama serve

# 2. Set your OpenAI API key in .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# 3. Start the application
docker-compose up -d

# 4. Open browser
open http://localhost:8501
```

### Option 2: Using Docker Run

```bash
# Build image
docker build -t rag-qa-system .

# Run container
docker run -d --name rage-qa
  -p 8501:8501 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  rag-qa-system
```

## Configuration

### Using Host Ollama (Default)

The docker-compose.yml is configured to connect to Ollama running on your host machine via `host.docker.internal:11434`.

**Mac/Windows:** Works automatically with Docker Desktop
**Linux:** Use `--network host` or update `OLLAMA_HOST` to your host IP:

```yaml
environment:
  - OLLAMA_HOST=http://172.17.0.1:11434
```

### Running Ollama in Docker

Uncomment the `ollama` service in `docker-compose.yml`:

```bash
# Start both services
docker-compose up -d

# Pull Ollama model
docker-compose exec ollama ollama pull qwen2.5-coder:7b

# Check models
docker-compose exec ollama ollama list
```

Then update the app service environment:
```yaml
environment:
  - OLLAMA_HOST=http://ollama:11434
```

## Useful Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# Check service health
docker-compose ps
```

## Troubleshooting

### Ollama Connection Failed

```bash
# Check if Ollama is reachable from container
docker-compose exec app curl http://host.docker.internal:11434/api/tags

# Expected: List of models
# If fails: Check Ollama is running on host (ollama serve)
```

### Missing Index Files

```bash
# Ensure index files exist
ls data/processed/rag_index.faiss
ls data/processed/chunks.jsonl

# If missing, build the index first (outside Docker)
python scripts/build_index.py
```

### Port Already in Use

```bash
# Change port in docker-compose.yml
ports:
  - "8502:8501"  # Use 8502 instead
```

## Production Considerations

1. **API Keys**: Use Docker secrets or external secret management
2. **Index Files**: Consider CDN or S3 for large indexes
3. **Ollama**: Run on GPU-enabled instance for better performance
4. **Monitoring**: Add Prometheus/Grafana for metrics
5. **Reverse Proxy**: Use nginx/Traefik for HTTPS and load balancing
