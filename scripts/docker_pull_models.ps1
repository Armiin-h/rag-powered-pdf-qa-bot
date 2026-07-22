Write-Host "Pulling Ollama models into the running compose stack..."
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull llama3.2
Write-Host "Models ready."
