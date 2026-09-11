# -------------------------------------------------------------------------
# LLM and embedding models configuration
#
# By default, both the LLM and the embedding model are served by a local
# Ollama instance through its OpenAI-compatible API (port 11434).
#
# LLM:
# - url: Base URL of the LLM server.
#        Examples:
#          Ollama: http://localhost:11434/v1
#          vLLM:   http://localhost:<PORT>/v1
#
# - model_name: Name of the LLM to use.
#               Default: gpt-oss:120b-cloud
#
# Embedder:
# - url: Base URL of the embedding model server.
#        Default: http://localhost:11434/v1
#
# - model_name: Name of the embedding model to use.
#               Default: qwen3-embedding:0.6b
#
# If you use a different LLM/embedding server or model, update the
# corresponding "url" and "model_name" parameters below.
# -------------------------------------------------------------------------

models = {
    "llm": {
        "url": "http://localhost:11434/v1",
        "model_name": "gpt-oss:120b-cloud"
    },
    "embedder": {
        "url": "http://localhost:11434/v1",
        "model_name": "qwen3-embedding:0.6b"
    }
}

# Milvus vector database URI.
# Set this to the URI of your Milvus instance.
milvus_uri = ""