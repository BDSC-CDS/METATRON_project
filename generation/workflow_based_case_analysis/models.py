# -------------------------------------------------------------------------
# LLM configuration
#
# This project requires an OpenAI-compatible LLM API.
#
# - url: Base URL of the LLM server.
#        Examples:
#          Ollama: http://localhost:11434/v1
#          vLLM:   http://localhost:<PORT>/v1
#
# - model_name: Name of the model to use.
#               The model must be available on the selected server.
#
# Update these two parameters according to your local LLM setup.
# -------------------------------------------------------------------------

# By default, the workflow connects to a local Ollama instance
# through its OpenAI-compatible API (port 11434) and uses the
# gpt-oss:120b-cloud model.

models = {
    "llm":{
        "url":"http://localhost:11434/v1",
        "model_name":"gpt-oss:120b-cloud"
    }
}