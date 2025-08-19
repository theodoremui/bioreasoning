import os
from textual.app import ComposeResult
from textual.widgets import Input

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_cloud import PipelineCreateEmbeddingConfig_OpenaiEmbedding

from ..base import ConfigurationScreen


class OpenAIEmbeddingScreen(ConfigurationScreen):
    """Configuration screen for OpenAI embeddings."""

    def get_title(self) -> str:
        return "OpenAI Embedding Configuration"

    def get_form_elements(self) -> list[ComposeResult]:
        return [
            Input(
                placeholder="API Key (optional if OPENAI_API_KEY set)",
                id="api_key",
                password=True,
                classes="form-control",
            ),
            Input(
                placeholder="Model",
                id="model",
                classes="form-control",
            ),
        ]

    def process_submission(self) -> None:
        """Handle form submission by creating OpenAI embedding configuration."""
        api_key = self.query_one("#api_key", Input).value or os.getenv("OPENAI_API_KEY")
        model = self.query_one("#model", Input).value

        if not api_key:
            self.notify(
                "No API key provided and OPENAI_API_KEY not set", severity="error"
            )
            return
        if not model:
            self.notify("Model name is required", severity="error")
            return

        try:
            embed_model = OpenAIEmbedding(model=model, api_key=api_key)
            embedding_config = PipelineCreateEmbeddingConfig_OpenaiEmbedding(
                type="OPENAI_EMBEDDING",
                component=embed_model,
            )
            self.app.config = embedding_config
            self.app.handle_completion(self.app.config)
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")
