from textual.app import ComposeResult
from textual.widgets import Input

from llama_index.embeddings.gemini import GeminiEmbedding
from llama_cloud import PipelineCreateEmbeddingConfig_GeminiEmbedding

from ..base import ConfigurationScreen


class GeminiEmbeddingScreen(ConfigurationScreen):
    """Configuration screen for Gemini embeddings."""

    # This is the only model currently available in the API
    DEFAULT_MODEL = "models/embedding-001"

    def get_title(self) -> str:
        return "Gemini Embedding Configuration"

    def get_form_elements(self) -> list[ComposeResult]:
        return [
            Input(
                value=self.DEFAULT_MODEL,
                id="model",
                classes="form-control",
                disabled=True,
            ),
            Input(
                placeholder="API Key",
                password=True,
                id="api_key",
                classes="form-control",
            ),
        ]

    def process_submission(self) -> None:
        api_key = self.query_one("#api_key", Input).value

        if not api_key:
            self.notify("API Key is required", severity="error")
            return

        try:
            embed_model = GeminiEmbedding(
                api_key=api_key, model_name=self.DEFAULT_MODEL
            )
            embedding_config = PipelineCreateEmbeddingConfig_GeminiEmbedding(
                type="GEMINI_EMBEDDING", component=embed_model
            )

            self.app.config = embedding_config
            self.app.handle_completion(self.app.config)
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")
