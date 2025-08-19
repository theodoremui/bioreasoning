from textual.app import ComposeResult
from textual.widgets import Input

from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_cloud import PipelineCreateEmbeddingConfig_HuggingfaceApiEmbedding

from ..base import ConfigurationScreen


class HuggingFaceEmbeddingScreen(ConfigurationScreen):
    """Configuration screen for HuggingFace embeddings."""

    def get_title(self) -> str:
        return "HuggingFace Embedding Configuration"

    def get_form_elements(self) -> list[ComposeResult]:
        return [
            Input(
                placeholder="HuggingFace API Token",
                password=True,
                id="api_key",
                classes="form-control",
            ),
            Input(
                placeholder="Model name",
                id="model",
                classes="form-control",
            ),
        ]

    def process_submission(self) -> None:
        """Handle form submission by creating HuggingFace embedding configuration."""
        api_key = self.query_one("#api_key", Input).value
        model = self.query_one("#model", Input).value

        if not api_key:
            self.notify("HuggingFace API Token is required", severity="error")
            return

        if not model:
            self.notify("Model name is required", severity="error")
            return

        try:
            embed_model = HuggingFaceInferenceAPIEmbedding(
                token=api_key, model_name=model
            )
            embedding_config = PipelineCreateEmbeddingConfig_HuggingfaceApiEmbedding(
                type="HUGGINGFACE_API_EMBEDDING",
                component=embed_model,
            )

            self.app.config = embedding_config
            self.app.handle_completion(self.app.config)
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")
