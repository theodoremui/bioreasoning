from textual.app import ComposeResult
from textual.widgets import Input

from llama_index.embeddings.azure_inference import AzureAIEmbeddingsModel
from llama_cloud import PipelineCreateEmbeddingConfig_AzureEmbedding

from ..base import ConfigurationScreen


class AzureEmbeddingScreen(ConfigurationScreen):
    """Configuration screen for Azure embeddings."""

    def get_title(self) -> str:
        return "Azure Embedding Configuration"

    def get_form_elements(self) -> list[ComposeResult]:
        return [
            Input(
                placeholder="API Key",
                password=True,
                id="api_key",
                classes="form-control",
            ),
            Input(placeholder="Endpoint URL", id="endpoint", classes="form-control"),
        ]

    def process_submission(self) -> None:
        api_key = self.query_one("#api_key", Input).value
        endpoint = self.query_one("#endpoint", Input).value

        if not all([api_key, endpoint]):
            self.notify("All fields are required.", severity="error")
            return

        try:
            embed_model = AzureAIEmbeddingsModel(credential=api_key, endpoint=endpoint)
            embedding_config = PipelineCreateEmbeddingConfig_AzureEmbedding(
                type="AZURE_EMBEDDING", component=embed_model
            )

            self.app.config = embedding_config
            self.app.handle_completion(self.app.config)
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")
