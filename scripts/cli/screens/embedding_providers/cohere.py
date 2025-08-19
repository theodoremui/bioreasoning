from textual.app import ComposeResult
from textual.widgets import Input

from llama_index.embeddings.cohere import CohereEmbedding
from llama_cloud import PipelineCreateEmbeddingConfig_CohereEmbedding

from ..base import ConfigurationScreen


class CohereEmbeddingScreen(ConfigurationScreen):
    """Configuration screen for Cohere embeddings."""

    def get_title(self) -> str:
        return "Cohere Embedding Configuration"

    def get_form_elements(self) -> list[ComposeResult]:
        return [
            Input(
                placeholder="API Key",
                password=True,
                id="api_key",
                classes="form-control",
            ),
            Input(placeholder="Model", id="model", classes="form-control"),
        ]

    def process_submission(self) -> None:
        api_key = self.query_one("#api_key", Input).value
        model = self.query_one("#model", Input).value

        if not all([api_key, model]):
            self.notify("All fields are required", severity="error")
            return

        try:
            embed_model = CohereEmbedding(
                model_name=model,
                api_key=api_key,
                input_type="search_document",
                embedding_type="float",
            )
            embedding_config = PipelineCreateEmbeddingConfig_CohereEmbedding(
                type="COHERE_EMBEDDING", component=embed_model
            )

            self.app.config = embedding_config
            self.app.handle_completion(self.app.config)
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")
