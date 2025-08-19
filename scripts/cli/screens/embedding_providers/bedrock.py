from textual.app import ComposeResult
from textual.widgets import Input, Select

from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_cloud import PipelineCreateEmbeddingConfig_BedrockEmbedding

from ..base import ConfigurationScreen


class BedrockEmbeddingScreen(ConfigurationScreen):
    """Configuration screen for Bedrock embeddings."""

    def get_title(self) -> str:
        return "Bedrock Embedding Configuration"

    def get_form_elements(self) -> list[ComposeResult]:
        model_options = []
        try:
            supported_models = BedrockEmbedding.list_supported_models()
            model_options = [
                (f"{provider.title()}: {model_id.split('.')[-1]}", model_id)
                for provider, models in supported_models.items()
                for model_id in models
            ]
        except Exception as e:
            self.notify(
                f"Could not fetch Bedrock models: {e}", severity="error", timeout=10
            )

        return [
            Select(
                options=model_options,
                prompt="Select Bedrock Model",
                id="model",
                classes="form-control",
            ),
            Input(
                placeholder="Region (e.g., us-east-1)",
                id="region",
                classes="form-control",
            ),
            Input(
                placeholder="Access Key ID (Optional)",
                id="access_key_id",
                classes="form-control",
            ),
            Input(
                placeholder="Secret Access Key (Optional)",
                password=True,
                id="secret_access_key",
                classes="form-control",
            ),
        ]

    def process_submission(self) -> None:
        model = self.query_one("#model", Select).value
        region = self.query_one("#region", Input).value
        access_key_id = self.query_one("#access_key_id", Input).value
        secret_access_key = self.query_one("#secret_access_key", Input).value

        if not all([model, region]):
            self.notify("All fields are required.", severity="error")
            return

        try:
            embed_model = BedrockEmbedding(
                model_name=model,
                region_name=region,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
            )
            embedding_config = PipelineCreateEmbeddingConfig_BedrockEmbedding(
                type="BEDROCK_EMBEDDING", component=embed_model
            )

            self.app.config = embedding_config
            self.app.handle_completion(self.app.config)
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")
