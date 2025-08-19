import os
from textual.app import App

from .config import EmbeddingConfig
from .screens import InitialScreen


class EmbeddingSetupApp(App):
    """Main application for embedding configuration setup."""

    CSS_PATH = "stylesheets/base.tcss"

    def __init__(self):
        super().__init__()
        self.config = EmbeddingConfig(provider="")

    def on_mount(self) -> None:
        self.push_screen(InitialScreen())

    def handle_completion(self, config: EmbeddingConfig) -> None:
        self.exit(config)

    def handle_default_setup(self) -> None:
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_cloud import PipelineCreateEmbeddingConfig_OpenaiEmbedding

        self.config.provider = "OpenAI"
        self.config.api_key = os.getenv("OPENAI_API_KEY")
        self.config.model = "text-embedding-3-small"

        embed_model = OpenAIEmbedding(
            model=self.config.model, api_key=self.config.api_key
        )
        embedding_config = PipelineCreateEmbeddingConfig_OpenaiEmbedding(
            type="OPENAI_EMBEDDING",
            component=embed_model,
        )
        self.config = embedding_config

        self.handle_completion(self.config)
