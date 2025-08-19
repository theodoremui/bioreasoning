from textual import on
from textual.widgets import Select

from .base import BaseScreen
from .embedding_providers import (
    OpenAIEmbeddingScreen,
    BedrockEmbeddingScreen,
    AzureEmbeddingScreen,
    GeminiEmbeddingScreen,
    CohereEmbeddingScreen,
    HuggingFaceEmbeddingScreen,
)


class ProviderSelectScreen(BaseScreen):
    """Screen for selecting embedding provider."""

    def get_title(self) -> str:
        return "Select an embedding provider"

    def get_form_elements(self) -> list:
        return [
            Select(
                options=[
                    ("OpenAI", "OpenAI"),
                    ("Cohere", "Cohere"),
                    ("Bedrock", "Bedrock"),
                    ("HuggingFace", "HuggingFace"),
                    ("Azure", "Azure"),
                    ("Gemini", "Gemini"),
                ],
                prompt="Please select an embedding provider",
                classes="form-control",
                id="provider_select",
            )
        ]

    @on(Select.Changed, "#provider_select")
    def handle_selection(self, event: Select.Changed) -> None:
        from ..embedding_app import EmbeddingSetupApp

        app = self.app
        if isinstance(app, EmbeddingSetupApp):
            app.config.provider = event.value
            self.handle_next()

    def handle_next(self) -> None:
        from ..embedding_app import EmbeddingSetupApp

        app = self.app
        if isinstance(app, EmbeddingSetupApp):
            provider_screens = {
                "OpenAI": OpenAIEmbeddingScreen,
                "Bedrock": BedrockEmbeddingScreen,
                "Azure": AzureEmbeddingScreen,
                "Gemini": GeminiEmbeddingScreen,
                "Cohere": CohereEmbeddingScreen,
                "HuggingFace": HuggingFaceEmbeddingScreen,
            }
            screen_class = provider_screens.get(app.config.provider)
            if screen_class:
                app.push_screen(screen_class())
