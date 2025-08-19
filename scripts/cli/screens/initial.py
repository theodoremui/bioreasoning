from textual import on
from textual.widgets import Select

from .base import BaseScreen


class InitialScreen(BaseScreen):
    """Initial screen for choosing between default or custom settings."""

    def get_title(self) -> str:
        return "How do you wish to proceed?"

    def get_form_elements(self) -> list:
        return [
            Select(
                options=[
                    ("With Default Settings", "default_settings"),
                    ("With Custom Settings", "custom_settings"),
                ],
                prompt="Please select one of the following",
                id="setup_type",
                classes="form-control",
            )
        ]

    @on(Select.Changed, "#setup_type")
    def handle_selection(self, event: Select.Changed) -> None:
        from ..embedding_app import EmbeddingSetupApp

        app = self.app
        if isinstance(app, EmbeddingSetupApp):
            app.config.setup_type = event.value
            self.handle_next()

    def handle_next(self) -> None:
        from ..embedding_app import EmbeddingSetupApp
        from .embedding_provider import ProviderSelectScreen

        app = self.app
        if isinstance(app, EmbeddingSetupApp):
            if app.config.setup_type == "default_settings":
                app.handle_default_setup()
            else:
                app.push_screen(ProviderSelectScreen())
