from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Label, Footer
from textual.binding import Binding
from textual.widgets import Input


class BaseScreen(Screen):
    """Base screen with common functionality for all screens."""

    BINDINGS = [
        Binding("ctrl+q", "quit", "Exit", key_display="ctrl+q"),
        Binding("ctrl+d", "toggle_dark", "Toggle Dark Theme", key_display="ctrl+d"),
    ]

    def action_toggle_dark(self) -> None:
        self.app.theme = (
            "textual-dark" if self.app.theme == "textual-light" else "textual-light"
        )

    def action_quit(self) -> None:
        self.app.exit()

    def compose(self) -> ComposeResult:
        yield Container(
            Label(self.get_title(), classes="form-title"),
            *self.get_form_elements(),
            classes="form-container",
        )
        yield Footer()

    def get_title(self) -> str:
        return "Base Screen"

    def get_form_elements(self) -> list[ComposeResult]:
        return []


class ConfigurationScreen(BaseScreen):
    """Base screen provider configuration with submit functionality."""

    BINDINGS = BaseScreen.BINDINGS + [
        Binding("shift+enter", "submit", "Submit"),
    ]

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Catches the Enter key press and delegates the work."""
        self.process_submission()

    def process_submission(self) -> None:
        """
        To be implemented by each specific provider screen.
        This method contains the unique logic for validating and creating
        the embedding configuration.
        """

        raise NotImplementedError(
            "Each configuration screen must implement 'process_submission'"
        )
