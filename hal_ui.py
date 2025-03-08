from textual.app import App, ComposeResult
from textual.widgets import Static, Header
from textual.containers import VerticalScroll
from textual.events import Key
import logging

logging.basicConfig(filename="hal_ui.log", level=logging.INFO, format="%(message)s")

class HALApp(App):
    TITLE = "HAL: Highly Adaptable Learning AI"
    CSS = """
    Screen {
        layout: vertical;
        background: black;
    }
    Header {
        color: cyan;
        background: black;
        height: 3;
    }
    HeaderIcon {
        visibility: hidden;
    }
    VerticalScroll {
        height: 1fr;
        border: tall green;
    }
    #console {
        color: cyan;
        background: black;
    }
    #status {
        color: cyan;
        background: black;
        padding-left: 2;
    }
    """

    def __init__(self):
        super().__init__()
        self.current_input = ""
        self.console_history = ""

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Static("", id="console")
        yield Static("14,658 Chunks | History 0.00s | Retrieval 0.00s | Generation 0.00s | Total 0.00s", id="status")

    def on_mount(self):
        self.query_one("#status").styles.dock = "bottom"
        self.query_one("#console").focus()

    def on_key(self, event: Key) -> None:
        console = self.query_one("#console")
        scroll = self.query_one(VerticalScroll)
        if event.key == "enter":
            if self.current_input:
                self.console_history += f"> {self.current_input}\nHAL: Understood - {self.current_input}\n"
                self.current_input = ""
                console.update(self.console_history)
            scroll.scroll_end()
        elif event.character and event.is_printable:
            self.current_input += event.character
            console.update(f"{self.console_history}> {self.current_input}")
            logging.info(f"Typed: {self.current_input}")
            scroll.scroll_end()
        elif event.key == "backspace":
            self.current_input = self.current_input[:-1]
            console.update(f"{self.console_history}> {self.current_input}")
            scroll.scroll_end()

if __name__ == "__main__":
    HALApp().run()