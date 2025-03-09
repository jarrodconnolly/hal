from textual.app import App, ComposeResult
from textual.widgets import Static, Header
from textual.containers import VerticalScroll
from textual.events import Key, Paste  # Add Paste
import httpx
import asyncio
import logging
from qdrant_client import QdrantClient
from rich.text import Text
import json

logging.basicConfig(filename="hal_ui.log", level=logging.INFO, format="%(message)s")

class HALApp(App):
    TITLE = "HAL: Highly Adaptable Learning AI"
    CSS = """
    Screen {
        layout: vertical;
        background: black;
    }
    Header {
        color: #E0FFFF;
        background: black;
        height: 3;
    }
    HeaderIcon {
        visibility: hidden;
    }
    VerticalScroll {
        height: 1fr;
    }
    #console {
        color: #E0FFFF;
        background: black;
        content-align: left top;
        padding: 0 1;
    }
    #status {
        color: #E0FFFF;
        background: black;
        padding-left: 1;
    }
    """

    def __init__(self):
        super().__init__()
        self.current_input = ""
        self.console_history = ""
        self.client = QdrantClient("localhost", port=6333)
        self.chunk_count = self.get_chunk_count()

    def get_chunk_count(self):
        try:
            collection_info = self.client.get_collection("hal_docs")
            return collection_info.points_count
        except Exception as e:
            logging.error(f"Failed to get chunk count: {e}")
            return 14658

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Static(Text("█"), id="console")
        yield Static(
            f"{self.chunk_count} Chunks | Hist 0.00s | Retr 0.00s | Gen 0.00s | Tot 0.00s | Tok 0 | TTFB 0.00s",
            id="status",
        )

    def on_mount(self):
        self.query_one("#status").styles.dock = "bottom"
        self.query_one("#console").focus()

    async def stream_query(self, query: str):
        console = self.query_one("#console")
        status = self.query_one("#status")
        self.console_history += f"{query}\n"
        console.update(Text(self.console_history + "█"))
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", "http://localhost:8001/query_stream", json={"query": query}
            ) as response:
                timings = {}
                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    if "\n\nTIMINGS:" in buffer:
                        parts = buffer.split("\n\nTIMINGS:", 1)
                        timings = json.loads(parts[1])
                        break
                    self.console_history += chunk
                    console.update(Text(self.console_history + "█"))
                    await asyncio.sleep(0.01)
                else:
                    self.console_history += buffer
                self.console_history += "\n\n"
                self.current_input = ""
                console.update(Text(self.console_history + "█"))
            token_count = len(self.console_history.split())
            status.update(
                f"{self.chunk_count} Chunks | Hist {timings.get('history', 0):.2f}s | "
                f"Retr {timings.get('qdrant', 0):.2f}s | Gen {timings.get('generation', 0):.2f}s | "
                f"Tot {timings.get('total', 0):.2f}s | Tok {token_count} | TTFB {timings.get('ttfb', 0):.2f}s"
            )

    def on_key(self, event: Key) -> None:
        console = self.query_one("#console")
        scroll = self.query_one(VerticalScroll)
        if event.key == "enter":
            if self.current_input:
                self.run_worker(self.stream_query(self.current_input))
            scroll.scroll_end()
        elif event.character and event.is_printable:
            self.current_input += event.character
            console.update(Text(f"{self.console_history}{self.current_input}█"))
            logging.info(f"Typed: {self.current_input}")
            scroll.scroll_end()
        elif event.key == "backspace":
            self.current_input = self.current_input[:-1]
            console.update(Text(f"{self.console_history}{self.current_input}█"))
            scroll.scroll_end()

    def on_paste(self, event: Paste) -> None:
        console = self.query_one("#console")
        scroll = self.query_one(VerticalScroll)
        pasted_text = event.text.strip()  # Get clipboard content, strip newlines
        if pasted_text:
            self.current_input += pasted_text
            console.update(Text(f"{self.console_history}{self.current_input}█"))
            logging.info(f"Pasted: {self.current_input}")
            scroll.scroll_end()

if __name__ == "__main__":
    HALApp().run()