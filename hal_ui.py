from textual.app import App, ComposeResult
from textual.widgets import Static, Header
from textual.containers import VerticalScroll
from textual.events import Key
import httpx
import asyncio
import logging
from qdrant_client import QdrantClient
from rich.text import Text

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
            f"{self.chunk_count} Chunks | History 0.00s | Retrieval 0.00s | Generation 0.00s | Total 0.00s",
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
                answer = ""
                async for chunk in response.aiter_text():
                    answer += chunk
                    self.console_history += chunk
                    console.update(Text(self.console_history + "█"))
                    await asyncio.sleep(0.01)
                self.console_history += "\n\n"
                self.current_input = ""
                console.update(Text(self.console_history + "█"))
            timings_response = await client.post(
                "http://localhost:8001/query", json={"query": query}
            )
            timings = timings_response.json().get("timings", {})
            status.update(
                f"{self.chunk_count} Chunks | History {timings.get('history', 0):.2f}s | "
                f"Retrieval {timings.get('qdrant', 0):.2f}s | Generation {timings.get('generation', 0):.2f}s | "
                f"Total {timings.get('total', 0):.2f}s"
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

if __name__ == "__main__":
    HALApp().run()