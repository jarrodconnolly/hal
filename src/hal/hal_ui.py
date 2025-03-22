"""Textual UI frontend for HAL, streaming queries and responses from the FastAPI server."""

import asyncio
import json
import logging
import uuid

import httpx
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.events import Key, Paste  # Add Paste
from textual.widgets import Header, Static

from .config import (
    API_HOST,
    API_PORT,
)

logging.basicConfig(filename="hal_ui.log", level=logging.INFO, format="%(message)s")


class HALApp(App):
    """HAL's interactive Textual UI for querying the FastAPI server and displaying responses.

    Attributes:
        TITLE: The app's title displayed in the header.
        CSS_PATH: Path to the CSS file for styling.
        current_input: Buffer for user-typed or pasted text.
        console_history: Accumulated query and response text.
        session_id: Unique identifier for the user session from login.
        chunk_count: Number of document chunks available in the server.
    """

    TITLE = "HAL: Highly Adaptable Learning AI"
    CSS_PATH = "./hal_ui.css"

    def __init__(self):
        super().__init__()
        self.current_input = ""
        self.console_history = ""
        login_response = asyncio.run(self.login("default_user"))
        self.session_id = login_response["session_id"]
        self.chunk_count = login_response["chunk_count"]
        if login_response.get("error"):
            self.console_history = f"{login_response['error']}\n"
        self._needs_update = False
        self.token_count = 0

    async def login(self, username: str) -> dict:
        """Log in to the HAL FastAPI server to obtain a session ID and chunk count.

        Args:
            username: The username to authenticate with the server.

        Returns:
            A dict with 'session_id' (str) and 'chunk_count' (int) from the server response,
            or a fallback with a random UUID and 0 chunks on failure.
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{API_HOST}:{API_PORT}/login",
                json={"username": username},
            )
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"Login failed: {response.text}")
                return {
                    "session_id": str(uuid.uuid4()),
                    "chunk_count": 0,
                    "error": "Login failed - using temp session",
                }

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Static(Text("█"), id="console")
        yield Static(
            f"{self.chunk_count} Chunks | Hist 0.00s | Retr 0.00s | Gen 0.00s | Tot 0.00s | Tok 0 | TTFB 0.00s",
            id="status",
        )

    def on_mount(self):
        """Set the focus on the console and dock the status bar at the bottom."""
        self.query_one("#status").styles.dock = "bottom"
        self.query_one("#console").focus()

    def on_unmount(self) -> None:
        """Log session termination when the UI shuts down."""
        logging.info(
            f"Session ended for {self.session_id} - Total tokens: {self.token_count}"
        )

    async def stream_query(self, query: str):
        """Stream a query to the HAL server and update the UI with responses and timings.

        Args:
            query: The user's input string to send to the server.

        Updates the console with streaming responses and the status bar with performance metrics.
        """
        console = self.query_one("#console")
        status = self.query_one("#status")
        self.console_history += f"{query}\n"
        # Use block cursor (█) to mimic terminal input focus (classic style)
        console.update(Text(self.console_history + "█"))
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"http://{API_HOST}:{API_PORT}/query_stream",
                    json={"query": query, "session_id": self.session_id},
                ) as response:
                    timings = {}
                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        if "\n\nUPDATED_TIMINGS:" in buffer:
                            parts = buffer.split("\n\nUPDATED_TIMINGS:", 1)
                            timings = json.loads(parts[1])  # Final timings with total
                            buffer = parts[0]
                            break
                        self.console_history += chunk
                        console.update(Text(self.console_history + "█"))
                    else:
                        self.console_history += buffer
            except (httpx.RequestError, httpx.TimeoutException) as e:
                error_msg = "Error: Server connection failed"
                logging.error(f"Stream failed: {str(e)}")
                self.console_history += f"\n{error_msg}\n"
                console.update(Text(self.console_history + "█"))
                timings = {}  # Ensure timings is defined for status update
            self.console_history += "\n\n"
            self.current_input = ""
            console.update(Text(self.console_history + "█"))
        self.token_count += len(query.split()) + len(buffer.split()) + 2
        status.update(
            f"{self.chunk_count} Chunks | Hist {timings.get('history', 0):.2f}s | "
            f"Retr {timings.get('qdrant', 0):.2f}s | Gen {timings.get('generation', 0):.2f}s | "
            f"Tot {timings.get('total', 0):.2f}s | Tok {self.token_count} | TTFB {timings.get('ttfb', 0):.2f}s"
        )

    def on_key(self, event: Key) -> None:
        """Handle keypress events to build input, trigger queries, or edit text.

        Args:
            event: The Key event containing the pressed key's details.

        Updates the console with the current input or streams a query on Enter.
        """
        console = self.query_one("#console")
        scroll = self.query_one(VerticalScroll)
        if event.key == "enter":
            if self.current_input:
                self.run_worker(self.stream_query(self.current_input))
                logging.info(f"Typed: {self.current_input}")
            scroll.scroll_end()
        elif event.character and event.is_printable:
            char = "".join(
                c for c in event.character if ord(c) >= 32
            )  # Strip control chars
            if char and len(self.current_input) < 1024:
                self.current_input += event.character
                if not self._needs_update:
                    # Throttle updates to avoid stutter during rapid typing
                    self._needs_update = True
                    self.call_later(self._update_console, console, scroll)
        elif event.key == "backspace":
            self.current_input = self.current_input[:-1]
            if not self._needs_update:
                self._needs_update = True
                self.call_later(self._update_console, console, scroll)

    async def _update_console(self, console: Static, scroll: VerticalScroll) -> None:
        """Update the console with current input and reset the update flag."""
        console.update(Text(f"{self.console_history}{self.current_input}█"))
        scroll.scroll_end()
        self._needs_update = False

    def on_paste(self, event: Paste) -> None:
        """Handle paste events to append clipboard text to the input buffer.

        Args:
            event: The Paste event containing the clipboard text.

        Updates the console with the pasted input, stripping newlines for clean display.
        """
        console = self.query_one("#console")
        scroll = self.query_one(VerticalScroll)
        pasted_text = "".join(
            c for c in event.text.strip() if ord(c) >= 32
        )  # Strip control chars
        if pasted_text:
            self.current_input = (self.current_input + pasted_text)[:1024]
            console.update(Text(f"{self.console_history}{self.current_input}█"))
            scroll.scroll_end()


if __name__ == "__main__":
    HALApp().run()
