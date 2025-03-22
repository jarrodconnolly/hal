import logging

import structlog
from structlog.processors import TimeStamper, add_log_level
from structlog.stdlib import LoggerFactory
from structlog.dev import ConsoleRenderer  # For dev readability

def configure_logging(level=logging.INFO, dev_mode=True):
    """
    Configure centralized structured logging for HAL to stdout, including FastAPI/Uvicorn.
    In dev mode, use plain text; in prod, use JSON.

    Args:
        level (int): Logging level (default: logging.INFO).
        dev_mode (bool): If True, use plain text logging for dev (default: False).
    """
    # Set up the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # Clear any existing handlers

    # Console handler for stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Structlog processor chain - dev vs prod
    processors = [
        TimeStamper(fmt="iso"),  # ISO 8601 timestamp
        add_log_level,  # Adds "level" field
    ]
    if dev_mode:
        processors.append(ConsoleRenderer())  # Plain text for dev
    else:
        processors.append(structlog.processors.JSONRenderer())  # JSON for prod

    structlog.configure(
        processors=processors,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Tie structlog to the console handler
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=ConsoleRenderer() if dev_mode else structlog.processors.JSONRenderer()
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Ensure Uvicorn/FastAPI loggers inherit this config
    for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
        logger = logging.getLogger(logger_name)
        logger.handlers = []  # Clear default handlers
        logger.propagate = True  # Propagate to root logger

    return structlog.get_logger()


if __name__ == "__main__":
    logger = configure_logging(dev_mode=True)  # Test dev mode
    logger.info("Logging configured", module="test", status="success")