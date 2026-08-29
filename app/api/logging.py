import logging
from logging.handlers import RotatingFileHandler


class ImportantEventFilter(logging.Filter):
    IMPORTANT_PATTERNS = (
        "WEBHOOK_VERIFIED",
        "Received call event",
        "Processing SDP Offer",
        "Received audio track",
        "Connection state",
        "terminated by peer",
        "Turn transcript",
        "Turn dropped",
        "Turn response",
        "Gemma response",
        "Local Gemma model",
        "RealtimeTTS complete",
        "Greeting timings",
        "Turn timings",
        "Turn stages",
        "Discarded",
        "input ended",
        "Stopping interrupted",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        message = record.getMessage()
        return any(pattern in message for pattern in self.IMPORTANT_PATTERNS)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    for noisy_logger in ("aioice", "httpx", "transformers", "bitsandbytes"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    configure_important_log()


def configure_important_log() -> None:
    import os

    path = "run_logs/important.log"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handler = RotatingFileHandler(
        path,
        maxBytes=1048576,
        backupCount=3,
    )
    handler.setLevel(logging.INFO)
    handler.addFilter(ImportantEventFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)
