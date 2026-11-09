# JSON logging, configured once by the application factory.

import json
import logging
import logging.config
from datetime import UTC, datetime
from typing import override

from app.core.request_context import request_id_var

MAX_MESSAGE_CHARS = 4000


class JsonFormatter(logging.Formatter):
    # One JSON object per line; json.dumps escapes CR/LF, so logged values cannot forge entries.

    @override
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()[:MAX_MESSAGE_CHARS],
            "request_id": request_id_var.get(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str) -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"json": {"()": JsonFormatter}},
            "handlers": {
                "stdout": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"level": level, "handlers": ["stdout"]},
        }
    )
