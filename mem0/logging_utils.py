import datetime
import logging
import os
import sys
from typing import Iterable, Tuple

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
COLORS = {
    "grey": "\033[90m",
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "magenta": "\033[95m",
}

LEVEL_COLORS = {
    "DEBUG": COLORS["blue"],
    "INFO": COLORS["cyan"],
    "WARNING": COLORS["yellow"],
    "ERROR": COLORS["red"],
    "CRITICAL": BOLD + COLORS["red"],
}

PHASE_COLORS = {
    "start": COLORS["green"],
    "end": COLORS["blue"],
    "error": COLORS["red"],
}


def _env_flag_true(value: str) -> bool:
    return value.lower() not in ("0", "false", "no", "off")


def should_use_color(env_var: str = "MEM0_COLOR_LOGS") -> bool:
    # Fall back to COLOR_LOGS for consistency with the server
    default_flag = os.getenv("COLOR_LOGS", "1")
    return _env_flag_true(os.getenv(env_var, default_flag)) and sys.stdout.isatty()


def colorize(text: str, color_code: str, enabled: bool) -> str:
    if not enabled or not color_code:
        return str(text)
    return f"{color_code}{text}{RESET}"


class Mem0ColorFormatter(logging.Formatter):
    def __init__(self, enable_color: bool):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        self.enable_color = enable_color

    def format(self, record: logging.LogRecord) -> str:
        original_level = record.levelname
        try:
            if self.enable_color:
                record.levelname = colorize(
                    original_level, LEVEL_COLORS.get(original_level, ""), True
                )
            return super().format(record)
        finally:
            record.levelname = original_level


def configure_mem0_logging(level: int = logging.INFO) -> bool:
    """
    Attach a colorized console handler to the mem0 logger (idempotent).
    """
    use_color = should_use_color()
    logger = logging.getLogger("mem0")
    logger.setLevel(level)
    logger.propagate = False  # prevent double-printing via root handlers

    stream_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            stream_handler = handler
            break

    if stream_handler is None:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

    stream_handler.setFormatter(Mem0ColorFormatter(enable_color=use_color))
    if stream_handler.level > level:
        stream_handler.setLevel(level)

    return use_color


def format_trace_line(
    event: str,
    phase: str,
    duration_ms: float,
    fields: Iterable[Tuple[str, object]],
    use_color: bool,
) -> str:
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    phase_display = phase.lower()
    header = [
        colorize(timestamp, COLORS["grey"], use_color),
        colorize(phase_display, PHASE_COLORS.get(phase_display, COLORS["grey"]), use_color),
        colorize(event, BOLD + COLORS["magenta"], use_color),
        colorize(f"{round(duration_ms)}ms", DIM, use_color),
    ]

    kv_parts = []
    for key, value in fields:
        if value is None:
            continue
        kv_parts.append(f"{colorize(f'{key}=', DIM, use_color)}{value}")

    return " | ".join(header) + (" | " + " ".join(kv_parts) if kv_parts else "")
