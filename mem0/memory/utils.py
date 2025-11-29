import re
from contextlib import contextmanager
from time import perf_counter

from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT


def get_fact_retrieval_messages(message):
    return FACT_RETRIEVAL_PROMPT, f"Input:\n{message}"


def parse_messages(messages):
    response = ""
    for msg in messages:
        if msg["role"] == "system":
            response += f"system: {msg['content']}\n"
        if msg["role"] == "user":
            response += f"user: {msg['content']}\n"
        if msg["role"] == "assistant":
            response += f"assistant: {msg['content']}\n"
    return response


def format_entities(entities):
    if not entities:
        return ""

    formatted_lines = []
    for entity in entities:
        simplified = f"{entity['source']} -- {entity['relatationship']} -- {entity['destination']}"
        formatted_lines.append(simplified)

    return "\n".join(formatted_lines)


def remove_code_blocks(content: str) -> str:
    """
    Removes enclosing code block markers ```[language] and ``` from a given string.

    Remarks:
    - The function uses a regex pattern to match code blocks that may start with ``` followed by an optional language tag (letters or numbers) and end with ```.
    - If a code block is detected, it returns only the inner content, stripping out the markers.
    - If no code block markers are found, the original content is returned as-is.
    """
    pattern = r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$"
    match = re.match(pattern, content.strip())
    return match.group(1).strip() if match else content.strip()


def get_image_description(image_obj, llm, vision_details):
    """
    Get the description of the image
    """

    if isinstance(image_obj, str):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "A user is providing an image. Provide a high level description of the image and do not include any additional text.",
                    },
                    {"type": "image_url", "image_url": {"url": image_obj, "detail": vision_details}},
                ],
            },
        ]
    else:
        messages = [image_obj]

    response = llm.generate_response(messages=messages)
    return response


def parse_vision_messages(messages, llm=None, vision_details="auto"):
    """
    Parse the vision messages from the messages
    """
    returned_messages = []
    for msg in messages:
        if msg["role"] == "system":
            returned_messages.append(msg)
            continue

        # Handle message content
        if isinstance(msg["content"], list):
            # Multiple image URLs in content
            description = get_image_description(msg, llm, vision_details)
            returned_messages.append({"role": msg["role"], "content": description})
        elif isinstance(msg["content"], dict) and msg["content"].get("type") == "image_url":
            # Single image content
            image_url = msg["content"]["image_url"]["url"]
            try:
                description = get_image_description(image_url, llm, vision_details)
                returned_messages.append({"role": msg["role"], "content": description})
            except Exception:
                raise Exception(f"Error while downloading {image_url}.")
        else:
            # Regular text content
            returned_messages.append(msg)

    return returned_messages


def _format_timing_details(details: dict) -> str:
    """Build a simple key=value string for timing logs."""
    parts = []
    for key, value in details.items():
        if value is None:
            continue
        try:
            parts.append(f"{key}={value}")
        except Exception:
            parts.append(f"{key}=<unserializable>")
    return ", ".join(parts)


def log_duration(logger, label: str, start_time: float, **details) -> float:
    """Log duration for an operation."""
    try:
        elapsed = perf_counter() - start_time
    except Exception:
        return 0.0
    detail_str = _format_timing_details(details)
    if detail_str:
        logger.info("[timing] %s=%.3fs | %s", label, elapsed, detail_str)
    else:
        logger.info("[timing] %s=%.3fs", label, elapsed)
    return elapsed


@contextmanager
def time_block(logger, label: str, collector: dict = None, log: bool = True, **details):
    """
    Context manager to log how long a block takes.
    
    Args:
        logger: Logger to use.
        label: Key for the timing entry.
        collector: Optional dict to collect timings instead of immediate logging.
        log: Whether to log immediately.
    """
    start_time = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - start_time
        if collector is not None:
            collector[label] = elapsed
        if log:
            detail_str = _format_timing_details(details)
            if detail_str:
                logger.info("[timing] %s=%.3fs | %s", label, elapsed, detail_str)
            else:
                logger.info("[timing] %s=%.3fs", label, elapsed)


def format_timing_summary(label: str, timings: dict, order=None) -> str:
    """
    Build a concise timing summary string.
    """
    parts = []
    ordered_keys = order or []
    seen = set()
    for key in ordered_keys:
        if key in timings:
            parts.append(f"{key}={timings[key]:.3f}s")
            seen.add(key)
    for key, value in timings.items():
        if key in seen:
            continue
        parts.append(f"{key}={value:.3f}s")
    return f"[timing_summary] {label} " + " | ".join(parts)
