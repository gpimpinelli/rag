import json
from pathlib import Path


def take_json(path: Path) -> dict[str, Any] | list[Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError):
        return {}

def take_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
