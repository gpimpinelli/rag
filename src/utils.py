import json
from pathlib import Path


def take_json(path: str | Path) -> dict[str, Any] | list[Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return {}

def take_txt(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""

def corpus_path(path: Path) -> str:
    """Restituisce il percorso relativo alla root del progetto (cwd).
 
    E' il formato richiesto dal grader, es.
    ``data/raw/vllm-0.10.1/docs/features/lora.md``.
    """
    try:
        path = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        pass
    return path.as_posix()
