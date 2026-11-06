"""Prova rapida di tokenize() su esempi e sul corpus reale."""

import time
from collections import Counter
from pathlib import Path

from src import take_txt
from src.rag.chunking import chunk_md, chunk_py
from src.rag.tokenizer import tokenize

EXAMPLES = [
    "getUserName",
    "get_user_name",
    "HTTPServer",
    "__init__",
    "utf8",
    "How does the async server handle LoRA adapters?",
]


def check_examples() -> None:
    """Stampa i token prodotti per ogni esempio."""
    for text in EXAMPLES:
        print(f"{text!r}\n  -> {tokenize(text)}")


def check_corpus(root: Path, max_chunk_size: int = 2000) -> None:
    """Tokenizza tutto il corpus e stampa statistiche."""
    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix in {".py", ".md"}
    ]
    df: Counter[str] = Counter()
    lengths: list[int] = []

    t0 = time.perf_counter()
    for path in files:
        txt = take_txt(path)
        if not txt:
            continue
        chunker = chunk_py if path.suffix == ".py" else chunk_md
        for s, e in chunker(txt, max_chunk_size):
            tokens = tokenize(txt[s:e])
            lengths.append(len(tokens))
            df.update(set(tokens))
    elapsed = time.perf_counter() - t0

    n = len(lengths)
    print(f"\nFile: {len(files)} | chunk: {n} | tempo: {elapsed:.1f}s")
    print(f"Token distinti: {len(df)} | lunghezza media: {sum(lengths) / n:.0f}")
    print("\nTop 30 token per document frequency:")
    for tok, count in df.most_common(30):
        print(f"  {tok:<20} {count:>7}  ({100 * count / n:.1f}% dei chunk)")

if __name__ == "__main__":
    check_examples()
    check_corpus(Path("data/datasets/raw"))
