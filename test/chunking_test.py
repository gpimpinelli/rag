from src.rag.chunking import chunk_md, chunk_py 
from tqdm import tqdm
from src.utils import take_txt
from pathlib import Path


def process_dir(
    directory_path: Path,
    max_chunk_size: int = 2000,
) -> dict[Path, list[tuple[int, int]]]:
    """Scansiona ricorsivamente una directory e calcola i chunk per i file .py e .md.
       Con una barra di caricamento per avere un riscontro visivo della scansione. (tqdm)

    Parameters
    ----------
    directory_path : Path
        Il percorso della directory radice da esplorare.
    max_chunk_size : int, optional
        Dimensione massima in caratteri per ogni chunk (default è 2000).

    Returns
    -------
    dict of {Path: list of tuple of (int, int)}
        Dizionario che mappa il percorso (`Path`) di ciascun file valido alla
        relativa lista di tuple `(start, end)`.
    """
    result = {}
    total_chunks_md = 0
    total_chunks_py = 0

    valid_files = [
        p for p in directory_path.rglob("*") 
        if p.is_file() and p.suffix in {".py", ".md"}
    ]

    for path in tqdm(valid_files, desc="File processing", unit="file"):
        txt = take_txt(path)
        if not txt:
            continue

        if path.suffix == ".py":
            chunks = chunk_py(txt, max_chunk_size)
            total_chunks_py += len(chunks)
        
        else:
            chunks = chunk_md(txt, max_chunk_size)
            total_chunks_md += len(chunks)

        result[path] = chunks
    print(f"Processing completed: {len(result)} files. Chunks generated: {total_chunks_md} MD, {total_chunks_py} PY.")

    return result


if __name__ == "__main__":
    print("--- Test con limite 2000 ---")
    process_dir(Path("data/datasets/raw"), max_chunk_size=2000)
    
    print("\n--- Test con limite 100 ---")
    process_dir(Path("data/datasets/raw"), max_chunk_size=100)