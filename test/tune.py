"""Griglia di esperimenti: max_chunk_size x k1 x b (non salva nulla su disco).

Per ogni max_chunk_size ricostruisce l'indice in memoria; k1 e b agiscono
solo in fase di ricerca, quindi non richiedono un nuovo indice. Valuta
Recall@k sui dataset AnsweredQuestions con la stessa regola della
moulinette (stesso file, IoU >= 0.05).

Uso, dalla root del progetto:
    uv run python tune.py
Modifica le liste qui sotto per cambiare la griglia.
"""

import json
import statistics as st
from pathlib import Path

from ..src.rag.orchestrator import RagOrchestrator

DATASETS = {
    "docs": "data/datasets/AnsweredQuestions/dataset_docs_public.json",
    "code": "data/datasets/AnsweredQuestions/dataset_code_public.json",
}
SIZES = [2000, 1500, 1000]
K1S = [0.9, 1.2, 1.5, 2.0]
BS = [0.5, 0.75, 0.9]
KS = (1, 3, 5, 10)
BASELINE = (2000, 1.5, 0.75)
THRESHOLD = 0.05


def iou(a, b):
    """IoU tra (file, start, end) e (file, start, end); 0 se file diversi."""
    if a[0] != b[0]:
        return 0.0
    inter = min(a[2], b[2]) - max(a[1], b[1])
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[1]) + (b[2] - b[1]) - inter)


def load(path):
    """Legge un dataset e restituisce [(domanda, [(file, start, end)])]."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for q in data["rag_questions"]:
        refs = [(s["file_path"], s["first_character_index"],
                 s["last_character_index"]) for s in q.get("sources", [])]
        if refs:
            out.append((q["question"], refs))
    return out


def recall(rag, questions):
    """Recall@k medio (quota di fonti di riferimento trovate nei primi k)."""
    totals = {k: 0.0 for k in KS}
    for text, refs in questions:
        got = [(s.file_path, s.first_character_index, s.last_character_index)
               for s in rag.search(text, top_k=max(KS))]
        for k in KS:
            found = sum(any(iou(r, g) >= THRESHOLD for g in got[:k])
                        for r in refs)
            totals[k] += found / len(refs)
    return {k: v / len(questions) for k, v in totals.items()}


def main():
    """Esegue la griglia e stampa una tabella ordinata per R@5 medio."""
    sets = {name: load(path) for name, path in DATASETS.items()}
    rows = []
    for size in SIZES:
        rag = RagOrchestrator()
        rag.build_index_from_directory(Path("data/raw"), size)
        for k1 in K1S:
            for b in BS:
                rag.k1, rag.b = k1, b
                res = {n: recall(rag, q) for n, q in sets.items()}
                rows.append((size, k1, b, res))
    rows.sort(key=lambda r: -st.mean(r[3][n][5] for n in sets))
    n_docs = len(sets["docs"])
    print(f"\n{'size':>5} {'k1':>4} {'b':>5} | docs R@1  R@5 R@10"
          f" | code R@1  R@5 R@10")
    for size, k1, b, res in rows:
        mark = "  <- baseline" if (size, k1, b) == BASELINE else ""
        d, c = res["docs"], res["code"]
        print(f"{size:>5} {k1:>4} {b:>5} |      {d[1]:.2f} {d[5]:.2f} "
              f"{d[10]:.2f} |      {c[1]:.2f} {c[5]:.2f} {c[10]:.2f}{mark}")
    print(f"\nCon {n_docs} domande docs, differenze di R@5 sotto ~0.05 sono "
          "rumore statistico: preferisci le configurazioni buone su ENTRAMBI "
          "i dataset e stabili nei dintorni.")


if __name__ == "__main__":
    main()