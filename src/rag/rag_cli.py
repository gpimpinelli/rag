"""CLI Fire per indicizzazione, ricerca e valutazione RAG."""

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from tqdm import tqdm

from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    AnsweredQuestion,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)

from src import take_json
from .rag_orchestrator import RagOrchestrator

if TYPE_CHECKING:
    from .answer_generator import AnswerGenerator

IOU_THRESHOLD = 0.05

from .answer_generator import NO_CONTEXT_REPLY


def _safe(func: Callable[..., Any]) -> Callable[..., Any]:
    """Trasforma le eccezioni attese in un messaggio, senza traceback."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (OSError, ValueError, RuntimeError, KeyError,
                TypeError) as exc:
            print(f"Error: {exc}")
            return None
    return wrapper

def _iou(a: MinimalSource, b: MinimalSource) -> float:
    """IoU tra due fonti (0.0 se file diversi)."""
    if a.file_path != b.file_path:
        return 0.0
    inter = (min(a.last_character_index, b.last_character_index)
             - max(a.first_character_index, b.first_character_index))
    if inter <= 0:
        return 0.0
    union = ((a.last_character_index - a.first_character_index)
             + (b.last_character_index - b.first_character_index) - inter)
    return inter / union if union > 0 else 0.0


class RagCLI:
    """Comandi esposti da Fire: index, search, search_dataset, evaluate."""

    def __init__(
        self,
        index_dir: str = "data/processed",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        """Parametri globali (prima del comando), es. ``--bm25_k1 1.2``.

        I nomi iniziano con ``bm25_`` di proposito: un parametro chiamato
        ``k1`` intercetterebbe il flag ``--k`` dei comandi (Fire accetta i
        prefissi), lasciando ``k`` al valore di default.

        Il costruttore e' volutamente leggero: Fire lo esegue a ogni
        comando, quindi l'indice viene caricato solo quando serve.

        Args:
            index_dir: Cartella dell'indice (default data/processed).
            bm25_k1: Saturazione della term frequency BM25.
            bm25_b: Penalita' di lunghezza BM25.
        """
        self.index_dir = Path(index_dir)
        self._rag: RagOrchestrator = RagOrchestrator(k1=bm25_k1, b=bm25_b)
        self._generator: "AnswerGenerator | None" = None

    def _load(self) -> RagOrchestrator:
        """Carica l'indice da disco solo la prima volta che serve."""
        if not self._rag.table.length:
            self._rag.load_index(self.index_dir)
        return self._rag

    def _load_generator(self) -> "AnswerGenerator":
        """Carica il modello LLM solo quando vengono chiamati answer o answer_dataset."""
        if self._generator is None:
            print("Loading language model Qwen/Qwen3-0.6B...")
            # Import pigro: torch/transformers non rallentano index e search.
            from .answer_generator import AnswerGenerator
            self._generator = AnswerGenerator()
        return self._generator

    @_safe
    def index(
        self,
        max_chunk_size: int = 2000,
        raw_directory: str = "data/raw",
    ) -> None:
        """Indicizza i file da raw_directory e salva l'indice in self.index_dir."""
        raw_path = Path(raw_directory)
        self._rag.build_index_from_directory(raw_path, max_chunk_size)
        self._rag.save_index(self.index_dir)
        print(f"Ingestion complete! Indexed {len(self._rag.table)} chunks "
              f"under {self.index_dir}/")

    @_safe
    def search(self, query: str, k: int = 10) -> None:
        """Stampa le top-k fonti per una singola query."""
        if not str(query).strip():
            print("Error: query is empty")
            return
        if k <= 0:
            print("Error: k must be greater than zero")
            return
        results = self._load().search(str(query), top_k=k)
        if not results:
            print("Nessun risultato.")
        for src in results:
            print(f"{src.file_path} [{src.first_character_index}:"
                  f"{src.last_character_index}]")

    @_safe
    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results/"
                              "UnansweredQuestions",
    ) -> None:
        """Cerca per tutte le domande e salva uno StudentSearchResults."""
        if k <= 0:
            print("Error: k deve essere maggiore di zero")
            return
        dataset = RagDataset.model_validate(take_json(dataset_path))
        rag = self._load()
        out = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=rag.search(q.question, top_k=k),
                )
                for q in tqdm(dataset.rag_questions, desc="Searching",
                              unit="question")
            ],
            k=k,
        )
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        target = save_dir / Path(dataset_path).name
        target.write_text(out.model_dump_json(indent=2), encoding="utf-8")
        print(f"Saved student_search_results to {target}")

    @_safe
    def answer(self, query: str, k: int = 5) -> None:
        """Risponde a una singola query stampando fonti e testo generato."""
        if not str(query).strip():
            print("Error: la query e' vuota")
            return
        if k <= 0:
            print("Error: k deve essere maggiore di zero")
            return

        # 1. Recupero delle fonti tramite l'indice
        rag = self._load()
        sources = rag.search(str(query), top_k=k)
        if not sources:
            print("Nessuna fonte trovata per la query.")
            return

        # 3. Generazione della risposta con il modello LLM
        generator = self._load_generator()
        response = generator.generate_answer(question=str(query), sources=sources)

        # Stampa dei risultati a terminale
        print("\n--- Sources Retrieved ---")
        for src in sources:
            print(f"{src.file_path} [{src.first_character_index}:{src.last_character_index}]")
            
        print("\n--- Generated Answer ---")
        print(response)

    @_safe
    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer/UnansweredQuestions",
    ) -> None:
        """Genera le risposte per un dataset partendo dai risultati di ricerca già salvati."""
        # 1. Carica e valida il file di ricerca esistente
        raw_data = take_json(student_search_results_path)
        search_res = StudentSearchResults.model_validate(raw_data)

        generator = self._load_generator()
        answers: list[MinimalAnswer] = []

        # 2. Itera sulle domande con la barra di avanzamento tqdm
        for item in tqdm(search_res.search_results, desc="Generating answers", unit="question"):
            try:
                # Genera il testo della risposta
                ans_text = generator.generate_answer(
                    question=item.question,
                    sources=item.retrieved_sources,
                )
            except (RuntimeError, ValueError, OSError) as exc:
                print(
                    f"Warning: generation failed for {item.question_id}: {exc}"
                )
                ans_text = NO_CONTEXT_REPLY

            # Crea il record MinimalAnswer
            answers.append(
                MinimalAnswer(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=item.retrieved_sources,
                    answer=ans_text,
                )
            )

        # 3. Costruisce il contenitore finale StudentSearchResultsAndAnswer
        output_payload = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=search_res.k,
        )

        # 4. Scrittura del file JSON nella cartella di destinazione
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        target_path = save_dir / Path(student_search_results_path).name
        target_path.write_text(output_payload.model_dump_json(indent=2), encoding="utf-8")
        
        print(f"Saved student_search_results_and_answer to {target_path}")

    @_safe
    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Recall@1/3/5/10 contro un dataset con risposte (IoU >= 0.05)."""
        dataset = RagDataset.model_validate(take_json(dataset_path))
        res = StudentSearchResults.model_validate(
            take_json(student_search_results_path)
        )
        got = {r.question_id: r.retrieved_sources
               for r in res.search_results}
        answered = [q for q in dataset.rag_questions
                    if isinstance(q, AnsweredQuestion)]
        if not answered:
            print("Error: il dataset non contiene fonti di riferimento")
            return
        print("Evaluation Results\n" + "=" * 40)
        for kk in (1, 3, 5, 10):
            if kk > res.k:
                continue
            total = 0.0
            for q in answered:
                top = got.get(q.question_id, [])[:kk]
                found = sum(
                    any(_iou(r, g) >= IOU_THRESHOLD for g in top)
                    for r in q.sources
                )
                total += found / len(q.sources) if q.sources else 0.0
            print(f"Recall@{kk}: {total / len(answered):.3f}")