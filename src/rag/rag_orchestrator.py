"""Orchestratore principale del motore di ricerca RAG.

Questo modulo definisce la classe `RagOrchestrator`, che agisce da "cervello" centrale
del sistema. Ha la responsabilità di coordinare l'estrazione del testo dai file, 
la suddivisione in frammenti (chunking), la tokenizzazione a doppio binario 
e l'indicizzazione matematica tramite l'algoritmo BM25.
"""

from pathlib import Path
from tqdm import tqdm
import pickle
import heapq

from .inverted_index import InvertedIndex
from .chunk_table import ChunkTable
from .chunking import chunk_md, chunk_py
from .tokenizer import tokenize

from src import take_txt, corpus_path
from src.models import MinimalSource

 
INDEX_FILE = "rag_index.pkl"
SUPPORTED_SUFFIXES = {".py", ".md", ".txt"}


class RagOrchestrator:
    """Motore di coordinamento per l'ingestion e la ricerca di documenti.

    Questa classe mantiene lo stato globale dell'indice e della tabella dei metadati.
    Non esegue direttamente calcoli pesanti, ma orchestra i moduli specializzati
    affinché i dati fluiscano in modo coerente dal disco fino alle strutture di ricerca.

    Attributes:
        table (ChunkTable): Anagrafica a colonne parallele per il tracciamento
            dei metadati fisici (file, posizioni, lunghezze) di ogni chunk.
        index (InvertedIndex): Motore matematico BM25 che mappa i vocabili 
            ai rispettivi `chunk_id` calcolandone la rilevanza sparsi.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Inizializza le strutture dati vuote (Indice Invertito e Tabella)."""
        self.table = ChunkTable()
        self.index = InvertedIndex()
        self.k1 = k1
        self.b = b
    
    def build_index_from_directory(
        self,
        directory_path: str | Path,
        max_chunk_size: int = 2000
    ) -> None:
        """Costruisce l'indice analizzando l'intera base di conoscenza su disco.

        Logica dell'algoritmo:
        1. Esplora ricorsivamente la cartella fornita filtrando solo file `.py` e `.md`.
        2. Per ogni file, legge il testo e sceglie la strategia di chunking appropriata.
        3. Esegue lo slicing della stringa per ogni coordinata, tokenizza il testo 
           estratto e ne calcola la lunghezza in termini di token validi.
        4. Registra i metadati nella `ChunkTable` per ottenere un `chunk_id` globale.
        5. Passa i token e il `chunk_id` all'`InvertedIndex` per l'inserimento.
        6. Chiude l'operazione trasferendo le statistiche globali (IDF e lunghezze).

        Args:
            directory_path (Path): Il percorso radice della cartella contenente 
                il dataset grezzo da indicizzare.
        """
        root = Path(directory_path)
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}")
        if max_chunk_size <= 0:
            raise ValueError("Max chunk must be greater than zero")

        # Ricostruzione da zero: niente dati residui di una build precedente.
        self.table = ChunkTable()
        self.index = InvertedIndex()


        # Filtra e raccoglie solo i file di testo supportati dal sistema
        # sorted garantisce il determinismo: i chunk_id saranno sempre identici
        valid_files = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )

        
        # Barra di caricamento per il processo lungo di ingestion
        for path in tqdm(
            valid_files,
            desc="Chunking and Indexing",
            unit="file"
        ):
            try:
                content_file = take_txt(path)
            except (OSError, UnicodeError):
                continue
            
            # Salta silenziosamente i file non leggibili o vuoti
            if not content_file:
                continue
                
            # Selezione dinamica del parser regex in base all'estensione del file
            if path.suffix.lower() == ".py":
                chunks_coords = chunk_py(content_file, max_chunk_size)
            else:
                chunks_coords = chunk_md(content_file, max_chunk_size)
                
            # Registra il nuovo documento nel sistema e ottiene l'ID file univoco
            current_file_id = self.table.add_file(corpus_path(path))

            parts = path.relative_to(root).with_suffix("").parts

            if len(parts) > 1:
                parts = parts[1:]

            file_tokens = tokenize(" ".join(parts))

            # Itera sulle coordinate (start, end) generate dalle espressioni regolari
            for start, end in chunks_coords:
                
                # Slicing: estrazione materiale del testo esatto per questo frammento
                text_chunk = content_file[start:end]
                
                # Tokenizzazione: pulizia del testo, gestione camelCase/snake_case 
                # ed estrazione dei termini validi.
                tokens = tokenize(text_chunk)
                
                # Ottimizzazione: se il chunk è costituito solo da scarti (es. simboli),
                # la lista di token sarà vuota. Evitiamo di inserire "spazzatura" nell'indice.
                if not tokens:
                    continue
                
                # Aggiungiamo i token del nome file ai token del testo per alzare il Recall
                tokens.extend(file_tokens * 3)

                # Registrazione anagrafica: inserisce i metadati a colonne e ottiene
                # il chunk_id globale e univoco per l'intero corpus.
                # NOTA: La lunghezza fornita è quella dei token estratti, vitale per il BM25.
                global_chunk_id = self.table.add_chunk(
                    f_id=current_file_id, 
                    start=start, 
                    end=end, 
                    length=len(tokens)
                )
                
                # Costruzione dell'indice: associa i token validi al nuovo chunk_id
                self.index.add_chunk(global_chunk_id, tokens)
        
        if not self.table.length:
            raise ValueError("No indexable chunks found")
        
        # Consolidamento: trasferisce la colonna delle lunghezze all'indice 
        # per calcolare le formule matematiche globali (avg_len e IDF)
        self.index.finalize(self.table.length)

    def search(self, query: str, top_k: int = 5) -> list[MinimalSource]:
        """Restituisce le fonti più pertinenti alla query secondo BM25.

        Args:
            query: Testo della domanda o della ricerca.
            top_k: Numero massimo di fonti da restituire.

        Returns:
            Fonti ordinate per rilevanza decrescente.
        """
        if top_k <= 0 or not self.table.length:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        
        # calcolo dello score tramite bm-25
        scores = self.index.score(
            query_tokens, self.table.length, self.k1, self.b
        )
        # ordino la heapq con gli n chunk piu grandi
        ranked_chunk_ids = heapq.nlargest(
            top_k,                       # numero di chunk da prender
            scores,                      # itera sulle chiavi (chunk_id)
            key=scores.__getitem__,      # ordina per score decrescente
        )
        return [self.table.get_source(chunk_id) for chunk_id in ranked_chunk_ids]

    def save_index(self, output_dir: str | Path) -> None:
        """Salva indice e metadati in un file pickle (scrittura atomica).
 
        Args:
            output_dir: Directory in cui salvare l'indice.
 
        Raises:
            RuntimeError: Se il salvataggio fallisce.
        """
        try:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / INDEX_FILE
            tmp = target.with_suffix(".tmp")
            with tmp.open("wb") as fh:
                pickle.dump(
                    {"table": self.table, "index": self.index},
                    fh,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            tmp.replace(target)
        except (OSError, pickle.PickleError) as exc:
            raise RuntimeError(
                f"Unable to save the index in {output_dir}: {exc}"
            ) from exc
 
    def load_index(self, input_dir: str | Path) -> None:
        """Carica indice e metadati salvati con `save_index`.
 
        Args:
            input_dir: Directory da cui caricare l'indice.
 
        Raises:
            FileNotFoundError: Se l'indice non esiste (eseguire `index`).
            RuntimeError: Se il file e' corrotto o incompatibile.
        """
        input_dir = Path(input_dir)
        index_path = input_dir / INDEX_FILE
        if not index_path.exists():
            raise FileNotFoundError(
                f"No index in {index_path}: run 'index' first"
            )
        try:
            with index_path.open("rb") as fh:
                data = pickle.load(fh)
            self.table = data["table"]
            self.index = data["index"]
        except (OSError, pickle.PickleError, EOFError, KeyError,
                AttributeError, ImportError) as exc:
            raise RuntimeError(
                f"Corrupted or incompatible index in {input_dir} ({exc}): "
                "Run 'index' again"
            ) from exc