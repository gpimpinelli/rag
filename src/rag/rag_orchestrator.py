"""Orchestratore principale del motore di ricerca RAG.

Questo modulo definisce la classe `RagOrchestrator`, che agisce da "cervello" centrale
del sistema. Ha la responsabilità di coordinare l'estrazione del testo dai file, 
la suddivisione in frammenti (chunking), la tokenizzazione a doppio binario 
e l'indicizzazione matematica tramite l'algoritmo BM25.
"""

from pathlib import Path
from tqdm import tqdm

from .inverted_index import InvertedIndex
from .chunk_table import ChunkTable
from .chunking import chunk_md, chunk_py
from .tokenizer import tokenize

# Si assume che take_txt sia importato correttamente dalle utility di base
from ..utils import take_txt 


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

    def __init__(self) -> None:
        """Inizializza le strutture dati vuote (Indice Invertito e Tabella)."""
        self.table = ChunkTable()
        self.index = InvertedIndex()

    def build_index_from_directory(self, directory_path: Path) -> None:
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
        # Filtra e raccoglie solo i file di testo supportati dal sistema
        # sorted garantisce il determinismo: i chunk_id saranno sempre identici
        valid_files = sorted([
            p for p in directory_path.rglob("*") 
            if p.is_file() and p.suffix.lower() in {".py", ".md", "txt"}
        ])
        
        # Barra di caricamento per il processo lungo di ingestion
        for path in tqdm(valid_files, desc="Chunking and Indexing", unit="file"):
            content_file = take_txt(path)
            
            # Salta silenziosamente i file non leggibili o vuoti
            if not content_file:
                continue
                
            # Selezione dinamica del parser regex in base all'estensione del file
            if path.suffix == ".py":
                chunks_coords = chunk_py(content_file)
            else:
                chunks_coords = chunk_md(content_file)

            # Tenta di convertire il path da relativo ad assoluto
            try:
                rel_path = path.relative_to(directory_path.parent)
            except ValueError:
                rel_path = path
                
            # Registra il nuovo documento nel sistema e ottiene l'ID file univoco
            current_file_id = self.table.add_file(rel_path.as_posix())

            file_tokens = tokenize(path.name)

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
                tokens.extend(file_tokens)

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
                
        # Consolidamento: trasferisce la colonna delle lunghezze all'indice 
        # per calcolare le formule matematiche globali (avg_len e IDF)
        self.index.finalize(self.table.length)


    def search(self, query: str, top_k: int = 5) -> list[MinimalSource]:
        """Esegue la ricerca usando l'indice in memoria.
        
        Args:
            query (str): La stringa di ricerca dell'utente.
            top_k (int): Il numero massimo di risultati da restituire.
            
        Returns:
            list[MinimalSource]: I migliori chunk trovati.
        """
        pass

    def save_index(self, output_dir: Path) -> None:
        """Salva indice e tabella su disco (Punto 5)."""
        pass