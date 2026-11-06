"""Tabella dei chunk: registro posizionale e anagrafico del corpus.

Questo modulo gestisce il tracciamento dei metadati di ogni frammento di testo.
Per massimizzare l'efficienza della memoria, utilizza un approccio a "colonne parallele":
invece di istanziare migliaia di oggetti pesanti, distribuisce i dati su liste native 
di primitive (interi), dove l'indice della lista corrisponde sempre al `chunk_id`.
"""

from src.models import MinimalSource


class ChunkTable:
    """Anagrafica dei chunk strutturata per colonne parallele indicizzate.

    Attributes:
        files (list[str]): Registro globale dei percorsi assoluti/relativi dei file. 
            L'indice in questa lista rappresenta il `file_id`.
        file_id (list[int]): Colonna che mappa il `chunk_id` (indice) al `file_id` (valore).
        start (list[int]): Colonna che mappa il `chunk_id` al suo carattere di inizio nel file.
        end (list[int]): Colonna che mappa il `chunk_id` al suo carattere di fine nel file.
        length (list[int]): Colonna che mappa il `chunk_id` al suo numero totale di token.
    """

    def __init__(self) -> None:
        """Inizializza le colonne vuote per lo stoccaggio dei metadati.
        
        Logica dell'algoritmo:
        1. Prepara una singola lista indipendente per archiviare i percorsi dei file univoci.
        2. Prepara quattro liste parallele (`file_id`, `start`, `end`, `length`) destinate 
           a crescere simultaneamente. L'indice `i` di queste liste sarà il `chunk_id`.
        """
        self.files: list[str] = []
        self.file_id: list[int] = []
        self.start: list[int] = []
        self.end: list[int] = []
        self.length: list[int] = []

    def add_file(self, path: str) -> int:
        """Registra un nuovo percorso file nel sistema e gli assegna un ID.

        Logica dell'algoritmo:
        1. Aggiunge la stringa del percorso alla lista globale dei file.
        2. Calcola l'ID di questo nuovo file deducendolo dalla sua posizione finale 
           nella lista (lunghezza totale - 1).

        Args:
            path (str): Il percorso testuale (assoluto o relativo) del documento sorgente.

        Returns:
            int: L'identificativo univoco del file appena registrato (`file_id`).
        """
        self.files.append(path)
        return len(self.files) - 1

    def add_chunk(self, f_id: int, start: int, end: int, length: int) -> int:
        """Archivia i metadati di un nuovo chunk e genera il suo ID univoco.

        Logica dell'algoritmo:
        1. Appende simultaneamente i quattro valori ricevuti alle rispettive liste parallele.
        2. Essendo un inserimento sincrono, tutte le liste si allungano di un elemento.
        3. Deduttivamente, la posizione (indice) dell'ultimo elemento inserito in `start` 
           diventa il `chunk_id` globale e univoco per questo frammento.

        Args:
            f_id (int): L'ID del file di origine (generato in precedenza da `add_file`).
            start (int): L'indice del primo carattere del chunk all'interno del file.
            end (int): L'indice dell'ultimo carattere del chunk all'interno del file.
            length (int): Il numero totale di token estratti validi in questo chunk.

        Returns:
            int: L'identificativo univoco posizionale del chunk appena inserito (`chunk_id`).
        """
        self.file_id.append(f_id)
        self.start.append(start)
        self.end.append(end)
        self.length.append(length)
        return len(self.start) - 1

    def get_source(self, c_id: int) -> MinimalSource:
        """Ricostruisce un oggetto strutturato interrogando le liste parallele.

        Logica dell'algoritmo:
        1. Sfrutta il `chunk_id` (ovvero `c_id`) come indice per leggere in O(1) 
           dalle liste parallele.
        2. Estrae il `file_id` associato al chunk e lo usa a sua volta come indice 
           nella lista `files` per recuperare il percorso testuale originale.
        3. Estrae le coordinate di inizio e fine.
        4. Impacchetta i tre dati raccolti nell'oggetto Pydantic `MinimalSource` finale.

        Args:
            c_id (int): L'identificativo univoco (`chunk_id`) del frammento richiesto.

        Returns:
            MinimalSource: Modello Pydantic convalidato contenente i confini del chunk 
            e il percorso del file, pronto per essere restituito all'utente finale.
        """
        return MinimalSource(
            file_path=self.files[self.file_id[c_id]],
            first_character_index=self.start[c_id],
            last_character_index=self.end[c_id],
        )

    def __len__(self) -> int:
        """Restituisce il conteggio totale dei frammenti (chunk) attualmente archiviati.

        Returns:
            int: Il numero di elementi presenti nelle liste parallele.
        """
        return len(self.start)