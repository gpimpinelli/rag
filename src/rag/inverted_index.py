import math


class InvertedIndex:
    """
    Indice invertito ottimizzato per il calcolo della similarità tramite BM25.

    Gestisce l'indicizzazione dei token estratti dai chunk di testo e fornisce
    metodi per il calcolo efficiente del ranking dei documenti,
    ottimizzando i pesi IDF tramite una cache pre-calcolata.

    Attributes:
        postings (dict[str, dict[int, int]]):
            - Mappa ogni termine a un dizionario interno.
            - Il dizionario interno mappa il `chunk_id`
              alla sua frequenza locale (TF).
        n_chunks (int):
            Numero totale di documenti inseriti nell'indice.
        avg_len (float):
            Lunghezza media (in token) di tutti i chunk indicizzati.
        idf_cache (dict[str, float]):
            Memoria cache pre-calcolata dei valori IDF per termine.
    """

    def __init__(self) -> None:
        """Inizializza le strutture dati vuote dell'indice invertito.

        Logica dell'algoritmo:
        1. Prepara il dizionario principale a tre livelli per l'indicizzazione
           sparsa (postings).
        2. Azzera il contatore globale dei documenti elaborati (n_chunks).
        3. Azzera la lunghezza media, che verrà calcolata in seguito (avg_len)
        4. Prepara il dizionario vuoto che fungerà da cache logaritmica
           (idf_cache).
        """
        self.postings: dict[str, dict[int, int]] = {}
        self.n_chunks: int = 0
        self.avg_len: float = 0.0
        self.idf_cache: dict[str, float] = {}

    def add_chunk(self, chunk_id: int, tokens: list[str]) -> None:
        """
        Aggiunge i token di un chunk all'indice invertito e mappa la frequenza

        Logica dell'algoritmo:
        1. Itera singolarmente su ogni token estratto dal documento.
        2. Se il token è totalmente nuovo nel corpus,
           crea il livello secondario e fissa la frequenza a 1.
        3. Se il token esiste ma è nuovo per QUESTO chunk,
           inizializza la chiave del chunk a 1.
        4. Se il token è già stato visto in questo chunk,
           incrementa la sua frequenza locale di 1.
        5. Inine, incrementa di 1 il contatore globale dei documenti

        Variabili interne principali:
        - token (str): Singola parola ora in esame estratta dalla lista

        Args:
            chunk_id (int): ID posizionale del chunk elaborato.
            tokens (list[str]): Lista di parole tokenizzate estratte dal chunk
        """
        for token in tokens:
            if token not in self.postings:
                self.postings[token] = {chunk_id: 1}
            else:
                if chunk_id not in self.postings[token]:
                    self.postings[token][chunk_id] = 1
                else:
                    self.postings[token][chunk_id] += 1
        self.n_chunks += 1

    def finalize(self, lengths: list[int]) -> None:
        """
        Conclude l'indicizzazione calcolando le stat globali e la cache IDF.

        Logica dell'algoritmo:
        1. Calcola la lunghezza totale del corpus sommando
           la dimensione di tutti i chunk.
        2. Determina la lunghezza media globale (avg_len)
           per la normalizzazione futura.
        3. Itera su ogni termine salvato nell'indice invertito.
        4. Applica la formula matematica del BM25-IDF con smoothing (+0.5, +1)
        5. Salva il risultato nella memoria cache per rendere istantanee le
           ricerche future.

        Variabili interne principali:
        - tot (int):
            Somma totale dei token presenti in tutto il corpus.
        - term (str):
            La singola parola attualmente in elaborazione nel ciclo.
        - chunks_dict (dict):
            Il dizionario secondario (chunk_id -> tf) associato al termine.
        - df (int):
            Document Frequency; il num tot di chunk che contengono il termine.
        - idf_score (float):
            Risultato matematico dell'equazione IDF per il singolo termine.

        Args:
            lengths (list[int]): Lista dove l'indice è il chunk_id e il valore
                è il numero totale di token presenti in quel chunk.
        """
        if self.n_chunks == 0:
            self.avg_len = 0.0
            return

        tot = sum(lengths)
        self.avg_len = tot / self.n_chunks

        for term, chunks_dict in self.postings.items():
            df = len(chunks_dict)
            idf_score = math.log(1 + (self.n_chunks - df + 0.5) / (df + 0.5))
            self.idf_cache[term] = float(idf_score)

    def idf(self, term: str) -> float:
        """Recupera il peso globale (Inverse Document Frequency) di un termine.

        Logica dell'algoritmo:
        1. Tenta di estrarre il valore associato alla parola dalla cache.
        2. Se la parola non è presente (cioè non esiste in nessun documento),
           l'algoritmo restituisce 0.0 per neutralizzarne l'impatto matematico

        Args:
            term (str): La parola di cui si desidera ottenere il peso.

        Returns:
            float: Il punteggio IDF, oppure 0.0 se il termine è inesistente.
        """
        return self.idf_cache.get(term, 0.0)

    def score(
        self,
        query_tokens: list[str],
        lengths: list[int],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> dict[int, float]:
        """Calcola la classifica di rilevanza BM25 tramite Sparse Evaluation.

        Logica dell'algoritmo:
        1. Inizializza un contenitore vuoto per accumulare i punteggi.
        2. Itera sulle parole della query:
           - scarta quelle inesistenti nell'indice (Sparse Evaluation).
        3. Interroga la cache per ottenere il peso globale (IDF) della parola.
        4. Itera unicamente sui chunk che contengono la parola.
        5. Calcola la normalizzazione locale (TF):
           - penalizza i chunk troppo lunghi (tramite b)
           - satura l'impatto delle parole ripetute troppe volte (tramite k1).
        6. Moltiplica l'IDF globale per la TF locale normalizzata e
           somma il risultato al chunk.

        Variabili interne principali:
        - chunk_scores (dict):
            L'accumulatore finale che mappa `chunk_id` al punteggio cumulativo.
        - token (str): La singola parola della query attualmente in esame.
        - idf_w (float):
            Il peso globale (IDF) recuperato per il token corrente.
        - chunks (dict):
            Il sotto-dizionario estratto dai postings (chunk_id -> frequenza).
        - chunk_id (int):
            L'ID del documento attualmente valutato nel ciclo interno.
        - tf (int): Term Frequency bruta;
            quante volte il token appare in questo specifico chunk.
        - tf_norm (float):
            Frequenza locale riadattata dalla formula BM25 (k1, b e avg_len).

        Args:
            query_tokens list[str]: I termini della domanda dell'utente.
            lengths list[int]: Lista delle dimensioni originarie di ogni chunk
            k1 float: Regola la saturazione della Term Frequency.
            b float: Regola la severità della penalità per i testi lunghi.

        Returns:
            dict[int, float]: Dizionario contenente solo i chunk con punteggio
            maggiore di zero, mappati al loro punteggio di rilevanza totale.
        """
        chunk_scores: dict[int, float] = {}

        for token in query_tokens:
            if token not in self.postings:
                continue

            idf_w = self.idf(token)

            chunks = self.postings[token]
            for chunk_id, tf in chunks.items():
                tf_norm = (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * (lengths[chunk_id] / self.avg_len))
                )

                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = 0.0
                chunk_scores[chunk_id] += idf_w * tf_norm

        return chunk_scores
