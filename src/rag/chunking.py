"""Modulo per la suddivisione (chunking) di file Markdown e Python tramite regex.

Questo modulo implementa un algoritmo a finestra scorrevole basato su livelli
di priorità per spezzare testi lunghi preservando gli indici assoluti
(start, end) del documento originale, senza mai alterare o copiare le stringhe.
"""

from pathlib import Path
import re
from src import take_txt

# ==============================================================================
# LEGENDA DEI SIMBOLI REGEX UTILIZZATI
# ==============================================================================
# (?m)       : Flag MULTILINE. Fa sì che '^' e '$' facciano match all'inizio e
#              alla fine di ogni singola riga, non solo dell'intera stringa.
# ^          : Inizio della riga (grazie al flag ?m).
# $          : Fine della riga.
# (?=...)    : Positive Lookahead. Verifica che il pattern seguente esista
#              senza "consumare" caratteri. Permette a m.start() di restituire
#              l'indice esatto di inizio riga senza mangiarsi spazi o parole.
# (?:...)    : Gruppo di non-cattura. Raggruppa espressioni (es. per usare '|')
#              senza salvare il sotto-match in memoria (più veloce e pulito).
#  {0,3}     : Da 0 a 3 spazi vuoti (standard Markdown per l'indentazione titoli).
# #{1,6}     : Da 1 a 6 caratteri cancelletto '#' consecutivi (titoli H1-H6).
# [ \t]      : Classe di caratteri che intercetta uno spazio o un tab.
# +          : Quantificatore: 1 o più occorrenze dell'elemento precedente.
# *          : Quantificatore: 0 o più occorrenze dell'elemento precedente.
# |          : Operatore logico OR (oppure) all'interno di un gruppo.
# \b         : Word boundary (confine di parola). Evita che 'def' faccia match
#              dentro parole come 'default' o 'define'. Notevole: '@' non è un
#              carattere parola, quindi in PY_PATTERNS è tenuto fuori da \b.
# \n         : Carattere di ritorno a capo (newline).
# ==============================================================================

# Pattern per Markdown ordinati per priorità decrescente:
# 1. Titoli Markdown (da # a ###### preceduti da max 3 spazi e seguiti da spazio o fine riga)
# 2. Doppio ritorno a capo (separatore di paragrafo, tollera spazi/tab sulla riga vuota)
# 3. Singolo ritorno a capo (fine riga standard)
MD_PATTERNS = [
    r"(?m)^(?= {0,3}#{1,6}(?:[ \t]+|$))",
    r"\n[ \t]*\n",
    r"\n",
]

# Pattern per Python ordinati per priorità decrescente:
# 1. Dichiarazioni di classi, funzioni (anche async) o decoratori '@', preservando l'indentazione
# 2. Doppio ritorno a capo (separatore logico tra blocchi di codice)
# 3. Singolo ritorno a capo (singola istruzione/riga)
PY_PATTERNS = [
    r"(?m)^(?=[ \t]*(?:(?:class|def|async[ \t]+def)\b|@))",
    r"\n[ \t]*\n",
    r"\n",
]


def chunk_text(
    file_content: str,
    patterns: list[str],
    max_chunk_size: int = 2000,
) -> list[tuple[int, int]]:
    """Suddivide un testo in blocchi (chunk) rispettando una gerarchia di pattern regex.

    Utilizza un approccio a finestra scorrevole per individuare i punti di taglio
    ottimali (es. inizio di sezioni, paragrafi o righe) senza mai superare la
    dimensione massima consentita e garantendo la continuità degli indici.

    Parameters
    ----------
    file_content : str
        Il contenuto testuale completo da suddividere.
    patterns : list of str
        Lista di espressioni regolari ordinate per priorità decrescente.
        Il primo pattern ha la priorità di taglio più alta.
    max_chunk_size : int, optional
        Numero massimo di caratteri consentito per singolo chunk (default è 2000).

    Returns
    -------
    list of tuple of (int, int)
        Lista di tuple `(start, end)` che rappresentano gli indici di slicing
        assoluti riferiti a `file_content`.

    Raises
    ------
    ValueError
        Se `max_chunk_size` è minore o uguale a zero.
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size deve essere maggiore di zero")

    if not file_content:
        return []

    # Pre-calcola gli indici di inizio per ogni livello di priorità in set dedicati
    boundary_tiers = [
        {m.start() for m in re.finditer(p, file_content)}
        for p in patterns
    ]

    chunks = []
    start = 0
    content_len = len(file_content)

    while start < content_len:
        # Se il testo rimanente rientra già nel limite, crea l'ultimo chunk ed esci
        if content_len - start <= max_chunk_size:
            chunks.append((start, content_len))
            break

        limit = start + max_chunk_size
        end = limit

        # Esplora i livelli di priorità dal più alto al più basso
        for tier in boundary_tiers:
            candidates = [pos for pos in tier if start < pos <= limit]
            if candidates:
                # Sceglie il confine valido più lontano possibile entro il limite
                end = max(candidates)
                break

        chunks.append((start, end))
        start = end

    return chunks


def chunk_md(file_content: str, max_chunk_size: int = 2000) -> list[tuple[int, int]]:
    """Suddivide un testo Markdown in chunk preservando titoli e paragrafi.

    Parameters
    ----------
    file_content : str
        Il contenuto del file Markdown da suddividere.
    max_chunk_size : int, optional
        Dimensione massima in caratteri di ciascun chunk (default è 2000).

    Returns
    -------
    list of tuple of (int, int)
        Lista di tuple `(start, end)` con gli indici assoluti dei chunk generati.
    """
    return chunk_text(file_content, MD_PATTERNS, max_chunk_size)


def chunk_py(file_content: str, max_chunk_size: int = 2000) -> list[tuple[int, int]]:
    """Suddivide codice sorgente Python in chunk preservando classi, funzioni e decoratori.

    Parameters
    ----------
    file_content : str
        Il codice sorgente Python da suddividere.
    max_chunk_size : int, optional
        Dimensione massima in caratteri di ciascun chunk (default è 2000).

    Returns
    -------
    list of tuple of (int, int)
        Lista di tuple `(start, end)` con gli indici assoluti dei chunk generati.
    """
    return chunk_text(file_content, PY_PATTERNS, max_chunk_size)


def process_dir(
    directory_path: Path,
    max_chunk_size: int = 2000,
) -> dict[Path, list[tuple[int, int]]]:
    """Scansiona ricorsivamente una directory e calcola i chunk per i file .py e .md.

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

    for path in directory_path.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".md"}:
            continue

        txt = take_txt(path)
        if not txt:
            continue

        if path.suffix == ".md":
            chunks = chunk_md(txt, max_chunk_size)
        else:
            chunks = chunk_py(txt, max_chunk_size)

        result[path] = chunks

    return result


if __name__ == "__main__":
    print(process_dir(Path("data/datasets/raw")))