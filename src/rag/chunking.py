"""Modulo per la suddivisione (chunking) di file testuali.

Implementa un algoritmo a finestra scorrevole basato su livelli di priorità.
I micro-chunk vengono fusi in sicurezza (senza mai superare il limite) e 
i decoratori Python vengono raggruppati correttamente con la loro definizione.
"""

import re
from bisect import bisect_right

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
# I pattern definiscono i confini "sicuri" dove è permesso tagliare il testo.
# Sono ordinati per priorità: Tier 0 (il più importante, non va spezzato) 
# scendendo fino a Tier 2 (tagli di emergenza su singole righe).

MD_PATTERNS = [
    r"(?m)^(?= {0,3}#{1,6}(?:[ \t]+|$))", # Tier 0: Titoli (fino a H6)
    r"\n[ \t]*\n",                        # Tier 1: Doppio a capo (paragrafi)
    r"\n",                                # Tier 2: Singolo a capo (righe)
]

# Il Tier 0 intercetta qualsiasi inizio di class, def (anche async) o decoratore.
# I finti confini interni al blocco verranno filtrati via codice in `chunk_py`.
PY_PATTERNS = [
    r"(?m)^(?=[ \t]*(?:class\b|def\b|async[ \t]+def\b|@))",
    r"\n[ \t]*\n",
    r"\n",
]


def _is_preceded_by_decorator(text: str, idx: int) -> bool:
    """Verifica se l'indice è immediatamente preceduto da una riga con un decoratore (@).
    
    Serve a evitare che un blocco di codice come:
        @property
        def func():
    venga spezzato in due chunk separati. Se il `def` è preceduto da un `@`, 
    non lo consideriamo un confine di taglio valido.
    """
    if idx == 0:
        return False
        
    # Estrae il testo precedente e lo divide in righe
    prev_lines = text[:idx].split("\n")
    
    # Scorre le righe all'indietro ignorando quelle vuote o composte solo da spazi
    for line in reversed(prev_lines):
        stripped = line.strip()
        if stripped:
            return stripped.startswith("@")
            
    return False


def _chunk_tiers(file_content: str, boundary_tiers: list[list[int]], max_chunk_size: int = 2000, min_chunk_size: int = 100) -> list[tuple[int, int]]:
    """Motore puro a finestra scorrevole con merge sicuro dei micro-chunk.
    
    Logica dell'algoritmo:
    1. Calcola i chunk ideali procedendo in avanti e tagliando sul confine
       più severo possibile senza superare `max_chunk_size`.
    2. Esegue una passata iterativa per fondere i "micro-chunk" (frammenti
       troppo corti che distruggerebbero il calcolo dell'IoU). La fusione
       avviene SOLO se la somma delle lunghezze non supera il limite massimo,
       evitando crash invalidanti.
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size deve essere maggiore di zero")

    if not file_content:
        return []

    chunks: list[tuple[int, int]] = []
    start = 0
    content_len = len(file_content)

    # ==========================================================================
    # 1. TAGLIO A FINESTRA SCORREVOLE
    # ==========================================================================
    while start < content_len:
        # Se il testo rimanente entra tutto nella finestra, chiudiamo
        if content_len - start <= max_chunk_size:
            chunks.append((start, content_len))
            break

        limit = start + max_chunk_size
        end = limit

        # Cerca il miglior punto di taglio esplorando i tier dal più importante (0)
        for tier in boundary_tiers:
            # bisect_right trova velocemente l'indice di inserimento nella lista ordinata
            i = bisect_right(tier, limit)
            # Se esiste un confine valido prima del limite, usalo e fermati
            if i and tier[i - 1] > start:
                end = tier[i - 1]
                break

        chunks.append((start, end))
        start = end

    # ==========================================================================
    # 2. MERGE SICURO DEI MICRO-CHUNK
    # ==========================================================================
    # Continua a fondere finché ci sono cambiamenti, controllando SEMPRE il limite.
    changed = True
    while changed:
        changed = False
        new_chunks = []
        i = 0
        while i < len(chunks):
            start, end = chunks[i]
            size = end - start
            
            if size < min_chunk_size:
                # Tenta di fondere a sinistra (col chunk precedente)
                if new_chunks and (end - new_chunks[-1][0]) <= max_chunk_size:
                    new_chunks[-1] = (new_chunks[-1][0], end)
                    changed = True
                # Altrimenti tenta di fondere a destra (col chunk successivo)
                elif i + 1 < len(chunks) and (chunks[i+1][1] - start) <= max_chunk_size:
                    chunks[i+1] = (start, chunks[i+1][1]) # Allarga il prossimo a coprire questo
                    changed = True
                else:
                    # Non entra da nessuna parte senza esplodere il max_chunk_size.
                    # Viene mantenuto come micro-chunk per non invalidare l'output.
                    new_chunks.append((start, end))
            else:
                new_chunks.append((start, end))
            i += 1
            
        chunks = new_chunks

    return chunks


def chunk_md(file_content: str, max_chunk_size: int = 2000) -> list[tuple[int, int]]:
    """Suddivide un testo Markdown mascherando i blocchi di codice.
    
    Maschera temporaneamente i blocchi di codice Bash/Python presenti nel Markdown
    sostituendo i '#' con spazi. In questo modo, i commenti nel codice (es. '# install')
    non vengono scambiati per titoli Markdown di livello 1, preservando l'integrità
    degli indici originali.
    """
    masked_text = re.sub(
        r"(?ms)^[ \t]*(?:```|~~~).*?^[ \t]*(?:```|~~~)",
        lambda m: m.group(0).replace("#", " "),
        file_content
    )
    
    # Crea i confini usando il testo mascherato per il Tier 0
    boundary_tiers = [
        [m.start() for m in re.finditer(p, masked_text)]
        for p in MD_PATTERNS
    ]
    
    return _chunk_tiers(file_content, boundary_tiers, max_chunk_size)


def chunk_py(file_content: str, max_chunk_size: int = 2000) -> list[tuple[int, int]]:
    """Suddivide codice Python raggruppando correttamente i decoratori."""
    boundary_tiers = []
    
    for tier_idx, p in enumerate(PY_PATTERNS):
        matches = []
        for m in re.finditer(p, file_content):
            idx = m.start()
            
            # FIX: Se siamo al Tier 0 (Definizioni) e la riga precedente 
            # è un decoratore, scartiamo il confine. Questo salda magicamente 
            # multipli decoratori impilati con la loro classe/funzione.
            if tier_idx == 0 and _is_preceded_by_decorator(file_content, idx):
                continue
                
            matches.append(idx)
        boundary_tiers.append(matches)
        
    return _chunk_tiers(file_content, boundary_tiers, max_chunk_size)