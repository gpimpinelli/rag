"""Tokenizzazione condivisa tra indicizzazione e ricerca.

Questo modulo fornisce le regole e le funzioni per scomporre il testo grezzo in 
token (parole chiave). Implementa un approccio a "doppio binario" per gli 
identificatori complessi del codice sorgente (camelCase e snake_case), 
mantenendo sia la parola intera che i suoi sotto-componenti.
"""

import re

# ==============================================================================
# COSTANTI E FILTRI
# ==============================================================================

_ENGLISH: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "also", "am", "an",
    "are", "at",
    "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by",
    "can", "could",
    "did", "do", "does", "doing", "down", "during",
    "each",
    "few", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how",
    "i", "into", "it", "its", "itself",
    "just",
    "may", "me", "might", "more", "most", "must", "my", "myself",
    "of", "on", "once", "only", "other", "our", "ours", "ourselves", "over",
    "own",
    "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up",
    "very",
    "was", "we", "were", "what", "when", "where", "which", "who", "whom",
    "why", "will", "would",
    "you", "your", "yours", "yourself", "yourselves",
})

_CONTRACTIONS: frozenset[str] = frozenset({
    "s", "t", "d", "ll", "m", "re", "ve", "don", "doesn", "isn", "aren",
    "wasn", "weren", "can", "couldn", "shouldn", "won", "wouldn",
})

_QUESTION_WORDS: frozenset[str] = frozenset({
    "describe", "explain", "tell",
})

_KEYWORD_LIKE: frozenset[str] = frozenset({
    "and", "as", "for", "in", "is", "or", "with",
})

_USAGE: frozenset[str] = frozenset({
    "use", "used", "uses", "using",
})

STOPWORDS: frozenset[str] = (_ENGLISH | _CONTRACTIONS | _QUESTION_WORDS
                             | _KEYWORD_LIKE | _USAGE)

MIN_LEN = 2        # Lunghezza minima di un token intero (es. "if" viene scartato, "for" no, ma "for" è stopword)
MAX_LEN = 40       # Soglia di rumore: stringhe più lunghe sono hash, base64 o dati binari

# ==============================================================================
# ESPRESSIONI REGOLARI (REGEX)
# ==============================================================================

# Leggenda WORD_RE:
# [A-Za-z0-9_] : Classe di caratteri. Accetta lettere (maiuscole e minuscole), numeri e underscore (_).
# +            : Quantificatore. Indica "uno o più" dei caratteri precedenti.
# Scopo        : Isola blocchi di codice o testo saltando punteggiatura, spazi e simboli speciali (es. !@#).
WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*")

# Leggenda CAMEL_RE:
# Divisa in tre blocchi separati dall'operatore OR (|) per intercettare pattern diversi nel camelCase:
#
# 1. [A-Z]+(?![a-z])
#    [A-Z]+    : Una o più lettere MAIUSCOLE consecutive.
#    (?![a-z]) : "Negative lookahead". Afferma che NON ci deve essere una minuscola subito dopo.
#    Scopo     : Cattura gli acronimi inseriti nel camelCase (es. "HTTP" in "HTTPServer").
#
# 2. [A-Z]?[a-z]+
#    [A-Z]?    : Zero o una lettera MAIUSCOLA.
#    [a-z]+    : Una o più lettere minuscole.
#    Scopo     : Cattura le normali parole (es. "Server", "get", "value").
#
# 3. [0-9]+
#    Scopo     : Cattura sequenze numeriche isolate (es. "64" in "base64Encode").
CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")

# ==============================================================================
# FUNZIONI
# ==============================================================================

def _split_identifier(word: str) -> list[str]:
    """Scompone uno snake_case o camelCase nei suoi pezzi atomici, in minuscolo.

    Logica dell'algoritmo:
    1. Divide preventivamente la parola originale a ogni underscore (snake_case).
    2. Per ogni segmento ottenuto, applica la regex CAMEL_RE per spezzare l'eventuale camelCase.
    3. Converte tutti i sotto-pezzi estratti in minuscolo.
    
    Variabili interne principali:
    - parts (list[str]): La lista accumulatrice dei frammenti atomici trovati.
    - segment (str): Una porzione della parola originaria separata da underscore.
    
    Args:
        word (str): L'identificatore grezzo da smembrare (es. "HTTPServer_Port").

    Returns:
        list[str]: La lista dei frammenti atomici in minuscolo (es. ["http", "server", "port"]).
    """
    parts: list[str] = []
    for segment in word.replace("-", "_").split("_"):
        parts.extend(p.lower() for p in CAMEL_RE.findall(segment))
    return parts


def _keep(token: str, min_len: int, max_len: int = MAX_LEN) -> bool:
    """Verifica se un token soddisfa i criteri di validità per l'indicizzazione.

    Logica dell'algoritmo:
    1. Controlla che la lunghezza del token sia superiore o uguale a min_len.
    2. Controlla che la lunghezza del token non superi il limite massimo (max_len).
    3. Verifica che il token non appartenga all'insieme delle STOPWORDS inutili.

    Args:
        token (str): La parola da valutare.
        min_len (int): La lunghezza minima richiesta per questo specifico token.
        max_len (int, optional): La lunghezza massima consentita. Default: MAX_LEN (40).

    Returns:
        bool: True se il token è valido e deve essere conservato, False se deve essere scartato.
    """
    return min_len <= len(token) <= max_len and token not in STOPWORDS


def tokenize(text: str) -> list[str]:
    """Trasforma un testo grezzo in una sequenza di token utili all'indicizzazione e ricerca.

    Logica dell'algoritmo:
    1. Estrae dal testo tutte le parole composte solo da caratteri alfanumerici e underscore (WORD_RE).
    2. Scompone ogni parola nei suoi frammenti base tramite `_split_identifier`.
    3. Se la parola è formata da un solo frammento (parola semplice): 
       ne valuta la validità tramite `_keep` (usando MIN_LEN) e la salva.
    4. Se la parola è composta (es. "user_id"): 
       salva la parola INTERA ripulita dagli underscore (doppio binario).
       salva in aggiunta tutti i SINGOLI FRAMMENTI estratti se superano il MIN_LEN.
    
    Variabili interne principali:
    - tokens (list[str]): L'accumulatore finale della sequenza di parole valide (con possibili ripetizioni).
    - word (str): Il blocco alfanumerico isolato dalla prima passata di Regex.
    - parts (list[str]): I frammenti smembrati della parola.
    - whole (str): L'identificatore composto originale, ripulito e passato in minuscolo.

    Args:
        text (str): Il testo originario da processare (es. il contenuto di un chunk o una query).

    Returns:
        list[str]: La lista piatta dei token validati, pronti per essere inseriti in InvertedIndex.
    """
    tokens: list[str] = []
    for word in WORD_RE.findall(text):
        if len(word) > MAX_LEN:
            continue

        parts = _split_identifier(word)
        
        # Ignora se l'estrazione non ha prodotto frammenti validi
        if not parts:
            continue
            
        # Caso 1: Parola semplice, non composta (es. "server")
        if len(parts) == 1:
            if _keep(parts[0], MIN_LEN):
                tokens.append(parts[0])
            continue
            
        # Caso 2: Identificatore composto, logica a "doppio binario" (es. "HTTPServer")
        whole = word.lower().replace("-", "_").strip("_")
        
        # Aggiunge la parola per intero (es. "httpserver")
        if _keep(whole, MIN_LEN):
            tokens.append(whole)
            
        # Aggiunge anche i frammenti separati (es. "http", "server")
        tokens.extend(p for p in parts if _keep(p, MIN_LEN))
        
    return tokens