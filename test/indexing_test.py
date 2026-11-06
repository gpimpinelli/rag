from pathlib import Path
from tqdm import tqdm

# Importa le tue strutture dati e il tokenizzatore
from src.rag.chunk_table import ChunkTable
from src.rag.inverted_index import InvertedIndex
from src.rag.tokenizer import tokenize

# Importa la funzione di test e take_txt dal tuo file precedente 
# (sostituisci 'test_chunking' con il nome reale del tuo file)
from .chunking_test import process_dir, take_txt

def test_build_index(directory_path: Path, max_chunk_size: int = 2000):
    """Testa l'indicizzazione partendo dai chunk generati da process_dir."""
    
    print("--- FASE 1: Estrazione Coordinate Chunk ---")
    # 1. Chiamiamo il tuo tester già fatto! Ci restituisce {Path: [(start, end), ...]}
    chunks_dict = process_dir(directory_path, max_chunk_size)
    
    print("\n--- FASE 2: Tokenizzazione e Indicizzazione ---")
    # 2. Inizializziamo il "cervello" e la "memoria"
    tabella = ChunkTable()
    indice = InvertedIndex()
    
    # 3. Iteriamo sui risultati che la tua funzione ha prodotto
    for path, coords_list in tqdm(chunks_dict.items(), desc="Popolamento Indice", unit="file"):
        
        # Dobbiamo rileggere il testo per poterne estrarre le parole
        txt = take_txt(path)
        if not txt:
            continue
            
        # Registra il file e ottieni il suo ID
        file_id = tabella.add_file(str(path))
        
        # Scorriamo le coordinate (start, end) generate in precedenza
        for start, end in coords_list:
            # Slicing: estraiamo la stringa esatta
            testo_chunk = txt[start:end]
            
            # Estraiamo i token (con la logica a doppio binario per camelCase/snake_case)
            tokens = tokenize(testo_chunk)
            
            # Se il chunk contiene solo simboli/spazi e zero token validi, lo ignoriamo
            if not tokens:
                continue
                
            # Salviamo i metadati a colonne nella tabella
            chunk_id = tabella.add_chunk(file_id, start, end, len(tokens))
            
            # Inseriamo nel motore matematico
            indice.add_chunk(chunk_id, tokens)

    # 4. Finalizzazione (pre-calcolo IDF e lunghezza media)
    print("\n--- FASE 3: Finalizzazione Indice ---")
    indice.finalize(tabella.length)
    
    # 5. Verifica e Report
    print("\n--- Report di Integrità ---")
    print(f"File processati: {len(tabella.files)}")
    print(f"Chunk archiviati in Tabella: {len(tabella)}")
    print(f"Chunk validi nell'Indice: {indice.n_chunks}")
    print(f"Lunghezza media di un chunk (in token): {indice.avg_len:.2f}")
    print(f"Vocabolario unico (Termini in IDF cache): {len(indice.idf_cache)}")
    
    # Asserzione vitale: la tabella e l'indice devono essere perfettamente allineati
    assert len(tabella) == indice.n_chunks, "ERRORE CRITICO: Disallineamento tra Tabella e Indice!"
    print("✅ Allineamento Tabella-Indice superato con successo.")

    return tabella, indice

if __name__ == "__main__":
    dataset_path = Path("data/datasets/raw")  # Modifica se necessario
    
    # Esegue il test completo e salva gli oggetti popolati
    tabella_popolata, indice_popolato = test_build_index(dataset_path, 2000)

