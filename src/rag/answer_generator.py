"""Modulo per la generazione delle risposte tramite LLM locale."""

import re
import torch
from pathlib import Path
from typing import Callable
from transformers import AutoModelForCausalLM, AutoTokenizer
from src import take_txt
from src.models import MinimalSource


# ==============================================================================
# COSTANTI E PROMPT
# ==============================================================================
# Sostituiamo il prompt generico con quello richiesto esplicitamente dal professore.
# È ottimizzato per un modello piccolo da 0.6B: frasi brevi, zero invenzioni.

SYSTEM_PROMPT = (
    "You answer questions about the vLLM codebase and documentation. "
    "Use ONLY the numbered context passages provided.\n"
    "Rules:\n"
    "1. Answer in 1-4 sentences that directly address the question.\n"
    "2. Copy identifiers, flag names, file names and values exactly "
    "as they appear in the context.\n"
    "3. Never add facts that are not in the context.\n"
    "4. If the context only partly answers, give what it supports and "
    "say what is missing. If it contains nothing relevant, reply: "
    "\"The provided context does not contain enough information to "
    "answer this question.\""
)

USER_PROMPT = "<context>\n{context}\n</context>\n\nQuestion: {question}"

# Risposta standardizzata per fallback
NO_CONTEXT_REPLY = "The provided context does not contain enough information to answer this question."

def build_context(
    sources: list[MinimalSource],
    count_tokens: Callable[[str], int],
    max_tokens: int = 3000,
) -> str:
    """Costruisce il contesto rispettando il budget di token.
    
    Legge il testo reale dal disco in base agli indici dei source. Include
    solo i chunk che rientrano nel budget di token per non saturare la memoria 
    del modello linguistico, prevenendo crolli di performance.
    """
    snippets: list[str] = []
    used = 0
    cache: dict[str, str] = {}
    
    # rank fornisce un numero [1], [2], ecc. per aiutare il LLM a citare le fonti
    for rank, src in enumerate(sources, start=1):
        try:
            if src.file_path not in cache:
                cache[src.file_path] = take_txt(Path(src.file_path))
        except (OSError, UnicodeError):
            continue # Ignora i file illeggibili senza far crashare il sistema
            
        # Slicing preciso basato sugli indici salvati
        chunk = cache[src.file_path][src.first_character_index:src.last_character_index]
        
        # Ignora i chunk composti solo da spazi/a capi
        if not chunk.strip():
            continue
            
        snippet = f"[{rank}] {src.file_path}\n{chunk}"
        cost = count_tokens(snippet)
        
        # Se sfondiamo il limite, interrompiamo (i source sono già ordinati per rilevanza!)
        if snippets and used + cost > max_tokens:
            break
            
        snippets.append(snippet)
        used += cost
        
    return "\n\n".join(snippets)

class AnswerGenerator:
    """Gestisce il caricamento del modello Qwen e l'inferenza del testo."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str | None = None
    ) -> None:
        """Inizializza il dispositivo e carica tokenizer e pesi del modello."""
        # 1. Rileva il device se non specificato (GPU se disponibile, altrimenti CPU) 
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 2. Determina il dtype migliore e più sicuro in base al device 
        if self.device == "cuda":
            # Solo se siamo su GPU chiediamo se supporta la matematica avanzata bfloat16
            best_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            # Sulla CPU usiamo sempre float32 nativo per evitare errori di compatibilità PyTorch
            best_dtype = torch.float32

        # 3. Carica il tokenizer associato al modello
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 4. Carica il modello sui pesi opportuni e spostalo su self.device 
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=best_dtype,
        ).to(self.device)
        
        self.model.eval()

    def generate_answer(
        self,
        question: str,
        sources: list[MinimalSource],
        max_new_tokens: int = 256,
    ) -> str:
        """Costruisce il contesto, esegue l'inferenza e pulisce la risposta."""
        
        # Se non ci sono fonti in partenza, ritorniamo la frase standard
        if not sources:
            return NO_CONTEXT_REPLY

        # Definisce la funzione per contare i token e costruisce il contesto limitato
        def count_tokens(text: str) -> int:
            return len(self.tokenizer.encode(text))

        context_str = build_context(sources, count_tokens)
        
        # Se il testo ricavato è vuoto (es. file illeggibili), ritorniamo errore
        if not context_str.strip():
            return NO_CONTEXT_REPLY

        # A. Costruisce i messaggi usando i prompt aggiornati
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(context=context_str, question=question)},
        ]

        # B. Applica il template di chat
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False, # Evita che si gonfi con pensieri interni
        )

        # C. Tokenizza il prompt per PyTorch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # D. Genera i nuovi token
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.05, # FIX Punto 3: Evita loop infiniti di ripetizione
            )

        # E. Taglia il prompt iniziale
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        # F. Decodifica
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # FIX Punto 5: Regex per eliminare forzatamente l'output <think> residuo (se presente)
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
        
        return answer.strip()