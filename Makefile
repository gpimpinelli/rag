# ==============================================================================
# VARIABILI DI CONFIGURAZIONE E PERCORSI
# ==============================================================================
PYTHON = uv run python -m src

# Cartelle Dati e Indice
RAW_DIR       = data/raw
INDEX_DIR     = data/processed
MAX_CHUNK     = 2000
K             = 10

# Percorsi Dataset (Adatta i nomi file se diversi nella tua cartella data/)
DS_ANSWERED_DOCS   = data/datasets/AnsweredQuestions/dataset_docs_public.json
DS_ANSWERED_CODE   = data/datasets/AnsweredQuestions/dataset_code_public.json
DS_UNANSWERED_DOCS = data/datasets/UnansweredQuestions/dataset_docs_public.json
DS_UNANSWERED_CODE = data/datasets/UnansweredQuestions/dataset_code_public.json

# Cartelle Output
OUT_SEARCH_ANS   = data/output/search_results/AnsweredQuestions
OUT_SEARCH_UNANS = data/output/search_results/UnansweredQuestions
OUT_ANS_UNANS    = data/output/search_results_and_answer/UnansweredQuestions

# Query di test rapido per la CLI
QUERY ?= "How to configure OpenAI server?"

.PHONY: install index search answer \
        search-ans-docs search-ans-code eval-docs eval-code eval-all \
        student-search-docs student-search-code student-answer-docs student-answer-code student-all \
        debug clean clean-all lint lint-strict format

# ==============================================================================
# SETUP & AMBIENTE
# ==============================================================================
install:
	uv sync

# ==============================================================================
# COMANDI INTERATTIVI SINGOLI (es. make search QUERY="what is PagedAttention?")
# ==============================================================================
index:
	$(PYTHON) index \
		--index_dir="$(INDEX_DIR)" \
		--raw_directory="$(RAW_DIR)" \
		--max_chunk_size=$(MAX_CHUNK)

search:
	$(PYTHON) search \
		--index_dir="$(INDEX_DIR)" \
		--query=$(QUERY) \
		--k=$(K)

answer:
	$(PYTHON) answer \
		--index_dir="$(INDEX_DIR)" \
		--query=$(QUERY) \
		--k=5

# ==============================================================================
# VALUTAZIONE (Recall@1/3/5/10 su AnsweredQuestions)
# ==============================================================================
search-ans-docs:
	$(PYTHON) search_dataset \
		--index_dir="$(INDEX_DIR)" \
		--dataset_path="$(DS_ANSWERED_DOCS)" \
		--k=$(K) \
		--save_directory="$(OUT_SEARCH_ANS)"

search-ans-code:
	$(PYTHON) search_dataset \
		--index_dir="$(INDEX_DIR)" \
		--dataset_path="$(DS_ANSWERED_CODE)" \
		--k=$(K) \
		--save_directory="$(OUT_SEARCH_ANS)"

eval-docs: search-ans-docs
	$(PYTHON) evaluate \
		--student_search_results_path="$(OUT_SEARCH_ANS)/$(notdir $(DS_ANSWERED_DOCS))" \
		--dataset_path="$(DS_ANSWERED_DOCS)"

eval-code: search-ans-code
	$(PYTHON) evaluate \
		--student_search_results_path="$(OUT_SEARCH_ANS)/$(notdir $(DS_ANSWERED_CODE))" \
		--dataset_path="$(DS_ANSWERED_CODE)"

eval-all: eval-docs eval-code

# ==============================================================================
# GENERAZIONE FILE DI CONSEGNA (UnansweredQuestions)
# ==============================================================================
student-search-docs:
	$(PYTHON) search_dataset \
		--index_dir="$(INDEX_DIR)" \
		--dataset_path="$(DS_UNANSWERED_DOCS)" \
		--k=$(K) \
		--save_directory="$(OUT_SEARCH_UNANS)"

student-search-code:
	$(PYTHON) search_dataset \
		--index_dir="$(INDEX_DIR)" \
		--dataset_path="$(DS_UNANSWERED_CODE)" \
		--k=$(K) \
		--save_directory="$(OUT_SEARCH_UNANS)"

student-answer-docs: student-search-docs
	$(PYTHON) answer_dataset \
		--index_dir="$(INDEX_DIR)" \
		--student_search_results_path="$(OUT_SEARCH_UNANS)/$(notdir $(DS_UNANSWERED_DOCS))" \
		--save_directory="$(OUT_ANS_UNANS)"

student-answer-code: student-search-code
	$(PYTHON) answer_dataset \
		--index_dir="$(INDEX_DIR)" \
		--student_search_results_path="$(OUT_SEARCH_UNANS)/$(notdir $(DS_UNANSWERED_CODE))" \
		--save_directory="$(OUT_ANS_UNANS)"

student-all: student-answer-docs student-answer-code

# ==============================================================================
# QUALITÀ DEL CODICE, DEBUG E PULIZIA
# ==============================================================================
debug:
	uv run python -m pdb -m src


lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src
	uv run mypy src --strict

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +

clean-all: clean
	rm -rf data/output data/processed .venv
	uv cache clean