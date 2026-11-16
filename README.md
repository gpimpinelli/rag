# RAG against the machine

*This project has been created as part of the 42 curriculum by gipimpin.*

## 1. Description

**RAG against the machine** is a custom Retrieval-Augmented Generation (RAG) pipeline built from scratch to answer technical questions about a specific codebase (in this case, the `vLLM` repository).

Large Language Models (LLMs) are frozen in time and often hallucinate when asked about private, recent, or highly specific codebases. Instead of fine-tuning, this project implements a complete RAG system that:

1. **Ingests and Indexes:** Parses Python and Markdown files, splitting them into manageable, semantically coherent chunks.
2. **Retrieves:** Uses a custom-built BM25 search engine to find the exact character ranges in the codebase that best match a user's query.
3. **Generates:** Injects the retrieved snippets into the context window of a local, lightweight LLM (`Qwen/Qwen3-0.6B`) to generate an accurate, grounded, and concise answer without hallucinations.

---

## 2. System Architecture

The project is built on a highly decoupled, modular architecture to ensure fast indexing and safe inference. The main components interact as follows:

* **Presentation Layer (`rag_cli.py` & `models.py`):** Uses Python `fire` to expose CLI commands. It handles lazy-loading of the ML models to keep indexing fast, wraps executions in safe `try/except` blocks to prevent tracebacks, and enforces strictly typed I/O using Pydantic models.
* **Orchestrator (`orchestrator.py`):** The "brain" of the pipeline. It coordinates the reading of raw files, applies the chunking strategies, tokenizes the text, and feeds data into the metadata table and the mathematical index. It also handles the atomic saving/loading of the `.pkl` index.
* **Data Storage (`chunk_table.py`):** A positional metadata registry. Instead of instantiating heavy Python objects, it uses a "Parallel Columns" approach (native lists of integers) to map a `chunk_id` to its `file_path`, `start`, `end`, and `length` in $O(1)$ time with minimal memory footprint.
* **Retrieval Engine (`inverted_index.py`):** A custom BM25 mathematical index mapped via an inverted dictionary, responsible for sparse evaluation and ranking.
* **Generation Engine (`answer_generator.py`):** Manages the Hugging Face `transformers` pipeline. It safely allocates `Qwen/Qwen3-0.6B` to the optimal hardware (CUDA or CPU) with the safest `dtype` (bfloat16, float16, or float32). It applies a strict prompt template, disables internal "thinking" tokens to save budget, and cleans the final output.

---

## 3. Chunking Strategy

To stay within the strict 2000-character limit required by the evaluator while maintaining context, the pipeline applies two distinct chunking strategies based on the file extension (`chunking.py`):

1. **Python Files (`chunk_py`):**
* Uses a multi-tiered Regex approach. Tier 0 cuts at `class`, `def`, and `async def`.
* **Decorator Protection:** A custom look-behind function prevents separating a `@decorator` from its underlying function.


2. **Markdown Files (`chunk_md`):**
* Cuts primarily at Headers (H1-H6) and double newlines (paragraphs).
* **Code-Block Masking Trick:** Markdown files often contain Bash or Python code blocks with `# comments`. To prevent the chunker from mistaking a code comment for an H1 Markdown title, the system temporarily masks `#` with spaces (`" "`) inside ````` blocks. This preserves the exact character length and index integrity while ensuring perfect semantic cuts.


3. **Safe Micro-Chunk Merging:** Both strategies run through a sliding window algorithm that safely merges micro-chunks (< 100 characters) with adjacent chunks, continuously checking that the 2000-character hard limit is never breached.

---

## 4. Retrieval Method (Deep Dive into BM25 & IDF)

The retrieval system does not rely on heavy vector embeddings; instead, it implements a highly optimized **BM25 (Best Matching 25)** algorithm, a state-of-the-art lexical bag-of-words ranking function.

BM25 improves upon standard TF-IDF by addressing term saturation and document length bias. The algorithm computes a relevance score for a document $D$ given a query $Q$ (containing tokens $q_1, q_2, ... , q_n$) as follows:

$$\text{Score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{\vert{}D\vert{}}{\text{avg\_len}}\right)}$$

### A. Inverse Document Frequency (IDF)

The IDF measures how much information a word provides. Rare words (e.g., `HTTPServer`) get a high score, while common words (e.g., `the`, `code`) get a low score.

* In `inverted_index.py`, during the `finalize()` step, the system pre-calculates the IDF for every token in the vocabulary using the standard smoothing formula:
`idf = math.log(1 + (N - df + 0.5) / (df + 0.5))`
*(Where `N` is total chunks, and `df` is the Document Frequency, i.e., how many chunks contain the term).*
* These scores are permanently saved in an `idf_cache` dictionary, making retrieval lookups instantaneous.

### B. Term Frequency (TF) Saturation ($k_1$)

Standard TF assumes that a word appearing 10 times is 10 times more relevant than a word appearing once. BM25 uses the $k_1$ parameter (default `1.5`) to create an asymptotic curve. If a token appears 2 or 3 times in a chunk, the score increases significantly; but if it appears 20 times, the score maxes out. This prevents spammy chunks from dominating the results.

### C. Document Length Normalization ($b$)

Since chunks vary in size, longer chunks naturally contain more words, giving them an unfair advantage. The parameter $b$ (default `0.75`) normalizes the term frequency based on the chunk's length (`lengths[chunk_id]`) relative to the average chunk length in the corpus (`self.avg_len`).

### D. Sparse Evaluation

To achieve sub-millisecond retrieval throughput, the `score()` method uses Sparse Evaluation. It only iterates through the specific tokens present in the user's query. If a query token exists in the inverted index (`self.postings`), the algorithm fetches its pre-calculated IDF and iterates **only** over the chunks that contain that specific token, entirely bypassing 99% of the database.

---

## 5. Design Decisions

* **Dual-Track Tokenization:** Code identifiers are complex. The custom `tokenizer.py` splits variables like `openai_compatible_server` into their atomic components (`openai`, `compatible`, `server`), but **also** indexes the whole cleaned word (`openaicompatibleserver`). This "dual-track" approach ensures high recall regardless of how the user types the query.
* **Path Boosting:** Codebase questions often implicitly target specific modules (e.g., "How does LoRA work?"). During indexing, the tokens extracted from the file's path are multiplied by 3 and injected into every chunk of that file. This artificially inflates the TF of module names, anchoring the chunks strongly to their domain.
* **Fallback Strategy:** If the retrieved context yields no useful information, the prompt explicitly instructs the LLM to output a standardized refusal (`"The provided context does not contain enough information..."`), preventing hallucinations.

---

## 6. Performance Analysis

* **Indexing Speed:** The use of `ChunkTable` parallel arrays and native dictionaries allows the ingestion, regex chunking, and BM25 tokenization of hundreds of files in just a few seconds, comfortably beating the 5-minute strict deadline.
* **Retrieval Throughput:** Sparse evaluation allows the system to process 200 questions in under a few seconds, easily bypassing the 90-second limit.
* **Recall@k:** Thanks to the synergistic effect of Dual-Track Tokenization, Path Boosting, and BM25's length normalization, the system aims to exceed the mandated baselines of 80% Recall@5 for documentation and 50% Recall@5 for raw code.

---

## 7. Challenges Faced

1. **Memory Exhaustion on Small Devices:** Loading a 0.6B LLM alongside a heavy object-oriented index caused memory spikes. *Solution:* Shifted to integer-based mapping for the metadata table and implemented lazy loading for the `AnswerGenerator`, ensuring `torch` is only imported when strictly necessary.
2. **Markdown Code Blocks vs. Headers:** Regex cutting on `#` destroyed code examples. *Solution:* The temporary mask-and-replace algorithm applied before calculating boundaries.
3. **Pydantic Edge Cases:** `take_json()` originally crashed when the moulinette fed missing or corrupted files. *Solution:* Wrapped file I/O operations to return safe defaults (`{}` or `[]`) and employed a generic `@_safe` wrapper to catch Pydantic validation errors gracefully.

---

## 8. Instructions & Example Usage

### Environment Setup (42 Cluster / Disk Quota)

If you are working in an environment with strict home directory storage limits (such as the 42 cluster), redirect `uv` caches, virtual environments, and Hugging Face model weights to `/goinfre` **before** running `make install` or any `uv` commands:

```bash
export UV_CACHE_DIR=/goinfre/$USER/uv_cache
export HF_HOME=/goinfre/$USER/hf_cache
export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT=/goinfre/$USER/rag_venv
```

### Prerequisites & Installation

Ensure you have Python 3.10+ and `uv` installed.

```bash
make install

```

### Running the System

The project is operated via a CLI built with Python Fire.

**1. Index the corpus (generates `data/processed/rag_index.pkl`):**

```bash
uv run python -m src index --max_chunk_size 2000 --raw_directory data/raw

```

**2. Search for the Top-K snippets for a single query:**

```bash
uv run python -m src search "How do I configure the OpenAI server?" --k 5

```

**3. Generate an AI answer for a single query:**

```bash
uv run python -m src answer "How do I configure the OpenAI server?" --k 5

```

**4. Process full datasets:**

```bash
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset.json
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset.json

```

**5. Evaluate your Recall@k locally:**

```bash
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset.json --dataset_path data/datasets/AnsweredQuestions/dataset.json

```

### Code Quality

To verify flake8 standards and strict MyPy static typing:

```bash
make lint

```

---

## 9. Resources & AI Usage

### References

* **BM25 Algorithm:** Robertson, S. E., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval.
* **Qwen Models:** [Qwen/Qwen3-0.6B Hugging Face Documentation](https://huggingface.co/Qwen/Qwen3-0.6B)
* **Python Fire:** [Google Python Fire Guide](https://google.github.io/python-fire/guide/)

### AI Usage

AI tools (LLMs) were utilized during this project exclusively for:

1. **Brainstorming:** Discussing edge cases in Regex logic for the Markdown parser (specifically for the non-destructive code-block masking algorithm).
2. **Type Checking:** Troubleshooting obscure `mypy` errors related to the `transformers` library type stubs (e.g., `_BaseModelWithGenerate` incompatibility).
3. **Drafting Documentation:** Generating the boilerplate structure and English phrasing for parts of this `README.md`.
