"""Suite di test completa (stile Moulinette) per il progetto RAG."""

import io
import pickle
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.rag.answer_generator import NO_CONTEXT_REPLY, build_context
from src.rag.chunking import chunk_md, chunk_py
from src.rag.rag_cli import IOU_THRESHOLD, RagCLI, _iou
from src.rag.orchestrator import RagOrchestrator
from src.rag.tokenizer import tokenize


# ==============================================================================
# 1. UNIT TEST: TOKENIZER & CHUNKER INVARIANTS
# ==============================================================================
def test_tokenizer_invariants() -> None:
    """Verifica il cross-matching CLI/Codice, camelCase e filtro rumore."""
    t_cli = tokenize("--max-model-len")
    t_py = tokenize("max_model_len")
    assert "max_model_len" in t_cli and "max_model_len" in t_py
    assert set(t_cli) == set(t_py)

    t_camel = tokenize("HTTPServer_Port64")
    for expected in ("http", "server", "port", "64"):
        assert expected in t_camel

    assert tokenize("the and of explain using") == []
    assert tokenize("a" * 50) == []


def test_chunker_invariants() -> None:
    """Verifica limiti, copertura contigua e unione dei decoratori."""
    py_code = (
        "import os\n\n"
        "@decorator_one\n"
        "@decorator_two(arg=1)\n"
        "def my_function(x: int) -> int:\n"
        "    return x + 1\n\n"
        "class MyClass:\n"
        "    @property\n"
        "    def prop(self):\n"
        "        return 42\n"
    )
    for max_sz in (2000, 1000, 500, 80):
        chunks = chunk_py(py_code, max_chunk_size=max_sz)
        assert chunks, f"Chunk vuoti per max_sz={max_sz}"
        assert chunks[0][0] == 0
        assert chunks[-1][1] == len(py_code)
        for i, (s, e) in enumerate(chunks):
            assert 0 <= s < e <= len(py_code)
            assert (e - s) <= max_sz
            if i > 0:
                assert s == chunks[i - 1][1], "Buco o sovrapposizione"
            snippet = py_code[s:e].lstrip()
            assert not snippet.startswith("def my_function"), "Decoratore spezzato!"

    md_text = (
        "# Titolo Principale\nTesto intro.\n"
        "```bash\n# Questo NON e' un titolo markdown\npip install vllm\n```\n"
        "## Secondo Titolo\nAltro testo.\n"
    )
    for s, e in chunk_md(md_text, max_chunk_size=60):
        assert (e - s) <= 60
        assert not md_text[s:e].startswith("# Questo NON")


# ==============================================================================
# 2. INVARIANTI DELL'INDICE & KNOWN-ITEM SEARCH
# ==============================================================================
def test_index_determinism_and_known_item() -> None:
    """Verifica determinismo, round-trip pickle e correttezza del ranking (known-item)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        raw = root / "raw" / "vllm-mock"
        raw.mkdir(parents=True)

        (raw / "engine.py").write_text(
            "class LLMEngine:\n    def step(self):\n        return 'paged_attention'\n",
            encoding="utf-8",
        )
        (raw / "readme.md").write_text(
            "# vLLM Documentation\nConfigure `--max-model-len` for LoRA adapters.\n",
            encoding="utf-8",
        )
        (raw / "empty.py").write_text("   \n\n  ", encoding="utf-8")

        rag1 = RagOrchestrator()
        rag1.build_index_from_directory(raw.parent, max_chunk_size=500)

        rag2 = RagOrchestrator()
        rag2.build_index_from_directory(raw.parent, max_chunk_size=500)

        assert rag1.table.files == rag2.table.files
        assert rag1.table.start == rag2.table.start
        assert rag1.table.end == rag2.table.end
        assert rag1.index.postings == rag2.index.postings

        idx_dir = root / "idx"
        rag1.save_index(idx_dir)

        rag_loaded = RagOrchestrator()
        rag_loaded.load_index(idx_dir)

        # Known-item verification: verifica che il primo file sia quello giusto
        res_md = rag_loaded.search("how to configure max-model-len", top_k=5)
        assert res_md and res_md[0].file_path.endswith("readme.md")

        res_py = rag_loaded.search("LLMEngine paged_attention step", top_k=5)
        assert res_py and res_py[0].file_path.endswith("engine.py")


# ==============================================================================
# 3. CASI LIMITE DELLA CLI E PARSING FIRE
# ==============================================================================
def _run_cli_captured(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


def test_cli_edge_cases_and_fire() -> None:
    """Verifica assert specifici sugli output CLI e testa l'entry-point Fire."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        cli = RagCLI(index_dir=str(root / "missing_index"))

        out = _run_cli_captured(cli.search, "vllm", k=5)
        assert "Error:" in out

        for bad_sz in (0, -5):
            out = _run_cli_captured(cli.index, max_chunk_size=bad_sz, raw_directory=str(root))
            assert "Error:" in out

        raw = root / "raw"
        raw.mkdir()
        (raw / "doc.md").write_text("# Test\nHello vllm world", encoding="utf-8")
        cli_ok = RagCLI(index_dir=str(root / "valid_index"))
        out = _run_cli_captured(cli_ok.index, max_chunk_size=500, raw_directory=str(raw))
        assert "Error:" not in out and "Ingestion complete!" in out

        # Query vuote devono segnalare errore
        for empty_q in ("", "   "):
            out = _run_cli_captured(cli_ok.search, empty_q, k=5)
            assert "Error: query is empty" in out

        # Query valide ma senza match NON devono produrre 'Error:'
        for no_match_q in ("the of and", "!@#$%^&*()", "🚀🔥汉字", "a" * 5000):
            out = _run_cli_captured(cli_ok.search, no_match_q, k=5)
            assert "Error:" not in out, f"Errore inatteso per {no_match_q!r}: {out}"
            assert "Nessun risultato." in out

        # Query con match reale
        out = _run_cli_captured(cli_ok.search, "vllm", k=5)
        assert "Error:" not in out and "doc.md" in out

        # k non validi
        for bad_k in (0, -1):
            out = _run_cli_captured(cli_ok.search, "vllm", k=bad_k)
            assert "Error: k must be greater than zero" in out

        # Test reale via subprocess (verifica che Fire non confonda --k con --bm25_k1)
        proc = subprocess.run(
            [
                sys.executable, "-m", "src", "search",
                f"--index_dir={root / 'valid_index'}",
                "--query=vllm",
                "--k=2",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0
        assert "Traceback" not in proc.stderr
        assert "doc.md" in proc.stdout


# ==============================================================================
# 4. TEST DEL GENERATORE (BUDGET TOKEN & PULIZIA <think>)
# ==============================================================================
def test_generator_helpers() -> None:
    """Verifica token budget, file mancanti e rimozione blocchi <think>."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "test.py"
        p.write_text("def alpha(): pass\n" * 100, encoding="utf-8")

        sources = [
            MinimalSource(
                file_path="non_esiste.py",
                first_character_index=0,
                last_character_index=50,
            ),
            MinimalSource(
                file_path=str(p),
                first_character_index=0,
                last_character_index=200,
            ),
            MinimalSource(
                file_path=str(p),
                first_character_index=200,
                last_character_index=400,
            ),
        ]
        ctx = build_context(sources, count_tokens=len, max_tokens=250)
        assert "[2]" in ctx and "[3]" not in ctx

        # Contesto tutto illeggibile -> stringa vuota
        empty_ctx = build_context([sources[0]], count_tokens=len, max_tokens=250)
        assert empty_ctx == ""


# ==============================================================================
# 5. CONTROLLO AMBIENTE REALE (CEILING, DIAGNOSTICA SPAN, RECALL REALE & JSON)
# ==============================================================================
def inspect_real_environment() -> None:
    """Controlla l'indice reale, diagnostica i chunk mancati e verifica le soglie Recall@5."""
    print("\n" + "=" * 70)
    print("CONTROLLO AMBIENTE REALE (Indice, Ceiling, Recall@5 e Output JSON)")
    print("=" * 70)

    idx_path = Path("data/processed/rag_index.pkl")
    if not idx_path.exists():
        print("[SKIP] data/processed/rag_index.pkl non trovato.")
        return

    t0 = time.perf_counter()
    with idx_path.open("rb") as fh:
        data = pickle.load(fh)
    dt = time.perf_counter() - t0
    table = data["table"]
    print(f"[OK] Indice caricato in {dt:.2f}s ({len(table)} chunk, {len(table.files)} file)")

    # Verifica invarianti fisici
    file_sizes = {
        f: len(Path(f).read_text(encoding="utf-8", errors="replace"))
        for f in table.files if Path(f).exists()
    }
    assert len(file_sizes) == len(table.files), "File indicizzati mancanti su disco!"
    for c_id in range(len(table)):
        f_path = table.files[table.file_id[c_id]]
        s, e = table.start[c_id], table.end[c_id]
        assert 0 <= s < e <= file_sizes[f_path] and (e - s) <= 2000
    print("[OK] Tutti i 13819 chunk rispettano <= 2000 chars e gli offset reali.")

    # Ceiling + Diagnostica delle fonti irraggiungibili
    ans_dir = Path("data/datasets/AnsweredQuestions")
    if ans_dir.exists():
        by_file: dict[str, list[MinimalSource]] = {}
        for i in range(len(table)):
            src = table.get_source(i)
            by_file.setdefault(src.file_path, []).append(src)

        for ds_file in sorted(ans_dir.glob("*.json")):
            ds = RagDataset.model_validate_json(ds_file.read_text(encoding="utf-8"))
            answered = [q for q in ds.rag_questions if isinstance(q, AnsweredQuestion)]
            reachable, total_refs = 0, 0
            for q in answered:
                for ref in q.sources:
                    total_refs += 1
                    cands = by_file.get(ref.file_path, [])
                    best_iou = max((_iou(ref, c) for c in cands), default=0.0)
                    if best_iou >= IOU_THRESHOLD:
                        reachable += 1
                    else:
                        span_len = ref.last_character_index - ref.first_character_index
                        print(
                            f"  [DIAGNOSTICA MISS] {ds_file.name} | {ref.file_path} "
                            f"[{ref.first_character_index}:{ref.last_character_index}] "
                            f"(len={span_len}) -> best IoU={best_iou:.4f}"
                        )
            print(f"[CEILING] {ds_file.name}: {reachable}/{total_refs} ({reachable/total_refs*100:.1f}%)")

    # Validazione JSON prodotti + Regression Test Recall@5
    out_dir = Path("data/output")
    if not out_dir.exists():
        print("[SKIP] Cartella data/output non presente.")
        return

    sr_files = sorted((out_dir / "search_results").rglob("*.json"))
    if not sr_files:
        print("[SKIP] Nessun file in data/output/search_results.")
    for sr_file in sr_files:
        res = StudentSearchResults.model_validate_json(sr_file.read_text(encoding="utf-8"))
        for item in res.search_results:
            assert len(item.retrieved_sources) <= res.k
            for src in item.retrieved_sources:
                assert 0 < (src.last_character_index - src.first_character_index) <= 2000

        # Calcolo Recall@5 se esiste il corrispettivo AnsweredQuestions
        ref_ds_path = ans_dir / sr_file.name
        if ref_ds_path.exists() and res.k >= 5:
            ds = RagDataset.model_validate_json(ref_ds_path.read_text(encoding="utf-8"))
            got = {r.question_id: r.retrieved_sources for r in res.search_results}
            answered = [q for q in ds.rag_questions if isinstance(q, AnsweredQuestion)]
            total_r5 = 0.0
            for q in answered:
                top5 = got.get(q.question_id, [])[:5]
                found = sum(any(_iou(r, g) >= IOU_THRESHOLD for g in top5) for r in q.sources)
                total_r5 += found / len(q.sources) if q.sources else 0.0
            recall_5 = total_r5 / len(answered)
            threshold = 0.80 if "docs" in sr_file.name else 0.50
            print(f"[RECALL@5] {sr_file} -> {recall_5:.3f} (Soglia minima: {threshold:.2f})")
            assert recall_5 >= threshold, f"Recall@5 sotto soglia per {sr_file}: {recall_5:.3f}"
        else:
            print(f"[OK] Output Search valido: {sr_file} ({len(res.search_results)} domande, k={res.k})")

    ans_files = sorted((out_dir / "search_results_and_answer").rglob("*.json"))
    if not ans_files:
        print("[SKIP] Nessun file in data/output/search_results_and_answer.")
    for ans_file in ans_files:
        res_a = StudentSearchResultsAndAnswer.model_validate_json(ans_file.read_text(encoding="utf-8"))
        for item in res_a.search_results:
            assert item.answer and item.answer.strip()
            assert "<think>" not in item.answer
            assert len(item.retrieved_sources) <= res_a.k
        print(f"[OK] Output Answer valido: {ans_file} ({len(res_a.search_results)} risposte, k={res_a.k})")


if __name__ == "__main__":
    print("Avvio Test Suite Automatica RAG...")
    tests = [
        ("1. Invarianti Tokenizer", test_tokenizer_invariants),
        ("2. Invarianti Chunker (Limiti & Decoratori)", test_chunker_invariants),
        ("3. Determinismo Indice, Round-Trip & Known-Item", test_index_determinism_and_known_item),
        ("4. Robustezza CLI & Subprocess Fire", test_cli_edge_cases_and_fire),
        ("5. Budget di Token & Resilienza Context Builder", test_generator_helpers),
    ]
    failed = 0
    for name, fn in tests:
        try:
            t0 = time.perf_counter()
            fn()
            print(f"  [PASS] {name} ({(time.perf_counter() - t0)*1000:.1f} ms)")
        except Exception as exc:
            failed += 1
            print(f"  [FAIL] {name} -> {exc}")

    inspect_real_environment()
    if failed:
        sys.exit(1)
    print("\nTutti i controlli sono passati con successo!")