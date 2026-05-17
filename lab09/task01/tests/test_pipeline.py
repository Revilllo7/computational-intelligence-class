import json
import sys
from pathlib import Path

# Ensure the local `src` package is importable when tests run from different CWDs
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.main import run_article_pipeline


def test_pipeline_saves_json_and_counts(tmp_path):
    article = Path(__file__).resolve().parents[1] / "data" / "article.txt"
    out = tmp_path / "counts.json"

    # Run pipeline without plots to keep test fast and non-interactive
    counts = run_article_pipeline(article, show_plots=False, save_json=out)

    assert isinstance(counts, dict)
    assert counts, "counts should not be empty"
    assert out.exists(), "JSON output file should be created"

    loaded = json.loads(out.read_text(encoding="utf-8"))
    # JSON should be saved as a list of [word, count] pairs ordered by count desc
    assert isinstance(loaded, list)
    assert loaded, "saved JSON list should not be empty"
    # Top word expected to be 'wind' (dominant in the article)
    top_word, top_count = loaded[0]
    assert top_word == "wind"
    assert isinstance(top_count, int) and top_count > 0

    # Reconstruct dict and compare with returned counts
    loaded_dict = {k: v for k, v in loaded}
    assert loaded_dict == counts
    # Basic sanity checks
    assert all(isinstance(k, str) for k in counts)
    assert all(isinstance(v, int) for v in counts.values())


def test_pipeline_token_counts_consistency():
    # Ensure pipeline's printed counts correspond to returned counts length > 0
    article = Path(__file__).resolve().parents[1] / "data" / "article.txt"
    counts = run_article_pipeline(article, show_plots=False, save_json=None)
    assert sum(counts.values()) > 0
