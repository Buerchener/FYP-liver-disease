import json
from pathlib import Path

from typer.testing import CliRunner

from liverkg_cli.app import app


def read_embedded_payload(path: Path) -> dict:
    html = path.read_text(encoding="utf-8")
    encoded = html.split('id="review-data">', 1)[1].split("</script>", 1)[0]
    return json.loads(encoded)


def test_demo_exports_real_verifier_decisions(tmp_path):
    destination = tmp_path / "defense-review.html"

    result = CliRunner().invoke(app, ["--lang", "zh", "review", "-o", str(destination)])

    assert result.exit_code == 0, result.output
    assert '<html lang="zh-CN">' in destination.read_text(encoding="utf-8")
    data = read_embedded_payload(destination)
    assert data["mode"] == "demo"
    relations = [article["relations"][0] for article in data["articles"]]
    assert [item["semantic_status"] for item in relations] == [
        "ACCEPTED", "ACCEPTED", "REVIEW", "REJECTED",
    ]
    assert [item["import_ready"] for item in relations] == [True, False, False, False]
    assert relations[1]["schema_name"] == "IllustrativeTrial"


def test_saved_run_escapes_quoted_markup_and_preserves_legacy_status(tmp_path, monkeypatch):
    results = tmp_path / "agent_results_old.json"
    title = "</script><img src=sample>"
    results.write_text(json.dumps({
        "report": {"run_id": "old"},
        "records": [{
            "pmid": "123", "title": title,
            "phases": {"verification": {"relations": [{
                "subject": "foo", "predicate": "ASSOCIATED_WITH", "object": "bar",
                "evidence": title, "import_ready": False,
            }]}},
        }],
    }), encoding="utf-8")
    runs = tmp_path / "runs"
    run = runs / "old"
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"result_path": str(results)}), encoding="utf-8")
    monkeypatch.setenv("LIVERKG_RUNS_DIR", str(runs))
    destination = tmp_path / "review.html"

    result = CliRunner().invoke(app, ["review", "--run-id", "old", "-o", str(destination)])

    assert result.exit_code == 0, result.output
    assert title not in destination.read_text(encoding="utf-8")
    data = read_embedded_payload(destination)
    assert data["mode"] == "run"
    assert data["articles"][0]["title"] == title
    relation = data["articles"][0]["relations"][0]
    assert relation["evidence"] == title
    assert relation["semantic_status"] == "UNKNOWN"
    assert relation["import_ready"] is False


def test_review_refuses_to_replace_its_input(tmp_path):
    results = tmp_path / "agent_results_example.json"
    results.write_text('{"records": []}', encoding="utf-8")

    result = CliRunner().invoke(app, ["review", str(results), "-o", str(results)])

    assert result.exit_code == 2
    assert "must differ" in result.output
    assert results.read_text(encoding="utf-8") == '{"records": []}'
