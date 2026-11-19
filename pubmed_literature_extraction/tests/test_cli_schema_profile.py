import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from cognitive_agent.schema.schema_profile import load_schema_profile
from liverkg_cli.app import app
from liverkg_cli.runs import RunSpec, load_run_spec
from liverkg_cli.service import _runtime_env, agent_command


class CliSchemaProfileTests(unittest.TestCase):
    def test_run_spec_schema_profile_reaches_agent_and_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "schema.json"
            profile.write_text("{}", encoding="utf-8")
            spec = RunSpec(
                run_id="schema-test", schema_profile_path=str(profile),
                normalized_input_path=str(Path(tmp) / "input.jsonl"),
                project_root=tmp,
            )
            command = agent_command(spec)
            flag = command.index("--schema-profile")
            self.assertEqual(command[flag + 1], str(profile))
            with patch("liverkg_cli.service.get_secret", return_value=""):
                self.assertEqual(_runtime_env(spec)["SCHEMA_PROFILE"], str(profile))

    def test_legacy_run_spec_defaults_to_no_custom_schema(self):
        legacy = RunSpec(**json.loads('{"run_id":"old-run"}'))
        self.assertEqual(legacy.schema_profile_path, "")
        self.assertNotIn("--schema-profile", agent_command(legacy))

    def test_cli_resume_uses_the_saved_schema_snapshot(self):
        profile = {
            "name": "Locations", "contract_version": "locations-v1",
            "predicates": {"LocatedIn": {"allowed_signatures": [["Facility", "City"]]}},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schema = root / "schema.json"
            schema.write_text(json.dumps(profile), encoding="utf-8")
            articles = root / "articles.jsonl"
            articles.write_text(json.dumps({
                "pmid": "1", "title": "Locations", "abstract": "The museum is in London.",
            }) + "\n", encoding="utf-8")
            runner = CliRunner()
            with (
                patch.dict("os.environ", {"LIVERKG_CONFIG": str(root / "config.toml"),
                                          "LIVERKG_RUNS_DIR": str(root / "runs")}),
                patch("liverkg_cli.worker.main", return_value=0),
            ):
                first = runner.invoke(app, [
                    "extract", str(articles), "--schema-profile", str(schema), "--run-id", "custom",
                ])
                self.assertEqual(first.exit_code, 0, first.output)
                spec = load_run_spec("custom")
                self.assertFalse(spec.second_llm_enabled)
                self.assertEqual(load_schema_profile(spec.schema_profile_path).name, "Locations")

                schema.write_text(json.dumps({**profile, "name": "Changed"}), encoding="utf-8")
                changed = runner.invoke(app, [
                    "extract", "--resume", "--run-id", "custom", "--schema-profile", str(schema),
                ])
                self.assertEqual(changed.exit_code, 2, changed.output)
                self.assertIn("schema profile cannot be changed", changed.output)
                schema.unlink()
                resumed = runner.invoke(app, ["extract", "--resume", "--run-id", "custom"])
                self.assertEqual(resumed.exit_code, 0, resumed.output)
                self.assertEqual(load_run_spec("custom").schema_profile_path, spec.schema_profile_path)


if __name__ == "__main__":
    unittest.main()
