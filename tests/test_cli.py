from typer.testing import CliRunner

from candidate_scoring.cli import app


def test_check_model_reports_a_missing_credential_without_a_provider_call(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = CliRunner().invoke(app, ["check-model"])

    assert result.exit_code == 1
    assert "OPENAI_API_KEY is not set" in result.output
