"""The documented setup is: copy .env.example to .env. Something has to read it."""

import os

from candidate_scoring.config import load_env_file


def test_a_dotenv_file_reaches_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_CREDENTIAL", raising=False)
    env = tmp_path / ".env"
    env.write_text("SOME_CREDENTIAL=from-the-file\n")

    load_env_file(env)

    assert os.environ["SOME_CREDENTIAL"] == "from-the-file"


def test_a_real_environment_variable_beats_the_file(tmp_path, monkeypatch):
    """An operator exporting a key for one command must not be overridden by a stale file."""
    monkeypatch.setenv("SOME_CREDENTIAL", "from-the-shell")
    env = tmp_path / ".env"
    env.write_text("SOME_CREDENTIAL=from-the-file\n")

    load_env_file(env)

    assert os.environ["SOME_CREDENTIAL"] == "from-the-shell"


def test_comments_blank_lines_and_padding_are_tolerated(tmp_path, monkeypatch):
    """The committed .env.example has comments, and a hand-edited file may pad the '='."""
    for name in ("PADDED_KEY", "QUOTED_KEY", "COMMENTED_KEY"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("# a comment\n\nPADDED_KEY = padded\nQUOTED_KEY=\"quoted\"\n#COMMENTED_KEY=no\n")

    load_env_file(env)

    assert os.environ["PADDED_KEY"] == "padded"
    assert os.environ["QUOTED_KEY"] == "quoted"
    assert "COMMENTED_KEY" not in os.environ


def test_a_missing_file_is_not_an_error(tmp_path):
    load_env_file(tmp_path / "nope.env")
