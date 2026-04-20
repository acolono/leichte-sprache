"""CLI smoke tests for tools.ml using Typer CliRunner."""

from typer.testing import CliRunner

from tools.ml.__main__ import app

runner = CliRunner()


def test_help_shows_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.output
    assert "evaluate" in result.output


def test_train_list():
    result = runner.invoke(app, ["train", "--list"])
    assert result.exit_code == 0
    assert "abkuerzungen" in result.output


def test_train_help_shows_options():
    result = runner.invoke(app, ["train", "--help"])
    assert "--list" in result.output
    assert "--device" in result.output
    assert "--seed" in result.output
    assert "--promote" in result.output
