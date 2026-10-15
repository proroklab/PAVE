"""Unit tests for :mod:`evaluations._common.latex`."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluations._common.latex import (
    bold_max,
    fmt,
    fmt_bold,
    fmt_pm,
    fmt_rate,
    row,
    write_table,
)


def test_fmt_nan():
    assert fmt(float("nan")) == "--"


def test_fmt_basic():
    assert fmt(3.14159) == "3.14"
    assert fmt(3.14159, digits=4) == "3.1416"


def test_fmt_pm_basic():
    assert fmt_pm(3.14, 0.5) == "3.14\\,$\\pm$\\,0.50"


def test_fmt_pm_nan_std():
    assert fmt_pm(3.14, float("nan")) == "3.14"


def test_fmt_pm_nan_mean():
    assert fmt_pm(float("nan"), 0.5) == "--"


def test_fmt_rate_basic():
    assert fmt_rate(0.857) == "85.70"


def test_fmt_bold():
    assert fmt_bold("foo") == "\\textbf{foo}"


def test_row_basic():
    assert row(["a", "b", "c"]) == "a & b & c \\\\"


def test_bold_max_basic():
    cells = ["1.0", "2.5", "1.8"]
    out = bold_max(cells, [1.0, 2.5, 1.8])
    assert out == ["1.0", "\\textbf{2.5}", "1.8"]


def test_bold_max_first_tie_wins():
    cells = ["1.0", "2.5", "2.5"]
    out = bold_max(cells, [1.0, 2.5, 2.5])
    assert out == ["1.0", "\\textbf{2.5}", "2.5"]


def test_bold_max_all_nan():
    cells = ["a", "b"]
    out = bold_max(cells, [float("nan"), float("nan")])
    assert out == cells


def test_bold_max_length_mismatch():
    with pytest.raises(ValueError):
        bold_max(["a", "b"], [1.0])


def test_write_table_contains_essentials(tmp_path: Path):
    out = tmp_path / "smoke.tex"
    header = row(["A", "B", "C"])
    rows = [row(["1", "2", "3"]), row(["4", "5", "6"])]
    write_table(out, "lcc", header, rows, "Caption", "tab:x")
    content = out.read_text()
    assert "\\caption{Caption}" in content
    assert "\\label{tab:x}" in content
    assert "\\begin{tabular}{lcc}" in content
    assert "\\toprule" in content
    assert "\\bottomrule" in content
    assert "1 & 2 & 3" in content
    assert "4 & 5 & 6" in content


def test_write_table_without_resizebox(tmp_path: Path):
    out = tmp_path / "noresize.tex"
    write_table(out, "l", "A \\\\", ["1 \\\\"], "c", "lbl",
                use_resizebox=False)
    content = out.read_text()
    assert "\\resizebox" not in content


def test_write_table_with_notes(tmp_path: Path):
    out = tmp_path / "notes.tex"
    write_table(out, "l", "A \\\\", ["1 \\\\"], "c", "lbl",
                notes="Means $\\pm$ std over 100 episodes.")
    content = out.read_text()
    assert "Means $\\pm$ std over 100 episodes." in content
