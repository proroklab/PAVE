"""LaTeX table writers — small, composable, with no internal cross-calls.

Each top-level function takes already-aggregated data and emits a
LaTeX fragment. There is no logic that loads CSVs, computes stats, or
calls another function in this module — every routine is independent.

Three groups:

  * cell formatters (``fmt_*``) — float → string
  * row builders (``row_*``)   — tuple-of-cells → one tabular line
  * table writers (``write_*``) — full ``\\begin{table}...\\end{table}``
                                  wrapper, takes already-built rows

``write_table`` also emits an HTML sibling next to each ``.tex`` file
(``foo.tex`` → ``foo.html``) so the same numbers are directly viewable
on the paper website / in a browser without a LaTeX build.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Cell formatters
# ---------------------------------------------------------------------------


def fmt(value: float, digits: int = 2) -> str:
    """Format one float at fixed precision; NaN → ``"--"``."""
    if not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def fmt_pm(mean: float, std: float, digits: int = 2) -> str:
    """Format a ``mean ± std`` cell. NaN std → just the mean."""
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        return fmt(mean, digits)
    return f"{mean:.{digits}f}\\,$\\pm$\\,{std:.{digits}f}"


def fmt_rate(rate: float, digits: int = 2) -> str:
    """Format a [0, 1] success rate as a percentage with no ``%`` glyph."""
    if not np.isfinite(rate):
        return "--"
    return f"{100.0 * rate:.{digits}f}"


def fmt_bold(text: str) -> str:
    """Wrap a pre-formatted cell in ``\\textbf{...}``."""
    return f"\\textbf{{{text}}}"


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def row(cells: Sequence[str]) -> str:
    """Join cells with ``&`` and terminate with ``\\\\``."""
    return " & ".join(cells) + " \\\\"


def bold_max(cells: Sequence[str], values: Sequence[float]) -> list[str]:
    """Return ``cells`` with the entry corresponding to ``max(values)`` bolded.

    Ties are broken by first occurrence. NaN entries are skipped from
    the max computation. ``len(cells)`` must equal ``len(values)``.
    """
    if len(cells) != len(values):
        raise ValueError("cells and values must have the same length")
    arr = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return list(cells)
    max_v = arr[finite_mask].max()
    out: list[str] = []
    bolded = False
    for c, v in zip(cells, arr):
        if not bolded and np.isfinite(v) and v == max_v:
            out.append(fmt_bold(c))
            bolded = True
        else:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Table writer
# ---------------------------------------------------------------------------


def write_table(
    output_path: Path,
    column_spec: str,
    header: str,
    rows: Iterable[str],
    caption: str,
    label: str,
    *,
    use_resizebox: bool = True,
    notes: str = "",
) -> None:
    """Write a complete ``\\begin{table}...\\end{table}`` to disk.

    The body is laid out as::

        \\begin{table}[t]
            \\centering
            \\caption{...} \\label{...}
            [\\resizebox{\\textwidth}{!}{]
                \\begin{tabular}{<column_spec>}
                    \\toprule
                    <header>
                    \\midrule
                    <rows...>
                    \\bottomrule
                \\end{tabular}
            [}]
            [<notes>]
        \\end{table}

    Caller is responsible for any ``\\midrule`` separators within
    ``rows`` (just include them as literal ``"\\midrule"`` strings).

    Also writes a sibling ``.html`` file with the same numbers, rendered
    as a plain HTML ``<table>`` — best-effort LaTeX→HTML conversion of
    each cell (``\\textbf`` → ``<b>``, ``\\texttt`` → ``<code>``,
    ``\\scenario`` → ``<span class="scenario">``, common inline-math
    symbols → HTML entities / sub/sup tags).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)

    pre_resize = "\\resizebox{\\textwidth}{!}{%\n" if use_resizebox else ""
    post_resize = "}\n" if use_resizebox else ""

    body = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{pre_resize}"
        f"\\begin{{tabular}}{{{column_spec}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        + "\n".join(rows_list) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"{post_resize}"
        + (f"\\par\\smallskip\\footnotesize {notes}\n" if notes else "")
        + "\\end{table}\n"
    )
    output_path.write_text(body)

    html_path = output_path.with_suffix(".html")
    html_path.write_text(_render_html_table(
        column_spec=column_spec,
        header=header,
        rows=rows_list,
        caption=caption,
        label=label,
        notes=notes,
    ))


def write_tabular(
    output_path: Path,
    column_spec: str,
    header: str,
    rows: Iterable[str],
) -> None:
    """Write only the ``tabular`` block: no table float, caption, label,
    sizing or notes. The paper wraps it (``\\input``) and owns all of that.

    Also writes a sibling ``.html`` with the same numbers (see
    :func:`write_table`).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    output_path.write_text(
        f"\\begin{{tabular}}{{{column_spec}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        + "\n".join(rows_list) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    output_path.with_suffix(".html").write_text(_render_html_table(
        column_spec=column_spec, header=header, rows=rows_list,
        caption="", label=output_path.stem, notes="",
    ))


# ---------------------------------------------------------------------------
# HTML sibling rendering
# ---------------------------------------------------------------------------


_MATH_SUBS: tuple[tuple[str, str], ...] = (
    (r"\cdot", "·"),
    (r"\pm", "±"),
    (r"\wedge", "∧"),
    (r"\vee", "∨"),
    (r"\land", "∧"),
    (r"\lor", "∨"),
    (r"\alpha", "α"),
    (r"\beta", "β"),
    (r"\gamma", "γ"),
    (r"\delta", "δ"),
    (r"\epsilon", "ε"),
    (r"\theta", "θ"),
    (r"\lambda", "λ"),
    (r"\mu", "μ"),
    (r"\pi", "π"),
    (r"\rho", "ρ"),
    (r"\sigma", "σ"),
    (r"\tau", "τ"),
    (r"\phi", "φ"),
    (r"\psi", "ψ"),
    (r"\omega", "ω"),
    (r"\times", "×"),
    (r"\to", "→"),
    (r"\leq", "≤"),
    (r"\geq", "≥"),
    (r"\neq", "≠"),
    (r"\approx", "≈"),
    (r"\infty", "∞"),
    (r"\!", ""),
    (r"\,", "&nbsp;"),
    (r"\ ", "&nbsp;"),
)


def _math_to_html(math: str) -> str:
    """Render inline math (``$...$`` contents) as loose HTML."""
    m = math
    # Word-like macros need to eat the trailing space LaTeX requires
    # after them (``\wedge F`` in source shouldn't leave ``∧ F`` in the
    # rendered HTML); do these before the literal-replace pass.
    m = re.sub(r"\\(wedge|vee|land|lor)\s?", lambda mm: {
        "wedge": "∧", "land": "∧", "vee": "∨", "lor": "∨",
    }[mm.group(1)], m)
    for tex, sub in _MATH_SUBS:
        m = m.replace(tex, sub)
    m = re.sub(r"_\{([^{}]+)\}", r"<sub>\1</sub>", m)
    m = re.sub(r"_(\w)", r"<sub>\1</sub>", m)
    m = re.sub(r"\^\{([^{}]+)\}", r"<sup>\1</sup>", m)
    m = re.sub(r"\^(\w)", r"<sup>\1</sup>", m)
    return f"<i>{m}</i>"


def _cell_to_html(cell: str) -> str:
    """Best-effort LaTeX-cell → HTML conversion."""
    s = cell
    s = re.sub(r"\\textbf\{([^{}]*)\}", r"<b>\1</b>", s)
    s = re.sub(r"\\texttt\{([^{}]*)\}", r"<code>\1</code>", s)
    s = re.sub(r"\\emph\{([^{}]*)\}", r"<em>\1</em>", s)
    s = re.sub(r"\\scenario\{([^{}]*)\}",
               r'<span class="scenario">\1</span>', s)
    s = re.sub(r"\$([^$]*)\$", lambda m: _math_to_html(m.group(1)), s)
    s = s.replace(r"\_", "_")
    s = s.replace(r"\%", "%")
    s = s.replace(r"\&", "&amp;")
    s = s.replace(r"\!", "")
    s = s.replace(r"\,", "&nbsp;")
    s = s.replace(r"\ ", "&nbsp;")
    return s


def _split_cells(row_str: str) -> list[str]:
    """Split a LaTeX row string back into its cells."""
    r = row_str.rstrip()
    if r.endswith("\\\\"):
        r = r[:-2].rstrip()
    return [c.strip() for c in r.split("&")]


def _count_cols(column_spec: str) -> int:
    """Count l/c/r/p{...} column entries in a booktabs-style spec."""
    n = 0
    i = 0
    while i < len(column_spec):
        c = column_spec[i]
        if c in "lcr":
            n += 1
            i += 1
            continue
        if c == "p" and i + 1 < len(column_spec) and column_spec[i + 1] == "{":
            n += 1
            end = column_spec.find("}", i)
            i = end + 1 if end > 0 else i + 1
            continue
        i += 1
    return n or 1


def _render_html_table(
    *,
    column_spec: str,
    header: str,
    rows: Sequence[str],
    caption: str,
    label: str,
    notes: str,
) -> str:
    n_cols = _count_cols(column_spec)
    header_cells = _split_cells(header)
    thead_row = "<tr>" + "".join(
        f"<th>{_cell_to_html(c)}</th>" for c in header_cells
    ) + "</tr>"
    body_rows: list[str] = []
    for r in rows:
        if r.strip() in ("\\midrule", r"\midrule"):
            body_rows.append(f'<tr><td colspan="{n_cols}"><hr></td></tr>')
            continue
        cells = _split_cells(r)
        body_rows.append(
            "<tr>"
            + "".join(f"<td>{_cell_to_html(c)}</td>" for c in cells)
            + "</tr>"
        )
    parts = [f'<table id="{label}">']
    if caption:
        parts.append(f"<caption>{_cell_to_html(caption)}</caption>")
    parts.append(thead_row)
    parts.extend(body_rows)
    parts.append("</table>")
    if notes:
        parts.append(f"<p>{_cell_to_html(notes)}</p>")
    return "\n".join(parts) + "\n"


__all__ = [
    "bold_max",
    "fmt",
    "fmt_bold",
    "fmt_pm",
    "fmt_rate",
    "row",
    "write_table",
    "write_tabular",
]
