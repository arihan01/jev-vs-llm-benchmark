"""report.md -> report.pdf without an external converter. Landscape pages,
small fixed-layout tables that wrap instead of overflowing, collapsed
sections expanded, table headers repeated across page breaks."""

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

CSS = """
@page { size: A4 landscape; margin: 12mm 10mm 14mm 10mm;
        @bottom-right { content: counter(page) " / " counter(pages); font-size: 7pt; color: #777; } }
html { font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 9pt; color: #222; }
h1 { font-size: 17pt; margin: 0 0 6pt; }
h2 { font-size: 13pt; margin: 14pt 0 5pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 10pt 0 4pt; page-break-after: avoid; }
h4 { font-size: 9.5pt; margin: 8pt 0 3pt; page-break-after: avoid; }
p, li { line-height: 1.35; }
code { font-family: Menlo, Consolas, monospace; font-size: 0.9em; background: #f3f3f3; padding: 0 2px;
       border-radius: 2px; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font-family: Menlo, Consolas, monospace; font-size: 6.8pt;
      line-height: 1.3; background: #f7f7f7; border: 1px solid #e3e3e3; padding: 4pt 6pt; margin: 3pt 0 8pt; }
pre code { background: none; padding: 0; font-size: inherit; }
blockquote { margin: 3pt 0 6pt; padding: 3pt 8pt; border-left: 3px solid #ccc; color: #444; background: #fafafa; }
table { border-collapse: collapse; width: 100%; table-layout: fixed; margin: 4pt 0 8pt; font-size: 7.4pt; }
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 2px 4px; vertical-align: top; text-align: left;
         overflow-wrap: anywhere; word-break: break-word; hyphens: auto; }
th { background: #eee; font-weight: 600; }
td code, th code { background: none; padding: 0; font-size: 0.95em; }
table.wide { font-size: 6.2pt; }
table.wide th, table.wide td { padding: 1px 2px; }
hr { border: 0; border-top: 1px solid #ddd; }
"""

WIDE_TABLE_COLUMNS = 12  # tables with more columns than this get the smaller font


def _import():
    try:
        import markdown
        import weasyprint
    except ImportError as e:  # pragma: no cover - exercised only without the pdf extra
        raise SystemExit(
            f"PDF export needs the 'pdf' extra ({e.name} missing): pip install -e '.[pdf]'  "
            "(WeasyPrint also needs Pango; on macOS: brew install pango)"
        ) from e
    return markdown, weasyprint


def to_html(md_text: str) -> str:
    markdown, _ = _import()
    # md_in_html only parses Markdown inside a raw HTML block when it is marked;
    # GitHub needs no marker, so the .md stays clean and the hint is added here.
    src = md_text.replace("<details>", '<details markdown="1">')
    body = markdown.markdown(src, extensions=["tables", "fenced_code", "md_in_html", "sane_lists"])
    body = body.replace("<details>", "<details open>")
    body = re.sub(r"<table>(\s*<thead>\s*<tr>((?:\s*<th[^>]*>.*?</th>)+))", _mark_wide, body, flags=re.S)
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"


def _mark_wide(m: re.Match) -> str:
    n = len(re.findall(r"<th[^>]*>", m.group(2)))
    return ("<table class='wide'>" if n > WIDE_TABLE_COLUMNS else "<table>") + m.group(1)


def write_pdf(md_path: Path, pdf_path: Path | None = None) -> Path:
    _, weasyprint = _import()
    pdf_path = pdf_path or md_path.with_suffix(".pdf")
    html = to_html(md_path.read_text())
    weasyprint.HTML(string=html, base_url=str(md_path.parent)).write_pdf(str(pdf_path))
    log.info("wrote %s", pdf_path)
    return pdf_path
