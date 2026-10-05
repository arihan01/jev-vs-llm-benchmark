import pytest

from bench import export_pdf

pytest.importorskip("weasyprint")
pytest.importorskip("markdown")

MD = """# Title

<details><summary>More</summary>

| a | b |
|---|---|
| 1 | x\\|y |

</details>

| c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10 | c11 | c12 | c13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
"""


def test_html_expands_details_and_marks_wide_tables():
    html = export_pdf.to_html(MD)
    assert "<details open>" in html
    assert html.count("<table class='wide'>") == 1 and html.count("<table>") == 1
    assert "x|y" in html and "<style>" in html


def test_write_pdf_produces_a_pdf(tmp_path):
    md = tmp_path / "report.md"
    md.write_text(MD)
    pdf = export_pdf.write_pdf(md)
    assert pdf == tmp_path / "report.pdf" and pdf.read_bytes()[:5] == b"%PDF-"
