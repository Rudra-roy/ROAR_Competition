"""tools/site_data.py: the book's code excerpts are pulled from git at the release commit and HTML-escaped, so the page
can never quote code that is not in the repository (or break its own markup with < and & in the quoted lines)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import site_data  # noqa: E402


def _lines(path, a, b):
    out = subprocess.run(["git", "-C", ROOT, "show", "%s:%s" % (site_data.RELEASE, path)], capture_output=True, text=True,
                         check=True).stdout.split("\n")
    return out[a - 1:b]


def test_fill_code_quotes_the_exact_release_lines_escaped(tmp_path):
    page = tmp_path / "p.html"
    page.write_text("<main>\n<!-- code:tools/gate.py:34-36 -->\nSTALE\n<!-- /code -->\n</main>\n")
    assert site_data.fill_code(str(page)) == 1
    html = page.read_text()
    assert "STALE" not in html
    assert "def verdict(rows, wall_min, need_walls=True):" in html
    assert "blob/%s/tools/gate.py#L34-L36" % site_data.RELEASE in html  # links to the commit it quotes
    assert "&quot;" in html or "&lt;" in html or "&gt;" in html or '"""rows' in html  # escaped, never raw markup
    assert all(line.strip() in html.replace("&quot;", '"').replace("&#x27;", "'").replace("&lt;", "<").replace("&gt;", ">")
               .replace("&amp;", "&") for line in _lines("tools/gate.py", 34, 36) if line.strip())


def test_fill_code_is_idempotent(tmp_path):
    page = tmp_path / "p.html"
    page.write_text("<!-- code:competition_code/submission.py:34-34 -->\n<!-- /code -->\n")
    site_data.fill_code(str(page))
    first = page.read_text()
    site_data.fill_code(str(page))
    assert page.read_text() == first
