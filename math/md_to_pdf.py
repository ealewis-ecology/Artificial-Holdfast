"""Render markdown (with $...$ LaTeX math) to PDF via Chrome headless + MathJax.

Why not pandoc? No LaTeX engine on this machine.
Why not weasyprint? It can't run JavaScript, so MathJax wouldn't render.
Chrome headless runs JS, so MathJax types out the math nicely.
"""
import sys
import re
import subprocess
import tempfile
import os
import html as _html

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Minimal markdown → HTML converter that preserves $...$ and $$...$$ math blocks.
# We can't use python-markdown without installing it; this hand-rolled converter
# handles only the subset our docs use: headings, paragraphs, lists, bold/italic,
# inline code, code fences. Math is left untouched for MathJax.

def md_to_html(md):
    # Pull out display math ($$...$$) and inline math ($...$) so they're not
    # mangled by other rules. Replace with placeholders and restore at the end.
    placeholders = []

    def stash(match):
        placeholders.append(match.group(0))
        return f"\x00MATH{len(placeholders) - 1}\x00"

    md = re.sub(r"\$\$.*?\$\$", stash, md, flags=re.DOTALL)
    md = re.sub(r"\$[^$\n]+?\$", stash, md)

    # Code fences first (so heading/list rules don't fire inside them).
    code_blocks = []

    def stash_code(match):
        code_blocks.append(match.group(1))
        return f"\x00CODE{len(code_blocks) - 1}\x00"

    md = re.sub(r"```[a-zA-Z]*\n(.*?)```", stash_code, md, flags=re.DOTALL)

    # Escape HTML in remaining text (math/code already stashed).
    md = _html.escape(md, quote=False)

    lines = md.split("\n")
    out = []
    in_ul = False
    in_para = False

    def close_para():
        nonlocal in_para
        if in_para:
            out.append("</p>")
            in_para = False

    def close_ul():
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    for line in lines:
        stripped = line.rstrip()

        # Horizontal rule
        if re.match(r"^\s*---+\s*$", stripped):
            close_para(); close_ul()
            out.append("<hr>")
            continue

        # Headings
        h = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if h:
            close_para(); close_ul()
            level = len(h.group(1))
            out.append(f"<h{level}>{h.group(2).strip()}</h{level}>")
            continue

        # Unordered list
        li = re.match(r"^\s*[-*]\s+(.*)$", stripped)
        if li:
            close_para()
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{li.group(1)}</li>")
            continue

        if not stripped:
            close_para(); close_ul()
            continue

        # Paragraph
        close_ul()
        if not in_para:
            out.append("<p>")
            in_para = True
        out.append(stripped)

    close_para(); close_ul()
    html_body = "\n".join(out)

    # Inline formatting on the joined body.
    # Bold **x**, italic *x*, inline code `x`. (Math/code already stashed.)
    html_body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html_body)
    html_body = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<em>\1</em>", html_body)
    html_body = re.sub(r"`([^`]+?)`", r"<code>\1</code>", html_body)

    # Restore code blocks.
    def restore_code(match):
        idx = int(match.group(1))
        return f"<pre><code>{_html.escape(code_blocks[idx], quote=False)}</code></pre>"

    html_body = re.sub(r"\x00CODE(\d+)\x00", restore_code, html_body)

    # Restore math placeholders (un-escaped, MathJax handles them).
    def restore_math(match):
        idx = int(match.group(1))
        return placeholders[idx]

    html_body = re.sub(r"\x00MATH(\d+)\x00", restore_math, html_body)

    return html_body


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$']],
    displayMath: [['$$', '$$']],
    processEscapes: true
  }},
  svg: {{ fontCache: 'global' }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
@page {{ size: letter; margin: 1in 1in 1in 1in; }}
body {{
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 11pt;
  line-height: 1.5;
  color: #1a1a1a;
  max-width: 100%;
}}
h1 {{ font-size: 18pt; border-bottom: 1px solid #888; padding-bottom: 0.2em;
      margin-top: 1.4em; }}
h1:first-child {{ margin-top: 0; }}
h2 {{ font-size: 14pt; margin-top: 1.4em; color: #2c3e50; }}
h3 {{ font-size: 12pt; margin-top: 1.2em; color: #34495e; }}
p {{ margin: 0.6em 0; }}
ul {{ margin: 0.5em 0; padding-left: 1.5em; }}
li {{ margin: 0.2em 0; }}
hr {{ border: none; border-top: 1px solid #ccc; margin: 1.5em 0; }}
code {{
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.92em;
  background: #f5f5f5;
  padding: 0.1em 0.3em;
  border-radius: 3px;
}}
pre {{
  background: #f5f5f5;
  padding: 0.7em 1em;
  border-radius: 4px;
  border-left: 3px solid #888;
  overflow-x: auto;
}}
pre code {{ background: none; padding: 0; }}
mjx-container[display="true"] {{
  margin: 0.7em 0 !important;
}}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def md_file_to_html_file(md_path, html_path, title):
    with open(md_path) as f:
        md = f.read()
    body = md_to_html(md)
    html = HTML_TEMPLATE.format(title=title, body=body)
    with open(html_path, "w") as f:
        f.write(html)


def html_to_pdf(html_path, pdf_path):
    """Use Chrome headless to print to PDF.
    --virtual-time-budget gives MathJax time to typeset before printing."""
    url = "file://" + os.path.abspath(html_path)
    cmd = [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=15000",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path}",
        url,
    ]
    subprocess.run(cmd, check=True)


def convert(md_path, pdf_path, title):
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        html_path = tmp.name
    try:
        md_file_to_html_file(md_path, html_path, title)
        html_to_pdf(html_path, pdf_path)
        print(f"  → {pdf_path}")
    finally:
        os.unlink(html_path)


if __name__ == "__main__":
    base = "/Users/elile/Documents/MLML/Thesis/Artificial Holdfast/Artificial-Holdfast"
    convert(
        f"{base}/haptera_math_overview.md",
        f"{base}/haptera_math_overview.pdf",
        "Haptera Mathematics — Overview",
    )
    convert(
        f"{base}/haptera_math_detailed.md",
        f"{base}/haptera_math_detailed.pdf",
        "Haptera Mathematics — Detailed Walk-Through",
    )
