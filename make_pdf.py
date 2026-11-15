"""
make_pdf.py - Renders report.md to publication-grade PDF using MathJax SVG
and headless Google Chrome over a temporary local HTTP server.
"""

import os
import sys
import time
import threading
import socketserver
import http.server
import subprocess
import markdown
import re

PORT = 8990
DIR = os.path.abspath(os.path.dirname(__file__))

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress server logs

class QuietServer(socketserver.TCPServer):
    allow_reuse_address = True

def generate_pdf():
    # 1. Read Markdown
    md_path = os.path.join(DIR, "report", "report.md")
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Pre-process math so Python markdown doesn't mess up underscores/asterisks inside math
    math_blocks = []
    def save_display_math(match):
        math_blocks.append(match.group(0))
        return f"@@DISPLAY_MATH_{len(math_blocks)-1}@@"

    def save_inline_math(match):
        math_blocks.append(match.group(0))
        return f"@@INLINE_MATH_{len(math_blocks)-1}@@"

    # Match display math $$...$$
    text = re.sub(r"\$\$[\s\S]*?\$\$", save_display_math, text)
    # Match inline math $...$
    text = re.sub(r"(?<!\$)\$([^\$\n]+?)\$(?!\$)", save_inline_math, text)

    # Render markdown to HTML
    html_body = markdown.markdown(text, extensions=["tables", "fenced_code"])

    # Restore math blocks untouched
    for idx, block in enumerate(math_blocks):
        html_body = html_body.replace(f"@@DISPLAY_MATH_{idx}@@", block)
        html_body = html_body.replace(f"@@INLINE_MATH_{idx}@@", block)

    # Fix relative image paths to work over HTTP
    html_body = html_body.replace("../figures/", "/figures/")

    styled_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>N-Queens Local Beam Search Report</title>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  }},
  svg: {{ fontCache: 'local' }}
}};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
@page {{
  size: letter;
  margin: 0.8cm 1.0cm;
}}

body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  color: #1a1e22;
  line-height: 1.30;
  font-size: 8.0pt;
  margin: 0;
  padding: 0;
}}

h1 {{
  font-size: 14pt;
  border-bottom: 1.5px solid #0366d6;
  padding-bottom: 3px;
  margin: 0 0 4px 0;
  color: #0366d6;
}}

h2 {{
  font-size: 10.5pt;
  border-bottom: 1px solid #eaecef;
  padding-bottom: 2px;
  margin: 8px 0 3px 0;
  color: #24292e;
  page-break-after: avoid;
}}

h3 {{
  font-size: 9.0pt;
  margin: 6px 0 2px 0;
  color: #24292e;
  page-break-after: avoid;
}}

h4 {{
  font-size: 8.5pt;
  margin: 5px 0 2px 0;
  color: #444d56;
  page-break-after: avoid;
}}

p, ul, ol {{
  margin: 2px 0 4px 0;
}}

li {{
  margin-bottom: 2px;
}}

table {{
  border-collapse: collapse;
  width: 100%;
  margin: 5px 0;
  font-size: 6.6pt;
  page-break-inside: avoid;
}}

th, td {{
  border: 1px solid #dfe2e5;
  padding: 2.5px 4px;
  text-align: left;
}}

th {{
  background-color: #f6f8fa;
  font-weight: 600;
  color: #24292e;
}}

tr:nth-child(even) {{
  background-color: #fafbfc;
}}

img {{
  max-width: 82%;
  max-height: 138px;
  display: block;
  margin: 3px auto;
  border: 1px solid #e1e4e8;
  border-radius: 4px;
  page-break-inside: avoid;
}}

code {{
  background-color: rgba(27,31,35,0.05);
  border-radius: 2px;
  font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 82%;
  padding: 1px 3px;
}}

hr {{
  border: 0;
  height: 1px;
  background: #e1e4e8;
  margin: 6px 0;
}}

mjx-container[jax="SVG"][display="true"] {{
  margin: 5px 0 !important;
}}

a {{
  color: #0366d6;
  text-decoration: none;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    html_path = os.path.join(DIR, "report", "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(styled_html)

    # 2. Start temporary HTTP server
    os.chdir(DIR)
    httpd = QuietServer(("127.0.0.1", PORT), QuietHandler)
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    time.sleep(0.5)

    # 3. Print to PDF using Chrome
    pdf_path = os.path.join(DIR, "report", "report.pdf")
    cmd = [
        "google-chrome",
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--virtual-time-budget=10000",
        f"--print-to-pdf={pdf_path}",
        f"http://127.0.0.1:{PORT}/report/report.html"
    ]
    subprocess.run(cmd, check=True)
    httpd.shutdown()

    # 4. Check page info
    res = subprocess.run(["pdfinfo", pdf_path], capture_output=True, text=True)
    print("PDF Generated successfully!")
    for line in res.stdout.splitlines():
        if "Pages:" in line or "File size:" in line:
            print(" ", line)

if __name__ == "__main__":
    generate_pdf()
