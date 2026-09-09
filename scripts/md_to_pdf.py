"""Render a project Markdown document to a print-ready PDF.

    python scripts/md_to_pdf.py docs/SETUP.md --out docs/SETUP.pdf

Converts the subset of Markdown these docs use (headings, fenced code, tables,
lists, blockquotes, rules, links, bold and inline code) into a styled HTML page,
then prints it with headless Chrome or Edge - both ship with Windows/macOS
installs, so there is no extra Python dependency.
"""
from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]

CSS = """
@page { size: A4; margin: 16mm 14mm 18mm; }
:root {
  --ink: #16191c; --ink-2: #545c62; --line: #d9dee2; --line-soft: #eef1f3;
  --accent: #0d5c68; --code-bg: #f4f6f7; --warn: #8a5a10;
}
* { box-sizing: border-box; }
body {
  margin: 0; color: var(--ink); background: #fff;
  font: 10.5pt/1.55 "Segoe UI", system-ui, -apple-system, sans-serif;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
.cover { padding: 26mm 0 10mm; border-bottom: 2px solid var(--accent); margin-bottom: 12mm; }
.cover .kicker { font: 600 9pt/1 "Segoe UI", sans-serif; letter-spacing: .16em; text-transform: uppercase; color: var(--accent); }
.cover h1 { font-size: 25pt; line-height: 1.12; margin: 10px 0 8px; letter-spacing: -.02em; }
.cover p { margin: 0; color: var(--ink-2); font-size: 11pt; max-width: 150mm; }
.cover .facts { margin-top: 14px; font-family: "Consolas", ui-monospace, monospace; font-size: 8.5pt; color: var(--ink-2); }
h1 { font-size: 17pt; margin: 0 0 8px; letter-spacing: -.01em; }
h2 {
  font-size: 13.5pt; margin: 20px 0 8px; padding-top: 10px; letter-spacing: -.01em;
  border-top: 1px solid var(--line); break-after: avoid; break-inside: avoid;
}
h3 { font-size: 11pt; margin: 14px 0 6px; break-after: avoid; }
p { margin: 0 0 8px; }
ul, ol { margin: 0 0 10px; padding-left: 18px; }
li { margin-bottom: 3px; }
a { color: var(--accent); text-decoration: none; word-break: break-all; }
strong { font-weight: 600; }
hr { border: none; border-top: 1px solid var(--line); margin: 16px 0; }
blockquote {
  margin: 10px 0; padding: 8px 12px; border-left: 3px solid var(--accent);
  background: #f2f7f8; color: var(--ink-2); break-inside: avoid;
}
blockquote p { margin: 0; }
code {
  font-family: "Consolas", ui-monospace, "SF Mono", monospace; font-size: 9pt;
  background: var(--code-bg); padding: 1px 4px; border-radius: 3px;
}
pre {
  background: var(--code-bg); border: 1px solid var(--line); border-left: 3px solid var(--accent);
  border-radius: 4px; padding: 9px 12px; margin: 0 0 10px; overflow: hidden;
  break-inside: avoid; white-space: pre-wrap; word-break: break-word;
}
pre code { background: none; padding: 0; font-size: 8.6pt; line-height: 1.5; }
table {
  width: 100%; border-collapse: collapse; margin: 0 0 12px; font-size: 9pt;
  break-inside: avoid;
}
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { background: var(--line-soft); font-weight: 600; font-size: 8.5pt; letter-spacing: .03em; text-transform: uppercase; color: var(--ink-2); }
td code { font-size: 8.4pt; }
tr { break-inside: avoid; }
.note { break-inside: avoid; }
"""


def find_browser() -> str:
    for candidate in BROWSERS:
        if os.path.exists(candidate):
            return candidate
    for name in ("chrome", "chromium", "msedge", "google-chrome"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("no Chrome or Edge found - install one, or print the HTML by hand")


def inline(text: str) -> str:
    """Inline markdown -> HTML, code spans protected from other rules."""
    spans: list[str] = []

    def stash(match: re.Match) -> str:
        spans.append(html.escape(match.group(1)))
        return f"\x00{len(spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"(?<![\">=])(https?://[^\s<)]+)", r'<a href="\1">\1</a>', text)
    for index, span in enumerate(spans):
        text = text.replace(f"\x00{index}\x00", f"<code>{span}</code>")
    return text


def convert(markdown: str) -> str:
    lines = markdown.split("\n")
    out: list[str] = []
    index, total = 0, len(lines)

    while index < total:
        line = lines[index]

        if line.startswith("```"):
            index += 1
            block = []
            while index < total and not lines[index].startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1
            out.append("<pre><code>" + html.escape("\n".join(block)) + "</code></pre>")
            continue

        if re.match(r"^\s*\|.+\|\s*$", line) and index + 1 < total and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[index + 1]):
            def cells(row: str) -> list[str]:
                return [c.strip() for c in row.strip().strip("|").split("|")]

            header = cells(line)
            index += 2
            body = []
            while index < total and re.match(r"^\s*\|.+\|\s*$", lines[index]):
                body.append(cells(lines[index]))
                index += 1
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in header) + "</tr></thead><tbody>")
            for row in body:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
            out.append("</tbody></table>")
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            out.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            index += 1
            continue

        if re.match(r"^\s*(---|___|\*\*\*)\s*$", line):
            out.append("<hr>")
            index += 1
            continue

        if line.startswith(">"):
            block = []
            while index < total and lines[index].startswith(">"):
                block.append(lines[index].lstrip("> ").rstrip())
                index += 1
            out.append("<blockquote><p>" + inline(" ".join(block)) + "</p></blockquote>")
            continue

        bullet = re.match(r"^\s*([-*+]|\d+\.)\s+", line)
        if bullet:
            ordered = bool(re.match(r"^\s*\d+\.", line))
            tag = "ol" if ordered else "ul"
            items: list[str] = []
            while index < total and re.match(r"^\s*([-*+]|\d+\.)\s+", lines[index]):
                items.append(re.sub(r"^\s*([-*+]|\d+\.)\s+", "", lines[index]).rstrip())
                index += 1
                while index < total and lines[index].startswith("  ") and lines[index].strip():
                    items[-1] += " " + lines[index].strip()
                    index += 1
            out.append(f"<{tag}>" + "".join(f"<li>{inline(i)}</li>" for i in items) + f"</{tag}>")
            continue

        if not line.strip():
            index += 1
            continue

        paragraph = [line.rstrip()]
        index += 1
        while index < total and lines[index].strip() and not re.match(r"^(#{1,4}\s|```|\s*\||>|\s*([-*+]|\d+\.)\s|---)", lines[index]):
            paragraph.append(lines[index].rstrip())
            index += 1
        out.append("<p>" + inline(" ".join(paragraph)) + "</p>")

    rendered = "\n".join(out)
    # section headings carry their own rule, so a preceding --- would double it
    return re.sub(r"<hr>\s*(?=<h2>)", "", rendered)


def build_html(markdown: str, title: str, subtitle: str, facts: str) -> str:
    body = convert(markdown)
    # the document's own H1 becomes the cover title
    body = re.sub(r"^<h1>.*?</h1>\s*", "", body, count=1, flags=re.S)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{CSS}</style></head>
<body>
<header class="cover">
  <div class="kicker">3D Liver Segmentation &middot; 3D U-Net</div>
  <h1>{html.escape(title)}</h1>
  <p>{html.escape(subtitle)}</p>
  <div class="facts">{facts}</div>
</header>
{body}
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Markdown doc to PDF")
    parser.add_argument("source", nargs="?", default="docs/SETUP.md")
    parser.add_argument("--out", default=None)
    parser.add_argument("--title", default="Setup guide: clone to running")
    parser.add_argument(
        "--subtitle",
        default="Every command verified on a clean clone. No dataset, no GPU and no accounts required.",
    )
    parser.add_argument("--keep-html", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        raise SystemExit(f"{args.source} not found")
    out_pdf = os.path.abspath(args.out or os.path.splitext(args.source)[0] + ".pdf")

    with open(args.source, "r", encoding="utf-8") as fh:
        markdown = fh.read()

    facts = "github.com/adityashuklaa/liver3d &nbsp;&middot;&nbsp; python 3.10-3.12 &nbsp;&middot;&nbsp; 45 tests &nbsp;&middot;&nbsp; research prototype, not a medical device"
    document = build_html(markdown, args.title, args.subtitle, facts)

    html_path = (os.path.splitext(out_pdf)[0] + ".html") if args.keep_html else os.path.join(
        tempfile.mkdtemp(prefix="liver3d_pdf_"), "document.html"
    )
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(document)

    browser = find_browser()
    profile = tempfile.mkdtemp(prefix="liver3d_profile_")
    command = [
        browser,
        "--headless=new",
        "--disable-gpu",
        f"--user-data-dir={profile}",
        "--no-pdf-header-footer",
        f"--print-to-pdf={out_pdf}",
        "file:///" + html_path.replace("\\", "/"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if not os.path.exists(out_pdf):
        sys.stderr.write(result.stderr[-2000:] + "\n")
        raise SystemExit("PDF was not produced")

    print(f"wrote {out_pdf} ({os.path.getsize(out_pdf) / 1024:.0f} KB) using {os.path.basename(browser)}")
    if args.keep_html:
        print(f"kept {html_path}")


if __name__ == "__main__":
    main()
