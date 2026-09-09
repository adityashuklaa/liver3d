"""Assemble the single-file review workstation from your own run.

    python scripts/export_viewer_data.py --checkpoint outputs/demo/best.pt --out outputs/viewer
    python scripts/build_viewer.py --data outputs/viewer --run outputs/demo --out viewer.html

Takes the exported study bundle (slice sprites, masks, metrics, meshes) and a
training run directory, inlines everything into web/template.html and writes one
self-contained HTML file: no server, no CDN, no network at all.
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PLOT_L, PLOT_R, PLOT_T, PLOT_B = 44, 306, 14, 136


def read_history(run_dir: str):
    """Epoch records from a training run, if one was given."""
    path = os.path.join(run_dir, "train_log.jsonl") if run_dir else None
    if not path or not os.path.exists(path):
        return []
    history = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                history.append(json.loads(line))
    return history


def chart(values, ymin, ymax, ticks, stroke, mark=None, fmt="{:.2f}") -> str:
    """Small line chart as SVG, drawn against the page's theme tokens."""
    if not values:
        return ('<text x="12" y="80" font-size="11" font-family="var(--mono)" '
                'fill="hsl(var(--muted-foreground))">no training log found</text>')
    n = len(values)
    span = max(1, n - 1)

    def X(i):
        return PLOT_L + (PLOT_R - PLOT_L) * i / span

    def Y(v):
        return PLOT_B - (PLOT_B - PLOT_T) * (v - ymin) / max(1e-6, ymax - ymin)

    out = []
    for tick in ticks:
        y = round(Y(tick), 1)
        out.append(f'<line x1="{PLOT_L}" y1="{y}" x2="{PLOT_R}" y2="{y}" '
                   f'stroke="hsl(var(--border))" stroke-width="1"/>')
        out.append(f'<text x="{PLOT_L - 7}" y="{y + 3.5}" text-anchor="end" font-size="9" '
                   f'font-family="var(--mono)" fill="hsl(var(--muted-foreground))">{fmt.format(tick)}</text>')
    for i in sorted({0, span // 2, span}):
        out.append(f'<text x="{round(X(i), 1)}" y="{PLOT_B + 14}" text-anchor="middle" font-size="9" '
                   f'font-family="var(--mono)" fill="hsl(var(--muted-foreground))">{i + 1}</text>')
    out.append(f'<text x="{(PLOT_L + PLOT_R) / 2}" y="{PLOT_B + 29}" text-anchor="middle" font-size="9" '
               f'font-family="var(--mono)" fill="hsl(var(--muted-foreground))">epoch</text>')

    points = " ".join(f"{round(X(i), 1)},{round(Y(v), 1)}" for i, v in enumerate(values))
    out.append(f'<polygon points="{PLOT_L},{PLOT_B} {points} {PLOT_R},{PLOT_B}" fill="{stroke}" opacity="0.10"/>')
    out.append(f'<polyline points="{points}" fill="none" stroke="{stroke}" stroke-width="1.8" stroke-linejoin="round"/>')
    last = n - 1
    out.append(f'<circle cx="{round(X(last), 1)}" cy="{round(Y(values[last]), 1)}" r="3.4" '
               f'fill="hsl(var(--card))" stroke="{stroke}" stroke-width="2"/>')
    out.append(f'<text x="{PLOT_R}" y="{round(Y(values[last]), 1) - 8}" text-anchor="end" font-size="10.5" '
               f'font-family="var(--mono)" fill="hsl(var(--foreground))">{fmt.format(values[last])}</text>')
    if mark:
        i = mark - 1
        x, y = round(X(i), 1), round(Y(values[i]), 1)
        out.append(f'<circle cx="{x}" cy="{y}" r="5" fill="none" stroke="{stroke}" stroke-width="1.5"/>')
        out.append(f'<text x="{x}" y="{y - 10}" text-anchor="middle" font-size="9" font-family="var(--mono)" '
                   f'fill="hsl(var(--foreground))">selected {fmt.format(values[i])}</text>')
    return "".join(out)


def training_charts(history):
    losses = [float(r["train_loss"]) for r in history if r.get("train_loss") is not None]
    dices = [float(r["val_dice"]) for r in history if r.get("val_dice") is not None]

    if losses:
        lo, hi = min(losses), max(losses)
        pad = max(0.05, (hi - lo) * 0.15)
        ticks = [round(lo - pad + (hi - lo + 2 * pad) * f, 2) for f in (0.0, 0.33, 0.66, 1.0)]
        loss_svg = chart(losses, lo - pad, hi + pad, ticks, "hsl(var(--band-weak))")
    else:
        loss_svg = chart([], 0, 1, [], "")

    if dices:
        lo, hi = min(dices), max(dices)
        pad = max(0.02, (hi - lo) * 0.2)
        ticks = [round(lo - pad + (hi - lo + 2 * pad) * f, 2) for f in (0.0, 0.33, 0.66, 1.0)]
        best = max(range(len(dices)), key=lambda i: dices[i]) + 1
        dice_svg = chart(dices, lo - pad, min(1.0, hi + pad), ticks, "hsl(var(--primary))", mark=best)
    else:
        dice_svg = chart([], 0, 1, [], "")
    return loss_svg, dice_svg


def data_uri(path: str) -> str:
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode("ascii")


def build(data_dir: str, run_dir: str, out_path: str, template_path: str) -> str:
    bundle_path = os.path.join(data_dir, "bundle.json")
    if not os.path.exists(bundle_path):
        raise SystemExit(
            f"{bundle_path} not found. Run scripts/export_viewer_data.py first."
        )
    with open(bundle_path, "r", encoding="utf-8") as fh:
        bundle = json.load(fh)

    images = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.png"))):
        images[os.path.splitext(os.path.basename(path))[0]] = data_uri(path)
    expected = {c["case_id"] + "_" + kind for c in bundle["cases"] for kind in ("ct", "pred", "ref")}
    missing = sorted(expected - set(images))
    if missing:
        raise SystemExit(f"sprite sheets missing for: {', '.join(missing)}")

    loss_svg, dice_svg = training_charts(read_history(run_dir))

    with open(template_path, "r", encoding="utf-8") as fh:
        html = fh.read()
    with open(os.path.join(os.path.dirname(template_path), "app.js"), "r", encoding="utf-8") as fh:
        app_js = fh.read()

    html = html.replace("__BUNDLE__", json.dumps(bundle, separators=(",", ":")))
    html = html.replace("__IMAGES__", json.dumps(images, separators=(",", ":")))
    html = html.replace("__LOSS_SVG__", loss_svg).replace("__DICE_SVG__", dice_svg)
    html = html.replace("__APP_JS__", app_js)
    for token in ("__BUNDLE__", "__IMAGES__", "__APP_JS__", "__LOSS_SVG__", "__DICE_SVG__"):
        if token in html:
            raise SystemExit(f"placeholder {token} was not filled")

    header = '<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(header + html)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the single-file review workstation")
    parser.add_argument("--data", default="outputs/viewer", help="output of export_viewer_data.py")
    parser.add_argument("--run", default="outputs/demo", help="training run directory (for the curves)")
    parser.add_argument("--out", default="viewer.html")
    parser.add_argument("--template", default=os.path.join(ROOT, "web", "template.html"))
    args = parser.parse_args()

    path = build(args.data, args.run, args.out, args.template)
    size_mb = os.path.getsize(path) / 1048576
    print(f"wrote {os.path.abspath(path)} ({size_mb:.2f} MB, self-contained)")
    print("open it directly, or serve it for the upload tab:")
    print("  python -m http.server 8099   ->  http://127.0.0.1:8099/" + os.path.basename(path))


if __name__ == "__main__":
    main()
