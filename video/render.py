#!/usr/bin/env python3
"""Render ClauseBench 30s demo video frames (1280x720, 30fps, 900 frames).

All on-screen numbers are verbatim from DEMO_TEST.md / FIXPASS_REPORT.md.
Sparklines use the real percentiles.json data (same code as app.py).
"""
import json, os, shutil, subprocess, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.expanduser("~/workspace/build/clausebench")
VID = os.path.join(ROOT, "video")
OUT = os.path.join(ROOT, "clausebench_demo.mp4")
W, H = 1280, 720

# --- verbatim strings ------------------------------------------------------
TERM_CLAUSE = ("Either party may terminate this Agreement upon thirty (30) "
               "days' prior written notice to the other party.")
GOV_CLAUSE = ("This Agreement shall be governed by and construed in accordance "
              "with the laws of the State of Delaware.")

TERM_HEADLINE = ("Your 30-day termination notice is longer than 54.1% of 883 "
                 "comparable notice clauses.")
TERM_FINE = ("Sample: n=883 comparable notice clauses with a stated notice "
             "period (CUAD+LEDGAR). Vertical line marks your clause.")

GOV_HEADLINE = ("Delaware is the governing law in 16.5% of 3,934 comparable "
                "provisions (rank #2 after New York).")
GOV_FINE = ("Sample: n=3,934 comparable provisions with an extracted "
            "governing law (CUAD+LEDGAR).")

BIAS_NOTE = ("Corpus bias: CUAD = 510 US commercial contracts, EDGAR-sourced, "
             "English only; LEDGAR = SEC-filed provisions. US-centric; not "
             "representative of non-US or consumer contracts.")
DISCLAIMER = "For information only. Not legal advice."

# colors (match app.py)
BG = "#f7f8fa"; CARD = "#ffffff"; BORDER = "#d7dce3"
HEAD = "#1a2332"; SUB = "#4b5563"; FINE = "#6b7280"; DISC = "#9ca3af"
ACCENT = "#1d4ed8"; BAR = "#93b4d8"; BTN = "#f97316"

FD = "/usr/share/fonts/truetype/dejavu"
def font(name, size):
    return ImageFont.truetype(os.path.join(FD, name), size)

def wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= max_w:
            cur = t
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines

def text_block(draw, cx, y, lines, fnt, fill, lh):
    for ln in lines:
        tw = draw.textlength(ln, font=fnt)
        draw.text((cx - tw / 2, y), ln, font=fnt, fill=fill)
        y += lh
    return y

# --- sparklines (same logic/colors as app.py, higher res) --------------------
def make_sparklines():
    dd = json.load(open(os.path.join(ROOT, "data/percentiles.json")))["dimensions"]
    n = dd["notice_days"]; vals = n["values"]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.hist(vals, bins=10, color=BAR, edgecolor="#ffffff")
    ax.axvline(30, color=ACCENT, linewidth=3)
    ax.set_xlabel("Notice days (corpus)", fontsize=10)
    ax.set_ylabel("Clauses", fontsize=10)
    ax.tick_params(labelsize=9)
    fig.tight_layout(pad=0.6)
    fig.savefig(os.path.join(VID, "spark_term.png"), dpi=130)
    plt.close(fig)

    counts = dd["governing_law"]["counts"]
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    labels = [k.replace("State of ", "") for k, _ in ranked]
    vals5 = [c for _, c in ranked]
    cols = [ACCENT if k in ("Delaware", "State of Delaware") else BAR for k, _ in ranked]
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.barh(labels[::-1], vals5[::-1], color=cols[::-1], edgecolor="#ffffff")
    ax.set_xlabel("Provisions", fontsize=10)
    ax.tick_params(labelsize=10)
    fig.tight_layout(pad=0.6)
    fig.savefig(os.path.join(VID, "spark_gov.png"), dpi=130)
    plt.close(fig)

# --- card frame --------------------------------------------------------------
def card_frame(headline, spark_path, fine_lines, out_path, card_w=940):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    pad = 30
    spark = Image.open(spark_path)
    sw, sh = spark.size
    target_w = card_w - 2 * pad
    sh = int(sh * target_w / sw); sw = target_w
    spark = spark.resize((sw, sh))
    fh = font("DejaVuSans-Bold.ttf", 30); ff = font("DejaVuSans.ttf", 18)
    fi = font("DejaVuSans.ttf", 18)
    hl = wrap(d, headline, fh, card_w - 2 * pad)
    fl = []
    for t in fine_lines:
        fl += wrap(d, t, ff, card_w - 2 * pad)
    dl = wrap(d, DISCLAIMER, fi, card_w - 2 * pad)
    body_h = len(hl) * 40 + 14 + sh + 14 + len(fl) * 26 + 10 + len(dl) * 26
    card_h = body_h + 2 * pad
    x0, y0 = (W - card_w) // 2, int((H - card_h) // 2)
    d.rounded_rectangle([x0, y0, x0 + card_w, y0 + card_h], radius=14,
                        fill=CARD, outline=BORDER, width=2)
    y = y0 + pad
    for ln in hl:
        tw = d.textlength(ln, font=fh)
        d.text(((W - tw) / 2, y), ln, font=fh, fill=HEAD); y += 40
    y += 14
    img.paste(spark, ((W - sw) // 2, y)); y += sh + 14
    for ln in fl:
        tw = d.textlength(ln, font=ff)
        d.text(((W - tw) / 2, y), ln, font=ff, fill=FINE); y += 26
    y += 10
    for ln in dl:
        tw = d.textlength(ln, font=fi)
        d.text(((W - tw) / 2, y), ln, font=fi, fill=DISC); y += 26
    img.save(out_path)

# --- scenes ------------------------------------------------------------------
def scene1():
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    y = text_block(d, W/2, 225, ["ClauseBench"],
                   font("DejaVuSans-Bold.ttf", 72), HEAD, 90)
    y = text_block(d, W/2, y + 18, ["Is your clause normal?"],
                   font("DejaVuSans.ttf", 38), SUB, 50)
    text_block(d, W/2, y + 26,
               ["Free percentile cards for contract clauses, from public corpora"],
               font("DejaVuSans.ttf", 24), FINE, 34)
    img.save(os.path.join(VID, "scene1.png"))

def scene2():
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    pw, ph = 1000, 590
    x0, y0 = (W - pw) // 2, (H - ph) // 2
    d.rounded_rectangle([x0, y0, x0 + pw, y0 + ph], radius=14,
                        fill=CARD, outline=BORDER, width=2)
    d.text((x0 + 40, y0 + 30), "ClauseBench", font=font("DejaVuSans-Bold.ttf", 26), fill=HEAD)
    d.text((x0 + 40, y0 + 68), "Free percentile cards for contract clauses",
           font=font("DejaVuSans.ttf", 16), fill=FINE)
    d.text((x0 + 40, y0 + 120), "Contract clause",
           font=font("DejaVuSans.ttf", 17), fill=SUB)
    tx0, ty0, tx1, ty1 = x0 + 40, y0 + 152, x0 + pw - 40, y0 + 330
    d.rounded_rectangle([tx0, ty0, tx1, ty1], radius=8, fill="#f3f4f6",
                        outline="#d1d5db", width=1)
    y = ty0 + 18
    for ln in wrap(d, TERM_CLAUSE, font("DejaVuSans.ttf", 22), tx1 - tx0 - 36):
        d.text((tx0 + 18, y), ln, font=font("DejaVuSans.ttf", 22), fill=HEAD); y += 34
    bw, bh = 300, 64
    bx0, by0 = x0 + 40, y0 + 368
    d.rounded_rectangle([bx0, by0, bx0 + bw, by0 + bh], radius=10, fill=BTN)
    bt = "Benchmark this clause"; bf = font("DejaVuSans-Bold.ttf", 20)
    d.text((bx0 + (bw - d.textlength(bt, font=bf)) / 2, by0 + 19), bt, font=bf, fill="white")
    img.save(os.path.join(VID, "scene2.png"))

def scene5():
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    y = text_block(d, W/2, 170, ["Free. No signup. Open source."],
                   font("DejaVuSans-Bold.ttf", 48), HEAD, 62)
    y = text_block(d, W/2, y + 22, ["Decision support, not legal advice."],
                   font("DejaVuSans.ttf", 30), SUB, 42)
    y += 30
    d.line([W/2 - 260, y, W/2 + 260, y], fill=BORDER, width=2)
    y += 26
    y = text_block(d, W/2, y,
                   ["Measured on a 200-row hand-labeled gold sample:"],
                   font("DejaVuSans.ttf", 22), FINE, 32)
    y = text_block(d, W/2, y + 8,
                   ["Classifier accuracy 0.7222",
                    "Extraction misses on stated clauses: 1/75 = 0.013"],
                   font("DejaVuSans-Bold.ttf", 28), HEAD, 40)
    text_block(d, W/2, y + 16, ["Kill gate: 0.013 vs 0.30  ->  PASS"],
               font("DejaVuSans.ttf", 24), SUB, 36)
    img.save(os.path.join(VID, "scene5.png"))

def main():
    os.makedirs(VID, exist_ok=True)
    make_sparklines()
    scene1(); scene2()
    card_frame(TERM_HEADLINE, os.path.join(VID, "spark_term.png"),
               [TERM_FINE, BIAS_NOTE], os.path.join(VID, "scene3.png"))
    card_frame(GOV_HEADLINE, os.path.join(VID, "spark_gov.png"),
               [GOV_FINE, BIAS_NOTE], os.path.join(VID, "scene4.png"))
    scene5()
    # 900 frames: S1 0-119, S2 120-299, S3 300-539, S4 540-749, S5 750-899
    spans = [(0, 120, "scene1.png"), (120, 300, "scene2.png"),
             (300, 540, "scene3.png"), (540, 750, "scene4.png"),
             (750, 900, "scene5.png")]
    for start, end, sc in spans:
        src = os.path.join(VID, sc)
        for i in range(start, end):
            shutil.copyfile(src, os.path.join(VID, "frame_%03d.png" % i))
    subprocess.run(
        ["/usr/bin/ffmpeg", "-y", "-framerate", "30", "-i",
         os.path.join(VID, "frame_%03d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", "30", OUT],
        check=True, capture_output=True, text=True)
    print("WROTE", OUT, os.path.getsize(OUT), "bytes")

if __name__ == "__main__":
    main()
