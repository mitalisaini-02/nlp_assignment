"""Render the Mermaid diagrams (Figures 1-4) in REPORT.md Appendix A to PNG, via headless Chrome + mermaid.js (CDN).

Each diagram is printed to a one-page PDF, then cropped to its content and rasterised with PyMuPDF.
Run:  C:/nlp_data/venv/Scripts/python.exe -I src/make_diagrams.py      (needs internet for mermaid.js)
"""
import os, re, subprocess, tempfile
import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
NAMES = ["fig1_architecture", "fig2_timeline", "fig3_training_example", "fig5_reranker"]
INIT = """{"theme":"base","themeVariables":{"primaryColor":"#f4f4f1","primaryBorderColor":"#52514e","primaryTextColor":"#0b0b0b",
 "lineColor":"#52514e","secondaryColor":"#ffffff","tertiaryColor":"#ffffff","clusterBkg":"#fafaf8","clusterBorder":"#a3a29b",
 "edgeLabelBackground":"#ffffff","fontFamily":"Arial, Helvetica, sans-serif","fontSize":"15px"},
 "htmlLabels":false,"flowchart":{"htmlLabels":false,"curve":"basis","nodeSpacing":40,"rankSpacing":45,"wrappingWidth":320,"padding":18}}"""

src = open(os.path.join(ROOT, "REPORT.md"), encoding="utf-8").read()
blocks = re.findall(r"```mermaid\n(.*?)```", src, re.S)
assert len(blocks) == 4, len(blocks)
tmp = tempfile.mkdtemp()
for name, code in zip(NAMES, blocks):
    html = os.path.join(tmp, name + ".html")
    pdf = os.path.join(tmp, name + ".pdf")
    open(html, "w", encoding="utf-8").write(
        "<!DOCTYPE html><html><head><meta charset='utf-8'><style>@page{size:1600px 1400px;margin:0}body{margin:20px;background:#fff;font-family:Arial,Helvetica,sans-serif}</style>"
        "<script src='https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js'></script></head><body>"
        "<div class='mermaid'>%%%%{init: %s}%%%%\n%s</div><script>mermaid.initialize({startOnLoad:true});</script></body></html>"
        % (INIT.replace("\n", ""), code))
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
                    "--print-to-pdf=" + pdf, "file:///" + html.replace("\\", "/")], capture_output=True, timeout=120)
    page = pymupdf.open(pdf)[0]
    big = page.rect.width * page.rect.height * 0.5     # skip the page-background rectangle
    boxes = [d["rect"] for d in page.get_drawings() if d["rect"].width * d["rect"].height < big]
    boxes += [pymupdf.Rect(b[:4]) for b in page.get_text("blocks")]
    assert boxes, name + ": nothing rendered (mermaid.js not loaded?)"
    r = boxes[0]
    for b in boxes[1:]:
        r |= b
    r = pymupdf.Rect(r.x0 - 8, r.y0 - 8, r.x1 + 8, r.y1 + 8) & page.rect
    page.get_pixmap(dpi=220, clip=r).save(os.path.join(ROOT, "figures", name + ".png"))
    print("wrote", name, "%.0fx%.0f pt" % (r.width, r.height))
