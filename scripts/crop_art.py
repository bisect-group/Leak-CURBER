"""Crop the illustration icons from the WSAI poster figures into webp files.

Run: python scripts/crop_art.py   (needs Pillow)
Boxes are measured on a 2000 px wide preview of each figure and scaled to the
full-size source. Each icon is resized to at most its max width.
"""
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SRC = Path.home() / "github/WSAI-Showcase-2026/figures"
OUT = Path(__file__).resolve().parent.parent / "static/img/art"
PREVIEW_W = 2000

# name: (source, (x0, y0, x1, y1) on the 2000 px preview, max width, [rects to clear],
#        optional (w, h) preview canvas to centre the crop on, so it matches its siblings)
CROPS = {
    # data_and_stats
    "db-reactions": ("data_and_stats", (40, 205, 595, 705), 420, [(482, 395, 600, 418)]),
    "db-enzymes": ("data_and_stats", (1480, 215, 1955, 715), 360, [(1480, 395, 1500, 418)]),
    "db-metabolites": ("data_and_stats", (150, 836, 476, 1335), 260, []),
    "db-properties": ("data_and_stats", (1558, 830, 1888, 1335), 260, []),
    "db-center": ("data_and_stats", (880, 565, 1115, 910), 200, []),
    # pipeline
    "stack-reactions": ("leak_curber_methodology", (45, 355, 345, 610), 260, []),
    "stack-metabolites": ("leak_curber_methodology", (515, 355, 815, 610), 260, []),
    "stack-proteins": ("leak_curber_methodology", (945, 355, 1236, 610), 260, []),
    "stack-properties": ("leak_curber_methodology", (1340, 355, 1640, 610), 260, []),
    "stack-literature": ("leak_curber_methodology", (1785, 355, 1950, 610), 260, [], (300, 255)),
    # modalities
    "mol-graph": ("multimodal_representations", (475, 210, 820, 390), 340, [(630, 366, 665, 390)]),
    "mol-conformer": ("multimodal_representations", (850, 215, 1190, 378), 340, []),
    "prot-structure": ("multimodal_representations", (1665, 195, 1890, 392), 220, []),
    # tasks
    "task-binding": ("benchmark_tasks", (245, 210, 810, 685), 420, []),
    "task-function": ("benchmark_tasks", (1070, 220, 1970, 598), 560, []),
    "task-ec-chain": ("benchmark_tasks", (1065, 680, 1970, 858), 560, []),
    "task-kinetics": ("benchmark_tasks", (140, 830, 890, 1150), 560, []),
    "task-outcome": ("benchmark_tasks", (35, 1318, 1940, 1458), 900, []),
    # splits
    "split-rxn-atoms": ("leak_control_splits", (125, 62, 1530, 170), 700, []),
    "split-mol-2d": ("leak_control_splits", (60, 190, 500, 525), 260, []),
    "split-mol-3d": ("leak_control_splits", (55, 545, 505, 858), 260, []),
    "split-similarity": ("leak_control_splits", (525, 248, 1118, 838), 300, []),
    "split-protein": ("leak_control_splits", (1130, 258, 1592, 802), 260, []),
    "split-seq-chain": ("leak_control_splits", (55, 880, 1165, 995), 640, []),
    "split-gr-chains": ("leak_control_splits", (78, 1092, 805, 1248), 420, []),
    "split-gr-blobs": ("leak_control_splits", (80, 1298, 1145, 1452), 600, []),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache: dict[str, Image.Image] = {}
    for name, (src, box, maxw, clear, *pad) in CROPS.items():
        if src not in cache:
            cache[src] = Image.open(SRC / f"{src}.png").convert("RGBA")
        img = cache[src]
        k = img.width / PREVIEW_W
        crop = img.crop(tuple(round(v * k) for v in box))
        for r in clear:
            x0, y0, x1, y1 = (round(v * k) for v in r)
            ox, oy = round(box[0] * k), round(box[1] * k)
            crop.paste((0, 0, 0, 0), (x0 - ox, y0 - oy, x1 - ox, y1 - oy))
        if pad:
            canvas = Image.new("RGBA", tuple(round(v * k) for v in pad[0]), (0, 0, 0, 0))
            canvas.paste(crop, ((canvas.width - crop.width) // 2, (canvas.height - crop.height) // 2))
            crop = canvas
        if crop.width > maxw:
            crop = crop.resize((maxw, round(crop.height * maxw / crop.width)), Image.LANCZOS)
        crop.save(OUT / f"{name}.webp", quality=88, method=6)
        print(f"{name}.webp {crop.size}")


if __name__ == "__main__":
    main()
