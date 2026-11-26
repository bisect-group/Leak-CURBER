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
    "db-reactions": ("data_and_stats", (32, 197, 605, 712), 420, [(478, 395, 610, 418), (300, 708, 340, 730)]),
    "db-enzymes": ("data_and_stats", (1470, 208, 1962, 722), 360, [(1462, 392, 1486, 414), (1700, 716, 1732, 740)]),
    "db-metabolites": ("data_and_stats", (142, 826, 484, 1340), 260, [(300, 812, 340, 831), (478, 1145, 492, 1165)]),
    "db-properties": ("data_and_stats", (1548, 822, 1898, 1342), 260, [(1700, 805, 1732, 832), (1540, 1145, 1562, 1165)]),
    "db-center": ("data_and_stats", (872, 545, 1117, 915), 200, []),
    # pipeline
    "stack-reactions": ("leak_curber_methodology", (38, 347, 352, 618), 260, []),
    "stack-metabolites": ("leak_curber_methodology", (508, 347, 822, 618), 260, []),
    "stack-proteins": ("leak_curber_methodology", (944, 347, 1240, 622), 260, []),
    "stack-properties": ("leak_curber_methodology", (1336, 347, 1648, 618), 260, []),
    "stack-literature": ("leak_curber_methodology", (1780, 347, 1958, 618), 260, [], (314, 271)),
    # modalities
    "mol-graph": ("multimodal_representations", (470, 205, 828, 394), 340, [(630, 366, 665, 410)]),
    "mol-conformer": ("multimodal_representations", (845, 208, 1189, 386), 340, [(1005, 380, 1035, 392)]),
    "prot-structure": ("multimodal_representations", (1652, 188, 1898, 396), 220, [(1765, 392, 1790, 410)]),
    # tasks
    "task-binding": ("benchmark_tasks", (238, 203, 818, 692), 420, []),
    "task-function": ("benchmark_tasks", (1062, 214, 1978, 605), 560, []),
    "task-ec-chain": ("benchmark_tasks", (1058, 674, 1975, 862), 560, [(1505, 852, 1535, 866)]),
    "task-kinetics": ("benchmark_tasks", (132, 822, 900, 1158), 560, []),
    "task-outcome": ("benchmark_tasks", (28, 1311, 1948, 1465), 900, []),
    # splits
    "split-rxn-atoms": ("leak_control_splits", (118, 55, 1538, 177), 700, []),
    "split-mol-2d": ("leak_control_splits", (52, 182, 508, 530), 260, []),
    "split-mol-3d": ("leak_control_splits", (48, 538, 512, 865), 260, []),
    "split-similarity": ("leak_control_splits", (518, 242, 1125, 845), 300, []),
    "split-protein": ("leak_control_splits", (1122, 250, 1600, 810), 260, []),
    "split-seq-chain": ("leak_control_splits", (48, 873, 1172, 1002), 640, []),
    "split-gr-chains": ("leak_control_splits", (70, 1084, 822, 1255), 420, []),
    "split-gr-blobs": ("leak_control_splits", (72, 1290, 1152, 1460), 600, []),
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
