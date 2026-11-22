"""Read the plotted curves of two published figures from the vector files in the papers' arXiv sources.

  osborne-fig3   bubble_profile.pdf in the source of arXiv:1305.1970 (Osborne, Senatore, Smith; Fig. 3)
  feeney-fig6    transferfunctions_l20.pdf in the source of arXiv:1506.01716 (Feeney, Elsner, Johnson,
                 Peiris; Fig. 6)

The PDF content stream is decompressed and its path operators are read; nothing is rendered or executed. The
axes are calibrated from the tick marks drawn in the same file (their labels are checked by eye once: 0, 5,
10, 15 deg and 0.0 ... 1.0 for the first figure; 13400 ... 14200 Mpc and 0, 1, 2 for the second).

Usage:
  python scripts/analysis/read_published_figure_curves.py osborne-fig3 bubble_profile.pdf out.csv
  python scripts/analysis/read_published_figure_curves.py feeney-fig6 transferfunctions_l20.pdf out.csv
"""
import re
import sys
import zlib

NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
TOKEN = re.compile(r"\((?:\\.|[^\\)])*\)|<[0-9A-Fa-f\s]*>|\[|\]|/[^\s/\[\]()<>]+|" + NUM + r"|[A-Za-z'\"*]+")


def content_stream(path):
    """The first compressed stream of the file that holds path operators."""
    raw = open(path, "rb").read()
    for m in re.finditer(rb"stream\r?\n", raw):
        blob = raw[m.end():raw.find(b"endstream", m.end())]
        try:
            data = zlib.decompress(blob)
        except zlib.error:
            continue
        if len(re.findall(rb"[\d.\-]+ [\d.\-]+ l\b", data)) > 50:
            return data.decode("latin-1")
    raise ValueError("no drawing stream found in " + path)


def mat_mul(a, b):
    return [a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3], a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3],
            a[4] * b[0] + a[5] * b[2] + b[4], a[4] * b[1] + a[5] * b[3] + b[5]]


def stroked_paths(src):
    """Every painted subpath as {op, stroke colour, dash, points in page coordinates}."""
    state = {"ctm": [1, 0, 0, 1, 0, 0], "stroke": None, "dash": None}
    stack, operands, paths, current, subpath, in_array = [], [], [], [], [], None

    def point(x, y):
        m = state["ctm"]
        return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])

    for t in TOKEN.findall(src):
        if t == "[":
            in_array = []
            continue
        if t == "]":
            operands.append(in_array)
            in_array = None
            continue
        is_num = re.fullmatch(NUM, t) is not None
        if in_array is not None:
            in_array.append(float(t) if is_num else t)
            continue
        if is_num:
            operands.append(float(t))
            continue
        if t[0] in "/(<":
            operands.append(t)
            continue
        nums = [x for x in operands if isinstance(x, float)]
        if t == "q":
            stack.append({k: (list(v) if isinstance(v, list) else v) for k, v in state.items()})
        elif t == "Q":
            state = stack.pop()
        elif t == "cm":
            state["ctm"] = mat_mul(nums[-6:], state["ctm"])
        elif t == "d":
            state["dash"] = operands[-2] if len(operands) >= 2 and isinstance(operands[-2], list) and operands[-2] else None
        elif t in ("RG", "G", "K"):
            state["stroke"] = nums[-{"RG": 3, "G": 1, "K": 4}[t]:]
        elif t == "m":
            if subpath:
                current.append(subpath)
            subpath = [point(nums[-2], nums[-1])]
        elif t in ("l", "c"):
            subpath.append(point(nums[-2], nums[-1]))
        elif t == "re":
            x, y, w, h = nums[-4:]
            if subpath:
                current.append(subpath)
            subpath = [point(*p) for p in ((x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y))]
        elif t == "h":
            if subpath:
                subpath.append(subpath[0])
        elif t in ("S", "s", "f", "F", "B", "b", "n", "f*", "B*", "b*"):
            if subpath:
                current.append(subpath)
            if t != "n":
                paths.extend({"op": t, "stroke": state["stroke"], "dash": state["dash"], "pts": sp} for sp in current)
            current, subpath = [], []
        operands = []
    return paths


def osborne_fig3(paths):
    two = [p["pts"] for p in paths if len(p["pts"]) == 2]
    bottom = min(p[0][1] for p in two)
    left = min(p[0][0] for p in two)
    xt = sorted({round(p[0][0], 3) for p in two if abs(p[0][1] - bottom) < 1e-6 and abs(p[0][0] - p[1][0]) < 1e-6
                 and 5.5 < abs(p[1][1] - p[0][1]) < 6.2})
    yt = sorted({round(p[0][1], 3) for p in two if abs(p[0][0] - left) < 1e-6 and abs(p[0][1] - p[1][1]) < 1e-6
                 and 8.0 < abs(p[1][0] - p[0][0]) < 8.6})
    assert len(xt) == 4 and len(yt) == 6, (xt, yt)             # ticks at 0, 5, 10, 15 deg and 0.0 ... 1.0
    per_deg, per_unit = (xt[3] - xt[0]) / 15.0, yt[5] - yt[0]
    assert max(abs((x - xt[0]) / per_deg - 5 * i) for i, x in enumerate(xt)) < 2e-3
    assert max(abs((y - yt[0]) / per_unit - 0.2 * i) for i, y in enumerate(yt)) < 2e-3
    styles = {"cosine": ([0.0, 0.0, 0.0], None), "ramp": ([1.0, 0.0, 0.0], 113.386), "step": ([0.0, 0.0, 1.0], 56.693)}
    rows = []
    for name, (colour, dash) in styles.items():
        pts = set()
        for p in paths:
            if p["op"] != "S" or p["stroke"] != colour or len(p["pts"]) < 3:
                continue
            first = p["dash"][0] if p["dash"] else None
            if (first is None) != (dash is None) or (dash is not None and abs(first - dash) > 1.0):
                continue
            pts.update((round((x - xt[0]) / per_deg, 5), round((y - yt[0]) / per_unit, 6)) for x, y in p["pts"])
        pts = sorted(pts)
        rows += [(name, th, v) for th, v in pts[::max(1, len(pts) // 800)]]
    return "curve,theta_deg,value", rows


def feeney_fig6(paths):
    ticks = sorted({round(p["pts"][0][0], 2) for p in paths if p["op"] == "B" and len(p["pts"]) == 2
                    and abs(p["pts"][0][1] - 57.3) < 0.1})
    yticks = sorted({round(p["pts"][0][1], 2) for p in paths if p["op"] == "B" and len(p["pts"]) == 2
                     and abs(p["pts"][0][0] - ticks[0]) < 0.1 and abs(p["pts"][0][1] - p["pts"][1][1]) < 0.01})
    assert len(ticks) == 5 and len(yticks) == 3, (ticks, yticks)   # 13400 ... 14200 Mpc; 0, 1, 2
    rows = []
    for name, colour in (("temperature", [0.0, 0.0, 1.0]), ("polarization", [0.0])):
        pts = [p["pts"] for p in paths if p["op"] == "S" and p["stroke"] == colour and len(p["pts"]) > 50][0]
        rows += [(name, 13400.0 + (x - ticks[0]) * 800.0 / (ticks[4] - ticks[0]), (y - yticks[0]) / (yticks[1] - yticks[0]))
                 for x, y in sorted(pts)]
    return "curve,r_mpc,value", rows


def main():
    which, pdf, out = sys.argv[1], sys.argv[2], sys.argv[3]
    paths = stroked_paths(content_stream(pdf))
    header, rows = {"osborne-fig3": osborne_fig3, "feeney-fig6": feeney_fig6}[which](paths)
    with open(out, "w", newline="\n") as f:
        f.write(header + "\n")
        for name, x, v in rows:
            f.write("%s,%.5f,%.6f\n" % (name, x, v))
    print(which, {name: sum(1 for r in rows if r[0] == name) for name in dict.fromkeys(r[0] for r in rows)}, "->", out)


if __name__ == "__main__":
    main()
