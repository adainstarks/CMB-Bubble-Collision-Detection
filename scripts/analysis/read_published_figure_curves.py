"""Read published figures from the files in the papers' arXiv sources.

  osborne-fig3    bubble_profile.pdf in the source of arXiv:1305.1970 (Osborne, Senatore, Smith; Fig. 3)
  feeney-fig6     transferfunctions_l20.pdf in the source of arXiv:1506.01716 (Feeney, Elsner, Johnson,
                  Peiris; Fig. 6)
  mcewen-snr      figures/snr_B1p80_A100uK_psfrag.pdf in the source of arXiv:1202.2861 (McEwen, Feeney,
                  Johnson, Peiris; Fig. 4a): signal-to-noise of a 100 uK collision against its size, for the
                  matched filter and for the unfiltered field
  mcewen-regions  figures/exclusion_regions_uK_nolabels_psfrag.pdf, same source (Fig. 6): the exclusion,
                  sensitivity and unprobed cells on their grid of sizes and amplitudes. This figure is a
                  picture inside the PDF; its pixels are decoded and the cell colours are read.
  mcewen-profile  figures/bubble_profile_thetac010_A100uK_psfrag.pdf, same source (Fig. 1a): the radial profile
                  of the collision template for z0 = 100 uK and a size of 10 deg
  mcewen-spectra  figures/bubble_spectra_thetac010_A100uK_psfrag.pdf, same source (Fig. 1c): the power spectrum
                  of that template and of the CMB (log axes, l from 1 to 1000, 1e-10 to 1e5 uK^2)

The PDF content stream is decompressed and its path operators are read; nothing is rendered or executed. The
axes are calibrated from the tick marks drawn in the same file (their labels are checked by eye once: 0, 5,
10, 15 deg and 0.0 ... 1.0 for the first figure; 13400 ... 14200 Mpc and 0, 1, 2 for the second; 0 ... 90 deg
and 0 ... 3.5 for the third, whose coordinates sit on a grid of about 0.2 deg and 0.01, with the curve points at
whole degrees; sizes 2, 5, 10, 25, 50, 90 deg and amplitudes 25, 50, 75, 100, 150, 200, 400 uK for the fourth;
0 ... 45 deg and 0 ... 100 uK for the profile, whose coordinates sit on a grid of 0.10 deg and 0.34 uK; the
frame from 1 to 1000 and from 1e-10 to 1e5 for the spectra).

Usage:
  python scripts/analysis/read_published_figure_curves.py osborne-fig3 bubble_profile.pdf out.csv
  python scripts/analysis/read_published_figure_curves.py feeney-fig6 transferfunctions_l20.pdf out.csv
  python scripts/analysis/read_published_figure_curves.py mcewen-snr snr_B1p80_A100uK_psfrag.pdf out.csv
  python scripts/analysis/read_published_figure_curves.py mcewen-regions exclusion_regions_uK_nolabels_psfrag.pdf out.csv
  python scripts/analysis/read_published_figure_curves.py mcewen-profile bubble_profile_thetac010_A100uK_psfrag.pdf out.csv
  python scripts/analysis/read_published_figure_curves.py mcewen-spectra bubble_spectra_thetac010_A100uK_psfrag.pdf out.csv
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


def mcewen_snr(paths):
    box = [p["pts"] for p in paths if p["op"] == "S" and len(p["pts"]) == 5 and p["pts"][0] == p["pts"][-1]][0]
    x0, x1 = min(q[0] for q in box), max(q[0] for q in box)        # 0 and 90 deg
    y0, y1 = min(q[1] for q in box), max(q[1] for q in box)        # 0 and 3.5
    rows = []
    for name, colour in (("matched_filter", [0.0, 0.0, 1.0]), ("unfiltered", [1.0, 0.0, 0.0])):
        pts = [p["pts"] for p in paths if p["op"] == "S" and p["stroke"] == colour and len(p["pts"]) > 50][0]
        rows += [(name, float(round((x - x0) / (x1 - x0) * 90.0)), (y - y0) / (y1 - y0) * 3.5) for x, y in sorted(pts)]
    return "curve,theta_crit_deg,snr", rows


def _frame(paths):
    box = [p["pts"] for p in paths if p["op"] == "S" and len(p["pts"]) == 5 and p["pts"][0] == p["pts"][-1]][0]
    return min(q[0] for q in box), max(q[0] for q in box), min(q[1] for q in box), max(q[1] for q in box)


def mcewen_profile(paths):
    x0, x1, _, _ = _frame(paths)                                   # 0 and 45 deg
    yt = sorted({round(p["pts"][0][1], 2) for p in paths if p["op"] == "S" and len(p["pts"]) == 2
                 and abs(p["pts"][0][0] - x0) < 0.01 and abs(p["pts"][0][1] - p["pts"][1][1]) < 0.01
                 and 3.0 < abs(p["pts"][1][0] - p["pts"][0][0]) < 6.0})
    assert len(yt) == 6, yt                                        # ticks at 0, 20, ..., 100 uK
    pts = [p["pts"] for p in paths if p["op"] == "S" and p["stroke"] == [0.0, 0.0, 1.0] and len(p["pts"]) > 50][0]
    return "curve,theta_deg,delta_t_uK", [("profile", (x - x0) / (x1 - x0) * 45.0, (y - yt[0]) / (yt[5] - yt[0]) * 100.0)
                                          for x, y in pts]


def mcewen_spectra(paths):
    x0, x1, y0, y1 = _frame(paths)                                 # l = 1 and 1000; 1e-10 and 1e5 uK^2
    rows = []
    for name, colour in (("template", [0.0, 0.0, 1.0]), ("cmb", [1.0, 0.0, 0.0])):
        pts = [p["pts"] for p in paths if p["op"] == "S" and p["stroke"] == colour and len(p["pts"]) > 50][0]
        rows += [(name, 10.0 ** ((x - x0) / (x1 - x0) * 3.0), 10.0 ** ((y - y0) / (y1 - y0) * 15.0 - 10.0)) for x, y in pts]
    return "curve,ell,value_uK2", rows


def image_rows(path):
    """The RGB picture strips of the file (PNG row predictors undone), stacked with the taller strip on top."""
    raw = open(path, "rb").read()
    strips = []
    pattern = rb"/Width (\d+)\s*/Height (\d+)[^>]*?/Columns (\d+)\s*/Colors 3>>/Length (\d+)>>\s*stream\r?\n"
    for m in re.finditer(pattern, raw, re.S):
        w, h, n = int(m.group(1)), int(m.group(2)), int(m.group(4))
        data = zlib.decompress(raw[m.end():m.end() + n])
        stride = w * 3
        assert len(data) == h * (stride + 1)
        rows, prev = [], bytearray(stride)
        for y in range(h):
            kind = data[y * (stride + 1)]
            line = bytearray(data[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
            for i in range(stride):
                a = line[i - 3] if i >= 3 else 0
                b = prev[i]
                c = prev[i - 3] if i >= 3 else 0
                if kind == 1:
                    line[i] = (line[i] + a) & 255
                elif kind == 2:
                    line[i] = (line[i] + b) & 255
                elif kind == 3:
                    line[i] = (line[i] + ((a + b) >> 1)) & 255
                elif kind == 4:
                    est = a + b - c
                    da, db, dc = abs(est - a), abs(est - b), abs(est - c)
                    line[i] = (line[i] + (a if (da <= db and da <= dc) else (b if db <= dc else c))) & 255
            rows.append(line)
            prev = line
        strips.append((h, w, rows))
    strips.sort(key=lambda t: -t[0])
    return strips[0][1], [r for t in strips for r in t[2]]


def mcewen_regions(path):
    w, rows = image_rows(path)
    h = len(rows)
    dark = [[rows[y][3 * x] < 60 for x in range(w)] for y in range(h)]

    def longest_run(values):
        best = run = 0
        for v in values:
            run = run + 1 if v else 0
            best = max(best, run)
        return best

    frame_rows = [y for y in range(h) if longest_run(dark[y]) > 0.6 * w]        # top block of black cells ... bottom axis
    frame_cols = [x for x in range(w) if longest_run([dark[y][x] for y in range(h)]) > 0.6 * h]   # left and right axis lines
    top, bottom, left, right = min(frame_rows), max(frame_rows), min(frame_cols), max(frame_cols)
    sizes, amps = [2.0, 5.0, 10.0, 25.0, 50.0, 90.0], [400.0, 200.0, 150.0, 100.0, 75.0, 50.0, 25.0]     # tick labels, read by eye
    out = []
    for i, size in enumerate(sizes):
        for j, amp in enumerate(amps):
            x = int(left + (i + 0.5) * (right - left) / len(sizes))
            y = int(top + (j + 0.5) * (bottom - top) / len(amps))
            grey = rows[y][3 * x]
            out.append(("exclusion" if grey < 60 else ("sensitivity" if grey < 200 else "unprobed"), size, amp))
    return "class,theta_crit_deg,z0_uK", out


def main():
    which, pdf, out = sys.argv[1], sys.argv[2], sys.argv[3]
    if which == "mcewen-regions":
        header, rows = mcewen_regions(pdf)
    else:
        paths = stroked_paths(content_stream(pdf))
        header, rows = {"osborne-fig3": osborne_fig3, "feeney-fig6": feeney_fig6, "mcewen-snr": mcewen_snr,
                        "mcewen-profile": mcewen_profile, "mcewen-spectra": mcewen_spectra}[which](paths)
    fmt = "%s,%.5f,%.6e\n" if which == "mcewen-spectra" else "%s,%.5f,%.6f\n"
    with open(out, "w", newline="\n") as f:
        f.write(header + "\n")
        for name, x, v in rows:
            f.write(fmt % (name, x, v))
    print(which, {name: sum(1 for r in rows if r[0] == name) for name in dict.fromkeys(r[0] for r in rows)}, "->", out)


if __name__ == "__main__":
    main()
