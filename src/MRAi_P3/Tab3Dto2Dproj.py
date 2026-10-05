import sys, collections, os

VARS_LINE = (
    "VARS   INDEX X_AXIS Y_AXIS DX DY X_PPM Y_PPM X_HZ Y_HZ XW YW XW_HZ YW_HZ "
    "X1 X3 Y1 Y3 HEIGHT DHEIGHT VOL PCHI2 TYPE ASS CLUSTID MEMCNT"
)
FORMAT_LINE = (
    "FORMAT %5d %9.3f %9.3f %6.3f %6.3f %8.3f %8.3f %9.3f %9.3f %7.3f %7.3f "
    "%8.3f %8.3f %4d %4d %4d %4d %+e %+e %+e %.5f %d %s %4d %4d"
)

def parse_header_and_rows(path):
    with open(path) as f:
        lines = f.readlines()

    # Header
    remark = None
    axes = {}  # "X_AXIS"/"Y_AXIS"/"Z_AXIS" -> {name,size,fppm,lppm}
    vars_order = None
    data_start_idx = None

    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        if s.startswith("REMARK") and remark is None:
            remark = s
        elif s.startswith("DATA"):
            # e.g. DATA  X_AXIS HN 1  642  11.000ppm  5.983ppm
            parts = s.split()
            axis = parts[1]
            name = parts[2]
            size = int(parts[4])
            fppm = float(parts[5].replace("ppm", ""))
            lppm = float(parts[6].replace("ppm", ""))
            axes[axis] = dict(name=name, size=size, fppm=fppm, lppm=lppm)
        elif s.startswith("VARS"):
            vars_order = s.split()[1:]
        elif s.startswith("FORMAT"):
            data_start_idx = i + 1
            break

    if vars_order is None or data_start_idx is None:
        sys.exit("Error: Missing VARS/FORMAT lines in input tab.")

    pos = {name: i for i, name in enumerate(vars_order)}

    # Rows (split on whitespace)
    rows = []
    for ln in lines[data_start_idx:]:
        t = ln.strip()
        if not t:
            continue
        parts = t.split()
        rows.append(parts)

    # sanity: required fields
    required = ["X_AXIS","Y_AXIS","Z_AXIS","X_PPM","Y_PPM","Z_PPM",
                "X1","Y1","Z1","HEIGHT","DHEIGHT","VOL","ASS","CLUSTID","MEMCNT"]
    for r in required:
        if r not in pos:
            sys.exit(f"Error: required column '{r}' not found in VARS.")

    if remark is None:
        remark = "REMARK ROI 2D Peak Detection System, File: XXXX"

    return remark, axes, pos, rows

def build_header_2d(remark, ax_keep, axes_info):
    A = axes_info[ax_keep[0]]
    B = axes_info[ax_keep[1]]
    hdr = []
    hdr.append(remark)
    hdr.append(f"DATA  X_AXIS {A['name']:<12} 1 {A['size']:5d}  {A['fppm']:9.3f}ppm {A['lppm']:9.3f}ppm")
    hdr.append(f"DATA  Y_AXIS {B['name']:<12} 1 {B['size']:5d}  {B['fppm']:9.3f}ppm {B['lppm']:9.3f}ppm")
    hdr.append(VARS_LINE)
    hdr.append(FORMAT_LINE)
    return hdr

def aggregate_projection(rows, pos, mode):
    agg = collections.defaultdict(lambda: {
        "H":0.0, "DH":0.0, "V":0.0, "MEM":0, "ASS":"0", "CLU":0,
        "XPPM":0.0, "YPPM":0.0, "seen":False
    })

    for parts in rows:
        # integer bins (use X1/Y1/Z1; if absent values are '', fall back to rounding X_AXIS etc.)
        def get_bin(name_idx, fallback_axis):
            s = parts[pos[name_idx]]
            if s != "":
                try: return int(s)
                except: pass
            return int(round(float(parts[pos[fallback_axis]])))

        x1 = get_bin("X1", "X_AXIS")
        y1 = get_bin("Y1", "Y_AXIS")
        z1 = get_bin("Z1", "Z_AXIS")

        xppm = float(parts[pos["X_PPM"]])
        yppm = float(parts[pos["Y_PPM"]])
        zppm = float(parts[pos["Z_PPM"]])

        H  = float(parts[pos["HEIGHT"]])
        DH = float(parts[pos["DHEIGHT"]])
        V  = float(parts[pos["VOL"]])
        ASS = parts[pos["ASS"]]
        CLU = int(parts[pos["CLUSTID"]])
        MEM = int(parts[pos["MEMCNT"]])

        if mode == "Y":
            key = (x1, z1); XPPM, YPPM = xppm, zppm
        elif mode == "Z":
            key = (x1, y1); XPPM, YPPM = xppm, yppm
        elif mode == "X":
            key = (y1, z1); XPPM, YPPM = yppm, zppm
        else:
            raise ValueError("mode must be 'C','N','HN'")

        a = agg[key]
        a["H"]  += H
        a["DH"] += DH
        a["V"]  += V
        a["MEM"] += MEM
        a["ASS"] = ASS      # last wins; change if you want first/min/etc.
        a["CLU"] = CLU
        if not a["seen"]:
            a["XPPM"] = XPPM
            a["YPPM"] = YPPM
            a["seen"] = True

    return agg

def write_2d(out_path, header_lines, agg):
    with open(out_path, "w") as out:
        for h in header_lines:
            out.write(h + "\n")
        # sorted by (xbin, ybin)
        keys = sorted(agg.keys())
        for idx, (xb, yb) in enumerate(keys, start=1):
            a = agg[(xb, yb)]
            # Fields (match FORMAT exactly)
            INDEX  = idx
            X_AXIS = float(xb)
            Y_AXIS = float(yb)
            DX = DY = 0.0
            X_PPM = a["XPPM"]; Y_PPM = a["YPPM"]
            X_HZ = Y_HZ = 0.0
            XW = YW = 0.0
            XW_HZ = YW_HZ = 0.0
            X1 = xb; X3 = xb + 1
            Y1 = yb; Y3 = yb + 1
            HEIGHT = a["H"]; DHEIGHT = a["DH"]; VOL = a["V"]
            PCHI2 = 0.0
            TYPE = 1
            ASS = a["ASS"]
            CLUSTID = a["CLU"]
            MEMCNT = a["MEM"]

            out.write((
                "%5d %9.3f %9.3f %6.3f %6.3f %8.3f %8.3f %9.3f %9.3f "
                "%7.3f %7.3f %8.3f %8.3f %4d %4d %4d %4d "
                "%+e %+e %+e %.5f %d %s %4d %4d\n"
            ) % (
                INDEX, X_AXIS, Y_AXIS, DX, DY, X_PPM, Y_PPM, X_HZ, Y_HZ,
                XW, YW, XW_HZ, YW_HZ, X1, X3, Y1, Y3,
                HEIGHT, DHEIGHT, VOL, PCHI2, TYPE, ASS, CLUSTID, MEMCNT
            ))

def main(infile):
    remark, axes, pos, rows = parse_header_and_rows(infile)
    base = os.path.splitext(infile)[0]

    # Build headers for each projection
    hdr_Y = build_header_2d(remark, ("X_AXIS","Z_AXIS"), axes)
    hdr_Z = build_header_2d(remark, ("X_AXIS","Y_AXIS"), axes)
    hdr_X = build_header_2d(remark, ("Y_AXIS","Z_AXIS"), axes)

    # Aggregate
    agg_Y = aggregate_projection(rows, pos, "Y")
    agg_Z = aggregate_projection(rows, pos, "Z")
    agg_X = aggregate_projection(rows, pos, "X")
    
    # Write
    outY = f"{base}_{axes['X_AXIS']['name']}.{axes['Z_AXIS']['name']}.tab"
    outZ = f"{base}_{axes['X_AXIS']['name']}.{axes['Y_AXIS']['name']}.tab"
    outX = f"{base}_{axes['Y_AXIS']['name']}.{axes['Z_AXIS']['name']}.tab"

    write_2d(outY, hdr_Y, agg_Y)
    write_2d(outZ, hdr_Z, agg_Z)
    write_2d(outX, hdr_X, agg_X)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {Path(sys.argv[0]).name} <3D_tab_file>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

