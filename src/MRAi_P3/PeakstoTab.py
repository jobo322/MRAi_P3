import argparse, os, sys
from . import headers


def point_from_ppm(ppm, first_ppm, last_ppm, size):
    return ((first_ppm - ppm) / (first_ppm - last_ppm)) * (size - 1) + 1


def main(peaks_path, spec_path, out=None):
    ND, sz, sw, pfst, plst, nam = headers.read_headers(spec_path)

    if out is None:
        out = os.path.splitext(peaks_path)[0] + '.tab'
    with open(out, "w") as out:
        # ----- Header (mirrors AWK) -----
        out.write(f"REMARK ROI {ND}D Peak Detection System, File: XXXX\n")

        # DATA lines: sizes are sz+2; upper ppm is plst - sw*2/sz
        def upper_ppm(axis):
            return plst[axis] - sw[axis] * 2.0 / sz[axis]

        # X is the 1H (direct) axis = NMRPipe dim 2; Y is the 15N axis = dim 1.
        # This matches the coordinate order P5toPeaks emits into the .peaks file.
        out.write(
            f"DATA  X_AXIS {nam[2]:3s}          1   {int(sz[2]) + 2:3d}  {pfst[2]:9.3f}ppm {upper_ppm(2):9.3f}ppm\n"
        )
        out.write(
            f"DATA  Y_AXIS {nam[1]:3s}          1   {int(sz[1]) + 2:3d}  {pfst[1]:9.3f}ppm {upper_ppm(1):9.3f}ppm\n"
        )
        if ND == 2:
            VARS = [
                "INDEX", "X_AXIS", "Y_AXIS", "DX", "DY", "X_PPM", "Y_PPM",
                "X_HZ", "Y_HZ", "XW", "YW", "XW_HZ", "YW_HZ", "X1", "X3",
                "Y1", "Y3", "HEIGHT", "DHEIGHT", "VOL", "PCHI2", "TYPE",
                "ASS", "CLUSTID", "MEMCNT",
            ]
            FMT = ("%5d %9.3f %9.3f %6.3f %6.3f %8.3f %8.3f %9.3f %9.3f %7.3f %7.3f %8.3f %8.3f %4d %4d %4d %4d %+e %+e %+e %.5f %d %s %4d %4d")
            out.write("VARS   " + " ".join(VARS) + "\n")
            out.write("FORMAT " + FMT.replace("%", "%%").replace("%% ", "% ").replace("%%", "%") + "\n")

        if ND == 3:
            out.write(
                f"DATA  Z_AXIS {nam[3]:3s}          1   {int(sz[3]) + 2:3d}  {pfst[3]:9.3f}ppm {upper_ppm(3):9.3f}ppm\n"
            )
            VARS = [
                "INDEX", "X_AXIS", "Y_AXIS", "Z_AXIS", "DX", "DY", "DZ",
                "X_PPM", "Y_PPM", "Z_PPM", "X_HZ", "Y_HZ", "Z_HZ",
                "XW", "YW", "ZW", "XW_HZ", "YW_HZ", "ZW_HZ", "X1", "X3",
                "Y1", "Y3", "Z1", "Z3", "HEIGHT", "DHEIGHT", "VOL", "PCHI2",
                "TYPE", "ASS", "CLUSTID", "MEMCNT",
            ]
            FMT = (
                "%5d %9.3f %9.3f %9.3f %6.3f %6.3f %6.3f %8.3f %8.3f %8.3f "
                "%9.3f %9.3f %9.3f %7.3f %7.3f %7.3f %8.3f %8.3f %8.3f %4d %4d %4d %4d %4d %4d "
                "+%e %+e +%e %.5f %d %s %4d %4d"
            )
            out.write("VARS   " + " ".join(VARS) + "\n")
            out.write("FORMAT " + FMT.replace("%", "%%").replace("%% ", "% ").replace("%%", "%") + "\n")

        # ----- Read peaks; emit rows -----
        # CYANA/xeasy-like peak lines (skip headers; start after '#FORMAT xeasy')
        in_cyana = False
        with open(peaks_path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#FORMAT xeasy"):
                    in_cyana = True
                    continue
                if not in_cyana:
                    continue
                parts = line.split()
                # need at least the 11 columns the AWK expects on peak rows
                if len(parts) < 11:
                    continue

                if ND == 3:
                    try:
                        n = int(float(parts[0]))
                        X = float(parts[1])
                        Y = float(parts[2])
                        Z = float(parts[3])
                        P = float(parts[6])  # HEIGHT in tab
                        I = float(parts[7])  # DHEIGHT (and VOL in AWK mapping)
                        A = str(parts[10])  # $(11) in AWK
                    except Exception:
                        continue

                if ND == 2:
                    try:
                        n = int(float(parts[0]))
                        X = float(parts[1])
                        Y = float(parts[2])
                        P = float(parts[5])  # HEIGHT in tab
                        I = float(parts[6])  # DHEIGHT (and VOL in AWK mapping)
                        A = str(parts[9])    # $(11) in AWK
                    except Exception:
                        continue

                # Compute NMRPipe point indices (1-based).
                # Each coordinate is calibrated with the axis it actually belongs to:
                # X -> 1H (dim 2), Y -> 15N (dim 1), Z -> dim 3.
                x_pnt = point_from_ppm(X, pfst[2], plst[2], sz[2])
                y_pnt = point_from_ppm(Y, pfst[1], plst[1], sz[1])
                if ND == 3:
                    z_pnt = point_from_ppm(Z, pfst[3], plst[3], sz[3])

                # integer bounding box indices
                X1 = int(x_pnt)
                X3 = X1 + 1
                Y1 = int(y_pnt)
                Y3 = Y1 + 1
                if ND == 3:
                    Z1 = int(z_pnt)
                    Z3 = Z1 + 1

                # Build row in the order of VARS
                if ND == 3:
                    row = [
                        n, x_pnt, y_pnt, z_pnt, 0.0, 0.0, 0.0,
                        X, Y, Z, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        X1, X3, Y1, Y3, Z1, Z3,
                        P, 0.0, I, 0.0, 1, A, n, 1,
                    ]
                if ND == 2:
                    row = [
                        n, x_pnt, y_pnt, 0.0, 0.0,
                        X, Y, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0,
                        X1, X3, Y1, Y3,
                        P, 0.0, I, 0.0, 1, A, n, 1,
                    ]

                out.write(FMT % tuple(row) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert CYANA .peaks + showhdr -verb(spec) -> NMRPipe .tab"
    )
    ap.add_argument("peaks", help="Path to CYANA .peaks file")
    ap.add_argument("-spec", required=True, help="Spectrum file to pass to `showhdr -verb` (e.g., HNCO.ft3 or .P3)")
    ap.add_argument("-o", default=None, help="Output file path (optional).")
    args = ap.parse_args()
    main(args.peaks, spec_path=args.spec, out=args.o)