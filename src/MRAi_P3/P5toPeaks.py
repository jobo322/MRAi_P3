import argparse, os, sys
import numpy as np
from . import headers


def main(P5_path, spec_path, spec_type=None, out=None):
    ND, _sz, _sw, _pfst, _plst, nam = headers.read_headers(spec_path)
    P5 = np.atleast_2d(np.loadtxt(P5_path, dtype=float))
    if spec_type is None:
        spec_type = os.path.splitext(os.path.basename(spec_path))[0]
    if out is None:
        out = os.path.splitext(P5_path)[0] + '.peaks'
    with open(out, "w") as out:
        # Emitted coordinate columns are, in 2D (F2, F1) and in 3D (F2, F1, F3).
        # nam is keyed by NMRPipe dimension (1 = FDF1, 2 = FDF2), so the INAME
        # labels must follow the emitted column order, not the dimension order.
        iname = {2: [2, 1], 3: [2, 1, 3]}[ND]
        out.write(f"# Number of dimensions {ND}\n")
        out.write(f"#FORMAT xeasy{ND}D\n")
        for i, d in enumerate(iname, start=1):
            out.write(f"#INAME {i} {nam[d]}\n")
        if ND == 3:
            out.write(f"#SPECTRUM {spec_type} {nam[2]} {nam[1]} {nam[3]}\n")
            for idx, row in enumerate(P5, start=1):
                if len(row) != ND + 2:
                    continue
                F3, F1, F2, p, i = row
                out.write(
                    f"{idx:7d} {F2:7.3f} {F1:7.3f} {F3:7.3f} 1 U {p:11.3E} {i:10.3E} e 0     0     0     0\n"
                )
        if ND == 2:
            out.write(f"#SPECTRUM {spec_type} {nam[2]} {nam[1]}\n")
            for idx, row in enumerate(P5, start=1):
                if len(row) != ND + 2:
                    continue
                F1, F2, p, i = row
                out.write(
                    f"{idx:7d} {F2:7.3f} {F1:7.3f} 1 U {p:11.3E} {i:10.3E} e 0     0     0\n"
                )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert P5 (format text to CYANA .peaks")
    ap.add_argument("P5", help="Path to input .P5 file")
    ap.add_argument("-spec", required=True, help="Spectrum file to pass to `showhdr -verb` (e.g., HNCO.ft3 or .P3)")
    ap.add_argument("-spec_type", default=None, help="Override spectrum name printed in #SPECTRUM")
    ap.add_argument("-o", default=None, help="Output file path (optional).")
    args = ap.parse_args()
    main(args.P5, spec_path=args.spec, spec_type=args.spec_type, out=args.o)