"""
Header reading: primary route via `showhdr -verb`, fallback via nmrglue.

Both paths return the same canonical six-tuple:
    (ND, sz, sw, pfst, plst, nam)

Each dict entry is keyed 1..ND (not 0-based).
"""

import os
import subprocess
import sys


def read_headers(spec_path):
    """Read NMR header information from a spectrum file.

    Parameters
    ----------
    spec_path : str
        Path to the spectrum file (`.ft2`, `.P3`, `.ft3`, etc.).

    Returns
    -------
    ND : int
        Number of dimensions.
    sz : dict {1..ND : float}
        True array size for each dimension (real FDSIZE, not stale FTSIZE).
    sw : dict {1..ND : float}
        Swelling in ppm per point (SW / OBS).
    pfst : dict {1..ND : float}
        First PPM position for each dimension (from ppm_limits upper bound).
    plst : dict {1..ND : float}
        Last PPM position for each dimension (from ppm_limits lower bound).
    nam : dict {1..ND : str}
        Axis names: `"HN"` if `"H"` in the label, else last character of label.
    """
    # ── Primary route: subprocess check_output("showhdr -verb", ...) ──
    try:
        text = subprocess.check_output(["showhdr", "-verb", str(spec_path)], text=True)
    except FileNotFoundError:
        # ── Fallback: use nmrglue ──
        return _read_headers_fallback(spec_path)
    except subprocess.CalledProcessError as e:
        sys.exit(f"`showhdr -verb` failed: {e}")

    # Parse showhdr -verb output
    ND = None
    raw_order = None
    nam = {}
    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("FILE:"):
            parts = line.split()
            ND = int(parts[parts.index("DIM:") + 1])
        if line.startswith("ORDER:"):
            # `ORDER: 2 1 PIPE: 0 CUBE: 0 ...` names the NMRPipe dimension
            # shown in each printed axis column. showhdr emits columns in
            # *display* order (X, Y[, Z] == FDF2, FDF1[, ...]), which is the
            # reverse of NMRPipe's own 1..ND numbering for the usual
            # non-transposed case, so the column index cannot be the key.
            nums = []
            for tok in line.split()[1:]:
                if tok == "PIPE:":
                    break
                nums.append(int(float(tok)))
            if len(nums) == ND:
                raw_order = nums
        if line.startswith("NAME:"):
            parts = line.split()
            for i in range(ND):
                lbl = str(parts[i - ND])
                dim = _showhdr_order(raw_order, ND)[i]
                nam[dim] = "HN" if "H" in lbl else (lbl[-1] if lbl else str(dim))

    if ND is None:
        raise ValueError(f"Could not parse header from {spec_path}")

    # Resolve once here so the value-block loop below reuses the same mapping.
    order = _showhdr_order(raw_order, ND)

    # showhdr reports absolute Hz values for PPMSIZE/PMTOTAL etc.,
    # but also provides SW_PPM and PPM_FIRST/PPM_LAST directly.
    # Each value line holds its axes in the same display order as ORDER,
    # so column i belongs to NMRPipe dimension `order[i]`.
    sz = {}
    sw = {}
    pfst = {}
    plst = {}
    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("PPM FIRST:"):
            parts = line.split()
            for i in range(ND):
                pfst[order[i]] = float(parts[i - ND])
        if line.startswith("PPM LAST:"):
            parts = line.split()
            for i in range(ND):
                plst[order[i]] = float(parts[i - ND])
        if line.startswith("SW PPM:"):
            parts = line.split()
            for i in range(ND):
                sw[order[i]] = float(parts[i - ND])
        if line.startswith("DATA SIZE:"):
            parts = line.split()
            for i in range(ND):
                sz[order[i]] = float(parts[i - ND])

    return ND, sz, sw, pfst, plst, nam


def _showhdr_order(raw_order, ND):
    """Map showhdr's printed axis columns to NMRPipe dimension numbers.

    showhdr emits one column per axis in *display* order (X, Y[, Z]). The
    `ORDER:` line carries exactly that mapping as NMRPipe dimension numbers,
    so column ``i`` holds the values of dimension ``raw_order[i]``.

    Returns ``raw_order`` when it parsed cleanly; otherwise falls back to the
    display order of the usual non-transposed file, which is the reverse of
    NMRPipe's 1..ND numbering.
    """
    if raw_order is not None and len(raw_order) == ND:
        return list(raw_order)
    return list(range(ND, 0, -1))


def _read_headers_fallback(spec_path):
    """Fallback: parse headers using nmrglue Pipe dictionary."""
    try:
        import nmrglue
    except ImportError:
        sys.exit(
            "nmrglue fallback path requires the 'nmrglue' package. "
            "Install it: pip install nmrglue"
        )

    dic, data = nmrglue.pipe.read(str(spec_path))

    # ND is authoritative: the array.ndim
    ND = data.ndim

    sz = {}
    sw = {}
    pfst = {}
    plst = {}
    nam = {}

    for i in range(1, ND + 1):
        dim_idx = i - 1

        # REAL array size for point-index scaling (not stale FTSIZE)
        sz[i] = float(data.shape[dim_idx])

        # Build uncertainty calibration with the REAL data array
        # (previously passed None which always raised AttributeError)
        uc = nmrglue.pipe.make_uc(dic, data, dim=dim_idx)
        # pmplimits returns (upper_ppm, lower_ppm) — we flip to [first, last]
        high_ppm, low_ppm = uc.ppm_limits()
        pfst[i] = float(high_ppm)
        plst[i] = float(low_ppm)

        # Swelling in ppm per point: SW / OBS (guard zero OBS)
        obs = dic.get(f"FDF{i}OBS", 0.0)
        if obs == 0:
            sw[i] = 0.0
        else:
            sw[i] = float(dic.get(f"FDF{i}SW", 0.0)) / float(obs)

        # Axis name convention
        label = str(dic.get(f"FDF{i}LABEL", f"dim{i}"))
        nam[i] = "HN" if "H" in label else (label[-1] if label else str(i))

    return ND, sz, sw, pfst, plst, nam