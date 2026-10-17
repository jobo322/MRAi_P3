#!/usr/bin/env python
"""
convert_2d.py — one-time conversion of the folder-51 NMRPipe inputs into
plain binary/JSON artifacts for the zero-dependency Node viewer `view2d.mjs`.

This script reuses the exact loaders from `interactive_2d.py` (nmrglue `.ft2`
read, `.tab`/`.peaks` peak parsing) so the web viewer sees byte-for-byte the
same data the matplotlib viewer would. It only needs to be run once (re-run
when the source `.ft2`/`.tab`/`.P3` files change).

    /Users/abolanos/.pyenv/versions/tflow/bin/python scripts/plotting/convert_2d.py
    ... python scripts/plotting/convert_2d.py --ft2 X --tab Y --p3 Z --out /tmp/p51_view2d

Output layout (--out, default /tmp/p51_view2d):

    meta.json      shapes, obs frequency, axis labels, peak list, file names
    spectrum.bin   float32 LE, C order, shape (F1 points, F2 points)
    p3.bin         float32 LE, same shape (only if --p3 exists)
    f2ppm.bin      float32 LE, F2 ppm scale (as uc.ppm_scale())
    f1hz.bin       float32 LE, F1 Hz scale (ppm * FDF1OBS)

Requires nmrglue, present in the `tflow` pyenv.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interactive_2d import (  # noqa: E402
    DEFAULT_FT2,
    DEFAULT_P3,
    DEFAULT_PEAKS,
    DEFAULT_TAB,
    load_peaks,
    load_spectrum,
)

DEFAULT_OUT = '/tmp/p51_view2d'


def load_tab_rows(tab_path, y_obs):
    """Extra per-peak fields for the viewer's peaks table (like nmrglue's .tab).

    Mirrors the VARS parsing of interactive_2d.load_peaks but keeps INDEX,
    X_PPM, Y_HZ, HEIGHT, VOL and TYPE for display.
    """
    rows = []
    if not os.path.isfile(tab_path):
        return rows
    vl = None
    for line in open(tab_path):
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if parts[0] == 'VARS':
            vl = parts[1:]
            continue
        if vl is None or not s[0].isdigit() or len(parts) != len(vl):
            continue
        try:
            rows.append({
                'index': int(float(parts[vl.index('INDEX')])),
                'xPpm': float(parts[vl.index('X_PPM')]),
                'yHz': float(parts[vl.index('Y_PPM')]) * y_obs,
                'height': float(parts[vl.index('HEIGHT')]),
                'vol': float(parts[vl.index('VOL')]),
                'type': parts[vl.index('TYPE')],
            })
        except (ValueError, IndexError):
            continue
    return rows


def _parser():
    ap = argparse.ArgumentParser(
        description='Convert folder-51 ft2/tab/peaks/P3 into artifacts for view2d.mjs.')
    ap.add_argument('--ft2', default=DEFAULT_FT2)
    ap.add_argument('--tab', default=DEFAULT_TAB)
    ap.add_argument('--peaks', default=DEFAULT_PEAKS)
    ap.add_argument('--p3', default=DEFAULT_P3)
    ap.add_argument('--no-p3', action='store_true',
                    help='skip the P3 probability map conversion')
    ap.add_argument('--out', default=DEFAULT_OUT,
                    help='output directory (default: %s)' % DEFAULT_OUT)
    return ap


def run(args=None):
    ns = _parser().parse_args(args)
    os.makedirs(ns.out, exist_ok=True)

    spectrum, ppm, f1_hz, y_obs, labels = load_spectrum(ns.ft2)

    p3 = None
    if not ns.no_p3 and os.path.isfile(ns.p3):
        p3, _, _, _, _ = load_spectrum(ns.p3)
        if p3.shape != spectrum.shape:
            sys.exit('Shape mismatch: P3 %s vs spectrum %s' % (p3.shape, spectrum.shape))

    px, py = load_peaks(ns.tab, ns.peaks, y_obs)
    peak_source = ns.tab if os.path.isfile(ns.tab) else ns.peaks

    def dump(arr, name):
        np.asarray(arr, dtype='<f4').tofile(os.path.join(ns.out, name))
        return name

    meta = {
        'shape': [int(spectrum.shape[0]), int(spectrum.shape[1])],  # (F1, F2)
        'obs': y_obs,
        'labels': [str(labels[0]), str(labels[1])],
        'spectrum': dump(spectrum, 'spectrum.bin'),
        'f2ppm': dump(ppm, 'f2ppm.bin'),
        'f1hz': dump(f1_hz, 'f1hz.bin'),
        'p3': dump(p3, 'p3.bin') if p3 is not None else None,
        'peaks': {'x': [float(v) for v in px], 'y': [float(v) for v in py],
                  'rows': load_tab_rows(ns.tab, y_obs)},
        'source': {'ft2': ns.ft2, 'tab': ns.tab, 'peaks': ns.peaks, 'p3': ns.p3,
                   'peakSource': peak_source},
    }
    with open(os.path.join(ns.out, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)

    print('converted -> %s' % ns.out)
    print('  spectrum %s  |  p3 %s  |  %d peaks from %s' %
          (spectrum.shape, 'ok' if p3 is not None else 'absent', len(px), peak_source))


if __name__ == '__main__':
    run()