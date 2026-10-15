"""
Interactive 2D contour viewer for folder-51 J-resolved spectrum + P3 peaks.

Adapted from ~/git/phenological/STRIP/spectrometer/Bruker/interactive_2d.py:
instead of STRIP ``.npy`` files it loads the folder-51 NMRPipe spectrum and
overlays the system-determined peaks from the MRAi-P3 ``.tab`` peak list.

    /Users/abolanos/.pyenv/versions/tflow/bin/python interactive_51.py          # spectrum + peaks
    /Users/abolanos/.pyenv/versions/tflow/bin/python interactive_51.py --dual   # spectrum vs P3, synchronized
    /Users/abolanos/.pyenv/versions/tflow/bin/python interactive_51.py --ft2 X --tab Y

Interactions (same engine as the original):

  * click + drag      -> rubber-band zoom to the selected region
  * scroll wheel      -> zoom about the cursor
  * 'r' or 'o'        -> reset to the full spectrum
  * '+' / '-'         -> show more / fewer contour levels
  * 'v'               -> toggle the peak overlay on/off
  * 'p' / toolbar     -> pan mode
  * 'f'               -> full screen
  * live readout      -> F2 shift (ppm) and F1 offset (Hz) under the cursor
  * side panels       -> F2 projection (top) and F1 J-slice at the cursor (right)

Requires nmrglue + an interactive matplotlib backend, both present in the
``tflow`` pyenv (``/Users/abolanos/.pyenv/versions/tflow/bin/python``).
"""

import argparse
import os
import sys

import numpy as np

DEFAULT_FT2 = '/tmp/p51/51.ft2'
DEFAULT_TAB = '/tmp/p51_run/51.tab'
DEFAULT_PEAKS = '/tmp/p51_run/51.peaks'
DEFAULT_P3 = '/tmp/p51_run/51.P3'


def load_spectrum(ft2_path):
    """Read the Pipe spectrum; return (spectrum, ppm, f1_hz, y_obs, labels)."""
    import nmrglue as ng

    if not os.path.isfile(ft2_path):
        sys.exit('No spectrum at %s.' % ft2_path)
    dic, spectrum = ng.pipe.read(ft2_path)
    uc_x = ng.pipe.make_uc(dic, spectrum, dim=1)   # F2, direct 1H (ppm)
    uc_y = ng.pipe.make_uc(dic, spectrum, dim=0)   # F1, J axis (ppm)
    ppm = uc_x.ppm_scale()
    y_obs = float(dic['FDF1OBS'])
    f1_hz = uc_y.ppm_scale() * y_obs
    labels = (str(dic.get('FDF2LABEL', 'F2')), str(dic.get('FDF1LABEL', 'F1')))
    return np.asarray(spectrum, dtype=float), ppm, f1_hz, y_obs, labels


def load_peaks(tab_path, peaks_path, y_obs):
    """Peak positions as (f2_ppm, f1_hz); prefers .tab, falls back to .peaks."""
    xp, yp = [], []
    if os.path.isfile(tab_path):
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
            xp.append(float(parts[vl.index('X_PPM')]))
            yp.append(float(parts[vl.index('Y_PPM')]) * y_obs)
        if xp:
            return np.array(xp), np.array(yp)
    if os.path.isfile(peaks_path):
        for line in open(peaks_path):
            s = line.strip()
            if not s or not s[0].isdigit():
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            xp.append(float(parts[1]))            # emitted order is (F2, F1)
            yp.append(float(parts[2]) * y_obs)
    return np.array(xp), np.array(yp)


def run(args=None):
    """Single-panel interactive view: spectrum contours + peak overlay."""
    ap = _parser()
    ns = ap.parse_args(args)
    spectrum, ppm, f1_hz, y_obs, _labels = load_spectrum(ns.ft2)
    if ns.p3only:
        p3, _, _, _, _ = load_spectrum(ns.p3)
        named = [(p3, 'P3 probability', 'tab:blue')]
    else:
        named = [(spectrum, 'Spectrum', 'black')]
        if ns.dual:
            p3, _, _, _, _ = load_spectrum(ns.p3)
            if p3.shape != spectrum.shape:
                sys.exit('Shape mismatch: P3 %s vs spectrum %s' % (p3.shape, spectrum.shape))
            named.append((p3, 'P3 probability', 'tab:blue'))
    px, py = load_peaks(ns.tab, ns.peaks, y_obs)
    print('spectrum %s  |  %d peaks from %s' %
          (spectrum.shape, len(px), ns.tab if os.path.isfile(ns.tab) else ns.peaks))
    return _run_interactive(named, ppm, f1_hz, peaks=(px, py))


def _parser():
    ap = argparse.ArgumentParser(description='Interactive folder-51 contour viewer + peaks.')
    ap.add_argument('--ft2', default=DEFAULT_FT2)
    ap.add_argument('--tab', default=DEFAULT_TAB)
    ap.add_argument('--peaks', default=DEFAULT_PEAKS)
    ap.add_argument('--p3', default=DEFAULT_P3)
    ap.add_argument('--dual', action='store_true',
                    help='spectrum vs P3 side by side, synchronized')
    ap.add_argument('--p3only', action='store_true',
                    help='show the P3 probability map + peaks alone')
    return ap


def _run_interactive(named, ppm, f1_hz, peaks=None, run_context=None):
    """Shared engine for synchronized multi-panel 2D contour views.

    named: [(spectrum, label, color), ...].  All spectra share the ppm/F1 axes.
    peaks: (px_ppm, py_hz) overlaid as red crosses on every main panel.
    Drag-zoom, wheel-zoom, slice/readout updates and contour-level keys act
    on every panel at once.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector, Cursor

    run_context = run_context or {}
    px, py = (np.asarray(peaks[0]), np.asarray(peaks[1])) if peaks is not None else (np.array([]), np.array([]))

    f1size, f2size = named[0][0].shape
    # Chemical-shift convention: high ppm (downfield) on the LEFT.
    f2_limits = (ppm[0], ppm[-1])            # (left, right) = (high ppm, low ppm)
    f1_limits = (f1_hz[0], f1_hz[-1])
    # Per-panel full-array scale so spectrum and P3 (very different units)
    # each get sensible contour levels; levels stay fixed while zooming.
    vmax = [max(float(np.max(np.abs(spec))), 1e-12) for spec, _, _ in named]
    state = {'low': 0.03,
             'show_peaks': True,
             'view': (f2_limits[0], f2_limits[1], f1_limits[0], f1_limits[1])}

    def contour_levels(v):
        lv = np.logspace(np.log10(max(state['low'], 1e-6)), 0.0, 14) * v
        return np.concatenate((-lv[::-1], lv))

    def usable_region(sub, levels):
        """Only call contour() when the region actually spans the levels."""
        if sub.size == 0 or len(levels) == 0:
            return False
        return float(np.min(sub)) < float(np.max(levels)) and \
            float(np.max(sub)) > float(np.min(levels))

    n = len(named)
    fig = plt.figure(figsize=(7 * n + 3, 9))
    gs = fig.add_gridspec(2, n + 1, width_ratios=[4] * n + [1.2], height_ratios=[1, 4],
                          hspace=0.12, wspace=0.30)
    ax_top = fig.add_subplot(gs[0, :])
    mains = [fig.add_subplot(gs[1, k]) for k in range(n)]
    ax_slice = fig.add_subplot(gs[1, n])
    # Keep references to cursors/annotations/selectors alive for the life of
    # the window (widget canvas callbacks vanish if these are garbage-collected).
    keep_alive = {'cursors': None, 'selectors': None}

    def draw_main(ax, spec, label, v):
        ax.clear()
        lo2, hi2 = min(state['view'][:2]), max(state['view'][:2])
        lo1, hi1 = min(state['view'][2:]), max(state['view'][2:])
        i_hi = int(np.argmin(np.abs(ppm - hi2)))
        i_lo = int(np.argmin(np.abs(ppm - lo2)))
        j_lo = int(np.argmin(np.abs(f1_hz - lo1)))
        j_hi = int(np.argmin(np.abs(f1_hz - hi1)))
        c0, c1 = min(i_hi, i_lo), max(i_hi, i_lo) + 1
        r0, r1 = min(j_lo, j_hi), max(j_lo, j_hi) + 1
        c0, c1 = max(c0, 0), min(c1, f2size)
        r0, r1 = max(r0, 0), min(r1, f1size)
        sub = spec[r0:r1, c0:c1]
        lv = contour_levels(v)
        neg = lv[lv < 0]
        pos = lv[lv > 0]
        if usable_region(sub, neg):
            ax.contour(ppm[c0:c1], f1_hz[r0:r1], sub, levels=neg,
                       colors='red', linewidths=0.5)
        if usable_region(sub, pos):
            ax.contour(ppm[c0:c1], f1_hz[r0:r1], sub, levels=pos,
                       colors='k', linewidths=0.5)
        if state['show_peaks'] and len(px):
            m = (px >= lo2) & (px <= hi2) & (py >= lo1) & (py <= hi1)
            if np.any(m):
                ax.scatter(px[m], py[m], s=25, marker='x', c='red',
                           linewidths=1.0, zorder=5)
        tag = '%d peaks' % len(px) if state['show_peaks'] else 'peaks hidden'
        ax.set_xlim(hi2, lo2)                # keep high ppm on the left
        ax.set_ylim(lo1, hi1)
        ax.set_title('%s  (%s, levels from %.4g)' % (label, tag, state['low']), fontsize=10)
        ax.set_xlabel('F2 (ppm)')
        ax.set_ylabel('F1 (Hz)')

    def draw_projection():
        ax_top.clear()
        lo2, hi2 = min(state['view'][:2]), max(state['view'][:2])
        npts = len(ppm)
        # ppm decreases with point index, so bracket the visible edges explicitly
        i_hi = int(np.argmin(np.abs(ppm - hi2)))   # high-ppm (left) edge
        i_lo = int(np.argmin(np.abs(ppm - lo2)))   # low-ppm (right) edge
        lo, hi = min(i_hi, i_lo), max(i_hi, i_lo) + 1
        lo, hi = max(lo, 0), min(hi, npts)
        for spec, label, color in named:
            ax_top.plot(ppm[lo:hi], spec.sum(axis=0)[lo:hi], lw=0.7,
                        color=color, label=label)
        if len(named) > 1:
            ax_top.legend(fontsize=8, loc='upper right')
        ax_top.tick_params(labelbottom=False)
        ax_top.set_xlim(hi2, lo2)

    def redraw():
        for ax, (spec, label, _color), v in zip(mains, named, vmax):
            draw_main(ax, spec, label, v)
        draw_projection()
        fig.canvas.draw_idle()

    # initial F1 slice: the strongest F2 column of the last panel
    col0 = int(np.argmax(np.abs(named[-1][0].sum(axis=0))))
    for spec, label, color in named:
        ax_slice.plot(spec[:, col0], f1_hz, lw=0.7, color=color, label=label)
    if len(named) > 1:
        ax_slice.legend(fontsize=8)
    ax_slice.set_ylim(f1_limits[0], f1_limits[1])
    ax_slice.tick_params(labelleft=False)
    ax_slice.set_title('F1 slice', fontsize=10)
    ax_slice.set_xlabel('intensity')
    ax_slice.ticklabel_format(useOffset=False, axis='x')

    redraw()


    # --- click + drag rubber-band zoom (synchronized across panels) --------------
    def on_select(eclick, erelease):
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if None in (x0, y0, x1, y1) or x0 == x1 or y0 == y1:
            return
        state['view'] = (x0, x1, y0, y1)
        redraw()

    # One selector per main panel; keep them alive so they are not
    # garbage-collected (their canvas callbacks vanish with them).
    keep_alive['selectors'] = [
        RectangleSelector(ax, on_select, useblit=True, button=[1],
                          props=dict(facecolor='blue', alpha=0.2, edgecolor='blue'))
        for ax in mains]


    # --- scroll-wheel zoom about the cursor (all panels together) ---------------
    def on_scroll(event):
        if event.xdata is None or event.inaxes not in mains:
            return
        x0, x1 = event.inaxes.get_xlim()
        y0, y1 = event.inaxes.get_ylim()
        scale = 0.8 if event.button == 'up' else 1.25
        xa = event.xdata - (event.xdata - x0) * scale
        xb = event.xdata + (x1 - event.xdata) * scale
        ya = event.ydata - (event.ydata - y0) * scale
        yb = event.ydata + (y1 - event.ydata) * scale
        state['view'] = (min(xa, xb), max(xa, xb), min(ya, yb), max(ya, yb))
        redraw()

    # --- live readout + F1 slice panel (tracks the cursor in any main panel) ---
    keep_alive['cursors'] = [Cursor(ax, horizOn=True, vertOn=True, useblit=True, color='r', lw=0.8)
                             for ax in mains]
    readout = mains[0].annotate('', xy=(0, 0), xytext=(10, 10), textcoords='offset points',
                                bbox=dict(boxstyle='round4', fc='linen', ec='k', lw=1))
    readout.set_visible(False)

    def on_move(event):
        if event.inaxes not in mains or event.xdata is None:
            return
        f2ppm = event.xdata
        jhz = event.ydata
        col = int(np.argmin(np.abs(ppm - f2ppm)))
        readout.xy = (f2ppm, jhz)
        readout.set_text('%.3f ppm,  %.1f Hz' % (f2ppm, jhz))
        readout.set_visible(True)
        ax_slice.clear()
        for spec, label, color in named:
            ax_slice.plot(spec[:, col] / 1e6, f1_hz, lw=0.7, color=color, label=label)
        if len(named) > 1:
            ax_slice.legend(fontsize=8)
        lo1, hi1 = min(state['view'][2:]), max(state['view'][2:])
        ax_slice.set_ylim(lo1, hi1)
        ax_slice.tick_params(labelleft=False)
        ax_slice.set_title('F1 slice @ %.3f ppm' % f2ppm, fontsize=10)
        ax_slice.set_xlabel(r'intensity ($\times 10^6$)')
        ax_slice.ticklabel_format(useOffset=False, axis='x')
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == '+':
            # '+' reveals more (weaker) contours: lower the base level
            state['low'] = max(state['low'] * 0.6, 1e-4)
            redraw()
        elif event.key == '-' or event.key == '_':
            # '-' hides weak contours: raise the base level
            state['low'] = min(state['low'] / 0.6, 0.9)
            redraw()
        elif event.key == 'v':
            # 'v' toggles the system-peak overlay
            state['show_peaks'] = not state['show_peaks']
            redraw()
        elif event.key in ('r', 'o'):
            state['view'] = (f2_limits[0], f2_limits[1], f1_limits[0], f1_limits[1])
            draw_projection()
            lo1, hi1 = f1_limits
            ax_slice.set_ylim(lo1, hi1)
            redraw()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('key_press_event', on_key)
    # Attach widgets/readout to the figure so nothing is garbage-collected
    # while the window lives; return the event handlers for scripted tests too.
    fig._strip_keep_alive = keep_alive
    fig._strip_readout = readout
    fig._strip_mains = mains
    handlers = {'on_select': on_select, 'on_scroll': on_scroll,
                'on_move': on_move, 'on_key': on_key,
                'state': state, 'named': named, 'ppm': ppm, 'f1_hz': f1_hz,
                'peaks': (px, py),
                'figure': fig, 'mains': mains}
    if run_context.get('show', True):
        plt.show()
    return handlers


if __name__ == '__main__':
    run()

