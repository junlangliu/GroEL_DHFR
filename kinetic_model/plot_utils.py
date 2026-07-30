"""Plotting helpers shared by figure scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

# Shared typography and canvas sizes for all figures.
FONT_LABEL = 14
FONT_TICK = 12
FONT_ANNOT = 12
FONT_CONTOUR = 10
FONT_LEGEND = 11
LINE_FIGSIZE = (5.5, 4.4)


def parse_ln_expression(expr_or_value):
    if isinstance(expr_or_value, (int, float)):
        return float(expr_or_value)
    if isinstance(expr_or_value, str):
        safe_globals = {"__builtins__": {}}
        safe_locals = {"ln": np.log}
        return float(eval(expr_or_value, safe_globals, safe_locals))
    raise TypeError(f"Unsupported ln expression type: {type(expr_or_value)}")


def set_frame_style(ax_obj, line_width):
    for spine in ax_obj.spines.values():
        spine.set_linewidth(line_width)
    ax_obj.tick_params(width=line_width)


def smooth_boundary_logx(x_vals, window):
    if window < 3 or window % 2 == 0 or len(x_vals) < window:
        return x_vals
    logx = np.log10(np.asarray(x_vals))
    pad = window // 2
    logx_pad = np.pad(logx, (pad, pad), mode="edge")
    smoothed = np.convolve(logx_pad, np.ones(window) / window, mode="valid")
    return 10**smoothed


def derivative_boundaries(matrix, k_oc_vals, consecutive_n, threshold_fraction):
    """Row-wise sensitivity boundaries along log10(k_oc)."""
    log_k_oc = np.log10(k_oc_vals)
    n_rows = matrix.shape[0]
    first = np.zeros(n_rows)
    second = np.zeros(n_rows)
    for i in range(n_rows):
        dv = np.gradient(matrix[i, :], log_k_oc)
        max_dv = np.max(dv)
        if not np.isfinite(max_dv) or max_dv <= 0:
            first[i] = k_oc_vals[0]
            second[i] = k_oc_vals[-1]
            continue
        thr = threshold_fraction * max_dv
        idx_first = 0
        for j in range(0, len(k_oc_vals) - consecutive_n + 1):
            if np.all(dv[j : j + consecutive_n] > thr):
                idx_first = j
                break
        first[i] = k_oc_vals[idx_first]
        idx_second = len(k_oc_vals) - 1
        for j in range(idx_first + consecutive_n, len(k_oc_vals) - consecutive_n + 1):
            if np.all(dv[j : j + consecutive_n] < thr):
                idx_second = j
                break
        second[i] = k_oc_vals[idx_second]
    return first, second


def add_contours(ax, x, y, z, levels, color="white", place_at_right=False):
    contour_set = ax.contour(
        x, y, z, levels=levels, colors=color, linewidths=0.9, alpha=0.85, zorder=6
    )
    if not place_at_right:
        labels = ax.clabel(
            contour_set,
            inline=False,
            fontsize=FONT_CONTOUR,
            fmt="%g",
            inline_spacing=3,
            use_clabeltext=True,
        )
        for txt in labels:
            txt.set_color(color)
            txt.set_alpha(0.95)
            txt.set_zorder(7)
        return

    x_max = float(np.max(x))
    for level, segments in zip(contour_set.levels, contour_set.allsegs):
        if not segments:
            continue
        best_x, best_y = -np.inf, None
        for seg in segments:
            if len(seg) == 0:
                continue
            idx = int(np.argmax(seg[:, 0]))
            if float(seg[idx, 0]) > best_x:
                best_x = float(seg[idx, 0])
                best_y = float(seg[idx, 1])
        if best_y is None:
            continue
        if best_x * 1.03 < x_max:
            text_x, text_ha = best_x * 1.03, "left"
        else:
            text_x, text_ha = best_x / 1.03, "right"
        ax.text(
            text_x,
            best_y,
            f"{level:g}",
            fontsize=FONT_CONTOUR,
            color=color,
            va="center",
            ha=text_ha,
            alpha=0.95,
            zorder=7,
        )


def heatmap_axes(fig, layout_fallback=True):
    heatmap_pos = [0.16, 0.18, 0.52, 0.66]
    fig_w, fig_h = fig.get_size_inches()
    heatmap_square_h = min(heatmap_pos[3], heatmap_pos[2] * (fig_w / fig_h))
    cbar_y = heatmap_pos[1] + 0.5 * (heatmap_pos[3] - heatmap_square_h)
    cbar_pos = [0.90, cbar_y, 0.03, heatmap_square_h]
    ax = fig.add_axes(heatmap_pos)
    return ax, cbar_pos


def save_png(fig, name: str, dpi: int = 300) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved {out}")
    return out


def positive_log_limits(matrix, floor=1e-6):
    """Return (vmin, vmax) for LogNorm from positive finite entries."""
    positive = matrix[np.isfinite(matrix) & (matrix > 0)]
    if positive.size == 0:
        return floor, 1.0
    vmin = float(np.nanmin(positive))
    vmax = float(np.nanmax(positive))
    if np.isclose(vmin, vmax):
        vmax = vmin * 1.000001
    return vmin, vmax


def contour_levels_in_range(candidates, vmin, vmax):
    return [level for level in candidates if vmin < level < vmax]
