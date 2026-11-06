"""
Stereo Correspondence Algorithms for Autostereogram Decoding
=============================================================
Implements three disparity estimation strategies:

1. SSD  — Sum of Squared Differences (window-based)
2. NCC  — Normalised Cross-Correlation (window-based)
3. DP   — Dynamic Programming scanline optimisation
           (smoothness-constrained, Viterbi-style)

Each function accepts a grayscale stereogram and the estimated
repeat period, and returns a disparity map (float32, in pixels).
"""

import numpy as np
from scipy.ndimage import uniform_filter, median_filter, gaussian_filter
from scipy.signal import fftconvolve


# ── Period estimation ─────────────────────────────────────────────────────────

def estimate_period(stereo, search_range=(60, 180)):
    """
    Estimate the texture repeat period from the horizontal autocorrelation
    of the stereogram. Returns the dominant period (int).
    
    Strategy: average normalised autocorrelation across all scanlines,
    then find the first strong peak in the search_range.
    """
    if stereo.ndim == 3:
        img = stereo.mean(axis=2).astype(np.float32)
    else:
        img = stereo.astype(np.float32)

    H, W = img.shape
    ac_sum = np.zeros(W, dtype=np.float64)

    for y in range(0, H, max(1, H//50)):   # sample 50 scanlines
        row = img[y] - img[y].mean()
        ac = np.correlate(row, row, mode='full')
        ac = ac[W-1:]   # positive lags only
        ac_sum += ac / (ac[0] + 1e-8)

    ac_avg = ac_sum / (H // max(1, H//50))

    lo, hi = search_range
    peaks = []
    for lag in range(lo, min(hi, W//2)):
        if ac_avg[lag] > ac_avg[lag-1] and ac_avg[lag] > ac_avg[lag+1]:
            peaks.append((ac_avg[lag], lag))

    if not peaks:
        return (lo + hi) // 2    # fallback: midpoint
    return max(peaks)[1]


# ── Cost volume helpers ───────────────────────────────────────────────────────

def _box_filter(arr, half_win):
    """Fast uniform box filter on a 2-D array."""
    return uniform_filter(arr.astype(np.float64), size=2*half_win+1)


def _shifted(img, d):
    """Return img shifted left by d pixels (fill right edge with zeros)."""
    if d == 0:
        return img.copy()
    shifted = np.zeros_like(img)
    if d > 0:
        shifted[:, :-d] = img[:, d:]
    else:
        shifted[:, -d:] = img[:, :d]
    return shifted


# ── Algorithm 1: SSD ──────────────────────────────────────────────────────────

def decode_ssd(stereo, period, half_win=5, search_delta=35):
    """
    Window-based Sum of Squared Differences decoder.

    For each pixel (y, x) and candidate disparity d in
    [period - search_delta, period + search_delta], compute the
    windowed SSD between the image and its d-shifted version.
    The disparity with minimum cost is selected.

    Parameters
    ----------
    stereo      : (H, W) uint8 or float array
    period      : estimated repeat period (pixels)
    half_win    : half-size of the matching window (default 5 → 11×11)
    search_delta: search radius around `period`

    Returns
    -------
    disp_map : (H, W) float32 — raw disparity in pixels
    """
    if stereo.ndim == 3:
        img = stereo.mean(axis=2).astype(np.float32) / 255.0
    else:
        img = stereo.astype(np.float32) / 255.0

    H, W = img.shape
    disp_lo = max(1, period - search_delta)
    disp_hi = min(W - 1, period + search_delta)
    n_disp   = disp_hi - disp_lo + 1

    cost_vol = np.full((H, W, n_disp), np.inf, dtype=np.float32)

    for i, d in enumerate(range(disp_lo, disp_hi + 1)):
        diff2 = (img - _shifted(img, d)) ** 2
        cost_vol[:, :, i] = _box_filter(diff2, half_win).astype(np.float32)

    best_i = np.argmin(cost_vol, axis=2)
    disp_map = (best_i + disp_lo).astype(np.float32)
    return disp_map


# ── Algorithm 2: NCC ──────────────────────────────────────────────────────────

def decode_ncc(stereo, period, half_win=7, search_delta=35):
    """
    Window-based Normalised Cross-Correlation decoder.

    NCC is invariant to local mean and contrast, making it more robust to
    texture brightness variation than SSD.

    NCC(x, d) = Σ[(I(x) - μ(x)) · (I(x-d) - μ(x-d))]
                ─────────────────────────────────────────
                    σ(x) · σ(x-d) · window_area

    Parameters
    ----------
    stereo      : (H, W) uint8 or float array
    period      : estimated repeat period (pixels)
    half_win    : half-size of the matching window
    search_delta: search radius around `period`

    Returns
    -------
    disp_map : (H, W) float32 — raw disparity in pixels
    """
    if stereo.ndim == 3:
        img = stereo.mean(axis=2).astype(np.float32) / 255.0
    else:
        img = stereo.astype(np.float32) / 255.0

    H, W    = img.shape
    disp_lo = max(1, period - search_delta)
    disp_hi = min(W - 1, period + search_delta)
    n_disp  = disp_hi - disp_lo + 1

    # Pre-compute local mean and std for the reference image
    mu_ref  = _box_filter(img, half_win)
    sq_ref  = _box_filter(img**2, half_win)
    var_ref = np.maximum(sq_ref - mu_ref**2, 1e-8)
    std_ref = np.sqrt(var_ref)

    score_vol = np.full((H, W, n_disp), -1.0, dtype=np.float32)

    for i, d in enumerate(range(disp_lo, disp_hi + 1)):
        shifted_img = _shifted(img, d)
        mu_sh   = _box_filter(shifted_img, half_win)
        sq_sh   = _box_filter(shifted_img**2, half_win)
        var_sh  = np.maximum(sq_sh - mu_sh**2, 1e-8)
        std_sh  = np.sqrt(var_sh)

        cross   = _box_filter(img * shifted_img, half_win) - mu_ref * mu_sh
        ncc     = cross / (std_ref * std_sh + 1e-8)
        score_vol[:, :, i] = ncc.astype(np.float32)

    best_i   = np.argmax(score_vol, axis=2)
    disp_map = (best_i + disp_lo).astype(np.float32)

    # Confidence: peak NCC score (used downstream for weighted averaging)
    confidence = np.max(score_vol, axis=2)
    return disp_map, confidence


# ── Algorithm 3: Dynamic Programming scanline optimisation ────────────────────

def decode_dp(stereo, period, search_delta=35, lambda_smooth=20.0):
    """
    Scanline Dynamic Programming decoder.

    Minimises a per-scanline energy:
        E = Σ_x [ data_cost(x, d_x) + λ · |d_x - d_{x-1}| ]

    Uses an L1 smoothness penalty (Viterbi/forward-backward on a 1-D MRF).

    Parameters
    ----------
    stereo        : (H, W) uint8 or float array
    period        : estimated repeat period (pixels)
    search_delta  : disparity search radius
    lambda_smooth : weight of the smoothness term (larger → smoother)

    Returns
    -------
    disp_map : (H, W) float32
    """
    if stereo.ndim == 3:
        img = stereo.mean(axis=2).astype(np.float32) / 255.0
    else:
        img = stereo.astype(np.float32) / 255.0

    H, W    = img.shape
    disp_lo = max(1, period - search_delta)
    disp_hi = min(W - 1, period + search_delta)
    D       = disp_hi - disp_lo + 1
    disp_map = np.zeros((H, W), dtype=np.float32)

    disp_vals = np.arange(disp_lo, disp_hi + 1, dtype=np.float32)

    # Pre-compute smoothness penalty matrix (D x D)
    smooth_mat = lambda_smooth * np.abs(disp_vals[:, None] - disp_vals[None, :]) / max(D - 1, 1)

    for y in range(H):
        row = img[y]
        # Data cost: pixel-wise squared difference for each candidate disparity (vectorised)
        data = np.zeros((W, D), dtype=np.float32)
        for i, d in enumerate(range(disp_lo, disp_hi + 1)):
            diff = row - np.roll(row, d)
            data[:, i] = diff ** 2

        # Forward pass (vectorised over D)
        cost  = data.copy()
        trace = np.zeros((W, D), dtype=np.int32)

        for x in range(1, W):
            # prev: (D,), smooth_mat: (D, D) — for each dst disparity j, min over src
            total = cost[x-1][None, :] + smooth_mat          # (D, D)
            best  = np.argmin(total, axis=1)                  # (D,)
            trace[x] = best
            cost[x]  = data[x] + total[np.arange(D), best]

        # Backward pass: traceback
        best_d = int(np.argmin(cost[W-1]))
        disp_map[y, W-1] = disp_vals[best_d]
        for x in range(W-2, -1, -1):
            best_d = int(trace[x+1, best_d])
            disp_map[y, x] = disp_vals[best_d]

    return disp_map


# ── Disparity → Depth conversion ──────────────────────────────────────────────

def disparity_to_depth(disp_map, period, max_shift, eps=1e-3):
    """
    Convert a disparity map (in pixels) to a relative depth map in [0, 1].

    depth = (disp - period) / max_shift   [clipped to [0, 1]]

    Parameters
    ----------
    disp_map  : (H, W) float array, disparity values in pixels
    period    : base repeat period used during encoding
    max_shift : maximum shift used during encoding

    Returns
    -------
    depth_map : (H, W) float32 in [0, 1]
    """
    depth = (disp_map - period) / (max_shift + eps)
    depth = np.clip(depth, 0, 1).astype(np.float32)
    return depth


# ── Post-processing ───────────────────────────────────────────────────────────

def postprocess(disp_map, method='median', **kwargs):
    """
    Apply post-processing to reduce noise in a disparity map.

    method : 'median'   — 2-D median filter (default, size=5)
             'gaussian' — Gaussian smoothing (sigma=1.5)
             'bilateral'— edge-preserving bilateral-style filter
                          (uses iterative weighted local means)
    """
    if method == 'median':
        size = kwargs.get('size', 5)
        return median_filter(disp_map, size=size).astype(np.float32)

    elif method == 'gaussian':
        sigma = kwargs.get('sigma', 1.5)
        return gaussian_filter(disp_map, sigma=sigma).astype(np.float32)

    elif method == 'bilateral':
        # Simplified bilateral: spatial Gaussian × intensity Gaussian
        sigma_s = kwargs.get('sigma_s', 3.0)
        sigma_r = kwargs.get('sigma_r', 0.1)
        out = disp_map.copy()
        for _ in range(kwargs.get('iters', 3)):
            blurred  = gaussian_filter(out, sigma=sigma_s)
            diff     = np.abs(out - blurred)
            w        = np.exp(-(diff**2) / (2 * sigma_r**2))
            out      = w * out + (1 - w) * blurred
        return out.astype(np.float32)

    else:
        raise ValueError(f"Unknown post-processing method: {method}")


def confidence_weighted_merge(disp_ssd, disp_ncc, conf_ncc, alpha=0.3):
    """
    Merge SSD and NCC disparity maps weighted by NCC confidence scores.
    Higher-confidence NCC regions get more NCC weight.
    """
    conf = np.clip((conf_ncc + 1) / 2, 0, 1)   # map [-1,1] → [0,1]
    w_ncc = conf
    w_ssd = alpha * (1 - conf) + (1 - alpha)
    merged = (w_ncc * disp_ncc + w_ssd * disp_ssd) / (w_ncc + w_ssd)
    return merged.astype(np.float32)
