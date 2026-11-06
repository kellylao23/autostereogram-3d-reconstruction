"""
Evaluation Metrics for Depth Map Reconstruction
================================================
Quantitative measures comparing reconstructed depth maps
to ground-truth depth maps from synthetic stereograms.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr


def mse(pred, gt):
    """Mean Squared Error."""
    return float(np.mean((pred - gt) ** 2))


def rmse(pred, gt):
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(pred, gt)))


def mae(pred, gt):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(pred - gt)))


def bad_pixel_ratio(pred, gt, threshold=0.05):
    """
    Fraction of pixels where |pred - gt| > threshold.
    Reports depth accuracy at a given tolerance level.
    """
    return float(np.mean(np.abs(pred - gt) > threshold))


def pearson_r(pred, gt):
    """Pearson correlation coefficient between flattened maps."""
    r, _ = pearsonr(pred.ravel(), gt.ravel())
    return float(r)


def spearman_r(pred, gt):
    """Spearman rank correlation — measures ordinal depth ordering."""
    r, _ = spearmanr(pred.ravel(), gt.ravel())
    return float(r)


def structural_similarity(pred, gt, win=7):
    """
    Simplified SSIM-like metric without external dependency.
    Measures local luminance, contrast, and structure agreement.
    """
    from scipy.ndimage import uniform_filter
    C1, C2 = 0.01**2, 0.03**2
    f = lambda x: uniform_filter(x.astype(np.float64), size=win)
    mu1, mu2 = f(pred), f(gt)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1*mu2
    s1 = f(pred**2) - mu1_sq
    s2 = f(gt**2) - mu2_sq
    s12 = f(pred * gt) - mu12
    num = (2*mu12 + C1) * (2*s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return float(np.mean(num / (den + 1e-12)))


def disparity_pixel_error(pred_disp, gt_disp, thresholds=(1, 2, 3)):
    """
    Fraction of pixels within N-pixel disparity error (EPE-style).
    Returns dict: {'epe': ..., 'pct_1px': ..., 'pct_2px': ..., ...}
    """
    diff = np.abs(pred_disp - gt_disp)
    result = {'epe': float(diff.mean())}
    for t in thresholds:
        result[f'pct_{t}px'] = float(np.mean(diff <= t))
    return result


def evaluate_all(pred_depth, gt_depth, pred_disp=None, gt_disp=None):
    """
    Compute a comprehensive suite of metrics.

    Returns
    -------
    dict of metric_name → value
    """
    # Align scale: normalise both to [0,1] for comparable MSE
    pred_n = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-8)
    gt_n   = (gt_depth   - gt_depth.min())   / (gt_depth.max() - gt_depth.min()   + 1e-8)

    metrics = {
        'mse':          mse(pred_n, gt_n),
        'rmse':         rmse(pred_n, gt_n),
        'mae':          mae(pred_n, gt_n),
        'bad_pix_5pct': bad_pixel_ratio(pred_n, gt_n, threshold=0.05),
        'bad_pix_10pct':bad_pixel_ratio(pred_n, gt_n, threshold=0.10),
        'pearson_r':    pearson_r(pred_n, gt_n),
        'spearman_r':   spearman_r(pred_n, gt_n),
        'ssim':         structural_similarity(pred_n, gt_n),
    }

    if pred_disp is not None and gt_disp is not None:
        disp_metrics = disparity_pixel_error(pred_disp, gt_disp)
        metrics.update(disp_metrics)

    return metrics


def print_metrics(metrics, scene='', method=''):
    """Pretty-print a metrics dict."""
    header = f"{'Scene':>15} | {'Method':>10} | " + \
             " | ".join(f"{k:>12}" for k in metrics)
    vals   = f"{scene:>15} | {method:>10} | " + \
             " | ".join(f"{v:>12.4f}" if isinstance(v, float) else f"{v:>12}" 
                        for v in metrics.values())
    return header, vals
