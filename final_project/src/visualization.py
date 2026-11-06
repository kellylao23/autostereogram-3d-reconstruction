"""
Visualization Module
====================
Generates publication-quality figures for the autostereogram project.
All figures are saved as high-resolution PNGs (300 dpi).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import os

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'image.cmap':        'viridis',
})

CMAP_DEPTH  = 'plasma'
CMAP_ERROR  = 'RdYlGn_r'
CMAP_STEREO = 'gray'


def _label(ax, text, loc='upper left', fontsize=9):
    ax.text(0.02 if 'left' in loc else 0.98,
            0.97 if 'upper' in loc else 0.03,
            text,
            transform=ax.transAxes, fontsize=fontsize,
            verticalalignment='top' if 'upper' in loc else 'bottom',
            horizontalalignment='left' if 'left' in loc else 'right',
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.4, ec='none'))


# ── Figure 1: Scene overview ──────────────────────────────────────────────────

def fig_scene_overview(dataset, out_path):
    """
    Panel grid showing each synthetic scene: depth map and stereogram side-by-side.
    """
    scenes = list(dataset.keys())
    n = len(scenes)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.2 * n))
    fig.suptitle('Synthetic Test Scenes: Depth Maps and Autostereograms',
                 fontsize=13, fontweight='bold', y=1.01)

    for row, name in enumerate(scenes):
        depth  = dataset[name]['depth']
        stereo = dataset[name]['stereo']

        ax_d = axes[row, 0]
        ax_s = axes[row, 1]

        im = ax_d.imshow(depth, cmap=CMAP_DEPTH, vmin=0, vmax=1)
        ax_d.set_title(f'{name.replace("_"," ").title()} — Ground-truth depth')
        ax_d.axis('off')
        plt.colorbar(im, ax=ax_d, fraction=0.03, pad=0.02).set_label('Depth')

        ax_s.imshow(stereo, cmap=CMAP_STEREO)
        ax_s.set_title(f'{name.replace("_"," ").title()} — Autostereogram')
        ax_s.axis('off')
        _label(ax_s, f'period={dataset[name]["period"]}px  '
                     f'max_shift={dataset[name]["max_shift"]}px')

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 2: Algorithm comparison for one scene ──────────────────────────────

def fig_algorithm_comparison(scene_name, gt_depth, results_dict, out_path):
    """
    Side-by-side comparison of GT depth and each algorithm's reconstruction.
    results_dict: {method_name: depth_map}
    """
    methods = list(results_dict.keys())
    n_cols  = len(methods) + 1   # GT + each method
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))
    fig.suptitle(f'Algorithm Comparison — {scene_name.replace("_"," ").title()}',
                 fontsize=13, fontweight='bold')

    # Row 0: depth maps
    im = axes[0, 0].imshow(gt_depth, cmap=CMAP_DEPTH, vmin=0, vmax=1)
    axes[0, 0].set_title('Ground Truth', fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    for col, (mname, pred) in enumerate(results_dict.items(), 1):
        pred_n = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        im = axes[0, col].imshow(pred_n, cmap=CMAP_DEPTH, vmin=0, vmax=1)
        axes[0, col].set_title(mname, fontweight='bold')
        axes[0, col].axis('off')
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

    # Row 1: absolute error maps
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, 'Error maps\n(|pred − GT|)',
                    ha='center', va='center', fontsize=11)

    for col, (mname, pred) in enumerate(results_dict.items(), 1):
        pred_n = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        err    = np.abs(pred_n - gt_depth)
        im = axes[1, col].imshow(err, cmap=CMAP_ERROR, vmin=0, vmax=0.3)
        axes[1, col].set_title(f'{mname} error')
        axes[1, col].axis('off')
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 3: Metrics bar chart ───────────────────────────────────────────────

def fig_metrics_summary(all_results, metric_keys, out_path):
    """
    Bar chart comparing methods across scenes and metrics.
    all_results: {scene: {method: {metric: val}}}
    metric_keys: list of metrics to plot
    """
    scenes  = list(all_results.keys())
    methods = list(next(iter(all_results.values())).keys())
    n_met   = len(metric_keys)

    fig, axes = plt.subplots(1, n_met, figsize=(5 * n_met, 5))
    if n_met == 1:
        axes = [axes]
    fig.suptitle('Reconstruction Quality — Method Comparison Across Scenes',
                 fontsize=13, fontweight='bold')

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for ax, metric in zip(axes, metric_keys):
        x      = np.arange(len(scenes))
        width  = 0.8 / len(methods)

        for j, (mname, color) in enumerate(zip(methods, colors)):
            vals = [all_results[s][mname].get(metric, np.nan) for s in scenes]
            bars = ax.bar(x + j * width, vals, width, label=mname,
                          color=color, alpha=0.88, edgecolor='white', linewidth=0.5)

        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([s.replace('_', '\n') for s in scenes], fontsize=8)
        ax.set_title(metric.upper().replace('_', ' '))
        ax.legend(fontsize=8)
        # Lower is better for error metrics
        if metric in ('mse', 'rmse', 'mae', 'bad_pix_5pct', 'bad_pix_10pct', 'epe'):
            ax.set_ylabel('↓ lower is better')
        else:
            ax.set_ylabel('↑ higher is better')

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 4: 3-D surface reconstruction ─────────────────────────────────────

def fig_3d_surface(scene_name, gt_depth, pred_depth, method_name, out_path):
    """
    Side-by-side 3-D surface plots: GT vs. reconstructed depth.
    """
    H, W   = gt_depth.shape
    step   = max(1, min(H, W) // 80)   # downsample for speed
    Y, X   = np.mgrid[0:H:step, 0:W:step]
    gt_ds  = gt_depth[::step, ::step]
    pr_ds  = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-8)
    pr_ds  = pr_ds[::step, ::step]

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(f'3-D Surface Reconstruction — {scene_name.replace("_"," ").title()}',
                 fontsize=13, fontweight='bold')

    for col, (data, title) in enumerate([(gt_ds, 'Ground Truth'),
                                          (pr_ds, f'{method_name} Reconstruction')]):
        ax = fig.add_subplot(1, 2, col+1, projection='3d')
        ax.plot_surface(X, Y, data, cmap=CMAP_DEPTH, linewidth=0,
                        antialiased=True, rcount=50, ccount=50, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel('x (px)', fontsize=8)
        ax.set_ylabel('y (px)', fontsize=8)
        ax.set_zlabel('Depth', fontsize=8)
        ax.view_init(elev=30, azim=-60)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 5: Period estimation diagnostics ───────────────────────────────────

def fig_period_estimation(stereo, true_period, estimated_period, out_path):
    """
    Autocorrelation plot used to estimate the repeat period.
    """
    if stereo.ndim == 3:
        img = stereo.mean(axis=2).astype(np.float32)
    else:
        img = stereo.astype(np.float32)

    H, W = img.shape
    ac_sum = np.zeros(W, dtype=np.float64)
    for y in range(0, H, max(1, H // 50)):
        row = img[y] - img[y].mean()
        ac  = np.correlate(row, row, mode='full')[W-1:]
        ac_sum += ac / (ac[0] + 1e-8)
    ac_avg = ac_sum / (H // max(1, H // 50))

    lags = np.arange(W)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Repeat Period Estimation via Autocorrelation', fontsize=13, fontweight='bold')

    axes[0].imshow(stereo[:80, :300], cmap=CMAP_STEREO)
    axes[0].set_title('Stereogram crop (first 80 rows)')
    axes[0].axis('off')

    ax = axes[1]
    ax.plot(lags[:W//2], ac_avg[:W//2], color='steelblue', lw=1.2, label='Avg autocorrelation')
    ax.axvline(true_period, color='green',  lw=1.5, ls='--', label=f'True period = {true_period}')
    ax.axvline(estimated_period, color='red', lw=1.5, ls=':', label=f'Estimated = {estimated_period}')
    ax.set_xlabel('Lag (pixels)')
    ax.set_ylabel('Normalised autocorrelation')
    ax.set_title('Horizontal autocorrelation')
    ax.legend()
    ax.set_xlim(0, W // 2)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 6: Post-processing comparison ─────────────────────────────────────

def fig_postprocessing(raw_depth, pp_results, gt_depth, out_path):
    """
    Compare raw disparity map vs. different post-processing strategies.
    pp_results: {'method_name': processed_depth}
    """
    items = [('Raw', raw_depth)] + list(pp_results.items())
    n     = len(items)

    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))
    fig.suptitle('Post-Processing Comparison', fontsize=13, fontweight='bold')

    for col, (name, depth) in enumerate(items):
        d_n = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        axes[0, col].imshow(d_n, cmap=CMAP_DEPTH, vmin=0, vmax=1)
        axes[0, col].set_title(name)
        axes[0, col].axis('off')

        err = np.abs(d_n - gt_depth)
        axes[1, col].imshow(err, cmap=CMAP_ERROR, vmin=0, vmax=0.3)
        rmse_val = np.sqrt(np.mean(err**2))
        axes[1, col].set_title(f'Error (RMSE={rmse_val:.3f})')
        axes[1, col].axis('off')

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 7: Scanline depth profile ─────────────────────────────────────────

def fig_scanline_profile(scene_name, gt_depth, results_dict, out_path, scanline=None):
    """
    Plot depth values along a representative horizontal scanline.
    """
    H, W = gt_depth.shape
    y    = scanline if scanline else H // 2

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gt_depth[y], color='black', lw=2, label='Ground Truth', zorder=10)

    colors = plt.cm.Set1(np.linspace(0, 0.8, len(results_dict)))
    for (mname, pred), color in zip(results_dict.items(), colors):
        pred_n = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        ax.plot(pred_n[y], color=color, lw=1.5, ls='--', label=mname, alpha=0.85)

    ax.set_title(f'Scanline Depth Profile — {scene_name.replace("_"," ").title()} (y={y})')
    ax.set_xlabel('Pixel (x)')
    ax.set_ylabel('Normalised depth')
    ax.legend()
    ax.set_xlim(0, W)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")
