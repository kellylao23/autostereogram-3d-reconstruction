"""
Main Experiment Runner
======================
Generates the full synthetic dataset, runs all decoding algorithms,
evaluates quantitatively, and produces all publication figures.

Usage:
    python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
import time

from src.generator   import build_synthetic_dataset, DEPTH_SCENES
from src.decoder     import (estimate_period, decode_ssd, decode_ncc,
                              decode_dp, disparity_to_depth,
                              postprocess, confidence_weighted_merge)
from src.evaluation  import evaluate_all
from src.visualization import (fig_scene_overview, fig_algorithm_comparison,
                                fig_metrics_summary, fig_3d_surface,
                                fig_period_estimation, fig_postprocessing,
                                fig_scanline_profile)

# ── Configuration ─────────────────────────────────────────────────────────────
PERIOD    = 120
MAX_SHIFT = 28
HEIGHT    = 400
WIDTH     = 600
DATA_DIR  = 'data'
FIG_DIR   = 'figures'
RES_DIR   = 'results'

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ── Step 1: Build dataset ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Generating synthetic dataset")
print("="*60)
dataset = build_synthetic_dataset(DATA_DIR, period=PERIOD, max_shift=MAX_SHIFT,
                                   height=HEIGHT, width=WIDTH)

# ── Step 2: Figure 1 — scene overview ────────────────────────────────────────
print("\nSTEP 2: Scene overview figure")
fig_scene_overview(dataset, os.path.join(FIG_DIR, 'fig1_scene_overview.png'))

# ── Step 3: Run decoders on all scenes ───────────────────────────────────────
print("\nSTEP 3: Running decoders on all scenes")
print("-"*60)

all_metrics = {}   # {scene: {method: {metric: val}}}
all_results = {}   # {scene: {method: depth_map}}

for scene_name, data in dataset.items():
    stereo    = data['stereo']
    gt_depth  = data['depth']
    true_per  = data['period']
    max_shift = data['max_shift']

    print(f"\n  Scene: {scene_name}")
    t0 = time.time()

    # ── Period estimation ──────────────────────────────────────────────────
    est_period = estimate_period(stereo)
    print(f"    Period: true={true_per}, estimated={est_period}")

    # ── SSD ───────────────────────────────────────────────────────────────
    t = time.time()
    disp_ssd = decode_ssd(stereo, est_period, half_win=5, search_delta=35)
    depth_ssd = disparity_to_depth(disp_ssd, est_period, max_shift)
    print(f"    SSD done in {time.time()-t:.1f}s")

    # ── NCC ───────────────────────────────────────────────────────────────
    t = time.time()
    disp_ncc, conf_ncc = decode_ncc(stereo, est_period, half_win=7, search_delta=35)
    depth_ncc = disparity_to_depth(disp_ncc, est_period, max_shift)
    print(f"    NCC done in {time.time()-t:.1f}s")

    # ── NCC + median post-processing ───────────────────────────────────────
    disp_ncc_pp  = postprocess(disp_ncc, method='median', size=5)
    depth_ncc_pp = disparity_to_depth(disp_ncc_pp, est_period, max_shift)

    # ── Confidence-weighted merge ──────────────────────────────────────────
    disp_merged  = confidence_weighted_merge(disp_ssd, disp_ncc, conf_ncc)
    depth_merged = disparity_to_depth(disp_merged, est_period, max_shift)
    disp_merged_pp = postprocess(disp_merged, method='median', size=5)
    depth_merged_pp = disparity_to_depth(disp_merged_pp, est_period, max_shift)

    # ── DP (only on first 3 scenes to save time) ───────────────────────────
    if scene_name in list(DEPTH_SCENES.keys())[:3]:
        t = time.time()
        disp_dp  = decode_dp(stereo, est_period, search_delta=25, lambda_smooth=15.0)
        depth_dp = disparity_to_depth(disp_dp, est_period, max_shift)
        print(f"    DP  done in {time.time()-t:.1f}s")
    else:
        depth_dp = None

    results_scene = {
        'SSD':           depth_ssd,
        'NCC':           depth_ncc,
        'NCC+Median':    depth_ncc_pp,
        'Merged':        depth_merged_pp,
    }
    if depth_dp is not None:
        results_scene['DP'] = depth_dp

    all_results[scene_name] = results_scene

    # ── Evaluate ───────────────────────────────────────────────────────────
    metrics_scene = {}
    for mname, pred in results_scene.items():
        m = evaluate_all(pred, gt_depth)
        metrics_scene[mname] = m
        print(f"    {mname:>12}: RMSE={m['rmse']:.4f}  "
              f"Pearson={m['pearson_r']:.3f}  SSIM={m['ssim']:.3f}")
    all_metrics[scene_name] = metrics_scene

    # ── Figure 2: algorithm comparison ────────────────────────────────────
    fig_algorithm_comparison(
        scene_name, gt_depth, results_scene,
        os.path.join(FIG_DIR, f'fig2_{scene_name}_comparison.png')
    )

    # ── Figure 5: period estimation (first scene only) ────────────────────
    if scene_name == 'sphere':
        fig_period_estimation(
            stereo, true_per, est_period,
            os.path.join(FIG_DIR, 'fig5_period_estimation.png')
        )
        # ── Figure 6: post-processing (sphere + NCC) ──────────────────────
        pp_results = {
            'Median-5':    disparity_to_depth(postprocess(disp_ncc, 'median', size=5), est_period, max_shift),
            'Gaussian':    disparity_to_depth(postprocess(disp_ncc, 'gaussian', sigma=2.0), est_period, max_shift),
            'Bilateral':   disparity_to_depth(postprocess(disp_ncc, 'bilateral'), est_period, max_shift),
        }
        fig_postprocessing(
            depth_ncc, pp_results, gt_depth,
            os.path.join(FIG_DIR, 'fig6_postprocessing.png')
        )

    # ── Figure 7: scanline profile ────────────────────────────────────────
    fig_scanline_profile(
        scene_name, gt_depth, results_scene,
        os.path.join(FIG_DIR, f'fig7_{scene_name}_scanline.png')
    )

    # ── Figure 4: 3-D surface (best method = Merged) ──────────────────────
    fig_3d_surface(
        scene_name, gt_depth, depth_merged_pp, 'Merged (NCC+SSD+Median)',
        os.path.join(FIG_DIR, f'fig4_{scene_name}_3d.png')
    )

    print(f"  Total scene time: {time.time()-t0:.1f}s")


# ── Step 4: Aggregate metrics figure ─────────────────────────────────────────
print("\nSTEP 4: Aggregate metrics figures")
# Only include methods that appear in every scene
common_methods = set(next(iter(all_metrics.values())).keys())
for sm in all_metrics.values():
    common_methods &= set(sm.keys())

all_metrics_filtered = {scene: {m: v for m, v in sm.items() if m in common_methods}
                        for scene, sm in all_metrics.items()}

fig_metrics_summary(
    all_metrics_filtered,
    metric_keys=['rmse', 'pearson_r', 'ssim', 'bad_pix_5pct'],
    out_path=os.path.join(FIG_DIR, 'fig3_metrics_summary.png')
)

# ── Step 5: Save metrics to JSON ──────────────────────────────────────────────
print("\nSTEP 5: Saving metrics JSON")
# Convert to serialisable format
def to_json(obj):
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, dict):        return {k: to_json(v) for k,v in obj.items()}
    return obj

with open(os.path.join(RES_DIR, 'metrics.json'), 'w') as f:
    json.dump(to_json(all_metrics), f, indent=2)
print("  Saved: results/metrics.json")

# ── Step 6: Print summary table ───────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY TABLE — RMSE (lower is better)")
print("="*60)
methods = list(next(iter(all_metrics.values())).keys())
print(f"{'Scene':<18}", end="")
for m in methods:
    print(f"  {m:>14}", end="")
print()
print("-" * (18 + 16 * len(methods)))
for scene, sm in all_metrics.items():
    print(f"{scene:<18}", end="")
    for m in methods:
        v = sm[m].get('rmse', float('nan'))
        print(f"  {v:>14.4f}", end="")
    print()

print("\n" + "="*60)
print("SUMMARY TABLE — Pearson r (higher is better)")
print("="*60)
print(f"{'Scene':<18}", end="")
for m in methods:
    print(f"  {m:>14}", end="")
print()
print("-" * (18 + 16 * len(methods)))
for scene, sm in all_metrics.items():
    print(f"{scene:<18}", end="")
    for m in methods:
        v = sm[m].get('pearson_r', float('nan'))
        print(f"  {v:>14.4f}", end="")
    print()

print("\nAll figures saved to:", os.path.abspath(FIG_DIR))
print("Metrics saved to:    ", os.path.abspath(RES_DIR))
print("\nDone.")
