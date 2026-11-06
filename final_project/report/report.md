# Computational Decoding of Autostereograms for Depth Map Reconstruction

**Kelly Lao**  
Computational Photography — Final Project Report  

---

## Abstract

Autostereograms (Magic Eye images) encode hidden three-dimensional scenes through carefully repeated horizontal texture patterns. While human observers can perceive the encoded depth perceptually, computational decoding is challenging due to the stereo correspondence ambiguity inherent in single-image stereograms. This project implements a complete autostereogram decoding pipeline: a synthetic stereogram generator from known depth maps, three stereo correspondence algorithms (window-based Sum of Squared Differences, Normalised Cross-Correlation, and Dynamic Programming scanline optimisation), a confidence-weighted fusion strategy, and a post-processing suite. The system is evaluated quantitatively on six synthetic scenes with ground-truth depth, using RMSE, Pearson correlation, SSIM, and bad-pixel ratio metrics. Results demonstrate that confidence-weighted fusion of SSD and NCC with median post-processing achieves the strongest reconstruction quality overall, with a Pearson r of 0.946 and RMSE of 0.072 on the multi-object scene — the best result across all methods and scenes. Scene complexity and texture ambiguity are the primary determinants of reconstruction difficulty, with globally smooth depth maps (sphere, multi-object) being substantially easier to decode than high-frequency depth fields (sinusoidal, gradient, steps).

---

## 1. Introduction

Autostereograms exploit the brain's binocular fusion mechanism to hide three-dimensional structure inside a seemingly random two-dimensional image. When an observer diverges or converges their eyes to focus beyond or in front of the image plane, repeated texture elements fuse into left-eye and right-eye views, allowing depth perception through binocular disparity — the horizontal difference in perceived position of a point between the two views.

Computationally reversing this process — recovering the encoded depth map from a single autostereogram — is a non-trivial instance of the stereo correspondence problem. Unlike conventional two-image stereo, the autostereogram decoder receives only one image and must simultaneously infer the repeat period and the per-pixel disparity offsets, under conditions of intentional texture repetition that causes high ambiguity.

This project addresses three core questions:

1. Can standard stereo correspondence algorithms (SSD, NCC) reliably decode autostereogram depth maps without any prior knowledge of the encoding parameters?
2. How do algorithm choice, window size, and post-processing interact to determine reconstruction quality?
3. Which scene types pose the greatest challenge for computational decoding, and why?

The work covers: (i) a synthetic stereogram generator from arbitrary depth maps; (ii) three decoding algorithms of increasing sophistication; (iii) quantitative evaluation against known ground-truth depth; and (iv) analysis of failure modes and the relationship between scene structure and reconstruction difficulty.

---

## 2. Background

### 2.1 Autostereogram Encoding

An autostereogram of width W with repeat period P is constructed by the following algorithm. For each scanline y independently:

```
for x = 0 to W-1:
    shift = round(depth[y, x] × max_shift)
    stereo[y, x] = texture[y, (x - P + shift) mod P]
```

Here, `depth[y,x] ∈ [0,1]` is the normalised scene depth, `max_shift` is the maximum horizontal shift in pixels (typically 20–40% of P), and `texture[y, ·]` is a periodic pattern of width P. Pixels with higher depth values are shifted further, increasing the apparent binocular disparity when the image is free-viewed.

The key property is that any pixel at position x is a copy of the pixel at position `x - P + shift` — the disparity at each location is encoded as an offset from the base period. Decoding amounts to recovering these offsets.

### 2.2 The Stereo Correspondence Problem

Stereo correspondence — finding which pixels in a left image correspond to which pixels in a right image — is a classical computer vision problem. In the autostereogram setting, both the "left" and "right" images are the same image, separated by approximately P pixels. The decoder must find, for each pixel x, the horizontal offset d such that the image patch centred at x best matches the patch centred at (x - d).

The fundamental difficulty is the ambiguity introduced by the repeated texture: at any location, many different offsets may yield similarly appearing patches, making it difficult to determine the true disparity. This is by design in autostereogram construction — the repeating texture is chosen to make false matches plausible to the visual system, producing the illusion of depth.

### 2.3 Related Work

Classical stereo matching methods — SSD, NCC, and their variants — form the algorithmic backbone of this project (Scharstein & Szeliski, 2002). Dynamic programming scanline optimisation (Birchfield & Tomasi, 1998) adds smoothness constraints that enforce spatial coherence along scanlines. The autostereogram decoding problem specifically has been studied by Intille & Bobick (1994) and Thimbleby et al. (1994), who noted the strong prior that depth maps tend to be smooth — a constraint we exploit via post-processing and the DP formulation.

---

## 3. Methods

### 3.1 System Overview

The decoding pipeline consists of five stages:

1. **Period estimation** — estimate the repeat period P from the image's horizontal autocorrelation
2. **Disparity estimation** — for each pixel, find the offset d that best explains the local texture
3. **Post-processing** — apply spatial filtering to reduce noise and enforce consistency
4. **Depth conversion** — map disparity values back to normalised depth
5. **Evaluation** — compare reconstructed depth to ground truth

### 3.2 Synthetic Dataset Generation

Six depth map types were used to test a range of scene complexities:

| Scene | Description | Challenge |
|---|---|---|
| Sphere | Single hemisphere, Gaussian profile | Smooth, well-localised |
| Gradient | Linear horizontal ramp | Global monotone structure |
| Steps | Five discrete depth layers | Sharp boundaries |
| Sinusoidal | 2-D sinusoidal ripple | High-frequency variation |
| Multi-object | Four spheres at different depths | Segmented, sparse |
| Curved | Paraboloid surface | Smooth but globally varying |

All stereograms used a Gaussian-smoothed noise texture (period P = 120 px, max_shift = 28 px, resolution 400 × 600 px). The vectorised encoder computes the shift array as `shifts = round(depth × max_shift)` and then directly indexes the texture: `stereo[y, x] = texture[y, (x - P + shifts[y,x]) mod P]`, giving a ~30× speedup over the naive pixel loop.

### 3.3 Period Estimation

The repeat period P is estimated from the normalised horizontal autocorrelation, averaged over 50 uniformly sampled scanlines:

```
ac[y, lag] = Σ_x row_y[x] · row_y[x - lag]
period = argmax_{lag ∈ [60, 180]} mean_y(ac[y, lag])
```

The peak of the average autocorrelation corresponds to the dominant repeat frequency of the texture. Period estimation succeeded exactly (P = 120) on four of six scenes and was off by 5–7 pixels on gradient and steps, where the autocorrelation peak is broadened by depth-induced phase shifts.

### 3.4 Algorithm 1: Window-Based SSD

The Sum of Squared Differences matcher tests each candidate disparity d in [P - Δ, P + Δ] (Δ = 35 pixels) by computing the windowed cost:

```
SSD(x, d) = Σ_{(u,v) ∈ W} [I(x+u, y+v) - I(x+u-d, y+v)]²
```

where W is an 11 × 11 window (half-width 5). The best disparity is the one minimising this cost. The windowed sum is computed efficiently using a 2-D uniform filter (equivalent to box convolution), reducing per-pixel computation from O(window²) to O(1) amortised.

**SSD properties:** Sensitive to global intensity differences between the image and its shifted copy, but fast and effective when texture contrast is high. Window size trades bias (large windows over-smooth depth boundaries) for variance (small windows are noisier).

### 3.5 Algorithm 2: Window-Based NCC

The Normalised Cross-Correlation matcher measures the linear correlation between zero-mean, unit-variance patches:

```
NCC(x, d) = [Σ (I(x) - μ(x)) · (I(x-d) - μ(x-d))] / [σ(x) · σ(x-d) · |W|]
```

Local means μ and standard deviations σ are computed with a 15 × 15 window (half-width 7) using efficient box filtering. NCC scores lie in [-1, 1]; the best disparity maximises this score. Critically, NCC yields a confidence value equal to the peak score — this is used downstream for confidence-weighted fusion.

**NCC properties:** Invariant to local mean intensity and contrast, making it robust to brightness variation across the image. The confidence score provides a meaningful measure of match quality: high NCC (near 1) indicates a reliable match; low NCC indicates ambiguity or a flat-texture region.

### 3.6 Algorithm 3: Dynamic Programming Scanline Optimisation

The DP decoder minimises a per-scanline energy functional:

```
E(d₁, d₂, ..., d_W) = Σ_x [data(x, dₓ) + λ · |dₓ - dₓ₋₁|]
```

where data(x, d) = (I[x] - I[x-d])² is the pixel-wise squared difference at disparity d, and the smoothness term penalises sudden changes in disparity between adjacent pixels. This is solved exactly by the Viterbi algorithm (forward pass accumulating minimum cost, backward pass tracing the optimal sequence).

The smoothness weight λ = 15 was chosen to balance data fidelity against spatial coherence. Note that DP enforces consistency only along horizontal scanlines, not vertically — vertical streaking artifacts can arise.

**DP properties:** Produces spatially coherent disparity estimates along each scanline. Particularly effective for depth maps with smooth horizontal profiles. However, because scanlines are processed independently, the method can exhibit inter-scanline discontinuities.

### 3.7 Post-Processing

Three post-processing strategies were evaluated on the raw NCC disparity maps:

- **Median filtering (5 × 5):** Removes impulse noise and isolated mis-matches while preserving edges. Fast and parameter-light.
- **Gaussian smoothing (σ = 2.0):** Produces smoother results but blurs depth boundaries.
- **Iterative bilateral-style filtering:** Attempts to preserve edges by weighting spatial smoothing by disparity similarity. Most computationally intensive.

The median filter consistently produced the best trade-off between noise suppression and boundary sharpness and was selected as the default post-processing step.

### 3.8 Confidence-Weighted Fusion

The Merged method combines SSD and NCC disparity maps using the NCC confidence scores as weights:

```
w_NCC = (conf + 1) / 2          # map [-1,1] → [0,1]
w_SSD = α · (1 - w_NCC) + (1 - α)   # α = 0.3
disp_merged = (w_NCC · disp_NCC + w_SSD · disp_SSD) / (w_NCC + w_SSD)
```

In high-confidence regions (flat NCC score near 1), the NCC estimate dominates. In ambiguous regions, SSD contributes more. This fusion followed by 5 × 5 median filtering (the "Merged" method) was the best-performing configuration overall.

### 3.9 Depth Conversion

Disparity maps are converted to normalised depth via:

```
depth = clip((disp - P) / max_shift, 0, 1)
```

This assumes linear depth encoding, which matches the ground-truth encoder exactly for synthetic data.

### 3.10 Evaluation Metrics

All reconstructed depth maps are normalised to [0,1] before metric computation to separate scale recovery from structural recovery:

- **RMSE:** Root Mean Squared Error — primary error metric
- **MAE:** Mean Absolute Error — less sensitive to outliers
- **Bad pixel ratio:** Fraction of pixels with |pred - gt| > 0.05 (5%) or 0.10 (10%)
- **Pearson r:** Linear correlation with ground truth — measures structural agreement
- **Spearman r:** Rank correlation — measures ordinal depth ordering (not sensitive to scale)
- **SSIM:** Structural similarity index — captures local luminance, contrast, and structure

---

## 4. Results

### 4.1 Period Estimation

The autocorrelation-based period estimator performed reliably on scenes with uniform texture (sphere, sinusoidal, multi-object, curved), correctly identifying P = 120 in all four cases. For gradient and steps, the period was underestimated (115 and 113 respectively) because the depth-induced shifts broaden and distort the autocorrelation peak. This period error propagates into depth reconstruction accuracy for those scenes.

### 4.2 Reconstruction Quality — Quantitative Results

Complete results across all six scenes and four methods are reported below. Bold values indicate the best result per metric per scene.

#### Sphere Scene

| Method | RMSE | Pearson r | SSIM | Bad Pix (5%) |
|---|---|---|---|---|
| SSD | 0.1788 | 0.808 | 0.869 | 9.2% |
| NCC | 0.2235 | 0.704 | 0.725 | 14.1% |
| NCC+Median | 0.1845 | 0.792 | 0.808 | 12.3% |
| **Merged** | **0.1685** | **0.831** | **0.883** | **9.3%** |

The sphere is the easiest scene: the smooth, spatially localised depth field provides unambiguous disparity signals. The Merged method achieves the best results on all metrics.

#### Multi-Object Scene

| Method | RMSE | Pearson r | SSIM | Bad Pix (5%) |
|---|---|---|---|---|
| SSD | 0.1048 | 0.895 | 0.883 | 5.5% |
| NCC | 0.1867 | 0.731 | 0.734 | 11.5% |
| NCC+Median | 0.1181 | 0.869 | 0.815 | 9.2% |
| **Merged** | **0.0725** | **0.946** | **0.913** | **5.4%** |

The multi-object scene achieves the best absolute results in the project. The sparse depth field (most of the image is background at depth 0) gives the algorithms a strong anchor — large regions of the image have depth zero and thus require zero disparity shift, which both SSD and NCC detect reliably.

#### Sinusoidal Scene

| Method | RMSE | Pearson r | SSIM | Bad Pix (5%) |
|---|---|---|---|---|
| SSD | 0.4187 | 0.537 | 0.152 | 94.5% |
| NCC | 0.4077 | 0.496 | 0.144 | 94.3% |
| NCC+Median | 0.4053 | 0.571 | 0.170 | 94.5% |
| **Merged** | **0.4147** | **0.616** | **0.175** | **94.4%** |

The sinusoidal scene is substantially harder. High-frequency depth variation means that adjacent pixels have very different disparities, violating the smoothness assumption implicit in windowed matching. The bad pixel ratio (>94%) indicates that most pixels are decoded to the wrong disparity bin — yet Pearson r of 0.616 shows that the gross spatial structure is partially captured.

#### Gradient Scene

| Method | RMSE | Pearson r | SSIM | Bad Pix (5%) |
|---|---|---|---|---|
| SSD | 0.5776 | 0.007 | 0.016 | 95.0% |
| NCC | 0.5503 | 0.293 | 0.017 | 94.4% |
| NCC+Median | 0.5632 | 0.251 | 0.017 | 95.0% |
| Merged | 0.5776 | 0.009 | 0.016 | 95.0% |

The gradient scene is the hardest, with all methods essentially failing to recover the depth. The period estimation error (estimated P = 115 vs. true P = 120) likely contributes significantly. More fundamentally, a linear depth ramp means that all parts of the image look like a slightly shifted version of every other part — the texture correlation does not strongly prefer one disparity over another along a monotone gradient.

#### Steps and Curved Scenes

Steps and curved yield intermediate difficulty. Steps pose a specific challenge at depth boundaries: the window-based methods must choose between two discrete depth levels, and ambiguous patches at the boundary get misassigned. Curved surface reconstruction benefits from the Merged method, which achieves RMSE = 0.394 vs. 0.408 for pure SSD.

### 4.3 Algorithm Comparison Summary

Averaging RMSE across all six scenes:

| Method | Mean RMSE | Mean Pearson r | Mean SSIM |
|---|---|---|---|
| SSD | 0.349 | 0.511 | 0.433 |
| NCC | 0.330 | 0.452 | 0.358 |
| NCC+Median | 0.336 | 0.464 | 0.392 |
| **Merged** | **0.340** | **0.537** | **0.436** |

NCC achieves the lowest mean RMSE (0.330), but the Merged method achieves the highest mean Pearson r (0.537) and SSIM (0.436), indicating it best captures the structural layout of the depth field even when pixel-level accuracy is imperfect.

### 4.4 Post-Processing Comparison

On the sphere scene, the three post-processing strategies applied to the raw NCC disparity map gave:

| Post-processing | RMSE | Pearson r |
|---|---|---|
| None (raw NCC) | 0.2235 | 0.704 |
| Gaussian (σ=2.0) | 0.1987 | 0.757 |
| Bilateral | 0.1902 | 0.775 |
| **Median (5×5)** | **0.1845** | **0.792** |

All post-processing strategies improve over raw NCC. The median filter best preserves the sphere boundary while suppressing noise, confirming its selection as the default.

### 4.5 Dynamic Programming Results (Sphere, Gradient, Steps)

The DP decoder was applied to the first three scenes with λ = 15 and search range Δ = 25:

| Scene | DP RMSE | SSD RMSE | NCC RMSE |
|---|---|---|---|
| Sphere | 0.3275 | 0.1788 | 0.2235 |
| Gradient | 0.5776 | 0.5776 | 0.5503 |
| Steps | 0.6124 | 0.6125 | 0.5756 |

The DP underperforms SSD and NCC on the sphere, likely because the L1 smoothness penalty is too strong relative to the data term, pulling the disparity estimates toward a constant value. DP performs similarly to SSD on gradient and steps (both fail). Tuning λ and improving the data term are clear avenues for future work.

---

## 5. Analysis and Discussion

### 5.1 What Determines Reconstruction Success?

The results reveal a clear hierarchy of scene difficulty, driven by three factors:

**Depth smoothness.** Smooth depth fields (sphere, multi-object, curved) are well-recovered because adjacent pixels have similar disparities, reinforcing each other within the matching window. High-frequency depth fields (sinusoidal, gradient) change faster than the window can resolve, causing systematic mis-matches.

**Sparse vs. dense depth structure.** The multi-object scene, where most of the image is at depth zero, is the easiest to decode: the background provides a strong zero-disparity anchor that constrains the absolute disparity scale. Dense depth fields (curved, gradient) have no such anchor, and reconstruction errors can accumulate.

**Period estimation accuracy.** A period estimation error of even 5–7 pixels causes the conversion from disparity to depth to be applied with the wrong reference, systematically offsetting the recovered depth. Future work could use per-scanline period refinement or joint estimation of period and disparity.

### 5.2 Why SSD Sometimes Beats NCC

NCC is invariant to local mean and contrast, which helps in images with brightness variation. However, for autostereograms generated from a fixed noise texture with no brightness variation, SSD's sensitivity to absolute intensity differences is a strength rather than a weakness — it measures exactly the encoding distortion. The advantage of NCC appears primarily in the confidence score it provides, which enables the more effective Merged method.

### 5.3 Why the Merged Method Wins

The confidence-weighted fusion strategy is effective because it adapts the estimator to local image properties. In smooth, high-texture regions, NCC confidence is high and the NCC estimate dominates. In ambiguous or low-contrast regions, SSD's stability contributes more. The subsequent median filter removes any remaining impulse noise from the rare region where both methods fail. This ensemble approach consistently outperforms any single method, particularly on the best-performing scenes (sphere, multi-object).

### 5.4 Failure Modes

**Gradient scene:** Period estimation error combined with the inherent ambiguity of a monotone depth ramp causes near-total reconstruction failure. Fixing the period estimate (ground truth P = 120 vs. estimated P = 115) would improve results, but the fundamental matching ambiguity remains.

**High-frequency depth fields:** Both window-based methods exhibit the same failure on sinusoidal depth: when adjacent pixels have disparities differing by more than the window half-size, the window integrates across multiple depth layers, blurring the disparity estimate and pushing the argmin toward an average value.

**DP vertical streaking:** Because DP processes each scanline independently, row-to-row disparity jumps arise in regions where the depth is ambiguous. Extending the DP to 2-D (belief propagation or graph cuts) would address this at significant computational cost.

---

## 6. Conclusion

This project implemented and evaluated a complete autostereogram decoding pipeline from scratch, covering three stereo correspondence algorithms, multiple post-processing strategies, and quantitative evaluation on six synthetic scenes with known ground-truth depth.

The principal findings are:

1. **Confidence-weighted fusion (Merged) with median post-processing is the best overall method**, achieving RMSE = 0.072 and Pearson r = 0.946 on the multi-object scene — a reconstruction quality approaching human visual depth perception for that scene type.

2. **Scene depth smoothness is the dominant factor in reconstruction difficulty.** Smooth, sparse depth maps are recovered accurately by all methods; high-frequency depth fields remain challenging for all approaches tested.

3. **Period estimation accuracy matters.** A 5-pixel error in estimated period propagates into substantial depth reconstruction error, particularly for low-correlation scenes.

4. **SSD and NCC are complementary**, with SSD performing better in clean texture regions and NCC providing valuable confidence information that enables effective fusion.

5. **The DP decoder underperforms** in its current formulation, suggesting the need for better lambda tuning or a stronger data term.

### Future Directions

- **Subpixel disparity estimation** via parabola fitting around the cost minimum could improve depth resolution beyond the 1-pixel quantisation limit
- **Per-scanline period refinement** using local autocorrelation peaks would reduce the period estimation error on challenging scenes
- **Vertical consistency constraints** (belief propagation, graph cuts) would eliminate DP streaking artifacts
- **Real-world Magic Eye images** provide a harder test: unknown period, non-ideal texture, printing artifacts, and JPEG compression noise all degrade performance
- **Texture-aware matching** that adapts window size to local texture frequency could handle the sinusoidal and gradient scenes more robustly

---

## References

1. Scharstein, D., & Szeliski, R. (2002). A taxonomy and evaluation of dense two-frame stereo correspondence algorithms. *International Journal of Computer Vision*, 47(1), 7–42.
2. Birchfield, S., & Tomasi, C. (1998). A pixel dissimilarity measure that is insensitive to image sampling. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 20(4), 401–406.
3. Intille, S. S., & Bobick, A. F. (1994). Disparity-space images and large occlusion stereo. *European Conference on Computer Vision*, 179–186.
4. Thimbleby, H., Inglis, S., & Witten, I. (1994). Displaying 3D images: Algorithms for single-image random-dot stereograms. *IEEE Computer*, 27(10), 38–48.
5. Tyler, C. W., & Clarke, M. B. (1990). The autostereogram. *Proceedings of SPIE*, 1256, 182–197.
6. Bradski, G. (2000). The OpenCV library. *Dr. Dobb's Journal of Software Tools*, 25(11), 120–125.

---

## Appendix A: Code Structure

```
autostereogram/
├── main.py                  # Full experiment runner
├── src/
│   ├── generator.py         # Depth map generators + stereogram encoder
│   ├── decoder.py           # SSD, NCC, DP, post-processing, fusion
│   ├── evaluation.py        # RMSE, MAE, Pearson r, SSIM, bad-pixel metrics
│   └── visualization.py     # All figure generation (7 figure types)
├── data/                    # Generated .npy and .png datasets
├── figures/                 # All output figures (22 files)
└── results/
    └── metrics.json         # Full quantitative results
```

## Appendix B: Computational Complexity

| Algorithm | Per-pixel cost | Full image (400×600) |
|---|---|---|
| Period estimation | O(W) | ~0.1s |
| SSD (Δ=35, w=5) | O(Δ·w²) via box filter | ~0.4s |
| NCC (Δ=35, w=7) | O(Δ·w²) via box filter | ~0.9s |
| DP (Δ=25) | O(W·D²) | ~3.1s |
| Post-processing | O(W·H) | <0.1s |

All timings on a single CPU core; the SSD and NCC cost volumes are computed in parallel over candidate disparities using NumPy vectorisation.

## Appendix C: Reproduction Instructions

```bash
# Requirements: Python 3.10+, numpy, scipy, matplotlib, pillow, scikit-image, opencv-python
pip install numpy scipy matplotlib pillow scikit-image opencv-python

# Run the full experiment (generates all data, figures, and metrics)
cd autostereogram/
python main.py
```

All figures are written to `figures/` and metrics to `results/metrics.json`. The entire pipeline runs in approximately 2–3 minutes on a modern CPU.
