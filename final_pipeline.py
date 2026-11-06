import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, median_filter, gaussian_filter

# ── Step 1: Load the image ─────────────────────────────────────────────────
img = np.array(Image.open("../images/img6.webp"))
# Use blue channel if color image — usually strongest signal
if img.ndim == 3:
    channel = img[:, :, 2].astype(np.float32) / 255.0
else:
    channel = img.astype(np.float32) / 255.0

H, W = channel.shape

# ── Step 2: Estimate the repeat period ────────────────────────────────────
# Average horizontal autocorrelation over all rows
ac_sum = np.zeros(W, dtype=np.float64)
for y in range(H):
    row = channel[y] - channel[y].mean()
    norm = np.dot(row, row)
    if norm < 1e-6:
        continue
    ac = np.correlate(row, row, mode='full')[W-1:]   # positive lags only
    ac_sum += ac / (norm + 1e-8)
ac_avg = ac_sum / H

# Find the strongest peak in a reasonable range
peaks = []
for lag in range(40, 200):
    if ac_avg[lag] > ac_avg[lag-1] and ac_avg[lag] > ac_avg[lag+1]:
        peaks.append((ac_avg[lag], lag))
period = max(peaks)[1]
print(f"Estimated period: {period}px")

# ── Step 3: NCC correspondence matching ───────────────────────────────────
# For each candidate disparity d near the period,
# compute how well the image matches a shifted copy of itself

def box(arr, hw):
    return uniform_filter(arr.astype(np.float64), size=2*hw+1)

hw = 6                              # half-window size (12x12 window)
search_lo = period - 25
search_hi = period + 25

best_score = np.full((H, W), -2.0, dtype=np.float32)
best_disp  = np.zeros((H, W), dtype=np.float32)

mu1 = box(channel, hw)
s1  = np.sqrt(np.maximum(box(channel**2, hw) - mu1**2, 1e-8))

for d in range(search_lo, search_hi + 1):
    shifted = np.roll(channel, d, axis=1)           # shift image left by d
    mu2  = box(shifted, hw)
    s2   = np.sqrt(np.maximum(box(shifted**2, hw) - mu2**2, 1e-8))
    ncc  = (box(channel * shifted, hw) - mu1 * mu2) / (s1 * s2 + 1e-8)
    
    mask = ncc > best_score
    best_score[mask] = ncc[mask]
    best_disp[mask]  = d

# ── Step 4: Post-process the disparity map ────────────────────────────────
disp_clean = median_filter(best_disp, size=7)       # remove noise
disp_clean = gaussian_filter(disp_clean, sigma=2.0) # smooth

# ── Step 5: Convert disparity → depth ─────────────────────────────────────
# Pixels shifted more than the period = closer to viewer
depth = (disp_clean - search_lo) / (search_hi - search_lo)
depth = np.clip(depth, 0, 1)

# Try both polarities — you don't know which is foreground
depth_fwd = depth
depth_inv = 1.0 - depth

# ── Step 6: Visualize ─────────────────────────────────────────────────────
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].imshow(img if img.ndim == 3 else img, cmap='gray')
axes[0].set_title('Input autostereogram')
axes[0].axis('off')

axes[1].imshow(depth_fwd, cmap='plasma')
axes[1].set_title(f'Depth map (period={period}px)')
axes[1].axis('off')

axes[2].imshow(depth_inv, cmap='plasma')
axes[2].set_title('Depth map (inverted)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('result_2d.png', dpi=200)

# 3D surface
step = 2
Y, X = np.mgrid[0:H:step, 0:W:step]
Z    = depth_fwd[::step, ::step]

fig3d = plt.figure(figsize=(10, 5))
for i, (elev, azim) in enumerate([(35, -60), (25, 30)]):
    ax = fig3d.add_subplot(1, 2, i+1, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma',
                    linewidth=0, antialiased=True,
                    rcount=60, ccount=60)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('depth')

plt.tight_layout()
plt.savefig('result_3d.png', dpi=200)
plt.show()