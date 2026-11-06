"""
Synthetic Autostereogram Generator
===================================
Generates autostereograms (Magic Eye images) from known depth maps.
The hidden 3D structure is encoded as horizontal texture shifts.

Algorithm:
    For each scanline y:
        For each pixel x (left-to-right):
            shift = round(depth[y,x] * max_shift)
            stereogram[y, x] = pattern[y, (x - period + shift) % period]
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os


# ── Depth map generators ──────────────────────────────────────────────────────

def depth_sphere(height=400, width=600, radius_frac=0.32, center=None, depth_scale=0.8):
    """Smooth hemisphere embedded in a zero background."""
    D = np.zeros((height, width), dtype=np.float32)
    cy, cx = center if center else (height // 2, width // 2)
    r = int(min(height, width) * radius_frac)
    Y, X = np.ogrid[:height, :width]
    dist2 = (Y - cy)**2 + (X - cx)**2
    mask = dist2 < r**2
    D[mask] = np.sqrt(np.maximum(r**2 - dist2[mask], 0)) / r * depth_scale
    return D


def depth_gradient(height=400, width=600):
    """Horizontal linear ramp — tests monotone disparity recovery."""
    x = np.linspace(0, 1, width, dtype=np.float32)
    return np.tile(x, (height, 1))


def depth_steps(height=400, width=600, n_steps=5):
    """Discrete depth layers — good for measuring step-edge accuracy."""
    D = np.zeros((height, width), dtype=np.float32)
    step_w = width // n_steps
    for i in range(n_steps):
        D[:, i*step_w:(i+1)*step_w] = i / (n_steps - 1)
    return D


def depth_sinusoidal(height=400, width=600, freq_x=3, freq_y=2):
    """Sinusoidal ripple — tests smooth varying disparity."""
    Y, X = np.mgrid[0:height, 0:width]
    D = (np.sin(2 * np.pi * freq_x * X / width) *
         np.cos(2 * np.pi * freq_y * Y / height))
    D = (D + 1) / 2   # normalise to [0, 1]
    return D.astype(np.float32)


def depth_multi_object(height=400, width=600):
    """Several spheres at different depths — tests multiple-object scenes."""
    D = np.zeros((height, width), dtype=np.float32)
    objects = [
        (height//4,     width//4,  0.15, 0.9),   # (cy, cx, r_frac, depth)
        (height//4,   3*width//4,  0.12, 0.6),
        (3*height//4,   width//2,  0.18, 0.75),
        (height//2,   2*width//5,  0.10, 0.5),
    ]
    for (cy, cx, r_frac, depth_val) in objects:
        r = int(min(height, width) * r_frac)
        Y, X = np.ogrid[:height, :width]
        dist2 = (Y - cy)**2 + (X - cx)**2
        mask = dist2 < r**2
        vals = np.sqrt(np.maximum(r**2 - dist2[mask], 0)) / r * depth_val
        D[mask] = np.maximum(D[mask], vals)
    return D


def depth_curved_surface(height=400, width=600):
    """Curved paraboloid — tests globally consistent reconstruction."""
    Y, X = np.mgrid[0:height, 0:width]
    cy, cx = height / 2, width / 2
    D = 1 - ((Y - cy)**2 / cy**2 + (X - cx)**2 / cx**2)
    D = np.clip(D, 0, 1).astype(np.float32)
    return D


# ── Texture generators ────────────────────────────────────────────────────────

def make_random_dot_texture(height, period, rng=None):
    """Classic random-dot pattern."""
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.integers(0, 256, (height, period), dtype=np.uint8)


def make_noise_texture(height, period, rng=None):
    """Smoothed Gaussian noise — more natural looking."""
    if rng is None:
        rng = np.random.default_rng(42)
    tex = rng.integers(0, 256, (height, period), dtype=np.uint8).astype(np.float32)
    tex = gaussian_filter(tex, sigma=1.2)
    tex = np.clip(tex, 0, 255).astype(np.uint8)
    return tex


def make_stripe_texture(height, period):
    """Vertical stripes — high contrast, easy to correlate."""
    tex = np.zeros((height, period), dtype=np.uint8)
    for i in range(period):
        if (i // 4) % 2 == 0:
            tex[:, i] = 200
        else:
            tex[:, i] = 55
    return tex


def make_color_noise_texture(height, period, rng=None):
    """RGB random-dot texture."""
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.integers(0, 256, (height, period, 3), dtype=np.uint8)


# ── Stereogram encoder ────────────────────────────────────────────────────────

def encode_autostereogram(depth_map, period=120, max_shift=30, texture=None, rng=None):
    """
    Encode a normalised depth map [0,1] into an autostereogram.

    Parameters
    ----------
    depth_map : 2-D float array, values in [0, 1]
    period    : base repeat period in pixels
    max_shift : maximum pixel shift at depth=1
    texture   : (H, period[, 3]) uint8 pattern; random dots if None
    rng       : numpy RNG for reproducibility

    Returns
    -------
    stereogram : uint8 array (H, W) or (H, W, 3)
    texture    : the texture used (for reference)
    """
    H, W = depth_map.shape
    if rng is None:
        rng = np.random.default_rng(42)

    if texture is None:
        texture = make_random_dot_texture(H, period, rng)

    is_color = texture.ndim == 3
    stereogram = np.zeros_like(texture[..., :W] if is_color else np.zeros((H, W), dtype=np.uint8))
    if is_color:
        stereogram = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        stereogram = np.zeros((H, W), dtype=np.uint8)

    for y in range(H):
        for x in range(W):
            shift = int(round(depth_map[y, x] * max_shift))
            src_x = (x - period + shift) % period
            if is_color:
                stereogram[y, x] = texture[y, src_x]
            else:
                stereogram[y, x] = texture[y, src_x]

    return stereogram, texture


def encode_autostereogram_fast(depth_map, period=120, max_shift=30, texture=None, rng=None):
    """
    Vectorised encoder — same result as encode_autostereogram but ~30× faster.
    Uses per-scanline vectorisation over x.
    """
    H, W = depth_map.shape
    if rng is None:
        rng = np.random.default_rng(42)
    if texture is None:
        texture = make_random_dot_texture(H, period, rng)

    is_color = texture.ndim == 3
    if is_color:
        stereogram = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        stereogram = np.zeros((H, W), dtype=np.uint8)

    shifts = np.round(depth_map * max_shift).astype(int)          # (H, W)
    x_idx  = np.arange(W)
    src_x  = (x_idx[None, :] - period + shifts) % period          # (H, W)
    y_idx  = np.arange(H)[:, None] * np.ones((1, W), dtype=int)   # (H, W)

    if is_color:
        stereogram = texture[y_idx, src_x]
    else:
        stereogram = texture[y_idx, src_x]

    return stereogram, texture


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_image(arr, path, cmap=None):
    """Save a numpy array as a PNG."""
    if arr.ndim == 2 and cmap is not None:
        fig, ax = plt.subplots(figsize=(arr.shape[1]/100, arr.shape[0]/100), dpi=100)
        ax.imshow(arr, cmap=cmap, vmin=0, vmax=1 if arr.max() <= 1 else 255)
        ax.axis('off')
        fig.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        img = Image.fromarray(arr.astype(np.uint8))
        img.save(path)


def load_image_gray(path):
    """Load an image as float32 grayscale in [0,1]."""
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float32) / 255.0


# ── Dataset builder ───────────────────────────────────────────────────────────

DEPTH_SCENES = {
    'sphere':        depth_sphere,
    'gradient':      depth_gradient,
    'steps':         depth_steps,
    'sinusoidal':    depth_sinusoidal,
    'multi_object':  depth_multi_object,
    'curved':        depth_curved_surface,
}

def build_synthetic_dataset(out_dir, period=120, max_shift=28, height=400, width=600):
    """
    Generate all synthetic scenes and save depth maps + stereograms.
    Returns a dict of {scene_name: {'depth': arr, 'stereo': arr, ...}}.
    """
    os.makedirs(out_dir, exist_ok=True)
    dataset = {}
    rng = np.random.default_rng(0)

    for name, fn in DEPTH_SCENES.items():
        depth = fn(height=height, width=width)
        depth = depth / depth.max() if depth.max() > 0 else depth   # normalise

        texture = make_noise_texture(height, period, rng=np.random.default_rng(1))
        stereo, tex = encode_autostereogram_fast(depth, period, max_shift, texture=texture)

        np.save(os.path.join(out_dir, f'{name}_depth.npy'), depth)
        np.save(os.path.join(out_dir, f'{name}_stereo.npy'), stereo)
        save_image((depth * 255).astype(np.uint8), os.path.join(out_dir, f'{name}_depth.png'))
        save_image(stereo, os.path.join(out_dir, f'{name}_stereo.png'))

        dataset[name] = {'depth': depth, 'stereo': stereo, 'texture': tex,
                         'period': period, 'max_shift': max_shift}
        print(f"  Generated: {name}  depth range [{depth.min():.2f}, {depth.max():.2f}]")

    return dataset
