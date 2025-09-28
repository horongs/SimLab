"""
math_art_shorts.py

This script generates a 30-second 1080x1920 vertical MP4 (30 FPS) featuring
continuous morphing mathematical neon curves suitable for YouTube Shorts.

Requirements satisfied:
- Uses only numpy and matplotlib (Agg backend) from the Python standard setup.
- Renders Lissajous, rose, and hypotrochoid families blended seamlessly.
- Produces "math_art_shorts.mp4" with FFMpegWriter at 30 FPS for 30.0 seconds.
- Deterministic output via fixed random seed and analytic curves.
- Artistic styling includes neon glow, gradient background, rotating camera,
  hue cycling, and subtle coordinate axes.

Run this file directly with Python 3 (ensure ffmpeg is available on PATH).
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import hsv_to_rgb

# Configuration constants
WIDTH = 1080
HEIGHT = 1920
FPS = 30
DURATION = 30.0
TOTAL_FRAMES = int(FPS * DURATION)
DPI = 100
FIGSIZE = (WIDTH / DPI, HEIGHT / DPI)
BITRATE = 8000
POINTS = 2400
ROTATION_CYCLES = 1.0
GLOW_LINEWIDTHS = [18, 12, 7, 3]
GLOW_ALPHAS = [0.08, 0.14, 0.3, 0.9]
AXIS_GLOW_LINEWIDTHS = [14, 8, 3]
AXIS_GLOW_ALPHAS = [0.04, 0.08, 0.2]


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Smooth cubic interpolation between 0 and 1."""
    if edge0 == edge1:
        return np.zeros_like(x)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def lissajous(u: np.ndarray, t: float) -> np.ndarray:
    """Generate a time-varying Lissajous curve normalized within [-1, 1]."""
    theta = 2 * np.pi * u
    a = 3 + 2 * np.sin(2 * np.pi * t)
    b = 4 + 1.5 * np.cos(2 * np.pi * t)
    delta = np.pi * np.sin(2 * np.pi * t)
    x = np.sin(a * theta + delta)
    y = np.sin(b * theta + 2 * np.pi * t)
    scale = 1.05
    return np.stack((x / scale, y / scale), axis=0)


def rose(u: np.ndarray, t: float) -> np.ndarray:
    """Generate a generalized rose curve."""
    theta = 2 * np.pi * u
    k = 4 + np.sin(2 * np.pi * t) * 2.5
    phase = 2 * np.pi * t
    r = np.cos(k * theta + phase)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    scale = 1.3
    return np.stack((x / scale, y / scale), axis=0)


def hypotrochoid(u: np.ndarray, t: float) -> np.ndarray:
    """Generate a hypotrochoid/spirograph curve."""
    theta = 2 * np.pi * u * (1.0 + 0.5 * np.sin(2 * np.pi * t))
    R = 5.0 + 1.5 * np.sin(2 * np.pi * t)
    r = 3.0 + 1.0 * np.cos(2 * np.pi * t)
    d = 1.6 + 0.6 * np.sin(4 * np.pi * t)
    ratio = (R - r) / r
    x = (R - r) * np.cos(theta) + d * np.cos(ratio * theta + np.pi * t)
    y = (R - r) * np.sin(theta) - d * np.sin(ratio * theta + np.pi * t)
    max_radius = (R - r) + d
    return np.stack((x / max_radius, y / max_radius), axis=0)


def blend_curves(u: np.ndarray, t: float) -> np.ndarray:
    """Blend between curve families across the loop."""
    t_mod = t % 1.0
    segment = t_mod * 3.0
    if segment < 1.0:
        blend = smoothstep(0.0, 1.0, segment)
        c0 = lissajous(u, t_mod)
        c1 = rose(u, t_mod)
    elif segment < 2.0:
        blend = smoothstep(1.0, 2.0, segment)
        c0 = rose(u, t_mod)
        c1 = hypotrochoid(u, t_mod)
    else:
        blend = smoothstep(2.0, 3.0, segment)
        c0 = hypotrochoid(u, t_mod)
        c1 = lissajous(u, t_mod)
    return (1.0 - blend) * c0 + blend * c1


def pattern_xy(u: np.ndarray, t: float) -> np.ndarray:
    """Return rotated, morphed coordinates for frame time t in [0, 1)."""
    base = blend_curves(u, t)
    wobble = 0.05 * np.sin(4 * np.pi * t)
    base = base * (1.0 + wobble)
    angle = 2 * np.pi * ROTATION_CYCLES * t + 0.3 * np.sin(2 * np.pi * t)
    c, s = np.cos(angle), np.sin(angle)
    rot_matrix = np.array([[c, -s], [s, c]])
    rotated = rot_matrix @ base
    return rotated


def neon_color(t: float) -> np.ndarray:
    """Return RGB neon color cycling smoothly over time."""
    hue = (t + 0.1) % 1.0
    saturation = 0.85 + 0.1 * np.sin(2 * np.pi * t)
    value = 0.9
    rgb = hsv_to_rgb([hue, saturation, value])
    return rgb


def draw_background(ax: plt.Axes) -> None:
    """Render a subtle navy-to-black vertical gradient background."""
    gradient_size = 512
    top_color = np.array([0.02, 0.04, 0.12])
    bottom_color = np.array([0.0, 0.0, 0.0])
    g = np.linspace(0.0, 1.0, gradient_size)[:, None]
    colors = top_color * (1.0 - g) + bottom_color * g
    gradient = np.repeat(colors[np.newaxis, :, :], 2, axis=0)
    ax.imshow(
        gradient.transpose(1, 0, 2),
        extent=(-1.2, 1.2, -1.2, 1.2),
        origin="lower",
        aspect="auto",
        zorder=-10,
    )


def draw_axes_with_glow(ax: plt.Axes) -> None:
    """Draw glowing coordinate axes with minimal ticks."""
    for lw, alpha in zip(AXIS_GLOW_LINEWIDTHS, AXIS_GLOW_ALPHAS):
        ax.axhline(0.0, color=(0.6, 0.8, 1.0, alpha), linewidth=lw, zorder=-5)
        ax.axvline(0.0, color=(0.6, 0.8, 1.0, alpha), linewidth=lw, zorder=-5)
    ticks = np.linspace(-1.0, 1.0, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(colors=(0.4, 0.6, 0.8), length=4, width=0.8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def create_figure() -> tuple[plt.Figure, plt.Axes, list]:
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="black")
    margin_x = 0.06
    margin_y = 0.05
    ax = fig.add_axes([
        margin_x,
        margin_y,
        1 - 2 * margin_x,
        1 - 2 * margin_y,
    ])
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_facecolor((0.0, 0.0, 0.0, 0.0))
    ax.set_frame_on(False)

    draw_background(ax)
    draw_axes_with_glow(ax)

    lines = []
    for lw, alpha in zip(GLOW_LINEWIDTHS, GLOW_ALPHAS):
        (line,) = ax.plot([], [], linewidth=lw, color=(1, 1, 1, alpha), solid_joinstyle="round")
        line.set_zorder(10)
        lines.append(line)
    return fig, ax, lines


def update_lines(lines: list, xy: np.ndarray, color: np.ndarray) -> None:
    for idx, line in enumerate(lines):
        line.set_data(xy[0], xy[1])
        alpha = GLOW_ALPHAS[idx]
        line.set_color((*color, alpha))


def render_animation() -> None:
    fig, ax, lines = create_figure()
    writer = FFMpegWriter(fps=FPS, metadata={"artist": "Python"}, bitrate=BITRATE)
    u = np.linspace(0.0, 1.0, POINTS)
    output_path = "math_art_shorts.mp4"

    with writer.saving(fig, output_path, DPI):
        for frame in range(TOTAL_FRAMES):
            t = frame / TOTAL_FRAMES
            xy = pattern_xy(u, t)
            color = neon_color(t)
            update_lines(lines, xy, color)
            writer.grab_frame()
            if frame % 60 == 0:
                print(f"Rendered frame {frame + 1}/{TOTAL_FRAMES}")

    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    render_animation()

# How to run:
#   python main.py
# Ensure numpy, matplotlib, and ffmpeg are installed/available.
