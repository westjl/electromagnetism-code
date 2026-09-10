#!/usr/bin/env python3
"""
Interactive rectangular-waveguide mode visualizer (3D panels).

Each of the 6 panels is a 3D waveguide view:
    Bx, By, Bz, Ex, Ey, Ez

The front x-y face and two longitudinal faces are color-mapped so the z axis
visually goes into the page, similar to textbook waveguide mode illustrations.

Controls:
    - Waveguide size a, b
    - Mode numbers m, n (0..6)
    - TE / TM selector
    - Time slider + Play/Pause + Reset
    - Optional per-panel colorbars

Run:
    python waveguide_modes_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons


def mode_fields_3d(x_grid, y_grid, z_grid, a, b, m, n, mode_type, t_now):
    """Compute synthetic TE/TM rectangular-waveguide field components."""
    eps = 1.0

    kx = m * np.pi / max(a, 1e-12)
    ky = n * np.pi / max(b, 1e-12)
    kc2 = kx**2 + ky**2
    kc = np.sqrt(kc2)

    if np.isclose(kc2, 0.0):
        zeros = np.zeros_like(x_grid)
        meta = {
            "kc": 0.0,
            "omega": 0.0,
            "beta": 0.0,
            "valid": False,
            "message": "m=n=0 is not a propagating hollow-waveguide TE/TM mode.",
        }
        return [zeros, zeros, zeros, zeros, zeros, zeros], meta

    omega = 1.6 * kc + 1.0
    beta = np.sqrt(max(omega**2 - kc2, 0.0))
    phase = beta * z_grid - omega * t_now

    cx = np.cos(kx * x_grid)
    sx = np.sin(kx * x_grid)
    cy = np.cos(ky * y_grid)
    sy = np.sin(ky * y_grid)
    cph = np.cos(phase)
    sph = np.sin(phase)

    denom = max(kc2, 1e-12)

    if mode_type == "TE":
        bz = cx * cy * cph
        ex = (omega * eps / denom) * ky * cx * sy * sph
        ey = -(omega * eps / denom) * kx * sx * cy * sph
        bx = (beta / denom) * kx * sx * cy * sph
        by = (beta / denom) * ky * cx * sy * sph
        ez = np.zeros_like(bz)
    else:
        ez = sx * sy * cph
        ex = -(beta / denom) * kx * cx * sy * sph
        ey = -(beta / denom) * ky * sx * cy * sph
        bx = -(omega / max(denom * eps, 1e-12)) * ky * sx * cy * sph
        by = (omega / max(denom * eps, 1e-12)) * kx * cx * sy * sph
        bz = np.zeros_like(ez)

    components = [bx, by, bz, ex, ey, ez]
    normalized = []
    for comp in components:
        scale = np.max(np.abs(comp))
        normalized.append(comp if scale < 1e-12 else comp / scale)

    meta = {
        "kc": kc,
        "omega": omega,
        "beta": beta,
        "valid": True,
        "message": "",
    }
    return normalized, meta


def draw_guide_component(ax, comp, title, a, b, length_z, cmap, norm):
    """Draw one component on three visible guide faces in a 3D axis."""
    ny, nx, nz = comp.shape

    x_vals = np.linspace(0.0, a, nx)
    y_vals = np.linspace(0.0, b, ny)
    z_vals = np.linspace(0.0, length_z, nz)

    # Front face (x-y at physical z=0)
    x_f, y_f = np.meshgrid(x_vals, y_vals)
    z_f = np.zeros_like(x_f)
    c_f = comp[:, :, 0]

    # Use interior longitudinal slices so wall boundary zeros do not hide structure.
    iy = int(round(0.55 * (ny - 1)))
    ix = int(round(0.85 * (nx - 1)))

    # Longitudinal x-z slice at y ~= 0.55 b
    x_t, z_t = np.meshgrid(x_vals, z_vals, indexing="ij")
    y_t = np.full_like(x_t, y_vals[iy])
    c_t = comp[iy, :, :]

    # Longitudinal y-z slice at x ~= 0.85 a
    y_s, z_s = np.meshgrid(y_vals, z_vals, indexing="ij")
    x_s = np.full_like(y_s, x_vals[ix])
    c_s = comp[:, ix, :]

    # Plot-coordinate remap:
    #   physical x -> plot x (horizontal)
    #   physical z -> plot y (depth, into page)
    #   physical y -> plot z (vertical)
    def pmap(xp, yp, zp):
        return xp, zp, yp

    ax.clear()

    xpf, ypf, zpf = pmap(x_f, y_f, z_f)
    xpt, ypt, zpt = pmap(x_t, y_t, z_t)
    xps, yps, zps = pmap(x_s, y_s, z_s)

    ax.plot_surface(xpf, ypf, zpf, facecolors=cmap(norm(c_f)), linewidth=0.0, antialiased=False, shade=False)
    ax.plot_surface(xpt, ypt, zpt, facecolors=cmap(norm(c_t)), linewidth=0.0, antialiased=False, shade=False)
    ax.plot_surface(xps, yps, zps, facecolors=cmap(norm(c_s)), linewidth=0.0, antialiased=False, shade=False)

    # Box edges for waveguide geometry clarity.
    edge_color = "#666666"
    lw = 0.8
    def draw_phys_line(x1, y1, z1, x2, y2, z2):
        xp1, yp1, zp1 = pmap(x1, y1, z1)
        xp2, yp2, zp2 = pmap(x2, y2, z2)
        ax.plot([xp1, xp2], [yp1, yp2], [zp1, zp2], color=edge_color, lw=lw)

    # 12 edges of rectangular guide in physical coordinates.
    draw_phys_line(0, 0, 0, a, 0, 0)
    draw_phys_line(0, b, 0, a, b, 0)
    draw_phys_line(0, 0, length_z, a, 0, length_z)
    draw_phys_line(0, b, length_z, a, b, length_z)

    draw_phys_line(0, 0, 0, 0, b, 0)
    draw_phys_line(a, 0, 0, a, b, 0)
    draw_phys_line(0, 0, length_z, 0, b, length_z)
    draw_phys_line(a, 0, length_z, a, b, length_z)

    draw_phys_line(0, 0, 0, 0, 0, length_z)
    draw_phys_line(a, 0, 0, a, 0, length_z)
    draw_phys_line(0, b, 0, 0, b, length_z)
    draw_phys_line(a, b, 0, a, b, length_z)

    # Limits in plot coordinates after remap.
    ax.set_xlim(0.0, a)          # physical x
    ax.set_ylim(0.0, length_z)   # physical z (depth)
    ax.set_zlim(0.0, b)          # physical y (vertical)
    ax.set_box_aspect((a, length_z, b))

    # Look almost straight at front face so depth axis goes into the page.
    ax.view_init(elev=10, azim=-90)
    # Smaller camera distance zooms the guide and reduces apparent blank margins.
    ax.dist = 7

    ax.set_xlabel("x", labelpad=-2)
    ax.set_ylabel("z", labelpad=-2)
    ax.set_zlabel("y", labelpad=-2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.tick_params(axis="both", labelsize=6, pad=0)
    ax.set_title("")
    ax.text2D(
        0.03,
        0.98,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )


def main():
    # Spatial grid in the guide volume.
    nx, ny, nz = 34, 26, 60
    z_length = 4.8

    a0, b0 = 2.0, 1.0
    m0, n0 = 2, 1
    mode0 = "TE"

    t_min, t_max = 0.0, 12.0
    t0 = 0.0

    fig = plt.figure(figsize=(14.6, 8.2))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.01,
        right=0.995,
        top=0.96,
        bottom=0.03,
        wspace=0.0,
        hspace=0.0,
    )

    axes = [fig.add_subplot(grid[i // 3, i % 3], projection="3d") for i in range(6)]
    titles = ["Bx", "By", "Bz", "Ex", "Ey", "Ez"]

    header = fig.suptitle("Rectangular Waveguide 3D Mode Fields", fontsize=15)

    cmap = plt.get_cmap("afmhot")
    norm = Normalize(vmin=-1.0, vmax=1.0)

    # Controls live in a separate window.
    fig_ctrl = plt.figure(figsize=(4.2, 8.6))
    fig_ctrl.canvas.manager.set_window_title("Waveguide Controls")
    fig_ctrl.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05)

    info_text = fig_ctrl.text(
        0.08,
        0.98,
        "",
        fontsize=8,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#999999"},
    )

    slider_face = "#efefef"
    ax_a = fig_ctrl.add_axes([0.12, 0.80, 0.76, 0.030], facecolor=slider_face)
    ax_b = fig_ctrl.add_axes([0.12, 0.75, 0.76, 0.030], facecolor=slider_face)
    ax_m = fig_ctrl.add_axes([0.12, 0.70, 0.76, 0.030], facecolor=slider_face)
    ax_n = fig_ctrl.add_axes([0.12, 0.65, 0.76, 0.030], facecolor=slider_face)

    ax_t = fig_ctrl.add_axes([0.12, 0.58, 0.76, 0.030], facecolor=slider_face)

    s_a = Slider(ax_a, "a", 0.5, 5.0, valinit=a0, valstep=0.02)
    s_b = Slider(ax_b, "b", 0.5, 5.0, valinit=b0, valstep=0.02)
    s_m = Slider(ax_m, "m", 0, 6, valinit=m0, valstep=1)
    s_n = Slider(ax_n, "n", 0, 6, valinit=n0, valstep=1)
    s_t = Slider(ax_t, "time t", t_min, t_max, valinit=t0, valstep=(t_max - t_min) / 240.0)

    ax_mode = fig_ctrl.add_axes([0.12, 0.38, 0.34, 0.14], facecolor="#f7f7f7")
    r_mode = RadioButtons(ax_mode, ("TE", "TM"), active=0)

    ax_play = fig_ctrl.add_axes([0.12, 0.28, 0.18, 0.040])
    ax_reset = fig_ctrl.add_axes([0.34, 0.28, 0.18, 0.040])
    b_play = Button(ax_play, "Play")
    b_reset = Button(ax_reset, "Reset")
    playing = {"value": False}

    def make_grids(a, b):
        x_vals = np.linspace(0.0, a, nx)
        y_vals = np.linspace(0.0, b, ny)
        z_vals = np.linspace(0.0, z_length, nz)
        y_grid, x_grid, z_grid = np.meshgrid(y_vals, x_vals, z_vals, indexing="ij")
        return x_grid, y_grid, z_grid

    def refresh(_=None):
        a = s_a.val
        b = s_b.val
        m = int(s_m.val)
        n = int(s_n.val)
        t_now = s_t.val
        mode_type = r_mode.value_selected

        x_grid, y_grid, z_grid = make_grids(a, b)
        fields, meta = mode_fields_3d(x_grid, y_grid, z_grid, a, b, m, n, mode_type, t_now)

        for ax, comp, title in zip(axes, fields, titles):
            draw_guide_component(ax, comp, title, a, b, z_length, cmap, norm)

        header.set_text(f"Rectangular Waveguide 3D Mode Fields ({mode_type})")
        if meta["valid"]:
            mode_note = ""
            if n == 0:
                mode_note = "  note: n=0 => Ex and By are physically zero for this model"
            info_text.set_text(
                f"type={mode_type}    m={m}, n={n}    a={a:.2f}, b={b:.2f}    t={t_now:.2f}    "
                f"kc={meta['kc']:.3f}, omega={meta['omega']:.3f}, beta={meta['beta']:.3f}{mode_note}"
            )
        else:
            info_text.set_text(
                f"type={mode_type}    m={m}, n={n}    a={a:.2f}, b={b:.2f}    {meta['message']}"
            )

        fig.canvas.draw_idle()
        fig_ctrl.canvas.draw_idle()

    def on_play(_event):
        playing["value"] = not playing["value"]
        b_play.label.set_text("Pause" if playing["value"] else "Play")

    def on_reset(_event):
        playing["value"] = False
        b_play.label.set_text("Play")
        s_t.set_val(t_min)

    def animate(_frame):
        if not playing["value"]:
            return
        dt = (t_max - t_min) / 240.0
        t_next = s_t.val + dt
        if t_next > t_max:
            t_next = t_min
        s_t.set_val(t_next)

    s_a.on_changed(refresh)
    s_b.on_changed(refresh)
    s_m.on_changed(refresh)
    s_n.on_changed(refresh)
    s_t.on_changed(refresh)
    r_mode.on_clicked(refresh)
    b_play.on_clicked(on_play)
    b_reset.on_clicked(on_reset)

    _ani = FuncAnimation(fig, animate, interval=60, blit=False, cache_frame_data=False)

    refresh()
    plt.show()


if __name__ == "__main__":
    main()
