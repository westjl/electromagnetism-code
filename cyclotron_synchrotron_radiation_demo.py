#!/usr/bin/env python3
"""
Cyclotron radiation animation with a particle-attached rotating donut.

This demo shows a non-relativistic charge in circular cyclotron motion:
    r(t) = R [cos(omega t), sin(omega t), 0]

The displayed donut is a visual proxy for the angular power pattern and is
attached to the moving particle. Its symmetry axis follows the instantaneous
acceleration direction, so the donut rotates with the orbit.

Controls:
    - Camera elevation slider
    - Camera azimuth slider
    - Speed slider (v/c)
    - Radiation scale slider
    - Scale mode: real / not-to-scale
    - Auto zoom
    - Pause/Play and Reset buttons

Run:
    python cyclotron_radiation_donut_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons


_ANIM = None


def charge_state(t, radius, omega_c):
    """Return position, velocity, acceleration for cyclotron motion."""
    c, s = np.cos(omega_c * t), np.sin(omega_c * t)
    pos = np.array([radius * c, radius * s, 0.0])
    vel = np.array([-radius * omega_c * s, radius * omega_c * c, 0.0])
    acc = np.array([-radius * omega_c**2 * c, -radius * omega_c**2 * s, 0.0])
    return pos, vel, acc


def charge_state_phase(phase, radius, beta, c_light):
    """Return position, velocity, acceleration using phase and speed beta=v/c."""
    c, s = np.cos(phase), np.sin(phase)
    speed = beta * c_light
    pos = np.array([radius * c, radius * s, 0.0])
    vel = np.array([-speed * s, speed * c, 0.0])
    acc_mag = (speed**2) / max(radius, 1e-12)
    acc = np.array([-acc_mag * c, -acc_mag * s, 0.0])
    return pos, vel, acc


def spherical_directions(n_theta=66, n_phi=132):
    """Return direction grid n-hat(th, ph) over the full sphere."""
    theta = np.linspace(1e-3, np.pi - 1e-3, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    nhat = np.stack(
        (
            np.sin(th) * np.cos(ph),
            np.sin(th) * np.sin(ph),
            np.cos(th),
        ),
        axis=-1,
    )
    return nhat


def lienard_angular_power(nhat, beta, beta_dot):
    """Compute Lienard angular power dP/dOmega on a direction grid.

    Formula (up to a global constant):
        dP/dOmega proportional to
        | n x [ (n - beta) x beta_dot ] |^2 / (1 - n . beta)^5
    """
    beta_mag = np.linalg.norm(beta)
    n_dot_beta = np.einsum("...i,i->...", nhat, beta)
    # Tight adaptive floor preserves sharp beaming while avoiding underflow.
    denom_floor = max(1e-12, 1e-6 * (1.0 - beta_mag) ** 2)
    denom = np.clip(1.0 - n_dot_beta, denom_floor, None) ** 5

    inner = np.cross(nhat - beta, beta_dot)
    outer = np.cross(nhat, inner)
    numer = np.einsum("...i,...i->...", outer, outer)

    return numer / denom


def lienard_surface(center, nhat, beta, beta_dot, power_ref_peak, scale=1.15, zoom=1.0):
    """Build a particle-centered surface with radius proportional to Lienard power."""
    power_raw = lienard_angular_power(nhat, beta, beta_dot)
    pnorm = power_raw / max(np.max(power_raw), 1e-12)
    prel = power_raw / max(power_ref_peak, 1e-12)
    radius = zoom * scale * np.power(np.clip(prel, 0.0, None), 0.72)
    pts = center[np.newaxis, np.newaxis, :] + radius[..., np.newaxis] * nhat
    return pts[..., 0], pts[..., 1], pts[..., 2], pnorm, float(np.max(prel))


def mapped_radius_from_power(prel, pnorm, mode):
    """Map power to radius for real-scale vs pedagogical not-to-scale views."""
    prel = np.clip(prel, 0.0, None)
    if mode == "real":
        return prel
    return np.power(np.clip(pnorm, 0.0, None), 0.72)


def choose_direction_resolution(beta_mag):
    """Adaptive angular sampling to resolve thinner high-beta beams."""
    if beta_mag < 0.80:
        return 66, 132
    if beta_mag < 0.95:
        return 96, 192
    return 140, 280


def power_facecolors(power_raw, power_norm, cmap_name="plasma"):
    """Map absolute power to RGBA using the provided color normalization."""
    cmap = plt.get_cmap(cmap_name)
    clipped = np.clip(power_raw, power_norm.vmin, power_norm.vmax)
    mapped = power_norm(clipped)
    rgba = cmap(np.clip(mapped, 0.0, 1.0))
    rgba[..., 3] = 0.28 + 0.68 * np.clip(mapped, 0.0, 1.0)
    return rgba


def auto_power_norm(power_raw):
    """Auto-scale color norm from current power distribution (per speed/frame)."""
    positive = power_raw[power_raw > 0]
    if positive.size == 0:
        return colors.LogNorm(vmin=1e-20, vmax=1.0)

    vmin = max(np.percentile(positive, 2.0), 1e-20)
    vmax = max(np.percentile(positive, 99.8), vmin * 1.05)
    return colors.LogNorm(vmin=vmin, vmax=vmax)


def main():
    global _ANIM

    # Cyclotron dynamics in dimensionless units.
    radius = 1.0
    c_light = 1.0
    beta_speed = [0.35]

    # Animation state.
    phase_step = 0.06
    phase_now = [0.0]
    paused = [False]

    # Camera controls.
    elev_cam = [24.0]
    azim_cam = [-56.0]
    zoom_surface = [1.0]
    scale_mode = ["not-to-scale"]

    # Auto-zoom controls (triggered only by control changes, not per-frame animation).
    auto_zoom_enabled = [True]
    axis_half_extent = [2.0]
    zoom_margin = 1.18

    fig = plt.figure(figsize=(12.4, 8.0), facecolor="#f5efe4")
    ax3d = fig.add_axes([0.04, 0.18, 0.92, 0.74], projection="3d")

    fig.suptitle(
        "Cyclotron/Synchrtron Radiation Demo",
        fontsize=17,
        fontweight="bold",
        color="#1f2a44",
    )

    ax3d.set_facecolor("#fffaf2")

    # Orbit guide ring and B-axis.
    phase = np.linspace(0.0, 2.0 * np.pi, 400)
    ax3d.plot(
        radius * np.cos(phase),
        radius * np.sin(phase),
        np.zeros_like(phase),
        color="#6d6d6d",
        lw=1.4,
        alpha=0.8,
    )

    ax3d.plot([0, 0], [0, 0], [-1.9, 1.9], color="#2a9d8f", lw=2.2, alpha=0.9)
    ax3d.text(0.03, 0.03, 1.80, "B axis", color="#2a9d8f", fontsize=9)

    # Dynamic artists: charge, acceleration axis, and rotating Lienard surface.
    charge_dot, = ax3d.plot([radius], [0.0], [0.0], "o", ms=8, color="#cf5c36")
    acc_axis_line, = ax3d.plot([], [], [], color="#1f2a44", lw=2.1)

    nhat_cache = {}

    def get_nhat_grid(beta_mag):
        key = choose_direction_resolution(beta_mag)
        if key not in nhat_cache:
            nhat_cache[key] = spherical_directions(*key)
        return nhat_cache[key], key

    nhat_grid, _ = get_nhat_grid(beta_speed[0])

    # Reference peak at beta=0.35 sets an absolute visual scale baseline.
    beta_ref = 0.35
    _, vel_ref, acc_ref = charge_state_phase(0.0, radius, beta_ref, c_light)
    nhat_ref = spherical_directions(140, 280)
    power_ref_peak = float(np.max(lienard_angular_power(nhat_ref, vel_ref / c_light, acc_ref / c_light)))

    init_pos, init_vel, init_acc = charge_state_phase(0.0, radius, beta_speed[0], c_light)
    power0 = lienard_angular_power(nhat_grid, init_vel / c_light, init_acc / c_light)
    power_norm = auto_power_norm(power0)
    pnorm0 = power0 / max(np.max(power0), 1e-12)
    prel0 = power0 / max(power_ref_peak, 1e-12)
    mapped0 = mapped_radius_from_power(prel0, pnorm0, scale_mode[0])
    pts0 = init_pos[np.newaxis, np.newaxis, :] + (zoom_surface[0] * 1.15 * mapped0)[..., np.newaxis] * nhat_grid
    donut_surface = [
        ax3d.plot_surface(
            pts0[..., 0],
            pts0[..., 1],
            pts0[..., 2],
            facecolors=power_facecolors(power0, power_norm),
            linewidth=0,
            antialiased=True,
            alpha=0.95,
            shade=False,
        )
    ]

    ax3d.set_xlim(-2.0, 2.0)
    ax3d.set_ylim(-2.0, 2.0)
    ax3d.set_zlim(-2.0, 2.0)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.view_init(elev=elev_cam[0], azim=azim_cam[0])
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("Lienard Angular Power Surface (Relativistic Beaming Enabled)", fontsize=12)

    fig.text(
        0.96,
        0.94,
        (
            r"$\frac{dP}{d\Omega}= \frac{\mu_0 q^2 a^2}{16\pi^2 c}"
            r"\frac{(1-\beta\cos\theta)^2-(1-\beta^2)\sin^2\theta\cos^2\phi}{(1-\beta\cos\theta)^5}$"
        ),
        fontsize=13,
        color="#1f2a44",
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fffaf2", "edgecolor": "#d8c8a8"},
    )

    # Camera controls.
    ax_zoom = fig.add_axes([0.12, 0.138, 0.27, 0.026], facecolor="#ece3d1")
    ax_elev = fig.add_axes([0.12, 0.106, 0.27, 0.026], facecolor="#ece3d1")
    ax_azim = fig.add_axes([0.12, 0.074, 0.27, 0.026], facecolor="#ece3d1")
    ax_beta = fig.add_axes([0.12, 0.042, 0.27, 0.026], facecolor="#ece3d1")

    ax_pause = fig.add_axes([0.55, 0.084, 0.11, 0.050])
    ax_reset = fig.add_axes([0.55, 0.030, 0.11, 0.050])
    ax_autozoom = fig.add_axes([0.69, 0.057, 0.15, 0.060])
    ax_mode = fig.add_axes([0.86, 0.028, 0.13, 0.11], facecolor="#fffaf2")

    s_zoom = Slider(ax_zoom, "radiation scale", 0.20, 3.00, valinit=zoom_surface[0], valstep=0.01)
    s_elev = Slider(ax_elev, "camera elev (deg)", -5.0, 85.0, valinit=elev_cam[0], valstep=1.0)
    s_azim = Slider(ax_azim, "camera azim (deg)", -180.0, 180.0, valinit=azim_cam[0], valstep=1.0)
    s_beta = Slider(ax_beta, "speed v/c", 0.05, 0.995, valinit=beta_speed[0], valstep=0.001)
    b_pause = Button(ax_pause, "Pause", color="#e6dac1", hovercolor="#d8c8a8")
    b_reset = Button(ax_reset, "Reset", color="#e6dac1", hovercolor="#d8c8a8")
    b_autozoom = Button(ax_autozoom, "Auto zoom: On", color="#e6dac1", hovercolor="#d8c8a8")
    rb_mode = RadioButtons(ax_mode, ("real", "not-to-scale"), active=1)

    ax_mode.set_title("scale mode", fontsize=9, color="#1f2a44", pad=3)
    for lbl in rb_mode.labels:
        lbl.set_color("#1f2a44")
        lbl.set_fontsize(9)

    cax = fig.add_axes([0.945, 0.20, 0.015, 0.63])
    mappable = plt.cm.ScalarMappable(cmap=plt.get_cmap("plasma"), norm=power_norm)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("dP/dOmega (arb. units, auto-scaled)", fontsize=9, color="#1f2a44")
    cbar.ax.tick_params(labelsize=8, colors="#1f2a44")

    def update_camera_from_sliders():
        elev_cam[0] = float(s_elev.val)
        azim_cam[0] = float(s_azim.val)
        ax3d.view_init(elev=elev_cam[0], azim=azim_cam[0])
        fig.canvas.draw_idle()

    def update_axis_limits(center, pts):
        if not auto_zoom_enabled[0]:
            return

        spread = pts - center[np.newaxis, np.newaxis, :]
        local_max = np.max(np.abs(spread))
        center_offset = np.max(np.abs(center))
        target_half_extent = zoom_margin * max(local_max + center_offset, 1.3)
        axis_half_extent[0] = 0.85 * axis_half_extent[0] + 0.15 * target_half_extent

        lim = axis_half_extent[0]
        ax3d.set_xlim(-lim, lim)
        ax3d.set_ylim(-lim, lim)
        ax3d.set_zlim(-lim, lim)
        ax3d.set_box_aspect((1.0, 1.0, 1.0))
        fig.canvas.draw_idle()

    def apply_autozoom_for_current_controls():
        beta = beta_speed[0]
        pos, vel, acc = charge_state_phase(phase_now[0], radius, beta, c_light)
        beta_mag = np.linalg.norm(vel / c_light)
        nhat_grid, _ = get_nhat_grid(beta_mag)
        power_raw = lienard_angular_power(nhat_grid, vel / c_light, acc / c_light)
        pnorm = power_raw / max(np.max(power_raw), 1e-12)
        prel = power_raw / max(power_ref_peak, 1e-12)

        mapped = mapped_radius_from_power(prel, pnorm, scale_mode[0])
        pts = pos[np.newaxis, np.newaxis, :] + (zoom_surface[0] * 1.15 * mapped)[..., np.newaxis] * nhat_grid

        update_axis_limits(pos, pts)

    def on_pause(_):
        paused[0] = not paused[0]
        b_pause.label.set_text("Play" if paused[0] else "Pause")

    def on_speed(val):
        beta_speed[0] = float(val)
        apply_autozoom_for_current_controls()

    def on_zoom(val):
        zoom_surface[0] = float(val)
        apply_autozoom_for_current_controls()

    def on_scale_mode(selected):
        scale_mode[0] = selected
        apply_autozoom_for_current_controls()

    def on_autozoom(_):
        auto_zoom_enabled[0] = not auto_zoom_enabled[0]
        b_autozoom.label.set_text("Auto zoom: On" if auto_zoom_enabled[0] else "Auto zoom: Off")
        if auto_zoom_enabled[0]:
            apply_autozoom_for_current_controls()

    def on_reset(_):
        phase_now[0] = 0.0
        s_elev.reset()
        s_azim.reset()
        s_beta.reset()
        s_zoom.reset()
        axis_half_extent[0] = 2.0
        paused[0] = False
        b_pause.label.set_text("Pause")
        update_camera_from_sliders()
        apply_autozoom_for_current_controls()
        b_autozoom.label.set_text("Auto zoom: On" if auto_zoom_enabled[0] else "Auto zoom: Off")

    s_zoom.on_changed(on_zoom)
    s_elev.on_changed(lambda _: update_camera_from_sliders())
    s_azim.on_changed(lambda _: update_camera_from_sliders())
    s_beta.on_changed(on_speed)
    rb_mode.on_clicked(on_scale_mode)
    b_pause.on_clicked(on_pause)
    b_reset.on_clicked(on_reset)
    b_autozoom.on_clicked(on_autozoom)

    def draw_frame(_):
        if not paused[0]:
            phase_now[0] += phase_step

        pos, vel, acc = charge_state_phase(phase_now[0], radius, beta_speed[0], c_light)
        beta_mag = np.linalg.norm(vel / c_light)
        nhat_grid, grid_key = get_nhat_grid(beta_mag)
        acc_hat = acc / max(np.linalg.norm(acc), 1e-12)

        # Update charge marker.
        charge_dot.set_data([pos[0]], [pos[1]])
        charge_dot.set_3d_properties([pos[2]])

        # Axis of instantaneous acceleration through the particle.
        axis_half_len = 0.52
        end1 = pos - axis_half_len * acc_hat
        end2 = pos + axis_half_len * acc_hat
        acc_axis_line.set_data([end1[0], end2[0]], [end1[1], end2[1]])
        acc_axis_line.set_3d_properties([end1[2], end2[2]])

        # Rebuild attached geometry and color from Lienard power.
        power_raw = lienard_angular_power(nhat_grid, vel / c_light, acc / c_light)
        peak_rel = float(np.max(power_raw) / max(power_ref_peak, 1e-12))
        pnorm = power_raw / max(np.max(power_raw), 1e-12)
        prel = power_raw / max(power_ref_peak, 1e-12)
        mapped = mapped_radius_from_power(prel, pnorm, scale_mode[0])
        pts = pos[np.newaxis, np.newaxis, :] + (zoom_surface[0] * 1.15 * mapped)[..., np.newaxis] * nhat_grid

        power_norm = auto_power_norm(power_raw)
        mappable.set_norm(power_norm)
        cbar.update_normal(mappable)

        donut_surface[0].remove()
        donut_surface[0] = ax3d.plot_surface(
            pts[..., 0],
            pts[..., 1],
            pts[..., 2],
            facecolors=power_facecolors(power_raw, power_norm),
            linewidth=0,
            antialiased=True,
            alpha=0.95,
            shade=False,
        )

        phase_deg = np.rad2deg(phase_now[0]) % 360.0
        gamma = 1.0 / np.sqrt(max(1.0 - beta_mag**2, 1e-12))
        beaming_hint = "strong" if beta_mag >= 0.8 else ("moderate" if beta_mag >= 0.5 else "weak")

        ax3d.set_title(
            "Lienard Angular Power Surface (Relativistic Beaming Enabled)"
            f"   beta={beta_mag:.3f}, gamma={gamma:.2f}, {beaming_hint}, mode={scale_mode[0]}",
            fontsize=12,
        )

        return (
            charge_dot,
            acc_axis_line,
        )

    update_camera_from_sliders()
    apply_autozoom_for_current_controls()
    _ANIM = FuncAnimation(fig, draw_frame, interval=28, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
