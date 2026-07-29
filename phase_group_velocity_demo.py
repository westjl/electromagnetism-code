#!/usr/bin/env python3
"""
Interactive Phase and Group Velocity Demo

This script visualizes a wave packet formed by adding two waves:

    y(x, t) = A1 cos(k1 x - w1 t) + A2 cos(k2 x - w2 t)

User controls are provided for A1, A2, k1, k2, w1, w2, and time t.
The play/pause button advances time automatically.

The equations are shown directly on the figure and update live.

Run:
    python phase_group_velocity_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.animation import FuncAnimation


# Spatial domain
x = np.linspace(0.0, 100.0, 2400)

# Time settings
fps = 30
t_max = 20.0
dt = 1.0 / fps
playing = True


# Figure setup: leave room at bottom for controls.
fig, ax = plt.subplots(figsize=(12.2, 8.0))
plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.36)

fig.patch.set_facecolor("white")
ax.set_xlim(0.0, 100.0)
ax.set_ylim(-4.6, 4.6)
ax.set_xlabel("x")
ax.set_ylabel("Amplitude")
ax.set_title("Interactive Demo: Phase Velocity vs Group Velocity")
ax.grid(True, alpha=0.25)

# Main lines
(wave1_line,) = ax.plot([], [], lw=1.2, color="#999999", alpha=0.8, label="Wave 1")
(wave2_line,) = ax.plot([], [], lw=1.2, color="#BBBBBB", alpha=0.8, label="Wave 2")
(packet_line,) = ax.plot([], [], lw=2.1, color="#0072B2", label="Wave packet (sum)")
(envelope_pos_line,) = ax.plot([], [], "--", lw=1.6, color="#009E73", label="Envelope (+ approx)")
(envelope_neg_line,) = ax.plot([], [], "--", lw=1.6, color="#009E73", label="Envelope (- approx)")

# Markers showing where each speed carries a reference point.
phase_marker = ax.axvline(0.0, color="#D55E00", lw=2.0, alpha=0.9, label="Phase marker")
group_marker = ax.axvline(0.0, color="#CC79A7", lw=2.0, alpha=0.9, label="Group marker")

# Text panels
info = ax.text(
    0.015,
    0.98,
    "",
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#999999"),
)

equations = ax.text(
    0.985,
    0.98,
    "",
    transform=ax.transAxes,
    va="top",
    ha="right",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#999999"),
)

ax.legend(loc="lower right", frameon=True)


# Initial values
init_vals = {
    "A1": 1.0,
    "A2": 1.0,
    "k1": 2.00,
    "k2": 2.25,
    "w1": 4.60,
    "w2": 4.70,
    "t": 0.0,
}

# Parameters used for envelope/velocity calculations.
applied_params = {
    "A1": init_vals["A1"],
    "A2": init_vals["A2"],
    "k1": init_vals["k1"],
    "k2": init_vals["k2"],
    "w1": init_vals["w1"],
    "w2": init_vals["w2"],
}


def wave1(x_vals: np.ndarray, t: float, a1: float, kk1: float, ww1: float) -> np.ndarray:
    return a1 * np.cos(kk1 * x_vals - ww1 * t)


def wave2(x_vals: np.ndarray, t: float, a2: float, kk2: float, ww2: float) -> np.ndarray:
    return a2 * np.cos(kk2 * x_vals - ww2 * t)


def envelope_approx(
    x_vals: np.ndarray,
    t: float,
    a1: float,
    a2: float,
    kk1: float,
    kk2: float,
    ww1: float,
    ww2: float,
) -> np.ndarray:
    # Equal amplitudes give the exact 2A cos(...) envelope; otherwise this is a useful approximation.
    amp = a1 + a2
    return amp * np.cos(0.5 * (kk2 - kk1) * x_vals - 0.5 * (ww2 - ww1) * t)


def compute_velocities(kk1: float, kk2: float, ww1: float, ww2: float) -> tuple[float, float]:
    k0 = 0.5 * (kk1 + kk2)
    w0 = 0.5 * (ww1 + ww2)

    v_phase = np.nan if np.isclose(k0, 0.0) else w0 / k0
    v_group = np.nan if np.isclose(kk2 - kk1, 0.0) else (ww2 - ww1) / (kk2 - kk1)
    return v_phase, v_group


def finite_str(val: float) -> str:
    return "undefined" if np.isnan(val) else f"{val:6.3f}"


def get_params() -> tuple[float, float, float, float, float, float, float]:
    return (
        s_a1.val,
        s_a2.val,
        s_k1.val,
        s_k2.val,
        s_w1.val,
        s_w2.val,
        s_t.val,
    )


def slider_params_dict() -> dict[str, float]:
    return {
        "A1": s_a1.val,
        "A2": s_a2.val,
        "k1": s_k1.val,
        "k2": s_k2.val,
        "w1": s_w1.val,
        "w2": s_w2.val,
    }


def has_pending_apply(curr: dict[str, float]) -> bool:
    tol = 1e-12
    return any(abs(curr[key] - applied_params[key]) > tol for key in applied_params)


def refresh_plot(_=None):
    a1, a2, kk1, kk2, ww1, ww2, t = get_params()

    y1 = wave1(x, t, a1, kk1, ww1)
    y2 = wave2(x, t, a2, kk2, ww2)
    y = y1 + y2
    env = envelope_approx(
        x,
        t,
        applied_params["A1"],
        applied_params["A2"],
        applied_params["k1"],
        applied_params["k2"],
        applied_params["w1"],
        applied_params["w2"],
    )

    wave1_line.set_data(x, y1)
    wave2_line.set_data(x, y2)
    packet_line.set_data(x, y)
    envelope_pos_line.set_data(x, env)
    envelope_neg_line.set_data(x, -env)

    v_phase, v_group = compute_velocities(
        applied_params["k1"],
        applied_params["k2"],
        applied_params["w1"],
        applied_params["w2"],
    )

    x_phase = 0.0 if np.isnan(v_phase) else v_phase * t
    x_group = 0.0 if np.isnan(v_group) else v_group * t
    phase_marker.set_xdata([x_phase, x_phase])
    group_marker.set_xdata([x_group, x_group])

    info.set_text(
        f"t = {t:5.2f} s\n"
        f"v_phase = {finite_str(v_phase)}\n"
        f"v_group = {finite_str(v_group)}\n"
        f"k0={(0.5 * (applied_params['k1'] + applied_params['k2'])):.3f}, "
        f"w0={(0.5 * (applied_params['w1'] + applied_params['w2'])):.3f}"
    )

    equations.set_text(
        "$y_1(x,t)=A_1\\cos(k_1x-\\omega_1t)$\n"
        "$y_2(x,t)=A_2\\cos(k_2x-\\omega_2t)$\n"
        "$y(x,t)=y_1+y_2$\n"
        "$v_p=\\omega_0/k_0,\\;k_0=(k_1+k_2)/2,\\;\\omega_0=(\\omega_1+\\omega_2)/2$\n"
        "$v_g\\approx(\\omega_2-\\omega_1)/(k_2-k_1)$\n"
        f"$A_1={a1:.2f},\\;A_2={a2:.2f},\\;k_1={kk1:.2f},\\;k_2={kk2:.2f}$\n"
        f"$\\omega_1={ww1:.2f},\\;\\omega_2={ww2:.2f}$"
    )

    pending_text.set_text("Pending changes: press Apply" if has_pending_apply(slider_params_dict()) else "Applied")

    fig.canvas.draw_idle()


# ----------------------
# UI controls
# ----------------------
slider_color = "#f5f5f5"

ax_a1 = fig.add_axes([0.10, 0.27, 0.32, 0.03], facecolor=slider_color)
ax_a2 = fig.add_axes([0.58, 0.27, 0.32, 0.03], facecolor=slider_color)
ax_k1 = fig.add_axes([0.10, 0.22, 0.32, 0.03], facecolor=slider_color)
ax_k2 = fig.add_axes([0.58, 0.22, 0.32, 0.03], facecolor=slider_color)
ax_w1 = fig.add_axes([0.10, 0.17, 0.32, 0.03], facecolor=slider_color)
ax_w2 = fig.add_axes([0.58, 0.17, 0.32, 0.03], facecolor=slider_color)
ax_t = fig.add_axes([0.10, 0.11, 0.80, 0.035], facecolor=slider_color)
ax_btn_play = fig.add_axes([0.10, 0.045, 0.16, 0.045])
ax_btn_apply = fig.add_axes([0.29, 0.045, 0.16, 0.045])

s_a1 = Slider(ax_a1, "A1", 0.0, 2.5, valinit=init_vals["A1"], valstep=0.01)
s_a2 = Slider(ax_a2, "A2", 0.0, 2.5, valinit=init_vals["A2"], valstep=0.01)
s_k1 = Slider(ax_k1, "k1", 0.2, 4.0, valinit=init_vals["k1"], valstep=0.01)
s_k2 = Slider(ax_k2, "k2", 0.2, 4.0, valinit=init_vals["k2"], valstep=0.01)
s_w1 = Slider(ax_w1, "omega1", 0.2, 10.0, valinit=init_vals["w1"], valstep=0.01)
s_w2 = Slider(ax_w2, "omega2", 0.2, 10.0, valinit=init_vals["w2"], valstep=0.01)
s_t = Slider(ax_t, "time t", 0.0, t_max, valinit=init_vals["t"], valstep=0.01)

btn_play = Button(ax_btn_play, "Pause", color="#e6e6e6", hovercolor="#d8d8d8")
btn_apply = Button(ax_btn_apply, "Apply", color="#dff0d8", hovercolor="#cfe9c5")

pending_text = fig.text(0.48, 0.063, "Applied", fontsize=10, ha="left", va="center")

component_labels = [
    "Wave 1",
    "Wave 2",
    "Packet",
    "Envelope +",
    "Envelope -",
    "Phase marker",
    "Group marker",
]

component_artists = {
    "Wave 1": wave1_line,
    "Wave 2": wave2_line,
    "Packet": packet_line,
    "Envelope +": envelope_pos_line,
    "Envelope -": envelope_neg_line,
    "Phase marker": phase_marker,
    "Group marker": group_marker,
}

# Put visibility toggles in a separate window to keep the main plot uncluttered.
checks_fig, ax_checks = plt.subplots(figsize=(3.1, 3.0))
checks_fig.patch.set_facecolor("white")
checks_fig.suptitle("Component Visibility", fontsize=11)
ax_checks.set_facecolor("#f5f5f5")
ax_checks.set_xticks([])
ax_checks.set_yticks([])

checks = CheckButtons(
    ax_checks,
    labels=component_labels,
    actives=[True for _ in component_labels],
)

# Keep checkbox text compact so the control fits below the sliders.
for label in checks.labels:
    label.set_fontsize(8)


def on_slider_change(_):
    refresh_plot()


for slider in (s_a1, s_a2, s_k1, s_k2, s_w1, s_w2, s_t):
    slider.on_changed(on_slider_change)


def toggle_play(_):
    global playing
    playing = not playing
    btn_play.label.set_text("Pause" if playing else "Play")


btn_play.on_clicked(toggle_play)


def apply_settings(_):
    curr = slider_params_dict()
    for key in applied_params:
        applied_params[key] = curr[key]
    refresh_plot()


btn_apply.on_clicked(apply_settings)


def toggle_component(label: str):
    artist = component_artists[label]
    artist.set_visible(not artist.get_visible())
    fig.canvas.draw_idle()


checks.on_clicked(toggle_component)


def animate(_frame):
    if playing:
        t_new = s_t.val + dt
        if t_new > t_max:
            t_new = 0.0
        s_t.set_val(t_new)

    return (
        wave1_line,
        wave2_line,
        packet_line,
        envelope_pos_line,
        envelope_neg_line,
        phase_marker,
        group_marker,
        info,
        equations,
    )


refresh_plot()

anim = FuncAnimation(fig, animate, interval=1000 / fps, blit=False)

plt.show()
