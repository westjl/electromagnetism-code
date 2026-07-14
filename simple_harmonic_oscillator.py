import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button


# Physical parameters for ideal SHO: m x'' + k x = 0
MASS = 1.0  # kg
SPRING_K = 4.0  # N/m
OMEGA = np.sqrt(SPRING_K / MASS)

# Geometry
WALL_X = 0.0
X_EQ = 0.8  # equilibrium position of mass center (m)
MASS_W = 0.16
MASS_H = 0.16
X_MIN, X_MAX = 0.2, 1.3


def clamp_x(x):
    return float(np.clip(x, X_MIN, X_MAX))


def main():
    state = {
        "A": 0.0,
        "t0": time.perf_counter(),
        "running": False,
        "dragging": False,
        "x": X_EQ,
    }

    fig, ax = plt.subplots(figsize=(9, 3))
    plt.subplots_adjust(bottom=0.25)

    # Wall
    ax.plot([WALL_X, WALL_X], [-0.25, 0.25], lw=6, color="black")

    # Spring and mass
    spring_line, = ax.plot([], [], lw=3, color="tab:gray")
    mass = Rectangle(
        (X_EQ - MASS_W / 2, -MASS_H / 2),
        MASS_W,
        MASS_H,
        facecolor="tab:blue",
        edgecolor="black",
    )
    ax.add_patch(mass)

    status_text = ax.text(
        0.02,
        0.92,
        "Drag the mass, then release.",
        transform=ax.transAxes,
        fontsize=11,
    )
    x_text = ax.text(
        0.02,
        0.82,
        "x - x_eq = +0.000 m",
        transform=ax.transAxes,
        fontsize=10,
    )

    ax.set_xlim(-0.05, 1.4)
    ax.set_ylim(-0.35, 0.35)
    ax.set_xlabel("x (m)")
    ax.set_yticks([])
    ax.set_title("Undamped Mass-Spring Oscillator")

    def draw_system(x):
        x = clamp_x(x)
        state["x"] = x

        mass_left = x - MASS_W / 2
        mass.set_x(mass_left)
        spring_line.set_data([WALL_X, mass_left], [0.0, 0.0])

        disp = x - X_EQ
        x_text.set_text(f"x - x_eq = {disp:+.3f} m")

    def reset(_event=None):
        state["running"] = False
        state["dragging"] = False
        state["A"] = 0.0
        draw_system(X_EQ)
        status_text.set_text("Reset. Drag the mass, then release.")
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes is not ax or event.xdata is None:
            return

        contains, _ = mass.contains(event)
        if contains:
            state["dragging"] = True
            state["running"] = False
            status_text.set_text("Dragging... release to start oscillation.")

    def on_motion(event):
        if not state["dragging"] or event.inaxes is not ax or event.xdata is None:
            return

        draw_system(event.xdata)
        fig.canvas.draw_idle()

    def on_release(_event):
        if not state["dragging"]:
            return

        state["dragging"] = False
        state["A"] = state["x"] - X_EQ
        state["t0"] = time.perf_counter()
        state["running"] = True
        status_text.set_text(
            "Oscillating (ideal, no damping). Drag mass to set new amplitude."
        )

    def update(_frame):
        if state["running"] and not state["dragging"]:
            t = time.perf_counter() - state["t0"]
            x = X_EQ + state["A"] * np.cos(OMEGA * t)
            draw_system(x)

        return spring_line, mass, status_text, x_text

    # Reset button
    ax_reset = plt.axes([0.80, 0.06, 0.12, 0.12])
    btn_reset = Button(ax_reset, "Reset")
    btn_reset.on_clicked(reset)

    # Mouse interactions
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    # Initial draw
    draw_system(X_EQ)

    # Keep a reference so animation is not garbage-collected
    _ani = FuncAnimation(fig, update, interval=16, blit=True)

    plt.show()


if __name__ == "__main__":
    main()
