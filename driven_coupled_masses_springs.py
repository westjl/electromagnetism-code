#!/usr/bin/env python3
"""
Driven Coupled Mass-Spring Demo

This is a modified version of the coupled mass-spring chain demo.
It adds a sinusoidal driving force at the left end of the chain and
lets the user adjust the driving frequency.

The driving force is applied to the first mass as:

    F_drive(t) = F0 sin(2 pi f_drive t)

A small damping term is included so the driven response stays bounded
and easier to observe.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, TextBox, Slider


DEFAULT_N = 3
DEFAULT_K = 4.0
DEFAULT_DRIVE_FREQ_HZ = 0.35
MASS = 1.0
DT = 0.002
SUBSTEPS = 5
DRIVE_FORCE = 1.0
DAMPING = 0.12


class DrivenCoupledMassSpringDemo:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(11, 4.4))
        plt.subplots_adjust(bottom=0.36)

        self.mass_w = 0.32
        self.mass_h = 0.20

        self.state = {
            "running": True,
            "paused": False,
            "drive_on": False,
            "drag_idx": None,
            "n": DEFAULT_N,
            "k": np.full(DEFAULT_N + 1, DEFAULT_K, dtype=float),
            "drive_freq_hz": DEFAULT_DRIVE_FREQ_HZ,
            "t": 0.0,
        }

        self.mass_patches = []
        self.spring_lines = []
        self.wall_artists = []
        self.driver_artists = []

        self.status_text = self.ax.text(
            0.01,
            0.95,
            "Stationary. Turn Drive On to start forcing the chain.",
            transform=self.ax.transAxes,
            fontsize=10,
        )
        self.info_text = self.ax.text(
            0.99,
            0.95,
            "",
            transform=self.ax.transAxes,
            ha="right",
            fontsize=9,
        )

        self._build_controls()
        self._setup_system(DEFAULT_N, np.full(DEFAULT_N + 1, DEFAULT_K, dtype=float), DEFAULT_DRIVE_FREQ_HZ)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

        self.ani = FuncAnimation(self.fig, self._update, interval=16, blit=False)

    def _build_controls(self):
        ax_n = plt.axes([0.08, 0.18, 0.10, 0.08])
        self.n_box = TextBox(ax_n, "N masses", initial=str(DEFAULT_N))

        ax_k = plt.axes([0.24, 0.18, 0.36, 0.08])
        self.k_box = TextBox(
            ax_k,
            "k list",
            initial=self._format_k_list(np.full(DEFAULT_N + 1, DEFAULT_K)),
        )

        ax_drive = plt.axes([0.66, 0.20, 0.23, 0.04])
        self.drive_slider = Slider(
            ax_drive,
            "drive f (Hz)",
            0.05,
            2.50,
            valinit=DEFAULT_DRIVE_FREQ_HZ,
            valstep=0.01,
        )
        self.drive_slider.on_changed(self._on_drive_slider_change)

        ax_apply = plt.axes([0.58, 0.08, 0.10, 0.08])
        self.apply_btn = Button(ax_apply, "Apply")
        self.apply_btn.on_clicked(self._apply_config)

        ax_drive_on = plt.axes([0.70, 0.08, 0.10, 0.08])
        self.drive_btn = Button(ax_drive_on, "Drive On")
        self.drive_btn.on_clicked(self._toggle_drive)

        ax_pause = plt.axes([0.81, 0.08, 0.08, 0.08])
        self.pause_btn = Button(ax_pause, "Pause")
        self.pause_btn.on_clicked(self._toggle_pause)

        ax_reset = plt.axes([0.91, 0.08, 0.08, 0.08])
        self.reset_btn = Button(ax_reset, "Reset")
        self.reset_btn.on_clicked(self._reset)

        self.fig.text(
            0.08,
            0.01,
            "k list accepts either one value (used for all springs) or N+1 comma-separated values.\n"
            "The sinusoidal drive acts on the first mass at the left end of the chain.",
            fontsize=8,
        )

    @staticmethod
    def _format_k_list(values):
        return ", ".join(f"{v:g}" for v in values)

    def _parse_k_values(self, text, n):
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            raise ValueError("Provide at least one spring constant.")

        nums = [float(p) for p in parts]
        if any(v <= 0 for v in nums):
            raise ValueError("All spring constants must be positive.")

        if len(nums) == 1:
            return np.full(n + 1, nums[0], dtype=float)

        if len(nums) != n + 1:
            raise ValueError(f"Need 1 value or exactly N+1 values (expected {n + 1}).")

        return np.array(nums, dtype=float)

    @staticmethod
    def _parse_drive_freq(text):
        value = float(text)
        if value <= 0:
            raise ValueError("Drive frequency must be positive.")
        return value

    def _setup_system(self, n, k_values, drive_freq_hz):
        spacing = 1.0
        left_wall = 0.0
        right_wall = (n + 1) * spacing
        x_eq = np.arange(1, n + 1, dtype=float) * spacing

        self.state["n"] = n
        self.state["k"] = np.array(k_values, dtype=float)
        self.state["drive_freq_hz"] = float(drive_freq_hz)
        self.state["left_wall"] = left_wall
        self.state["right_wall"] = right_wall
        self.state["l0"] = spacing
        self.state["x_eq"] = x_eq
        self.state["x"] = x_eq.copy()
        self.state["v"] = np.zeros(n, dtype=float)
        self.state["t"] = 0.0
        self.state["running"] = True
        self.state["paused"] = False
        self.state["drive_on"] = False
        self.state["drag_idx"] = None

        self._rebuild_artists()
        self._draw_system()

        self.n_box.set_val(str(n))
        self.k_box.set_val(self._format_k_list(self.state["k"]))
        self.drive_slider.set_val(drive_freq_hz)
        self.pause_btn.label.set_text("Pause")
        self.drive_btn.label.set_text("Drive On")
        self.status_text.set_text("Stationary. Turn Drive On to start forcing the chain.")

    def _rebuild_artists(self):
        for art in self.mass_patches + self.spring_lines + self.wall_artists:
            art.remove()
        for art in self.driver_artists:
            art.remove()

        self.mass_patches = []
        self.spring_lines = []
        self.wall_artists = []
        self.driver_artists = []

        n = self.state["n"]
        left_wall = self.state["left_wall"]
        right_wall = self.state["right_wall"]

        wall_l, = self.ax.plot([left_wall, left_wall], [-0.35, 0.35], lw=6, color="black")
        wall_r, = self.ax.plot([right_wall, right_wall], [-0.35, 0.35], lw=6, color="black")
        self.wall_artists.extend([wall_l, wall_r])

        # Driver graphic on the left side of the plot.
        driver_base_x = left_wall - 0.42
        driver_rect = Rectangle(
            (driver_base_x, -0.14),
            0.22,
            0.28,
            facecolor="tab:red",
            edgecolor="black",
            linewidth=1.2,
            alpha=0.9,
        )
        self.ax.add_patch(driver_rect)
        driver_link, = self.ax.plot([driver_base_x + 0.22, left_wall], [0.0, 0.0], lw=2.0, color="tab:red")
        driver_label = self.ax.text(driver_base_x + 0.11, 0.23, "driver", ha="center", va="bottom", fontsize=8)
        self.driver_artists.extend([driver_rect, driver_link, driver_label])

        for _ in range(n + 1):
            line, = self.ax.plot([], [], lw=2.8, color="tab:gray")
            self.spring_lines.append(line)

        for _ in range(n):
            rect = Rectangle(
                (0, -self.mass_h / 2),
                self.mass_w,
                self.mass_h,
                facecolor="tab:blue",
                edgecolor="black",
                linewidth=1.5,
            )
            self.ax.add_patch(rect)
            self.mass_patches.append(rect)

        self.ax.set_xlim(left_wall - 0.6, right_wall + 0.6)
        self.ax.set_ylim(-0.55, 0.55)
        self.ax.set_xlabel("x")
        self.ax.set_yticks([])
        self.ax.set_title("Driven Coupled Mass-Spring Chain")

    def _draw_system(self):
        x = self.state["x"]
        left_wall = self.state["left_wall"]
        right_wall = self.state["right_wall"]
        n = self.state["n"]
        drive_phase = 2.0 * np.pi * self.state["drive_freq_hz"] * self.state["t"]
        drive_offset = 0.0 if not self.state["drive_on"] else 0.08 * np.sin(drive_phase)

        for i in range(n):
            self.mass_patches[i].set_x(x[i] - self.mass_w / 2)

        left_points = [left_wall] + [x[i] + self.mass_w / 2 for i in range(n - 1)] + [x[-1] + self.mass_w / 2]
        right_points = [x[0] - self.mass_w / 2] + [x[i + 1] - self.mass_w / 2 for i in range(n - 1)] + [right_wall]

        for j in range(n + 1):
            self.spring_lines[j].set_data([left_points[j], right_points[j]], [0.0, 0.0])

        driver_rect, driver_link, driver_label = self.driver_artists
        driver_base_x = left_wall - 0.42
        driver_rect.set_x(driver_base_x + drive_offset)
        driver_link.set_data([driver_base_x + 0.22 + drive_offset, left_wall], [0.0, 0.0])
        driver_label.set_position((driver_base_x + 0.11 + drive_offset, 0.23))
        driver_rect.set_facecolor("tab:red" if self.state["drive_on"] else "#bbbbbb")

        disp = x - self.state["x_eq"]
        disp_str = ", ".join(f"{v:+.2f}" for v in disp)
        self.info_text.set_text(
            f"Displacements: [{disp_str}]\n"
            f"Drive: {'ON' if self.state['drive_on'] else 'OFF'} | f = {self.state['drive_freq_hz']:.3f} Hz, F0 = {DRIVE_FORCE:g}, damping = {DAMPING:g}\n"
            f"Drive phase = {drive_phase:.2f} rad"
        )

    def _compute_accel(self):
        x = self.state["x"]
        v = self.state["v"]
        n = self.state["n"]
        k = self.state["k"]
        left_wall = self.state["left_wall"]
        right_wall = self.state["right_wall"]
        l0 = self.state["l0"]
        t = self.state["t"]

        ext = np.zeros(n + 1, dtype=float)
        ext[0] = x[0] - left_wall - l0
        for j in range(1, n):
            ext[j] = x[j] - x[j - 1] - l0
        ext[n] = right_wall - x[n - 1] - l0

        force = np.zeros(n, dtype=float)

        force[0] += -k[0] * ext[0]

        for j in range(1, n):
            force[j - 1] += k[j] * ext[j]
            force[j] += -k[j] * ext[j]

        force[n - 1] += k[n] * ext[n]

        # Light damping keeps the driven response bounded and easier to observe.
        force += -DAMPING * v

        # External sinusoidal drive applied to the first mass at the left end.
        if self.state["drive_on"]:
            force[0] += DRIVE_FORCE * np.sin(2.0 * np.pi * self.state["drive_freq_hz"] * t)

        return force / MASS

    def _apply_config(self, _event):
        try:
            n = int(self.n_box.text)
            if n < 1 or n > 20:
                raise ValueError("N must be between 1 and 20.")
            k_values = self._parse_k_values(self.k_box.text, n)
            drive_freq_hz = float(self.drive_slider.val)
        except Exception as exc:
            self.status_text.set_text(f"Config error: {exc}")
            self.fig.canvas.draw_idle()
            return

        self._setup_system(n, k_values, drive_freq_hz)
        self.fig.canvas.draw_idle()

    def _reset(self, _event):
        self.state["x"] = self.state["x_eq"].copy()
        self.state["v"] = np.zeros(self.state["n"], dtype=float)
        self.state["t"] = 0.0
        self.state["running"] = True
        self.state["paused"] = False
        self.state["drive_on"] = False
        self.state["drag_idx"] = None
        self.pause_btn.label.set_text("Pause")
        self.drive_btn.label.set_text("Drive On")
        self.status_text.set_text("Reset. Stationary at equilibrium.")
        self._draw_system()
        self.fig.canvas.draw_idle()

    def _toggle_pause(self, _event):
        if self.state["drag_idx"] is not None:
            return

        if self.state["running"]:
            self.state["running"] = False
            self.state["paused"] = True
            self.pause_btn.label.set_text("Resume")
            self.status_text.set_text("Paused.")
        elif self.state["paused"]:
            self.state["running"] = True
            self.state["paused"] = False
            self.pause_btn.label.set_text("Pause")
            self.status_text.set_text("Resumed driven oscillation.")
        else:
            self.state["running"] = True
            self.pause_btn.label.set_text("Pause")
            self.status_text.set_text("Started driven oscillation.")

        self.fig.canvas.draw_idle()

    def _toggle_drive(self, _event):
        self.state["drive_on"] = not self.state["drive_on"]
        self.drive_btn.label.set_text("Drive Off" if self.state["drive_on"] else "Drive On")
        if self.state["drive_on"]:
            self.status_text.set_text("Driving force enabled.")
        elif np.allclose(self.state["v"], 0.0) and np.allclose(self.state["x"], self.state["x_eq"]):
            self.status_text.set_text("Driving force disabled; system is stationary at equilibrium.")
        else:
            self.status_text.set_text("Driving force disabled.")
        self.fig.canvas.draw_idle()

    def _on_drive_slider_change(self, _value):
        self.state["drive_freq_hz"] = float(self.drive_slider.val)
        self.fig.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return

        for i, patch in enumerate(self.mass_patches):
            contains, _ = patch.contains(event)
            if contains:
                self.state["drag_idx"] = i
                self.state["running"] = False
                self.state["paused"] = False
                self.pause_btn.label.set_text("Pause")
                self.state["v"][i] = 0.0
                self.status_text.set_text(f"Dragging mass {i + 1}. Release to resume driving.")
                break

    def _on_motion(self, event):
        idx = self.state["drag_idx"]
        if idx is None or event.inaxes is not self.ax or event.xdata is None:
            return

        x = self.state["x"]
        left_wall = self.state["left_wall"]
        right_wall = self.state["right_wall"]

        lo = left_wall + self.mass_w / 2
        hi = right_wall - self.mass_w / 2

        if idx > 0:
            lo = max(lo, x[idx - 1] + 0.9 * self.mass_w)
        if idx < self.state["n"] - 1:
            hi = min(hi, x[idx + 1] - 0.9 * self.mass_w)

        x[idx] = float(np.clip(event.xdata, lo, hi))
        self.state["v"][idx] = 0.0
        self._draw_system()
        self.fig.canvas.draw_idle()

    def _on_release(self, _event):
        if self.state["drag_idx"] is None:
            return

        self.state["drag_idx"] = None
        self.state["running"] = True
        self.state["paused"] = False
        self.pause_btn.label.set_text("Pause")
        self.status_text.set_text("Drag released. The chain responds to the external force when Drive On is enabled.")

    def _update(self, _frame):
        if self.state["running"] and self.state["drag_idx"] is None:
            for _ in range(SUBSTEPS):
                a = self._compute_accel()
                self.state["v"] += a * DT
                self.state["x"] += self.state["v"] * DT
                self.state["t"] += DT

        self._draw_system()
        return []

    def run(self):
        plt.show()


def main():
    demo = DrivenCoupledMassSpringDemo()
    demo.run()


if __name__ == "__main__":
    main()
