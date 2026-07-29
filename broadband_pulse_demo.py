#!/usr/bin/env python3
"""
Broadband Wave Packet Demo: Many-Frequency Pulse Simulation

This script simulates a broadband pulse formed from many closely spaced
frequency components (like a laser pulse), rather than just two waves.

The pulse is:
    y(x, t) = sum_{j=0}^{N-1} A_j cos(k_j x - omega_j t)

where the frequencies and wavenumbers are evenly spaced over user-defined ranges.

User controls:
- Number of waves (typically 50-200)
- Center frequency and frequency bandwidth
- Center wavenumber and wavenumber bandwidth
- Time slider and animation controls

The plot shows:
- Instantaneous displacement (sum of all components)
- Envelope (amplitude modulation, from Hilbert transform)
- Group velocity marker (envelope peak)
- Phase velocity marker (nearest crest to envelope peak)
- Measured and theoretical velocities
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation


class BroadbandPulseDemo:
    def __init__(self):
        # Spatial domain
        self.x = np.linspace(0.0, 100.0, 2400)
        self.dx = self.x[1] - self.x[0]

        # Time settings
        self.fps = 30
        self.t_max = 20.0
        self.dt = 1.0 / 100.0
        self.playing = True

        # State for parameters
        self.state = {
            "n_waves": 60,
            "f_center": 7.0,
            "f_bandwidth": 1.0,
            "k_center": 2.5,
            "k_bandwidth": 0.5,
            "t": 0.0,
        }

        # History for velocity tracking
        self.group_hist = []
        self.phase_hist = []

        self._build_figure()
        self._compute_waves()
        self._draw_once()

        self.anim = FuncAnimation(self.fig, self._update, interval=1000 / self.fps, blit=False)

    def _build_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(12.2, 6.0))
        plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.36)

        self.fig.patch.set_facecolor("white")
        self.ax.set_xlim(self.x.min(), self.x.max())
        self.ax.set_ylim(-2.5, 2.5)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_title("Broadband Pulse: Sum of Many Frequency Components (Laser-like Pulse)")
        self.ax.grid(True, alpha=0.25)

        (self.packet_line,) = self.ax.plot([], [], lw=2.0, color="#0072B2", label="Pulse (sum of all waves)")
        (self.envelope_pos_line,) = self.ax.plot([], [], "--", lw=1.8, color="#009E73", label="Envelope (+)")
        (self.envelope_neg_line,) = self.ax.plot([], [], "--", lw=1.8, color="#009E73", label="Envelope (-)")

        self.status_text = self.ax.text(
            0.01,
            0.97,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
        )

        self.vel_text = self.ax.text(
            0.99,
            0.97,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="right",
            fontsize=10,
        )

        self.ax.legend(loc="lower right", frameon=True)

        # Controls
        slider_color = "#f5f5f5"

        ax_nwaves = plt.axes([0.10, 0.27, 0.28, 0.03], facecolor=slider_color)
        self.s_nwaves = Slider(ax_nwaves, "N waves", 1, 20, valinit=self.state["n_waves"], valstep=1)
        self.s_nwaves.on_changed(self._on_param_change)

        ax_fcenter = plt.axes([0.10, 0.22, 0.28, 0.03], facecolor=slider_color)
        self.s_fcenter = Slider(ax_fcenter, "f center (Hz)", 0.05, 50.0, valinit=self.state["f_center"], valstep=0.01)
        self.s_fcenter.on_changed(self._on_param_change)

        ax_fbw = plt.axes([0.10, 0.17, 0.28, 0.03], facecolor=slider_color)
        self.s_fbw = Slider(ax_fbw, "f bandwidth (Hz)", 0.02, 10.0, valinit=self.state["f_bandwidth"], valstep=0.01)
        self.s_fbw.on_changed(self._on_param_change)

        ax_kcenter = plt.axes([0.60, 0.27, 0.28, 0.03], facecolor=slider_color)
        self.s_kcenter = Slider(ax_kcenter, "k center", 0.1, 5.0, valinit=self.state["k_center"], valstep=0.05)
        self.s_kcenter.on_changed(self._on_param_change)

        ax_kbw = plt.axes([0.60, 0.22, 0.28, 0.03], facecolor=slider_color)
        self.s_kbw = Slider(ax_kbw, "k bandwidth", 0.1, 3.0, valinit=self.state["k_bandwidth"], valstep=0.05)
        self.s_kbw.on_changed(self._on_param_change)

        ax_t = plt.axes([0.10, 0.11, 0.78, 0.035], facecolor=slider_color)
        self.s_t = Slider(ax_t, "time t", 0.0, self.t_max, valinit=self.state["t"], valstep=0.01)
        self.s_t.on_changed(self._on_time_change)

        ax_pause = plt.axes([0.10, 0.045, 0.12, 0.05])
        self.b_pause = Button(ax_pause, "Pause")
        self.b_pause.on_clicked(self._toggle_pause)

        ax_reset = plt.axes([0.24, 0.045, 0.12, 0.05])
        self.b_reset = Button(ax_reset, "Reset")
        self.b_reset.on_clicked(self._reset)

        self.fig.text(
            0.10,
            0.005,
            "This demo sums N evenly-spaced frequency components to simulate a broadband pulse (laser-like).\n"
            "Adjust N, frequency range, and wavenumber range to explore how bandwidth affects dispersion and pulse shape.",
            fontsize=9,
        )

    def _compute_waves(self):
        n = int(self.s_nwaves.val)
        f_center = float(self.s_fcenter.val)
        f_bw = float(self.s_fbw.val)
        k_center = float(self.s_kcenter.val)
        k_bw = float(self.s_kbw.val)

        # Create frequency and wavenumber arrays
        if n > 1:
            self.freqs = np.linspace(f_center - f_bw / 2.0, f_center + f_bw / 2.0, n)
            self.ks = np.linspace(k_center - k_bw / 2.0, k_center + k_bw / 2.0, n)
        else:
            self.freqs = np.array([f_center])
            self.ks = np.array([k_center])

        self.omegas = 2.0 * np.pi * self.freqs

        self.state["n_waves"] = n
        self.state["f_center"] = f_center
        self.state["f_bandwidth"] = f_bw
        self.state["k_center"] = k_center
        self.state["k_bandwidth"] = k_bw

    def _compute_pulse(self, t):
        n = len(self.freqs)
        y = np.zeros_like(self.x)

        for j in range(n):
            y += np.cos(self.ks[j] * self.x - self.omegas[j] * t)

        y /= n  # Normalize by number of components

        return y

    def _compute_envelope(self, y):
        # Use analytic signal (Hilbert transform) to extract envelope
        analytic = hilbert(y, N=len(y))
        envelope = np.abs(analytic)
        return envelope

    def _track_markers(self, t):
        y = self._compute_pulse(t)
        env = self._compute_envelope(y)

        # Track group velocity (envelope peak in second half, to avoid startup transient)
        search_start = len(self.x) // 3
        search_end = len(self.x) - 40
        if search_end > search_start:
            seg_env = env[search_start:search_end]
            if np.max(seg_env) > 0.05:
                group_idx = search_start + int(np.argmax(seg_env))
                self.group_hist.append((t, self.x[group_idx]))
                if len(self.group_hist) > 300:
                    self.group_hist = self.group_hist[-300:]

        # Track phase velocity (nearest local maximum crest near envelope peak)
        if len(self.group_hist) >= 1:
            _, gx = self.group_hist[-1]
            closest_idx = int(np.argmin(np.abs(self.x - gx)))
            window = 12
            lo = max(0, closest_idx - window)
            hi = min(len(y), closest_idx + window + 1)
            local_y = y[lo:hi]
            
            if len(local_y) >= 3:
                # Find all local maxima (peaks where y[i-1] < y[i] > y[i+1])
                local_maxima_indices = []
                for i in range(1, len(local_y) - 1):
                    if local_y[i] > local_y[i-1] and local_y[i] > local_y[i+1]:
                        local_maxima_indices.append(lo + i)
                
                if len(local_maxima_indices) > 0:
                    # Pick the local maximum closest to the group marker x-position
                    distances = np.abs(self.x[local_maxima_indices] - gx)
                    closest_peak_idx = local_maxima_indices[int(np.argmin(distances))]
                    
                    self.phase_hist.append((t, self.x[closest_peak_idx]))
                    if len(self.phase_hist) > 300:
                        self.phase_hist = self.phase_hist[-300:]

    @staticmethod
    def _history_velocity(hist, min_dt=0.4):
        if len(hist) < 2:
            return np.nan
        t0, x0 = hist[0]
        t1, x1 = hist[-1]
        dt = t1 - t0
        if dt < min_dt:
            return np.nan
        return (x1 - x0) / dt

    def _compute_theory_vp_vg(self):
        # Simple linear dispersion: omega = c * k
        k_avg = np.mean(self.ks)
        omega_avg = np.mean(self.omegas)
        vp = omega_avg / k_avg if k_avg > 1e-12 else np.nan

        # Group velocity from slope of omega-k relation
        k_min, k_max = np.min(self.ks), np.max(self.ks)
        omega_min, omega_max = np.min(self.omegas), np.max(self.omegas)
        dk = k_max - k_min
        domega = omega_max - omega_min

        vg = domega / dk if dk > 1e-12 else np.nan

        return vp, vg

    def _update_display(self):
        t = self.state["t"]

        y = self._compute_pulse(t)
        env = self._compute_envelope(y)

        self.packet_line.set_data(self.x, y)
        self.envelope_pos_line.set_data(self.x, env)
        self.envelope_neg_line.set_data(self.x, -env)

        self._track_markers(t)

        vp_th, vg_th = self._compute_theory_vp_vg()

        self.status_text.set_text(
            f"N={int(self.s_nwaves.val)}, "
            f"f∈[{self.state['f_center']-self.state['f_bandwidth']/2:.3f}, {self.state['f_center']+self.state['f_bandwidth']/2:.3f}] Hz, "
            f"k∈[{self.state['k_center']-self.state['k_bandwidth']/2:.3f}, {self.state['k_center']+self.state['k_bandwidth']/2:.3f}]"
        )

        self.vel_text.set_text(
            f"t = {t:6.2f} s\n"
            f"Theory: vp = {vp_th:6.3f}, vg = {vg_th:6.3f}"
        )

    def _on_param_change(self, _value):
        self._compute_waves()
        self._draw_once()

    def _on_time_change(self, _value):
        self.state["t"] = float(self.s_t.val)
        self._update_display()
        self.fig.canvas.draw_idle()

    def _draw_once(self):
        self._update_display()
        self.fig.canvas.draw_idle()

    def _toggle_pause(self, _event):
        self.playing = not self.playing
        self.b_pause.label.set_text("Pause" if self.playing else "Play")

    def _reset(self, _event):
        self.state["t"] = 0.0
        self.s_t.set_val(0.0)
        self.group_hist = []
        self.phase_hist = []
        self.playing = True
        self.b_pause.label.set_text("Pause")

    def _update(self, _frame):
        if self.playing:
            t_new = self.state["t"] + self.dt
            if t_new > self.t_max:
                t_new = 0.0
            self.s_t.set_val(t_new)

        self._update_display()
        self.fig.canvas.draw_idle()
        return []

    def run(self):
        plt.show()


def main():
    demo = BroadbandPulseDemo()
    demo.run()


if __name__ == "__main__":
    main()
