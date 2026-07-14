import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, CheckButtons, TextBox


DEFAULT_NZ = 100
DEFAULT_NT = 100
DEFAULT_TMIN = 0.0
DEFAULT_TMAX = 10.0
DEFAULT_ZMIN = 0.0
DEFAULT_ZMAX = 10.0
DEFAULT_A = 1.0
DEFAULT_B = 1.0
DEFAULT_V = 1.0
DEFAULT_Z_FIXED = 1.0
DEFAULT_EXPRS = [
    "A * np.exp(-b * (z - v * t)**2)",
    "A * np.sin(b * (z - v * t))",
    "A / (b * (z - v * t)**2 + 1)",
    "A * np.exp(-b * (b * z**2 + v * t))",
    "A * np.sin(b * z) * np.power(np.cos(b * v * t), 3)",
    "",
]
NUM_EQUATIONS = 6


class FunctionZTTwoWindowAnimator:
    def __init__(self):
        self.default_enabled = [True, False, False, False, False, False]
        self.default_der_enabled = [False, False, False, False, False, False]
        self.state = {
            "playing": False,
            "frame_idx": 0,
            "nz": DEFAULT_NZ,
            "nt": DEFAULT_NT,
            "tmin": DEFAULT_TMIN,
            "tmax": DEFAULT_TMAX,
            "zmin": DEFAULT_ZMIN,
            "zmax": DEFAULT_ZMAX,
            "A": DEFAULT_A,
            "b": DEFAULT_B,
            "v": DEFAULT_V,
            "z_fixed": DEFAULT_Z_FIXED,
            "exprs": DEFAULT_EXPRS.copy(),
            "enabled": self.default_enabled.copy(),
            "der_enabled": self.default_der_enabled.copy(),
            "ymin": None,
            "ymax": None,
            "dragging_zfixed": False,
            "dragging_time_cursor": False,
        }

        self.z = np.linspace(DEFAULT_ZMIN, DEFAULT_ZMAX, DEFAULT_NZ)
        self.t = np.linspace(DEFAULT_TMIN, DEFAULT_TMAX, DEFAULT_NT)
        self.colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

        self._build_plot_window()
        self._build_control_window()
        self._connect_events()

        self._apply_current_settings(reset_frame=True)

        self.ani = FuncAnimation(
            self.fig_plot,
            self._update,
            interval=60,
            blit=False,
            cache_frame_data=False,
        )

    def _build_plot_window(self):
        self.fig_plot, (self.ax_main, self.ax_time) = plt.subplots(
            2,
            1,
            figsize=(11, 8),
            gridspec_kw={"height_ratios": [1, 1]},
        )
        self.fig_plot.canvas.manager.set_window_title("f(z,t) Plots")
        self.fig_plot.subplots_adjust(hspace=0.35, left=0.10, right=0.97, bottom=0.13)
        self.fig_plot.text(0.10, 0.03, "Dotted line indicates the derivative curve.", fontsize=9)

        self.plot_play_btn = Button(self.fig_plot.add_axes([0.84, 0.02, 0.11, 0.055]), "Play")
        self.plot_play_btn.on_clicked(self._toggle_play)

        self.main_lines = []
        self.time_lines = []
        self.main_der_lines = []
        self.time_der_lines = []
        self.main_tangent_lines = []
        self.time_tangent_lines = []
        self.time_markers = []
        for i in range(NUM_EQUATIONS):
            main_line, = self.ax_main.plot([], [], lw=2.2, color=self.colors[i], label=f"f{i + 1}")
            time_line, = self.ax_time.plot([], [], lw=1.8, color=self.colors[i], alpha=0.9, label=f"f{i + 1}")
            main_der_line, = self.ax_main.plot(
                [], [], lw=1.8, ls=":", color=self.colors[i], alpha=0.95, label="_nolegend_"
            )
            time_der_line, = self.ax_time.plot(
                [], [], lw=1.6, ls=":", color=self.colors[i], alpha=0.95, label="_nolegend_"
            )
            main_tangent_line, = self.ax_main.plot(
                [], [], lw=2.6, ls="-", color=self.colors[i], alpha=0.98, label="_nolegend_"
            )
            time_tangent_line, = self.ax_time.plot(
                [], [], lw=2.6, ls="-", color=self.colors[i], alpha=0.98, label="_nolegend_"
            )
            marker, = self.ax_time.plot([], [], marker="o", ms=6, color=self.colors[i], linestyle="None")
            self.main_lines.append(main_line)
            self.time_lines.append(time_line)
            self.main_der_lines.append(main_der_line)
            self.time_der_lines.append(time_der_line)
            self.main_tangent_lines.append(main_tangent_line)
            self.time_tangent_lines.append(time_tangent_line)
            self.time_markers.append(marker)

        self.zfixed_line = self.ax_main.axvline(
            self.state["z_fixed"],
            color="black",
            linestyle="--",
            linewidth=1.4,
            alpha=0.75,
        )
        self.time_cursor_line = self.ax_time.axvline(
            self.state["tmin"],
            color="black",
            linestyle=":",
            linewidth=1.4,
            alpha=0.85,
            label="_nolegend_",
        )

        self.time_text = self.ax_main.text(0.02, 0.93, "", transform=self.ax_main.transAxes, fontsize=11)
        self.formula_text = self.ax_main.text(0.02, 0.85, "", transform=self.ax_main.transAxes, fontsize=10)
        self.fixed_text = self.ax_time.text(0.02, 0.92, "", transform=self.ax_time.transAxes, fontsize=10)

    def _make_tangent_segment(self, x0, y0, slope, x_halfspan):
        x1 = x0 - x_halfspan
        x2 = x0 + x_halfspan
        y1 = y0 + slope * (x1 - x0)
        y2 = y0 + slope * (x2 - x0)
        return [x1, x2], [y1, y2]

    def _build_control_window(self):
        self.fig_ctrl = plt.figure(figsize=(11, 7))
        self.fig_ctrl.canvas.manager.set_window_title("f(z,t) Controls")

        def box(pos, label, initial):
            return TextBox(self.fig_ctrl.add_axes(pos), label, initial=initial)

        self.fig_ctrl.text(0.06, 0.96, "Grid and range", fontsize=10, fontweight="bold")
        self.nz_box = box([0.06, 0.90, 0.09, 0.050], "nz", str(DEFAULT_NZ))
        self.nt_box = box([0.20, 0.90, 0.09, 0.050], "nt", str(DEFAULT_NT))
        self.tmin_box = box([0.34, 0.90, 0.10, 0.050], "tmin", str(DEFAULT_TMIN))
        self.tmax_box = box([0.50, 0.90, 0.10, 0.050], "tmax", str(DEFAULT_TMAX))
        self.zmin_box = box([0.64, 0.90, 0.10, 0.050], "zmin", str(DEFAULT_ZMIN))
        self.zmax_box = box([0.80, 0.90, 0.10, 0.050], "zmax", str(DEFAULT_ZMAX))

        self.fig_ctrl.text(0.06, 0.85, "Parameters and scale", fontsize=10, fontweight="bold")
        self.a_box = box([0.06, 0.78, 0.08, 0.050], "A", str(DEFAULT_A))
        self.b_box = box([0.19, 0.78, 0.08, 0.050], "b", str(DEFAULT_B))
        self.v_box = box([0.32, 0.78, 0.08, 0.050], "v", str(DEFAULT_V))
        self.zfixed_box = box([0.47, 0.78, 0.11, 0.050], "zfix", str(DEFAULT_Z_FIXED))
        self.ymin_box = box([0.63, 0.78, 0.11, 0.050], "ymin", "")
        self.ymax_box = box([0.79, 0.78, 0.11, 0.050], "ymax", "")

        self.apply_btn = Button(self.fig_ctrl.add_axes([0.06, 0.70, 0.10, 0.055]), "Apply")
        self.reset_btn = Button(self.fig_ctrl.add_axes([0.33, 0.70, 0.10, 0.055]), "Reset")

        self.fig_ctrl.text(0.06, 0.67, "Equations", fontsize=10, fontweight="bold")
        self.expr_boxes = []
        y_rows = [0.59, 0.54, 0.49, 0.44, 0.39, 0.34]
        for i in range(NUM_EQUATIONS):
            self.expr_boxes.append(box([0.06, y_rows[i], 0.60, 0.050], f"f{i + 1}", self.state["exprs"][i]))

        self.fig_ctrl.text(0.69, 0.71, "Curves", fontsize=10, fontweight="bold")
        val_ax = self.fig_ctrl.add_axes([0.69, 0.30, 0.13, 0.38])
        der_ax = self.fig_ctrl.add_axes([0.84, 0.30, 0.13, 0.38])
        labels = [f"f{i + 1}" for i in range(NUM_EQUATIONS)]
        der_labels = [f"f{i + 1}'" for i in range(NUM_EQUATIONS)]
        self.eq_checks = CheckButtons(val_ax, labels, self.state["enabled"])
        self.der_checks = CheckButtons(der_ax, der_labels, self.state["der_enabled"])
        val_ax.text(0.02, 0.98, "Value", transform=val_ax.transAxes, ha="left", va="top", fontsize=9)
        der_ax.text(0.02, 0.98, "Deriv", transform=der_ax.transAxes, ha="left", va="top", fontsize=9)

        for txt in self.eq_checks.labels + self.der_checks.labels:
            txt.set_fontsize(10)

        for tb in [
            self.nz_box,
            self.nt_box,
            self.tmin_box,
            self.tmax_box,
            self.zmin_box,
            self.zmax_box,
            self.a_box,
            self.b_box,
            self.v_box,
            self.zfixed_box,
            self.ymin_box,
            self.ymax_box,
        ]:
            tb.label.set_fontsize(9)
            tb.label.set_horizontalalignment("right")
            tb.label.set_x(-0.08)

        for tb in self.expr_boxes:
            tb.label.set_fontsize(9)
            tb.label.set_horizontalalignment("right")
            tb.label.set_x(-0.03)

        self.status_text = self.fig_ctrl.text(0.06, 0.20, "Ready.", fontsize=10)
        self.fig_ctrl.text(
            0.06,
            0.14,
            "Use this window for inputs. Drag the dashed line in the plot window to set z fix interactively.",
            fontsize=9,
        )

        self.apply_btn.on_clicked(self._apply_from_controls)
        self.reset_btn.on_clicked(self._reset)
        self.eq_checks.on_clicked(self._on_toggle_eq)
        self.der_checks.on_clicked(self._on_toggle_der)

    def _connect_events(self):
        self.fig_plot.canvas.mpl_connect("button_press_event", self._on_press_plot)
        self.fig_plot.canvas.mpl_connect("motion_notify_event", self._on_motion_plot)
        self.fig_plot.canvas.mpl_connect("button_release_event", self._on_release_plot)

    def _set_checkbuttons(self, checks, target_status):
        current = list(checks.get_status())
        for i, target in enumerate(target_status):
            if current[i] != target:
                checks.set_active(i)

    def _parse_int(self, text, name, minimum=2):
        try:
            value = int(float(text))
        except Exception as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        return value

    def _parse_float(self, text, name):
        try:
            return float(text)
        except Exception as exc:
            raise ValueError(f"{name} must be numeric") from exc

    def _evaluate_expr(self, expr, z, t):
        local_vars = {
            "np": np,
            "z": z,
            "t": t,
            "A": self.state["A"],
            "b": self.state["b"],
            "v": self.state["v"],
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "sqrt": np.sqrt,
            "pi": np.pi,
        }
        y = eval(expr, {"__builtins__": {}}, local_vars)
        y_arr = np.asarray(y, dtype=float)

        if y_arr.ndim == 0:
            return np.full_like(np.asarray(z, dtype=float), float(y_arr))

        ref = np.asarray(z, dtype=float)
        if y_arr.shape == ref.shape:
            return y_arr

        try:
            return np.broadcast_to(y_arr, ref.shape).astype(float)
        except Exception as exc:
            raise ValueError("Expression result is not compatible with the sampling grid") from exc

    def _compute_auto_ylim(self, panel):
        y_vals = []
        for i in range(NUM_EQUATIONS):
            if not ((self.state["enabled"][i] or self.state["der_enabled"][i]) and self.state["exprs"][i]):
                continue

            expr = self.state["exprs"][i]
            try:
                if panel == "main":
                    sample_t = self.t[::max(1, len(self.t) // 20)]
                    for t_sample in sample_t:
                        y = self._evaluate_expr(expr, self.z, t_sample)
                        if self.state["enabled"][i]:
                            finite = y[np.isfinite(y)]
                            if finite.size:
                                y_vals.extend([float(finite.min()), float(finite.max())])
                        if self.state["der_enabled"][i]:
                            dy_dz = np.gradient(y, self.z)
                            finite_d = dy_dz[np.isfinite(dy_dz)]
                            if finite_d.size:
                                y_vals.extend([float(finite_d.min()), float(finite_d.max())])
                else:
                    y = self._evaluate_expr(expr, np.full_like(self.t, self.state["z_fixed"]), self.t)
                    if self.state["enabled"][i]:
                        finite = y[np.isfinite(y)]
                        if finite.size:
                            y_vals.extend([float(finite.min()), float(finite.max())])
                    if self.state["der_enabled"][i]:
                        dy_dt = np.gradient(y, self.t)
                        finite_d = dy_dt[np.isfinite(dy_dt)]
                        if finite_d.size:
                            y_vals.extend([float(finite_d.min()), float(finite_d.max())])
            except Exception:
                pass

        if not y_vals:
            return -1.5, 1.5
        lo, hi = min(y_vals), max(y_vals)
        margin = max(0.1 * (hi - lo), 0.2)
        return lo - margin, hi + margin

    def _apply_current_settings(self, reset_frame=False):
        self.z = np.linspace(self.state["zmin"], self.state["zmax"], self.state["nz"])
        self.t = np.linspace(self.state["tmin"], self.state["tmax"], self.state["nt"])

        self.ax_main.set_xlim(self.state["zmin"], self.state["zmax"])
        self.ax_main.set_xlabel("z")
        self.ax_main.set_ylabel("f(z, t)")
        self.ax_main.set_title("Spatial Profile at Current Time")

        self.ax_time.set_xlim(self.state["tmin"], self.state["tmax"])
        self.ax_time.set_xlabel("t")
        self.ax_time.set_ylabel(f"f(z={self.state['z_fixed']:.3f}, t)")
        self.ax_time.set_title("Time Series at Fixed z")

        if self.state["ymin"] is not None and self.state["ymax"] is not None:
            main_ymin = time_ymin = self.state["ymin"]
            main_ymax = time_ymax = self.state["ymax"]
        else:
            main_auto_ymin, main_auto_ymax = self._compute_auto_ylim("main")
            time_auto_ymin, time_auto_ymax = self._compute_auto_ylim("time")
            main_ymin = self.state["ymin"] if self.state["ymin"] is not None else main_auto_ymin
            main_ymax = self.state["ymax"] if self.state["ymax"] is not None else main_auto_ymax
            time_ymin = self.state["ymin"] if self.state["ymin"] is not None else time_auto_ymin
            time_ymax = self.state["ymax"] if self.state["ymax"] is not None else time_auto_ymax

        self.ax_main.set_ylim(main_ymin, main_ymax)
        self.ax_time.set_ylim(time_ymin, time_ymax)

        self.zfixed_line.set_xdata([self.state["z_fixed"], self.state["z_fixed"]])
        self.zfixed_box.set_val(f"{self.state['z_fixed']:.3f}")

        if reset_frame:
            self.state["frame_idx"] = 0
            self.state["playing"] = False
            self.plot_play_btn.label.set_text("Play")

        self.ax_main.legend(loc="upper right")
        self.ax_time.legend(loc="upper right")
        self._draw_current_frame()

    def _refresh_for_zfixed_change(self):
        self.ax_time.set_ylabel(f"f(z={self.state['z_fixed']:.3f}, t)")
        self.zfixed_line.set_xdata([self.state["z_fixed"], self.state["z_fixed"]])
        self.zfixed_box.set_val(f"{self.state['z_fixed']:.3f}")

        if self.state["ymin"] is None and self.state["ymax"] is None:
            time_ymin, time_ymax = self._compute_auto_ylim("time")
            self.ax_time.set_ylim(time_ymin, time_ymax)

        self._draw_current_frame()
        self.fig_plot.canvas.draw_idle()
        self.fig_ctrl.canvas.draw_idle()

    def _apply_from_controls(self, _event):
        try:
            nz = self._parse_int(self.nz_box.text, "nz", minimum=2)
            nt = self._parse_int(self.nt_box.text, "nt", minimum=2)
            tmin = self._parse_float(self.tmin_box.text, "t min")
            tmax = self._parse_float(self.tmax_box.text, "t max")
            zmin = self._parse_float(self.zmin_box.text, "z min")
            zmax = self._parse_float(self.zmax_box.text, "z max")
            A = self._parse_float(self.a_box.text, "A")
            b = self._parse_float(self.b_box.text, "b")
            v = self._parse_float(self.v_box.text, "v")
            z_fixed = self._parse_float(self.zfixed_box.text, "z fix")
            exprs = [box.text.strip() for box in self.expr_boxes]
            enabled = list(self.eq_checks.get_status())
            der_enabled = list(self.der_checks.get_status())

            ymin_text = self.ymin_box.text.strip()
            ymax_text = self.ymax_box.text.strip()
            ymin_val = self._parse_float(ymin_text, "y min") if ymin_text else None
            ymax_val = self._parse_float(ymax_text, "y max") if ymax_text else None
            if ymin_val is not None and ymax_val is not None and ymin_val >= ymax_val:
                raise ValueError("y min must be less than y max")

            if tmax == tmin:
                raise ValueError("t max must be different from t min")
            if zmax == zmin:
                raise ValueError("z max must be different from z min")
            if not (zmin <= z_fixed <= zmax):
                raise ValueError("z fix must be between z min and z max")
            if not any(enabled) and not any(der_enabled):
                raise ValueError("Select at least one value or derivative curve")

            test_z = np.linspace(zmin, zmax, max(8, min(64, nz)))
            test_t = np.linspace(tmin, tmax, max(8, min(64, nt)))
            self.state.update({"A": A, "b": b, "v": v, "z_fixed": z_fixed})

            for i in range(NUM_EQUATIONS):
                if enabled[i] or der_enabled[i]:
                    if not exprs[i]:
                        raise ValueError(f"f{i + 1} is selected but empty")
                    self._evaluate_expr(exprs[i], test_z, tmin)
                    self._evaluate_expr(exprs[i], np.full_like(test_t, z_fixed), test_t)
        except Exception as exc:
            self.status_text.set_text(f"Input error: {exc}")
            self.fig_ctrl.canvas.draw_idle()
            return

        self.state.update(
            {
                "nz": nz,
                "nt": nt,
                "tmin": tmin,
                "tmax": tmax,
                "zmin": zmin,
                "zmax": zmax,
                "A": A,
                "b": b,
                "v": v,
                "z_fixed": z_fixed,
                "exprs": exprs,
                "enabled": enabled,
                "der_enabled": der_enabled,
                "ymin": ymin_val,
                "ymax": ymax_val,
                "frame_idx": 0,
            }
        )

        self.status_text.set_text("Applied.")
        self._apply_current_settings(reset_frame=True)
        self.fig_plot.canvas.draw_idle()
        self.fig_ctrl.canvas.draw_idle()

    def _draw_current_frame(self):
        idx = self.state["frame_idx"]
        t_now = self.t[idx]
        active_desc = []

        for i in range(NUM_EQUATIONS):
            expr = self.state["exprs"][i]
            enabled = self.state["enabled"][i]
            der_enabled = self.state["der_enabled"][i]

            if (enabled or der_enabled) and expr:
                try:
                    y_z = self._evaluate_expr(expr, self.z, t_now)
                    y_t = self._evaluate_expr(expr, np.full_like(self.t, self.state["z_fixed"]), self.t)
                    y_marker = self._evaluate_expr(expr, np.array([self.state["z_fixed"]]), np.array([t_now]))
                except Exception as exc:
                    self.status_text.set_text(f"Evaluation error in f{i + 1}: {exc}")
                    self.main_lines[i].set_data([], [])
                    self.time_lines[i].set_data([], [])
                    self.main_der_lines[i].set_data([], [])
                    self.time_der_lines[i].set_data([], [])
                    self.main_tangent_lines[i].set_data([], [])
                    self.time_tangent_lines[i].set_data([], [])
                    self.time_markers[i].set_data([], [])
                    continue

                if enabled:
                    self.main_lines[i].set_data(self.z, y_z)
                    self.time_lines[i].set_data(self.t, y_t)
                    self.time_markers[i].set_data([t_now], [float(y_marker[0])])
                    active_desc.append(f"f{i + 1}")
                else:
                    self.main_lines[i].set_data([], [])
                    self.time_lines[i].set_data([], [])
                    self.time_markers[i].set_data([], [])

                if der_enabled:
                    dy_dz = np.gradient(y_z, self.z)
                    dy_dt = np.gradient(y_t, self.t)
                    self.main_der_lines[i].set_data(self.z, dy_dz)
                    self.time_der_lines[i].set_data(self.t, dy_dt)

                    z0 = self.state["z_fixed"]
                    y0_dz = float(np.interp(z0, self.z, dy_dz))
                    d2y_dz2 = np.gradient(dy_dz, self.z)
                    slope_dz = float(np.interp(z0, self.z, d2y_dz2))
                    z_halfspan = 0.05 * (self.state["zmax"] - self.state["zmin"])
                    tx, ty = self._make_tangent_segment(z0, y0_dz, slope_dz, z_halfspan)
                    self.main_tangent_lines[i].set_data(tx, ty)

                    t0 = t_now
                    y0_dt = float(np.interp(t0, self.t, dy_dt))
                    d2y_dt2 = np.gradient(dy_dt, self.t)
                    slope_dt = float(np.interp(t0, self.t, d2y_dt2))
                    t_halfspan = 0.05 * (self.state["tmax"] - self.state["tmin"])
                    bx, by = self._make_tangent_segment(t0, y0_dt, slope_dt, t_halfspan)
                    self.time_tangent_lines[i].set_data(bx, by)

                    active_desc.append(f"df{i + 1}")
                else:
                    self.main_der_lines[i].set_data([], [])
                    self.time_der_lines[i].set_data([], [])
                    self.main_tangent_lines[i].set_data([], [])
                    self.time_tangent_lines[i].set_data([], [])

                if not der_enabled:
                    self.main_tangent_lines[i].set_data([], [])
                    self.time_tangent_lines[i].set_data([], [])
            else:
                self.main_lines[i].set_data([], [])
                self.time_lines[i].set_data([], [])
                self.main_der_lines[i].set_data([], [])
                self.time_der_lines[i].set_data([], [])
                self.main_tangent_lines[i].set_data([], [])
                self.time_tangent_lines[i].set_data([], [])
                self.time_markers[i].set_data([], [])

        self.time_text.set_text(f"t = {t_now:.3f}  |  frame {idx + 1}/{len(self.t)}")
        self.fixed_text.set_text(f"Bottom plot uses z_fixed = {self.state['z_fixed']:.3f}")
        self.time_cursor_line.set_xdata([t_now, t_now])

        if active_desc:
            short_desc = " | ".join(active_desc[:4])
            if len(active_desc) > 4:
                short_desc += f" | +{len(active_desc) - 4} more"
            self.formula_text.set_text(short_desc)
        else:
            self.formula_text.set_text("No equations selected")

    def _on_toggle_eq(self, _label):
        self.state["enabled"] = list(self.eq_checks.get_status())
        self._draw_current_frame()
        self.fig_plot.canvas.draw_idle()

    def _on_toggle_der(self, _label):
        self.state["der_enabled"] = list(self.der_checks.get_status())
        self._draw_current_frame()
        self.fig_plot.canvas.draw_idle()

    def _toggle_play(self, _event):
        self.state["playing"] = not self.state["playing"]
        self.plot_play_btn.label.set_text("Pause" if self.state["playing"] else "Play")
        self.fig_plot.canvas.draw_idle()
        self.fig_ctrl.canvas.draw_idle()

    def _reset(self, _event):
        self.state.update(
            {
                "playing": False,
                "frame_idx": 0,
                "nz": DEFAULT_NZ,
                "nt": DEFAULT_NT,
                "tmin": DEFAULT_TMIN,
                "tmax": DEFAULT_TMAX,
                "zmin": DEFAULT_ZMIN,
                "zmax": DEFAULT_ZMAX,
                "A": DEFAULT_A,
                "b": DEFAULT_B,
                "v": DEFAULT_V,
                "z_fixed": DEFAULT_Z_FIXED,
                "exprs": DEFAULT_EXPRS.copy(),
                "enabled": self.default_enabled.copy(),
                "der_enabled": self.default_der_enabled.copy(),
                "ymin": None,
                "ymax": None,
                "dragging_zfixed": False,
                "dragging_time_cursor": False,
            }
        )

        self.nz_box.set_val(str(DEFAULT_NZ))
        self.nt_box.set_val(str(DEFAULT_NT))
        self.tmin_box.set_val(str(DEFAULT_TMIN))
        self.tmax_box.set_val(str(DEFAULT_TMAX))
        self.zmin_box.set_val(str(DEFAULT_ZMIN))
        self.zmax_box.set_val(str(DEFAULT_ZMAX))
        self.a_box.set_val(str(DEFAULT_A))
        self.b_box.set_val(str(DEFAULT_B))
        self.v_box.set_val(str(DEFAULT_V))
        self.zfixed_box.set_val(str(DEFAULT_Z_FIXED))
        self.ymin_box.set_val("")
        self.ymax_box.set_val("")
        for i in range(NUM_EQUATIONS):
            self.expr_boxes[i].set_val(DEFAULT_EXPRS[i])

        self._set_checkbuttons(self.eq_checks, self.default_enabled)
        self._set_checkbuttons(self.der_checks, self.default_der_enabled)

        self.status_text.set_text("Reset to defaults.")
        self.plot_play_btn.label.set_text("Play")
        self._apply_current_settings(reset_frame=True)
        self.fig_plot.canvas.draw_idle()
        self.fig_ctrl.canvas.draw_idle()

    def _on_press_plot(self, event):
        if event.xdata is None:
            return

        if event.inaxes is self.ax_main:
            z_range = self.state["zmax"] - self.state["zmin"]
            tolerance = 0.02 * z_range
            if abs(event.xdata - self.state["z_fixed"]) <= tolerance:
                self.state["dragging_zfixed"] = True
            return

        if event.inaxes is self.ax_time:
            t_range = self.state["tmax"] - self.state["tmin"]
            tolerance = 0.02 * t_range
            t_cursor = self.t[self.state["frame_idx"]]
            if abs(event.xdata - t_cursor) <= tolerance:
                self.state["dragging_time_cursor"] = True

    def _on_motion_plot(self, event):
        if event.xdata is None:
            return

        if self.state["dragging_zfixed"] and event.inaxes is self.ax_main:
            self.state["z_fixed"] = float(np.clip(event.xdata, self.state["zmin"], self.state["zmax"]))
            self._refresh_for_zfixed_change()
            return

        if self.state["dragging_time_cursor"] and event.inaxes is self.ax_time:
            t_new = float(np.clip(event.xdata, self.state["tmin"], self.state["tmax"]))
            idx = int(np.argmin(np.abs(self.t - t_new)))
            self.state["frame_idx"] = idx
            self.state["playing"] = False
            self.plot_play_btn.label.set_text("Play")
            self._draw_current_frame()
            self.fig_plot.canvas.draw_idle()
            self.fig_ctrl.canvas.draw_idle()

    def _on_release_plot(self, _event):
        self.state["dragging_zfixed"] = False
        self.state["dragging_time_cursor"] = False

    def _update(self, _frame):
        if self.state["playing"] and len(self.t) > 1:
            self.state["frame_idx"] = (self.state["frame_idx"] + 1) % len(self.t)
            self._draw_current_frame()
        return (
            self.main_lines
            + self.time_lines
            + self.main_der_lines
            + self.time_der_lines
            + self.main_tangent_lines
            + self.time_tangent_lines
            + self.time_markers
            + [self.time_cursor_line]
            + [self.time_text, self.formula_text, self.fixed_text]
        )

    def run(self):
        self._draw_current_frame()
        plt.show()


def main():
    app = FunctionZTTwoWindowAnimator()
    app.run()


if __name__ == "__main__":
    main()
