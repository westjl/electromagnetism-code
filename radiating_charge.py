import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons, Button, Slider

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
C = 1.0                  # Speed of light (simulation units)
NUM_LINES = 16           # Electric field lines
# History needs to reach GRID_RADIUS/C = 10 time-units back; 300 * 0.05 = 15 s
STEPS_HISTORY = 300
GRID_RADIUS = 10.0
EDGE_MARGIN = 1.0
NUM_POINTS_PER_LINE = 60
dt = 0.05

# Adjustable parameters (controlled by sliders)
osc_amp    = [0.8]   # Oscillation amplitude
osc_omega  = [2.0]   # Oscillation angular frequency
charge_vel = [0.4]   # Speed for constant-velocity, acceleration, and circular modes
lin_accel  = [0.02]  # Constant acceleration for linear-acceleration mode
orbit_r    = [2.0]   # Orbital radius for circular-orbit mode

# ---------------------------------------------------------------------------
# Mutable simulation state (lists so closures can rebind contents)
# ---------------------------------------------------------------------------
current_step       = [0]
time_history       = np.zeros(STEPS_HISTORY)
pos_history        = np.zeros((STEPS_HISTORY, 2))

wall_t             = [0.0]   # Monotonically increasing simulation clock
mode               = ['constant']
mode_start_wall_t  = [0.0]   # wall_t value when the current mode was activated

drag_pos           = [np.array([0.0, 0.0])]
is_dragging        = [False]
paused             = [False]
charge_stopped     = [False]
stopped_local_t    = [None]
stopped_position   = [np.array([0.0, 0.0])]

# ---------------------------------------------------------------------------
# Charge trajectory
# local_t = wall_t - mode_start_wall_t  (can be negative in pre-populated history)
# ---------------------------------------------------------------------------

def reached_box_edge(pos):
    limit = GRID_RADIUS - EDGE_MARGIN
    return (pos[0] <= -limit or pos[0] >= limit or
            pos[1] <= -limit or pos[1] >= limit)


def get_charge_position(wt):
    lt = wt - mode_start_wall_t[0]
    m  = mode[0]
    if charge_stopped[0] and stopped_local_t[0] is not None and lt >= stopped_local_t[0]:
        return stopped_position[0].copy()
    if m == 'oscillate':
        return np.array([0.0, osc_amp[0] * np.sin(osc_omega[0] * lt)])
    elif m == 'constant':
        # Charge moves right at constant speed — no acceleration, no radiation
        v = charge_vel[0]
        return np.array([-2.0 + v * lt, 0.0])
    elif m == 'linear':
        v0 = charge_vel[0]
        a = lin_accel[0]
        return np.array([-3.0 + v0 * lt + 0.5 * a * lt * lt, 0.0])
    elif m == 'drag':
        return drag_pos[0].copy()
    elif m == 'circular':
        r = orbit_r[0]
        v = charge_vel[0]
        omega = v / r          # v = r*omega  =>  omega = v/r
        return np.array([r * np.cos(omega * lt), r * np.sin(omega * lt)])
    return np.array([0.0, 0.0])

# ---------------------------------------------------------------------------
# History reset  — fills buffer with STEPS_HISTORY frames going *back* from
# the current wall_t, using the current mode's trajectory extrapolated backwards.
# ---------------------------------------------------------------------------
def reset_simulation():
    t_now = wall_t[0]
    for i in range(STEPS_HISTORY):
        t_i = t_now - (STEPS_HISTORY - 1 - i) * dt
        time_history[i] = t_i
        pos_history[i]  = get_charge_position(t_i)
    current_step[0] = STEPS_HISTORY - 1

# Initial population
reset_simulation()

# ---------------------------------------------------------------------------
# Figure layout — left panel for controls, right panel for animation
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(11, 7))
fig.patch.set_facecolor('#111111')

ax = fig.add_axes([0.23, 0.03, 0.75, 0.94])
ax.set_xlim(-GRID_RADIUS, GRID_RADIUS)
ax.set_ylim(-GRID_RADIUS, GRID_RADIUS)
ax.set_aspect('equal')
ax.axis('off')
ax.set_facecolor('#111111')

ax_radio   = fig.add_axes([0.01, 0.70, 0.20, 0.28], facecolor='#1e1e1e')
ax_pause   = fig.add_axes([0.03, 0.62, 0.16, 0.05])
ax_stop    = fig.add_axes([0.03, 0.56, 0.16, 0.05])
ax_reset   = fig.add_axes([0.03, 0.50, 0.16, 0.05])
ax_sl_amp  = fig.add_axes([0.07, 0.41, 0.12, 0.025], facecolor='#2a2a2a')
ax_sl_freq = fig.add_axes([0.07, 0.34, 0.12, 0.025], facecolor='#2a2a2a')
ax_sl_vel  = fig.add_axes([0.07, 0.27, 0.12, 0.025], facecolor='#2a2a2a')
ax_sl_acc  = fig.add_axes([0.07, 0.20, 0.12, 0.025], facecolor='#2a2a2a')
ax_sl_orb  = fig.add_axes([0.07, 0.13, 0.12, 0.025], facecolor='#2a2a2a')
ax_info    = fig.add_axes([0.01, 0.02, 0.20, 0.08], facecolor='#1a1a1a')
ax_info.axis('off')

# --- Radio buttons ---
radio = RadioButtons(
    ax_radio,
    ('Const. Velocity', 'Linear Acceleration', 'Sinusoidal', 'Circular', 'Manual mode'),
    activecolor='#4da6ff',
)
for lbl in radio.labels:
    lbl.set_color('white')
    lbl.set_fontsize(9)
ax_radio.set_title('Motion Mode', color='#aaaaaa', fontsize=9, pad=3)

# --- Pause button ---
btn_pause = Button(ax_pause, 'Pause', color='#2a2a2a', hovercolor='#444444')
btn_pause.label.set_color('white')
btn_pause.label.set_fontsize(9)

# --- Reset button ---
btn_reset = Button(ax_reset, 'Reset', color='#2a2a2a', hovercolor='#444444')
btn_reset.label.set_color('white')
btn_reset.label.set_fontsize(9)

# --- Stop charge button ---
btn_stop = Button(ax_stop, 'Stop charge', color='#2a2a2a', hovercolor='#5a1010')
btn_stop.label.set_color('#ff8888')
btn_stop.label.set_fontsize(9)

# --- Sliders ---
SLIDER_COLOR = '#4da6ff'
sl_amp = Slider(ax_sl_amp, 'Amp', 0.1, 3.0, valinit=osc_amp[0],
                color=SLIDER_COLOR, initcolor='none')
sl_freq = Slider(ax_sl_freq, 'Freq', 0.2, 6.0, valinit=osc_omega[0],
                 color=SLIDER_COLOR, initcolor='none')
sl_vel = Slider(ax_sl_vel, 'Vel/c', 0.05, 0.95, valinit=charge_vel[0],
                color=SLIDER_COLOR, initcolor='none')
sl_acc = Slider(ax_sl_acc, 'Accel', -0.08, 0.08, valinit=lin_accel[0],
                color='#ff8844', initcolor='none')
sl_orb = Slider(ax_sl_orb, 'Orb. r', 0.5, 4.0, valinit=orbit_r[0],
                color='#ffaa44', initcolor='none')
for sl in (sl_amp, sl_freq, sl_vel, sl_acc, sl_orb):
    sl.label.set_color('#cccccc')
    sl.label.set_fontsize(8)
    sl.valtext.set_color('#cccccc')
    sl.valtext.set_fontsize(8)
    sl.ax.tick_params(colors='#555555')

# Label the slider group
fig.text(0.115, 0.455, 'Parameters', color='#aaaaaa', fontsize=8,
         ha='center', va='bottom')

def on_sl_amp(val):
    osc_amp[0] = val
    charge_stopped[0] = False
    stopped_local_t[0] = None
    btn_stop.label.set_text('Stop charge')
    if mode[0] == 'oscillate':
        mode_start_wall_t[0] = wall_t[0]
        reset_simulation()

def on_sl_freq(val):
    osc_omega[0] = val
    charge_stopped[0] = False
    stopped_local_t[0] = None
    btn_stop.label.set_text('Stop charge')
    if mode[0] == 'oscillate':
        mode_start_wall_t[0] = wall_t[0]
        reset_simulation()

def on_sl_vel(val):
    charge_vel[0] = val
    charge_stopped[0] = False
    stopped_local_t[0] = None
    btn_stop.label.set_text('Stop charge')
    if mode[0] in ('constant', 'linear', 'circular'):
        mode_start_wall_t[0] = wall_t[0]
        reset_simulation()

def on_sl_acc(val):
    lin_accel[0] = val
    charge_stopped[0] = False
    stopped_local_t[0] = None
    btn_stop.label.set_text('Stop charge')
    if mode[0] == 'linear':
        mode_start_wall_t[0] = wall_t[0]
        reset_simulation()

def on_sl_orb(val):
    orbit_r[0] = val
    charge_stopped[0] = False
    stopped_local_t[0] = None
    btn_stop.label.set_text('Stop charge')
    if mode[0] == 'circular':
        mode_start_wall_t[0] = wall_t[0]
        reset_simulation()

sl_amp.on_changed(on_sl_amp)
sl_freq.on_changed(on_sl_freq)
sl_vel.on_changed(on_sl_vel)
sl_acc.on_changed(on_sl_acc)
sl_orb.on_changed(on_sl_orb)

# --- Mode description panel ---
MODE_INFO = {
    'oscillate': (
        'SINUSOIDAL\n\n'
        'Charge oscillates\n'
        'sinusoidally.\n\n'
        'Kinks in field lines\n'
        'carry radiated energy\n'
        'outward — classic\n'
        'electric dipole\n'
        'radiation.'
    ),
    'constant': (
        'CONST. VELOCITY\n\n'
        'Charge moves at\n'
        'constant speed.\n\n'
        'No acceleration\n'
        '→ no radiation.\n'
        'Field lines remain\n'
        'smooth and kink-free.'
    ),
    'linear': (
        'LINEAR ACCELERATION\n\n'
        'Charge has constant\n'
        'acceleration along x.\n\n'
        'This produces a\n'
        'growing asymmetry\n'
        'in the field and\n'
        'clear radiation.'
    ),
    'drag': (
        'MANUAL DRAG\n\n'
        'Click and drag\n'
        'the charge freely.\n\n'
        'Any non-uniform\n'
        'motion produces\n'
        'visible kinks —\n'
        'radiation.'
    ),
    'circular': (
        'CIRCULAR ORBIT\n\n'
        'Vel/c < 0.3: kinks\n'
        'spread isotropically\n'
        '→ cyclotron.\n\n'
        'Vel/c > 0.6: kinks\n'
        'beam forward\n'
        '→ synchrotron.\n\n'
        'Orb. r sets radius.'
    ),
}

info_label = ax_info.text(
    0.07, 0.97, MODE_INFO['constant'],
    transform=ax_info.transAxes,
    color='#cccccc', fontsize=8, va='top',
    linespacing=1.4, fontfamily='monospace',
)
ax_info.set_title('About', color='#aaaaaa', fontsize=8, pad=2)

# ---------------------------------------------------------------------------
# Animation artists
# ---------------------------------------------------------------------------
charge_plot, = ax.plot([], [], 'o', color='#ff4444', ms=10, zorder=5)
field_lines  = [ax.plot([], [], color='#4da6ff', lw=1.5, alpha=0.85)[0]
                for _ in range(NUM_LINES)]
emission_angles = np.linspace(0, 2 * np.pi, NUM_LINES, endpoint=False)

# ---------------------------------------------------------------------------
# Widget callbacks
# ---------------------------------------------------------------------------
MODE_MAP = {
    'Const. Velocity':    'constant',
    'Linear Acceleration': 'linear',
    'Sinusoidal':         'oscillate',
    'Circular':           'circular',
    'Manual mode':        'drag',
}

def on_mode_change(label):
    # Snapshot position so drag mode inherits wherever the charge currently is
    drag_pos[0] = get_charge_position(wall_t[0])
    mode[0] = MODE_MAP[label]
    mode_start_wall_t[0] = wall_t[0]
    paused[0] = False
    btn_pause.label.set_text('Pause')
    charge_stopped[0] = False
    stopped_local_t[0] = None
    btn_stop.label.set_text('Stop charge')
    reset_simulation()
    info_label.set_text(MODE_INFO[mode[0]])
    fig.canvas.draw_idle()

radio.on_clicked(on_mode_change)

def on_reset(event):
    drag_pos[0] = get_charge_position(wall_t[0])
    mode_start_wall_t[0] = wall_t[0]
    paused[0] = False
    btn_pause.label.set_text('Pause')
    charge_stopped[0] = False
    stopped_local_t[0] = None
    btn_stop.label.set_text('Stop charge')
    reset_simulation()

btn_reset.on_clicked(on_reset)

def on_stop(event):
    """Freeze the charge position while leaving the field history evolving."""
    if not charge_stopped[0]:
        stopped_position[0] = get_charge_position(wall_t[0])
        stopped_local_t[0] = wall_t[0] - mode_start_wall_t[0]
        charge_stopped[0] = True
        btn_stop.label.set_text('Charge stopped')
    else:
        charge_stopped[0] = False
        stopped_local_t[0] = None
        btn_stop.label.set_text('Stop charge')
    fig.canvas.draw_idle()

btn_stop.on_clicked(on_stop)

def on_pause(event):
    paused[0] = not paused[0]
    btn_pause.label.set_text('Resume' if paused[0] else 'Pause')
    fig.canvas.draw_idle()

btn_pause.on_clicked(on_pause)

# ---------------------------------------------------------------------------
# Mouse drag (Manual Drag mode)
# ---------------------------------------------------------------------------
def on_mouse_press(event):
    if mode[0] == 'drag' and event.inaxes == ax:
        is_dragging[0] = True
        drag_pos[0] = np.array([event.xdata, event.ydata])

def on_mouse_move(event):
    if mode[0] == 'drag' and is_dragging[0] and event.inaxes == ax:
        if event.xdata is not None and event.ydata is not None:
            drag_pos[0] = np.array([event.xdata, event.ydata])

def on_mouse_release(event):
    is_dragging[0] = False

fig.canvas.mpl_connect('button_press_event',   on_mouse_press)
fig.canvas.mpl_connect('motion_notify_event',  on_mouse_move)
fig.canvas.mpl_connect('button_release_event', on_mouse_release)

# ---------------------------------------------------------------------------
# Animation loop
# ---------------------------------------------------------------------------

def _ray_length_to_box_edge(pos, angle):
    """Return the distance from `pos` to the box edge along direction `angle`."""
    ux = np.cos(angle)
    uy = np.sin(angle)
    candidates = []

    visible_limit = GRID_RADIUS - EDGE_MARGIN * 0.5
    for limit, coord in ((visible_limit, 0), (-visible_limit, 0), (visible_limit, 1), (-visible_limit, 1)):
        if coord == 0:
            u = ux
            if abs(u) < 1e-12:
                continue
            d = (limit - pos[0]) / u
        else:
            u = uy
            if abs(u) < 1e-12:
                continue
            d = (limit - pos[1]) / u
        if d > 0:
            candidates.append(d)

    if not candidates:
        return 0.01
    return max(0.01, min(candidates))


def animate(frame):
    if paused[0]:
        return [charge_plot] + field_lines

    curr_idx = current_step[0]
    next_idx = (curr_idx + 1) % STEPS_HISTORY

    wall_t[0] += dt
    t   = wall_t[0]
    pos = get_charge_position(t)

    if reached_box_edge(pos):
        paused[0] = True
        btn_pause.label.set_text('Resume')

    time_history[next_idx] = t
    pos_history[next_idx]  = pos
    current_step[0]        = next_idx

    charge_plot.set_data([pos[0]], [pos[1]])

    for i, angle in enumerate(emission_angles):
        max_r = _ray_length_to_box_edge(pos, angle)
        R_values = np.linspace(0.01, max_r, NUM_POINTS_PER_LINE)
        x_pts = np.empty(NUM_POINTS_PER_LINE)
        y_pts = np.empty(NUM_POINTS_PER_LINE)
        for j, R in enumerate(R_values):
            ret_t    = t - R / C
            idx      = np.argmin(np.abs(time_history - ret_t))
            past_pos = pos_history[idx]
            x_pts[j] = past_pos[0] + R * np.cos(angle)
            y_pts[j] = past_pos[1] + R * np.sin(angle)
        field_lines[i].set_data(x_pts, y_pts)

    return [charge_plot] + field_lines

ani = FuncAnimation(fig, animate, frames=None, interval=30, blit=True)
plt.show()
