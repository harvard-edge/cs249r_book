# viz/timing_strip.py
# The timing-strip figure family: time becoming distance on one to-scale axis.
#
# One horizontal time axis carries lanes of intervals (sensing, proposal,
# permission, body), vertical event markers, and brackets; a distance row
# beneath converts the same axis into travel at a constant speed, so every
# millisecond on the strip is also v * t millimeters of floor. Chapters build a
# TimingStrip from MLSysIM values (or with one of the instance builders below)
# and draw it with plot_timing_strip. Nothing here holds a physical number: the
# builders read every duration and speed from the machine and scenario objects
# they are handed, and the caller passes any chapter-derived quantity (the
# timing allowance, the spare) from its own LEGO cell.

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .plots import COLORS, set_book_style

__all__ = [
    "VOL4_ROLES",
    "Span",
    "Event",
    "Bracket",
    "TimingStrip",
    "plot_timing_strip",
    "capture_to_onset_strip",
    "permission_tick_strip",
    "chunk_renewal_strip",
    "takeover_strip",
    "perception_age_strip",
    "decoder_budget_strip",
]

# Volume IV diagram semantic roles, mirrored from books/vol4/tex/theme-colors.tex
# (\colorlet structural/measured/proposal/consequence/authority). theme-colors.tex
# is authoritative; keep these in sync with it. "consequence" is the volume's red
# role and means hazard or physical consequence only.
VOL4_ROLES = {
    "structural": "#1A4D3E",   # ethdarkblue, aliased to emerald pine in Vol IV
    "measured": "#007A87",     # ethpetrol
    "proposal": "#B87333",     # ethbronze
    "consequence": "#A51C30",  # harvardcrimson
    "authority": "#5B4B8A",    # ethpurple
    "neutral": "#4A5568",      # softink
}


def _role_color(role):
    return VOL4_ROLES.get(role, VOL4_ROLES["neutral"])


def _tint(hex_color, amount):
    """Blend a hex color toward white; amount = 1 keeps the color, 0 gives white."""
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(1 - amount * (1 - c) for c in rgb)


@dataclass
class Span:
    """One interval on a lane. Times are in the strip's time unit.

    hatch marks an interval that is allocated but not qualified (Delta_da).
    label_pos is "in", "above", "below", "left", or "right"; "auto" puts the
    label inside when it fits and above otherwise.
    """

    label: str
    start: float
    end: float
    role: str = "structural"
    lane: int = 0
    hatch: Optional[str] = None
    label_pos: str = "auto"
    label_dx: float = 0.0
    solid: bool = False          # fill with the full role color (white label)
    outline_only: bool = False   # a bound, not an occupied interval


@dataclass
class Event:
    """A vertical marker across the lanes, labeled above the strip."""

    t: float
    label: str
    role: str = "neutral"
    level: int = 0               # stagger level for the label row above the strip
    style: str = "solid"         # "solid", "dashed", or "dotted"
    ha: str = "center"


@dataclass
class Bracket:
    """A labeled interval bracket.

    row = "time" draws it above the lanes; row = "distance" draws it in the
    distance row, where the label usually states the interval in millimeters.
    """

    start: float
    end: float
    label: str
    role: str = "neutral"
    row: str = "time"
    level: int = 0


@dataclass
class TimingStrip:
    """Everything one timing strip draws."""

    spans: Sequence[Span]
    lanes: Sequence[str]
    events: Sequence[Event] = field(default_factory=list)
    brackets: Sequence[Bracket] = field(default_factory=list)
    xlim: Optional[tuple] = None
    time_unit: str = "ms"
    speed_mm_per_unit: Optional[float] = None   # travel per time unit; None hides the distance row
    speed_label: str = ""
    distance_origin: float = 0.0                # time at which travel is counted from zero
    distance_step_mm: Optional[float] = None    # distance-row tick spacing
    distance_notes: Sequence[tuple] = field(default_factory=list)  # (t_start, t_end, text)
    offaxis_note: Optional[tuple] = None        # (lane, text, role): arrow off the right edge
    figsize: tuple = (10, 4.6)
    xlabel: Optional[str] = None


# ── drawing ─────────────────────────────────────────────────────────────────

_LANE_H = 0.56


def _text_width_data(ax, text, fontsize, renderer):
    """Width of a text string in data units of ax's x axis."""
    t = ax.text(0, 0, text, fontsize=fontsize)
    bb = t.get_window_extent(renderer=renderer)
    t.remove()
    inv = ax.transData.inverted()
    x0 = inv.transform((0, 0))[0]
    x1 = inv.transform((bb.width, 0))[0]
    return abs(x1 - x0)


def _draw_bracket(ax, x0, x1, y, label, color, up=True, fontsize=8.5):
    tick = 0.09 if up else -0.09
    ax.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick], color=color, lw=1.0, clip_on=False, zorder=6)
    ax.text(
        (x0 + x1) / 2,
        y + (0.05 if up else -0.05),
        label,
        ha="center",
        va="bottom" if up else "top",
        fontsize=fontsize,
        color=color,
        zorder=7,
        bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"),
    )


def plot_timing_strip(strip, ax=None):
    """Draw a TimingStrip. Returns (fig, ax_time, ax_distance or None).

    Lanes run top to bottom in the order given; time runs left to right. The
    distance row, when strip.speed_mm_per_unit is set, shares the time axis and
    labels it in millimeters of travel from strip.distance_origin.
    """
    set_book_style()
    import matplotlib.pyplot as plt

    n = len(strip.lanes)
    show_dist = strip.speed_mm_per_unit is not None
    n_time_levels = max([b.level for b in strip.brackets if b.row == "time"] + [-1]) + 1
    n_event_levels = max([e.level for e in strip.events] + [-1]) + 1
    head = 0.42 * (n_time_levels + n_event_levels) + 0.15

    if show_dist:
        fig, (ax, axd) = plt.subplots(
            2, 1, figsize=strip.figsize, sharex=True,
            gridspec_kw=dict(height_ratios=[n * 0.62 + head, 1.05], hspace=0.08),
        )
    else:
        fig, ax = plt.subplots(figsize=strip.figsize)
        axd = None

    x0, x1 = strip.xlim if strip.xlim else (
        min(s.start for s in strip.spans), max(s.end for s in strip.spans))
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.78, n - 0.4 + head)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # lanes: top lane has the highest y
    lane_y = {i: n - 1 - i for i in range(n)}
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(lane_y[i] - 0.5, lane_y[i] + 0.5, color="#F4F6F9", zorder=0, lw=0)
    ax.set_yticks([lane_y[i] for i in range(n)])
    ax.set_yticklabels(strip.lanes, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)

    # spans
    for s in strip.spans:
        y = lane_y[s.lane]
        c = _role_color(s.role)
        width = s.end - s.start
        if s.outline_only:
            ax.barh(y, width, left=s.start, height=_LANE_H, facecolor="none",
                    edgecolor=c, lw=1.1, ls=(0, (3, 2)), zorder=3)
        else:
            ax.barh(y, width, left=s.start, height=_LANE_H,
                    facecolor=c if s.solid else _tint(c, 0.22),
                    edgecolor=c, lw=1.0, hatch=s.hatch, zorder=3)
        fs = 8.5
        tw = _text_width_data(ax, s.label, fs, renderer)
        pos = s.label_pos
        if pos == "auto":
            pos = "in" if tw < 0.92 * width else "above"
        xm = (s.start + s.end) / 2 + s.label_dx
        tc = "white" if (s.solid and pos == "in") else c
        lane_bg = "#F4F6F9" if s.lane % 2 == 0 else "white"
        obox = dict(boxstyle="square,pad=0.08", fc=lane_bg, ec="none")
        box = dict(boxstyle="square,pad=0.1", fc=_tint(c, 0.22), ec="none") if (pos == "in" and s.hatch) else None
        if pos == "in":
            ax.text(xm, y, s.label, ha="center", va="center", fontsize=fs, color=tc, zorder=5, bbox=box)
        elif pos == "above":
            ax.text(xm, y + _LANE_H / 2 + 0.04, s.label, ha="center", va="bottom", fontsize=fs, color=c, zorder=5,
                    bbox=obox)
        elif pos == "below":
            ax.text(xm, y - _LANE_H / 2 - 0.04, s.label, ha="center", va="top", fontsize=fs, color=c, zorder=5,
                    bbox=obox)
        elif pos == "right":
            ax.text(s.end + (x1 - x0) * 0.006 + s.label_dx, y, s.label, ha="left", va="center", fontsize=fs, color=c,
                    zorder=5, bbox=obox)
        elif pos == "left":
            ax.text(s.start - (x1 - x0) * 0.006 + s.label_dx, y, s.label, ha="right", va="center", fontsize=fs, color=c,
                    zorder=5, bbox=obox)

    # events: lines through the lanes, labels stacked above them
    top = n - 0.5
    styles = {"solid": "-", "dashed": (0, (4, 3)), "dotted": (0, (1, 2))}
    for e in strip.events:
        c = _role_color(e.role)
        ax.plot([e.t, e.t], [-0.5, top + 0.12 + 0.42 * e.level], color=c, lw=0.9,
                ls=styles[e.style], zorder=2)
        ax.text(e.t, top + 0.16 + 0.42 * e.level, e.label, ha=e.ha, va="bottom",
                fontsize=8.5, color=c, zorder=6,
                bbox=dict(boxstyle="square,pad=0.06", fc="white", ec="none"))
        if axd is not None:
            axd.axvline(e.t, color=c, lw=0.9, ls=styles[e.style], zorder=2)

    # time brackets above the event labels
    base = top + 0.42 * n_event_levels + 0.2
    for b in strip.brackets:
        if b.row != "time":
            continue
        _draw_bracket(ax, b.start, b.end, base + 0.42 * b.level, b.label, _role_color(b.role))

    # arrow off the right edge (e.g. a stop that outlasts the window)
    if strip.offaxis_note:
        lane, text, role = strip.offaxis_note
        c = _role_color(role)
        y = lane_y[lane]
        ax.annotate("", xy=(x1, y), xytext=(x1 - (x1 - x0) * 0.035, y),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2), zorder=6, annotation_clip=False)
        ax.text(x1 - (x1 - x0) * 0.04, y, text, ha="right", va="center",
                fontsize=8.5, color=c, zorder=6)

    ax.tick_params(axis="x", labelbottom=not show_dist)
    if not show_dist:
        ax.set_xlabel(strip.xlabel or f"Time ({strip.time_unit})")
        return fig, ax, None

    # distance row: the same axis, relabeled as travel v * (t - origin)
    v = strip.speed_mm_per_unit
    n_dist_levels = max([b.level for b in strip.brackets if b.row == "distance"] + [0])
    # each extra distance-bracket level sits 0.3 lower; leave room for its label
    axd.set_ylim(-0.3 * n_dist_levels, 1)
    axd.set_yticks([0.5])
    axd.set_yticklabels([strip.speed_label or "travel"], fontsize=9)
    axd.tick_params(axis="y", length=0)
    axd.grid(False)
    for side in ("left", "right", "top"):
        axd.spines[side].set_visible(False)
    t_lo = max(x0, strip.distance_origin)
    axd.fill_between([t_lo, x1], 0.62, 0.9, color=_tint(VOL4_ROLES["neutral"], 0.12), lw=0, zorder=1)
    step = strip.distance_step_mm or 50.0
    d_hi = v * (x1 - strip.distance_origin)
    k = 0
    while k * step <= d_hi + 1e-9:
        t = strip.distance_origin + k * step / v
        if t >= x0 - 1e-9:
            axd.plot([t, t], [0.62, 0.9], color=VOL4_ROLES["neutral"], lw=0.7, zorder=2)
            axd.text(t, 0.93, f"{k * step:g}" + (" mm" if k == 0 else ""), ha="center", va="bottom", fontsize=8, color=VOL4_ROLES["neutral"],
                     zorder=4, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
        k += 1
    for (ta, tb, text) in strip.distance_notes:
        axd.text((ta + tb) / 2, 0.76, text, ha="center", va="center", fontsize=8,
                 color=VOL4_ROLES["neutral"], zorder=4,
                 bbox=dict(boxstyle="square,pad=0.06", fc=_tint(VOL4_ROLES["neutral"], 0.12), ec="none"))
    for b in strip.brackets:
        if b.row != "distance":
            continue
        _draw_bracket(axd, b.start, b.end, 0.44 - 0.3 * b.level, b.label, _role_color(b.role), up=False)
    axd.set_xlabel(strip.xlabel or f"Time ({strip.time_unit})")
    return fig, ax, axd


# ── instance builders (each reads its durations from MLSysIM objects) ─────────

def _ms(q):
    return float(q.to("millisecond").magnitude)


def _mps(q):
    return float(q.to("meter / second").magnitude)


def capture_to_onset_strip(scenario, speed, t_allowance=None, spare=None, delta_da=None,
                           stop_time=None, stop_distance=None, xlim=None):
    """Capture-to-brake-onset strip for the permission path (Ch12 instance).

    t = 0 is the exposure midpoint of the frame that first shows the hazard.
    Durations come from scenario (an EmbodiedSiteScenario): exposure, the
    observation age to dispatch, Delta_da (scenario.t_dispatch_to_admission
    unless delta_da is given), lease, tick, bus, and brake onset. t_allowance
    and spare are the chapter's unallocated timing allowance and the matching
    spare distance at `speed`; stop_time and stop_distance label the stop that
    runs off the right edge.
    """
    cam = scenario.machine.nav_camera
    t_exp = _ms(cam.exposure_time)
    t_read = _ms(cam.readout_time)
    t_age = _ms(scenario.perception_age)
    d_da = _ms(delta_da if delta_da is not None else scenario.t_dispatch_to_admission)
    t_lease, t_tick, t_bus, t_act = (_ms(scenario.t_lease), _ms(scenario.t_tick),
                                     _ms(scenario.t_bus), _ms(scenario.t_brake_onset))
    v = _mps(speed)

    t_disp = t_age
    t_adm = t_disp + d_da
    t_exp_lease = t_adm + t_lease
    t_onset = t_exp_lease + t_tick + t_bus + t_act

    right = xlim[1] if xlim else (t_onset + 60)
    spans = [
        Span(f"exposure {t_exp:.0f} ms", -t_exp / 2, t_exp / 2, "measured", 0, solid=True, label_pos="above"),
        Span(f"readout {t_read:.1f} ms", t_exp / 2, t_exp / 2 + t_read, "measured", 0, label_pos="right"),
        Span(f"observation age {t_age:.1f} ms", 0, t_disp, "measured", 1),
        Span(f"$\\Delta_{{da}}$ {d_da:.1f} ms, not qualified", t_disp, t_adm, "proposal", 2, hatch="///",
             label_pos="in"),
        Span(f"lease {t_lease:.0f} ms", t_adm, t_exp_lease, "structural", 3),
        Span(f"tick + bus {t_tick + t_bus:.0f} ms", t_exp_lease, t_exp_lease + t_tick + t_bus,
             "structural", 3, label_pos="above", solid=True),
        # The onset label sits inside its bar: below it, the lease-expiry and
        # brake-onset lines would cross the text.
        Span(f"onset {t_act:.0f}", t_exp_lease + t_tick + t_bus, t_onset, "consequence", 4,
             label_pos="in"),
    ]
    events = [
        Event(0, "capture (exposure midpoint)", "measured", 1, ha="left"),
        Event(t_disp, "dispatch", "proposal", 0),
        Event(t_adm, "admission", "structural", 0),
        Event(t_exp_lease, "lease expiry", "structural", 0, ha="right"),
        Event(t_onset, "brake onset", "consequence", 0, ha="left"),
    ]
    brackets = []
    dnotes = [(0, t_disp, f"{v * t_disp:.1f} mm"), (t_disp, t_adm, f"{v * d_da:.1f} mm"),
              (t_adm, t_onset, f"{v * (t_onset - t_adm):.1f} mm")]
    if t_allowance is not None:
        t_allow = _ms(t_allowance)
        latest_adm = t_disp + t_allow
        latest_onset = latest_adm + t_lease + t_tick + t_bus + t_act
        brackets.append(Bracket(t_disp, latest_adm, f"timing allowance {t_allow:.1f} ms", "consequence", "time", 0))
        events.append(Event(latest_adm, f"latest admission {latest_adm:.1f} ms", "consequence", 1, "dashed"))
        events.append(Event(latest_onset, f"latest onset {latest_onset:.1f} ms", "consequence", 1, "dashed",
                            ha="right" if latest_onset > right - 25 else "center"))
        if spare is not None:
            mm = float(spare.to("millimeter").magnitude)
            brackets.append(Bracket(t_disp, latest_adm, f"spare {mm:.1f} mm", "consequence", "distance", 0))
    if stop_time is not None and stop_distance is not None:
        spans.append(Span(f"stop {_ms(stop_time):.0f} ms, {float(stop_distance.to('millimeter').magnitude):.0f} mm",
                          t_onset, right, "consequence", 4, solid=True, label_pos="in"))
    return TimingStrip(
        spans=spans,
        lanes=["camera", "perception", "proposal", "permission", "drive"],
        events=events,
        brackets=brackets,
        xlim=xlim or (-t_exp / 2 - 6, right),
        speed_mm_per_unit=v,
        speed_label=f"travel at {v:.1f} m/s (mm)",
        distance_step_mm=50.0,
        distance_notes=dnotes,
        figsize=(10, 5.4),
        xlabel="Time from the exposure midpoint (ms)",
    )


def permission_tick_strip(scenario, speed):
    """One permission tick (Ch4 instance): release, execution, deadline, next release.

    Period is scenario.t_tick; the unloaded execution, declared WCET, and
    deadline are scenario.enforcer_unloaded / enforcer_wcet / enforcer_deadline.
    Time is in microseconds; the distance row counts travel at `speed`.
    """
    T = float(scenario.t_tick.to("microsecond").magnitude)
    c_u = float(scenario.enforcer_unloaded.to("microsecond").magnitude)
    c_w = float(scenario.enforcer_wcet.to("microsecond").magnitude)
    dl = float(scenario.enforcer_deadline.to("microsecond").magnitude)
    v = _mps(speed) / 1000.0  # mm per microsecond
    # Lane 0 is what a stress-free run measures; lane 1 is what the schedule
    # must assume. Every label sits right of its bar, clear of the lane edges.
    spans = [
        Span(f"unloaded run {c_u:.0f} µs", 0, c_u, "measured", 0, label_pos="right"),
        Span("", 0, c_u, "structural", 1),
        Span(f"declared WCET {c_w:.0f} µs, {c_w - c_u:.0f} µs to justify", c_u, c_w, "structural", 1,
             hatch="///", label_pos="right", label_dx=(dl - c_w) + 10.0),
        Span(f"slack {dl - c_w:.0f} µs", c_w, dl, "structural", 1, outline_only=True, label_pos="in"),
    ]
    events = [
        Event(0, "release", "structural", 0, ha="left"),
        Event(dl, f"deadline {dl:.0f} µs", "consequence", 0, "dashed"),
        Event(T, f"next release {T:.0f} µs", "structural", 0, ha="right"),
    ]
    brackets = [Bracket(0, T, f"period {T / 1000:.0f} ms", "structural", "time", 0),
                Bracket(0, T, f"{v * T:.1f} mm per tick", "neutral", "distance", 0)]
    return TimingStrip(
        spans=spans, lanes=["measured", "declared"], events=events, brackets=brackets,
        xlim=(-30, T + 30), time_unit="µs", speed_mm_per_unit=v,
        speed_label=f"travel at {_mps(speed):.1f} m/s (mm)", distance_step_mm=0.25,
        figsize=(9, 3.4), xlabel="Time from the tick's release (µs)",
    )


def chunk_renewal_strip(machine, evidence_horizon=None, n_renewals=4):
    """Chunk renewal schedule (Ch3 instance): what a chunk supplies vs. what its evidence covers.

    Time zero is the capture that the first chunk is computed from. Reads
    machine.chunk_step (setpoint step), machine.chunk_period (renewal period),
    machine.chunk_inference_latency, machine.chunk_horizon, and
    machine.chunk_schedule (open-loop span = age at replacement, chunk span).
    evidence_horizon (optional Quantity, measured from capture) splits the
    first chunk's unexecuted tail into the part its evidence still covers and
    the part that is supply buffer only (hatched). No distance row.

    Lanes, top to bottom: capture, inference, chunk in force (which chunk is
    feeding setpoints), first chunk (its whole predicted span), setpoints.
    """
    step = _ms(machine.chunk_step)
    period = _ms(machine.chunk_period)
    infer = _ms(machine.chunk_inference_latency)
    sched = machine.chunk_schedule
    span = _ms(sched["chunk_span"])
    age = _ms(sched["open_loop_span"])          # age of the evidence when the successor takes over
    horizon_n = machine.chunk_horizon
    t_end = span  # prediction support ends relative to capture, not availability
    spans = []
    # capture instants and inference bars
    for k in range(n_renewals + 1):
        spans.append(Span("", k * period - 0.8, k * period + 0.8, "measured", 0, solid=True))
    for k in range(n_renewals):
        t0 = k * period
        spans.append(Span(f"{infer:.0f} ms" if k == 0 else "", t0, t0 + infer, "proposal", 1, label_pos="in"))
    # which chunk is in force: chunk k drives setpoints from its availability to its successor's
    for k in range(n_renewals):
        a = k * period + infer
        spans.append(Span(f"chunk {k}", a, a + period, "proposal", 2, solid=(k % 2 == 0),
                          label_pos="in"))
    # the first chunk's whole predicted span
    replaced = period + infer                     # successor available
    if not infer < replaced <= t_end:
        raise ValueError("Chunk support must cover availability through replacement.")
    spans.append(Span("elapsed prefix", 0, infer, "neutral", 3, outline_only=True, label_pos="in"))
    spans.append(Span("executed", infer, replaced, "proposal", 3, solid=True, label_pos="in"))
    if evidence_horizon is not None:
        t_ev = _ms(evidence_horizon)
        if not replaced < t_ev < t_end:
            raise ValueError("Evidence horizon must lie between replacement and support end.")
        spans.append(Span("evidence still valid", replaced, t_ev, "proposal", 3, label_pos="in"))
        spans.append(Span("buffer, not permission", t_ev, t_end, "neutral", 3, hatch="///", label_pos="in"))
    else:
        spans.append(Span("unexecuted tail", replaced, t_end, "proposal", 3, outline_only=True, label_pos="in"))
    # setpoint grid
    t = 0.0
    while t <= t_end + 1e-9:
        spans.append(Span("", t - 0.6, t + 0.6, "structural", 4, solid=True))
        t += step
    events = [Event(infer, "first chunk available", "proposal", 0, "dotted", ha="left"),
              Event(replaced, "successor available", "proposal", 1, "dotted", ha="left")]
    if evidence_horizon is not None:
        events.append(Event(t_ev, f"evidence horizon {t_ev:.0f} ms", "consequence", 0, "dashed", ha="left"))
    events.append(Event(t_end, f"end of chunk, H = {horizon_n}", "neutral", 1, "dotted", ha="right"))
    brackets = [Bracket(0, step, f"step {step:.0f} ms", "structural", "time", 0),
                Bracket(0, period, f"renewal {period:.0f} ms", "proposal", "time", 1),
                Bracket(0, age, f"age at replacement {age:.0f} ms", "measured", "time", 2)]
    return TimingStrip(
        spans=spans,
        lanes=["capture", "inference", "chunk in force", "first chunk", "setpoints"],
        events=events, brackets=brackets, xlim=(-8, t_end + 8), figsize=(10, 4.6),
        xlabel="Time from the first chunk's capture (ms)",
    )


def takeover_strip(scenario, speed=None):
    """In-loop remote takeover (Ch14 instance): video, human response, handshake.

    Reads the scenario's slow-bound Model Human Processor terms, the console
    glass-to-glass latency, and the four-phase handshake; speed defaults to
    scenario.v_takeover_in_loop.
    """
    s = scenario
    v = _mps(speed if speed is not None else s.v_takeover_in_loop)
    seq = [
        ("video", _ms(s.t_video_glass_to_glass), "measured", 0),
        ("perceive", _ms(s.t_human_perceptual_slow), "authority", 1),
        ("decide", _ms(s.t_human_cognitive_slow), "authority", 1),
        ("act", _ms(s.t_human_motor_slow), "authority", 1),
        ("handshake", _ms(s.t_handshake), "structural", 2),
    ]
    spans, t = [], 0.0
    for name, d, role, lane in seq:
        spans.append(Span(f"{name} {d:.0f}", t, t + d, role, lane))
        t += d
    total = t
    events = [Event(0, "request", "neutral", 0), Event(total, "commit notification", "structural", 0)]
    brackets = [Bracket(0, total, f"in-loop takeover {total:.0f} ms", "authority", "time", 0),
                Bracket(0, total, f"{v * total:.0f} mm", "consequence", "distance", 0)]
    return TimingStrip(
        spans=spans, lanes=["console", "operator", "permission"], events=events,
        brackets=brackets, xlim=(-15, total + 25), speed_mm_per_unit=v,
        speed_label=f"travel at {v:.2f} m/s (mm)", distance_step_mm=100.0, figsize=(10, 3.6),
    )


def perception_age_strip(scenario, speed):
    """Capture-to-brake strip for the perception chapter (Ch8 instance).

    Same clock as capture_to_onset_strip: t = 0 is the exposure midpoint. The
    camera lane carries the second half of the exposure and the rolling
    readout; the host lane carries link + DMA + ISP, the vision backbone, and
    the hand-off, ending at dispatch (scenario.perception_age). The response
    lane then adds lease, tick + bus, and brake onset with Delta_da taken as
    zero: the perception chapter does not bound Delta_da, so the strip marks
    where it enters instead of drawing it at an invented width. Every
    duration is read from scenario (an EmbodiedSiteScenario); travel is
    counted at `speed`.
    """
    cam = scenario.machine.nav_camera
    t_exp = _ms(cam.exposure_time)
    t_read = _ms(cam.readout_time)
    t_link = _ms(scenario.t_transport) + _ms(scenario.t_dma) + _ms(scenario.t_isp)
    t_bb, t_ipc = _ms(scenario.t_backbone), _ms(scenario.t_ipc)
    t_age = _ms(scenario.perception_age)
    t_lease, t_tb, t_act = (_ms(scenario.t_lease), _ms(scenario.t_tick) + _ms(scenario.t_bus),
                            _ms(scenario.t_brake_onset))
    v = _mps(speed)

    a = t_exp / 2
    b = a + t_read
    c = b + t_link
    d = c + t_bb
    t_disp = d + t_ipc
    t_exp_lease = t_disp + t_lease
    t_onset = t_exp_lease + t_tb + t_act
    spans = [
        Span(f"exposure {t_exp:.0f} ms", -a, a, "measured", 0, solid=True, label_pos="above"),
        Span(f"readout {t_read:.1f} ms", a, b, "measured", 0, label_pos="right"),
        Span(f"link, DMA, ISP {t_link:.1f} ms", b, c, "measured", 1, label_pos="below",
             label_dx=-6.0),
        Span(f"backbone {t_bb:.0f} ms", c, d, "measured", 1, label_pos="above"),
        Span(f"IPC {t_ipc:.1f} ms", d, t_disp, "measured", 1, label_pos="below", label_dx=5.0),
        Span(f"lease {t_lease:.0f} ms", t_disp, t_exp_lease, "structural", 2, label_pos="in"),
        Span(f"tick + bus {t_tb:.0f} ms", t_exp_lease, t_exp_lease + t_tb, "structural", 2, solid=True,
             label_pos="below"),
        Span(f"onset {t_act:.0f} ms", t_exp_lease + t_tb, t_onset, "consequence", 2, label_pos="in"),
    ]
    events = [
        Event(0, "capture", "measured", 1, ha="left"),
        Event(t_disp, f"dispatch, $t_{{\\mathrm{{age}}}}$ = {t_age:.1f} ms", "measured", 0),
        Event(t_disp, "$+\\,\\Delta_{\\mathrm{da}}$ enters here, not bounded", "proposal", 1, "dashed", ha="left"),
        Event(t_onset, "brake onset", "consequence", 0, ha="right"),
    ]
    brackets = [Bracket(0, t_onset, f"$\\tau_{{\\mathrm{{delay}}}}$ = {t_onset:.1f} ms + $\\Delta_{{\\mathrm{{da}}}}$",
                        "neutral", "time", 0)]
    dnotes = [(0, t_disp, f"{v * t_disp:.1f} mm"), (t_disp, t_onset, f"{v * (t_onset - t_disp):.1f} mm")]
    return TimingStrip(
        spans=spans,
        lanes=["camera", "host", "response"],
        events=events,
        brackets=brackets,
        xlim=(-a - 3, t_onset + 4),
        speed_mm_per_unit=v,
        speed_label=f"travel at {v:.1f} m/s (mm)",
        distance_step_mm=50.0,
        distance_notes=dnotes,
        figsize=(7.0, 3.5),
        xlabel="Time from the exposure midpoint (ms)",
    )


def decoder_budget_strip(budget, t_lease, n_renewals=2):
    """Decoder renewal budget (Ch6 instance): ACT vs. diffusion on the renewal schedule.

    Reads budget.t_step (setpoint step), budget.t_renew (renewal period),
    budget.t_act and budget.t_ddim (decoder inference), budget.t_jitter (the
    renewal-jitter stretch), and budget.ddim_stretch_over (how far the
    stretched diffusion run ends past the renewal); t_lease is the lease from
    admission. Slack is drawn as an outlined bound, the stretched overrun in
    the consequence role, the lease in the structural role. No distance row.

    Lanes, top to bottom: setpoints, ACT decoder, diffusion head, diffusion
    under the jitter stretch, lease from admission.
    """
    step, renew = _ms(budget.t_step), _ms(budget.t_renew)
    t_act, t_ddim, t_jit = _ms(budget.t_act), _ms(budget.t_ddim), _ms(budget.t_jitter)
    over = _ms(budget.ddim_stretch_over)
    lease = _ms(t_lease)
    t_end = n_renewals * renew
    spans = []
    t = 0.0
    while t <= t_end + 1e-9:
        spans.append(Span("", t - 0.35, t + 0.35, "structural", 0, solid=True))
        t += step
    for k in range(n_renewals):
        r = k * renew
        spans.append(Span(f"{t_act:.0f} ms", r, r + t_act, "proposal", 1, solid=True, label_pos="in"))
        spans.append(Span(f"{renew - t_act:.0f} ms slack", r + t_act, r + renew, "proposal", 1,
                          outline_only=True, label_pos="in"))
        spans.append(Span(f"{t_ddim:.0f} ms, {renew - t_ddim:.0f} ms slack" if k == 0 else f"{t_ddim:.0f} ms",
                          r, r + t_ddim, "proposal", 2, solid=True, label_pos="in"))
    spans.append(Span(f"{t_ddim:.0f} ms", 0.0, t_ddim, "proposal", 3, solid=True, label_pos="in"))
    spans.append(Span("", t_ddim, renew, "proposal", 3, hatch="////"))
    spans.append(Span(f"ends {over:.0f} ms past renewal", renew, t_ddim + t_jit, "consequence", 3,
                      solid=True, label_pos="right"))
    spans.append(Span(f"lease {lease:.0f} ms, expires unless renewed", 0.0, lease, "structural", 4,
                      solid=True, label_pos="in"))
    events = [Event(k * renew, "renewal", "neutral", 0, "dashed",
                    ha="left" if k == 0 else ("right" if k == n_renewals else "center"))
              for k in range(n_renewals + 1)]
    brackets = [Bracket(0, step, f"setpoint {step:.0f} ms", "structural", "time", 0),
                Bracket(0, renew, f"renewal {renew:.0f} ms", "proposal", "time", 1)]
    return TimingStrip(
        spans=spans,
        lanes=["setpoints", "ACT decoder", "diffusion head", f"diffusion, {t_jit:.0f} ms stretch",
               "lease from admission"],
        events=events, brackets=brackets, xlim=(-3, t_end + 3), figsize=(9, 3.9),
        xlabel="Time from admission (ms)",
    )
