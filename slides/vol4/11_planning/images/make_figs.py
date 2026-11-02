#!/usr/bin/env python3
"""Figures for Lecture 11 (Trajectory Planning), Volume IV.

Every value is sourced. "TP" is the rendered chapter, Trajectory Planning.
Every derived value comes from calcs.py (imported as K); values labeled
"illustrative" on a figure are the chapter's illustrative inputs.

  - hook: TP Napkin Math 11.1 (3,000 handoffs, p99, mission target 1e-3).
  - obligations: the arm stall of TP Napkin Math 11.4 (1 m/s, lease 60 ms,
    onset at 82 ms, at rest 357 ms); the seam mismatch is schematic.
  - jac_ellipse, dls: TP Sec. 11.2.1 (Panda planar links 0.316 and 0.384 m,
    damping 50 mm illustrative); computed in calcs.py.
  - record: TP Sec. 11.2.2 and 11.8 (fields, deadlines of the alternate
    schedule, stop of NM 11.4).
  - repl: TP equation 3 (51.6 + 63.3 = 114.9 ms; 40 ms inference, 18 ms
    validation); 50 ms renewal and 60 ms lease (TP Sec. 11.2.2).
  - chunks: published chunk configurations: Diffusion Policy Push-T 16/6
    (Chi et al. 2025, via TP Sec. 11.3.2); pi0 50/25 at 50 Hz (Black et al.
    2024, via TP table 4); the running machine 16 setpoints, renewed every
    50 ms (2 setpoints executed).
  - stale: TP Sec. 11.3.3 (0.50 rad/s, 80 ms, illustrative).
  - overshoot: TP equation 9 and 10, schematic.
  - seam: TP Napkin Math 11.2 (reflected inertia from the MLSysIM registry).
  - phase: TP Napkin Math 11.3.
  - ruckig_arm, base: Ruckig 0.19.4 output (ruckig_stops.csv, written by
    ruckig_run.py) with the chapter's bounds; constant-deceleration and
    polynomial curves from TP equations 16 and 17.
  - runway: TP Napkin Math 11.4 and Sec. 11.5.2 (open mug).
  - timeline: TP figure 2 values (both schedules).
  - gen_timing: Ruckig 19.8 us mean, 123 us worst (Berscheid and Kroger
    2021, via TP table 4); 1 ms tick, 18 ms validation, 40 ms chunk pass
    (illustrative), 114.9 ms replacement (TP Sec. 11.1, 11.3.1); pi0 0.5 s
    between inferences (TP table 4).
  - base_view: TP figure 6 (differential drive), schematic.

Run from this folder: python3 make_figs.py
"""
import contextlib
import csv
import io
import math
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon
plt = sp.plt
C = sp.C
out = Path(__file__).parent
V4 = {'proposal': '#2563EB', 'proposal_l': '#EFF6FF', 'nervous': '#059669', 'nervous_l': '#DCFCE7', 'veto': '#DC2626', 'veto_l': '#FEE2E2', 'lag': '#D97706', 'lag_l': '#FEF3C7', 'plant': '#334155', 'plant_l': '#F8FAFC', 'navy': '#1F407A', 'gray': '#666666', 'gray_l': '#DDDDDD', 'text': '#1A1A2E'}
FS = 11
FS_S = 10

def cm(w, h):
    return (w / 2.54, h / 2.54)

def save_slot(fig, name):
    fig.savefig(out / name, format='svg', facecolor='white')
    plt.close(fig)

def hide_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

def read_ruckig():
    data = {}
    with (out / 'ruckig_stops.csv').open() as f:
        for r in csv.DictReader(f):
            d = data.setdefault(r['case'], {'t': [], 'p': [], 'v': [], 'a': []})
            d['t'].append(float(r['t_s']))
            d['p'].append(float(r['position_m']))
            d['v'].append(float(r['velocity_mps']))
            d['a'].append(float(r['acceleration_mps2']))
    return {k: {kk: np.array(vv) for kk, vv in d.items()} for k, d in data.items()}
RK = read_ruckig()

def hook(step):
    rows = [('p99 budget,\nper handoff', 0.01, V4['gray'], '1 in 100'), ('Mission target,\nwhole run', 0.001, V4['lag'], '1 in 1,000'), ('Design target,\nper handoff', 3.333333333e-07, V4['veto'], '1 in 3,000,000')]
    fig, ax = plt.subplots(figsize=cm(14.2, 4.6))
    fig.subplots_adjust(left=0.24, right=0.97, top=0.95, bottom=0.27)
    for i, (lab, p, col, txt) in enumerate(rows):
        y = 2 - i
        shown = i < step
        n = -math.log10(p)
        ax.barh(y, n, color=col, height=0.6, alpha=1 if shown else 0)
        ax.text(-0.15, y, lab, ha='right', va='center', fontsize=FS, fontweight='bold', color=V4['text'])
        ax.text(n + 0.12, y, txt, va='center', fontsize=FS, fontweight='bold', color=col, alpha=1 if shown else 0)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-0.65, 2.45)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('How rare a late handoff must be (nines: $-\\log_{10}$ of the probability)', fontsize=FS_S)
    ax.tick_params(labelsize=FS_S)
    if step >= 3:
        ax.annotate('', xy=(-math.log10(3.333333333e-07), -0.42), xytext=(2, -0.42), arrowprops=dict(arrowstyle='<->', color=V4['veto'], lw=1.4))
        ax.text(-math.log10(3.333333333e-07) + 0.15, -0.42, '4.5 orders deeper than p99', fontsize=FS_S, color=V4['veto'], ha='left', va='center')
    save_slot(fig, f'hook_step{step}.svg')
for s in (1, 2, 3):
    hook(s)

def obligations(step):
    """Tool speed against time: the three things a plan carries."""
    fig, ax = plt.subplots(figsize=cm(9.4, 5.6))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.93, bottom=0.2)
    t_pred = np.linspace(0, 400.0, 200)
    ax.plot(t_pred, np.ones_like(t_pred), color=V4['proposal'], lw=1.6, ls='--')
    ax.text(395, 0.95, 'predicted to 400 ms', color=V4['proposal'], fontsize=FS_S, ha='right', va='top')
    ax.plot([0, 50.0], [1, 1], color=V4['nervous'], lw=3.5)
    ax.plot(np.arange(0, 50.0 + 1, 25.0), [1, 1, 1], 'o', color=V4['nervous'], ms=6)
    ax.text(5, 0.86, '1. timed prefix', color=V4['nervous'], fontsize=FS, fontweight='bold', va='top')
    if step >= 2:
        tb = np.linspace(0, 1, 60)
        vb = 1 + 0.2 * (10 * tb ** 3 - 15 * tb ** 4 + 6 * tb ** 5)
        ab = 0.3 if step >= 3 else 1.0
        ax.plot(50 + 50 * tb, vb, color=V4['nervous'], lw=3.5, alpha=ab)
        ax.plot([100, 160], [1.2, 1.2], color=V4['proposal'], lw=1.6, ls='--', alpha=ab)
        ax.plot([50, 50], [1.0, 1.2], color=V4['veto'], lw=1.4, ls=':', alpha=ab)
        ax.text(105, 1.27, '2. smooth join', color=V4['nervous'], fontsize=FS, fontweight='bold', va='bottom', alpha=ab)
    if step >= 3:
        d = RK['arm']
        t0 = 82.0
        ax.plot(t0 + 1000 * d['t'], d['v'], color=V4['veto'], lw=3)
        ax.plot([60.0, t0], [1, 1], color=V4['lag'], lw=3)
        ax.text(88, 0.22, '3. stop without\nthe proposer', color=V4['veto'], fontsize=FS, fontweight='bold', ha='left', va='center')
    ax.set_xlim(0, 410)
    ax.set_ylim(0, 1.45)
    ax.set_xlabel('Time after admission (ms)', fontsize=FS)
    ax.set_ylabel('Tool speed (m/s)', fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, f'obligations_step{step}.svg')
for s in (1, 2, 3):
    obligations(s)

def jac_ellipse():
    fig, axes = plt.subplots(1, 2, figsize=cm(9.6, 5.4))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.05)
    for ax, (q2d, q1d) in zip(axes, ((90, 15), (10, 30))):
        q1, q2 = (math.radians(q1d), math.radians(q2d))
        p0 = np.array([0.0, 0.0])
        p1 = p0 + 0.316 * np.array([math.cos(q1), math.sin(q1)])
        p2 = p1 + 0.384 * np.array([math.cos(q1 + q2), math.sin(q1 + q2)])
        th = np.linspace(0, math.pi / 2 + 0.6, 60)
        R = 0.316 + 0.384
        ax.plot(R * np.cos(th - 0.3), R * np.sin(th - 0.3), color=V4['gray'], lw=1, ls='--')
        ax.plot(*zip(p0, p1, p2), color=V4['plant'], lw=5, solid_capstyle='round')
        ax.plot(*p0, 's', color=V4['plant'], ms=10)
        ax.plot(*p1, 'o', color='white', mec=V4['plant'], mew=2.5, ms=9)
        J = np.array([[-0.316 * math.sin(q1) - 0.384 * math.sin(q1 + q2), -0.384 * math.sin(q1 + q2)], [0.316 * math.cos(q1) + 0.384 * math.cos(q1 + q2), 0.384 * math.cos(q1 + q2)]])
        U, S, _ = np.linalg.svd(J)
        sc = 0.22
        ang = math.degrees(math.atan2(U[1, 0], U[0, 0]))
        ax.add_patch(Ellipse(p2, 2 * sc * S[0], 2 * sc * S[1] + 0.004, angle=ang, fc=V4['proposal_l'], ec=V4['proposal'], lw=1.6))
        minor = U[:, 1] * 0.13
        ax.annotate('', xy=p2 + minor, xytext=p2 - minor, arrowprops=dict(arrowstyle='<->', color=V4['veto'], lw=1.8))
        ax.plot(*p2, 'o', color=V4['proposal'], ms=6)
        key = q2d
        ax.text(0.98, 0.17, f'Elbow {q2d}°', transform=ax.transAxes, fontsize=FS, fontweight='bold', va='bottom', ha='right', color=V4['text'], bbox=dict(fc='white', ec='none', pad=1))
        ax.text(0.98, 0.03, f"{ {15: '4.9', 30: '37.8'}[q1d]} rad/s", transform=ax.transAxes, fontsize=FS + 2, fontweight='bold', va='bottom', ha='right', color=V4['veto'], bbox=dict(fc='white', ec='none', pad=1))
        ax.set_aspect('equal')
        ax.set_xlim(-0.12, 0.85)
        ax.set_ylim(-0.3, 0.72)
        hide_axis(ax)
    save_slot(fig, 'jac_ellipse.svg')
jac_ellipse()

def dls():
    fig, ax = plt.subplots(figsize=cm(14.2, 5.0))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.24)
    el = np.linspace(90, 3, 300)
    ex, dm = ([], [])
    for e in el:
        smin = {90.0: 0.2042170293, 89.7090301: 0.2038112822, 89.4180602: 0.2034024181, 89.1270903: 0.2029904636, 88.8361204: 0.2025754448, 88.5451505: 0.2021573876, 88.2541806: 0.2017363175, 87.9632107: 0.2013122598, 87.6722408: 0.2008852391, 87.3812709: 0.2004552802, 87.090301: 0.2000224071, 86.7993311: 0.199586644, 86.5083612: 0.1991480143, 86.2173913: 0.1987065415, 85.9264214: 0.1982622486, 85.63545151: 0.1978151583, 85.34448161: 0.1973652931, 85.05351171: 0.1969126752, 84.76254181: 0.1964573266, 84.47157191: 0.1959992689, 84.18060201: 0.1955385236, 83.88963211: 0.1950751117, 83.59866221: 0.1946090542, 83.30769231: 0.1941403717, 83.01672241: 0.1936690847, 82.72575251: 0.1931952132, 82.43478261: 0.1927187772, 82.14381271: 0.1922397964, 81.85284281: 0.1917582903, 81.56187291: 0.1912742781, 81.27090301: 0.1907877788, 80.97993311: 0.1902988112, 80.68896321: 0.1898073939, 80.39799331: 0.1893135454, 80.10702341: 0.1888172836, 79.81605351: 0.1883186268, 79.52508361: 0.1878175925, 79.23411371: 0.1873141984, 78.94314381: 0.1868084619, 78.65217391: 0.1863004001, 78.36120401: 0.1857900301, 78.07023411: 0.1852773688, 77.77926421: 0.1847624327, 77.48829431: 0.1842452383, 77.19732441: 0.1837258019, 76.90635452: 0.1832041397, 76.61538462: 0.1826802677, 76.32441472: 0.1821542015, 76.03344482: 0.1816259569, 75.74247492: 0.1810955494, 75.45150502: 0.1805629942, 75.16053512: 0.1800283066, 74.86956522: 0.1794915015, 74.57859532: 0.1789525939, 74.28762542: 0.1784115984, 73.99665552: 0.1778685296, 73.70568562: 0.177323402, 73.41471572: 0.1767762299, 73.12374582: 0.1762270274, 72.83277592: 0.1756758087, 72.54180602: 0.1751225875, 72.25083612: 0.1745673777, 71.95986622: 0.1740101929, 71.66889632: 0.1734510466, 71.37792642: 0.1728899523, 71.08695652: 0.1723269232, 70.79598662: 0.1717619724, 70.50501672: 0.1711951132, 70.21404682: 0.1706263582, 69.92307692: 0.1700557205, 69.63210702: 0.1694832127, 69.34113712: 0.1689088473, 69.05016722: 0.168332637, 68.75919732: 0.167754594, 68.46822742: 0.1671747307, 68.17725753: 0.1665930593, 67.88628763: 0.1660095918, 67.59531773: 0.1654243403, 67.30434783: 0.1648373165, 67.01337793: 0.1642485324, 66.72240803: 0.1636579996, 66.43143813: 0.1630657298, 66.14046823: 0.1624717344, 65.84949833: 0.161876025, 65.55852843: 0.1612786127, 65.26755853: 0.160679509, 64.97658863: 0.160078725, 64.68561873: 0.1594762718, 64.39464883: 0.1588721604, 64.10367893: 0.1582664017, 63.81270903: 0.1576590066, 63.52173913: 0.1570499859, 63.23076923: 0.1564393503, 62.93979933: 0.1558271104, 62.64882943: 0.1552132768, 62.35785953: 0.1545978599, 62.06688963: 0.1539808702, 61.77591973: 0.1533623181, 61.48494983: 0.1527422137, 61.19397993: 0.1521205673, 60.90301003: 0.151497389, 60.61204013: 0.150872689, 60.32107023: 0.1502464772, 60.03010033: 0.1496187636, 59.73913043: 0.148989558, 59.44816054: 0.1483588704, 59.15719064: 0.1477267104, 58.86622074: 0.1470930878, 58.57525084: 0.1464580122, 58.28428094: 0.1458214933, 57.99331104: 0.1451835405, 57.70234114: 0.1445441633, 57.41137124: 0.1439033712, 57.12040134: 0.1432611736, 56.82943144: 0.1426175797, 56.53846154: 0.1419725988, 56.24749164: 0.1413262403, 55.95652174: 0.1406785131, 55.66555184: 0.1400294265, 55.37458194: 0.1393789896, 55.08361204: 0.1387272112, 54.79264214: 0.1380741006, 54.50167224: 0.1374196665, 54.21070234: 0.1367639178, 53.91973244: 0.1361068634, 53.62876254: 0.1354485122, 53.33779264: 0.1347888728, 53.04682274: 0.134127954, 52.75585284: 0.1334657644, 52.46488294: 0.1328023128, 52.17391304: 0.1321376075, 51.88294314: 0.1314716574, 51.59197324: 0.1308044708, 51.30100334: 0.1301360561, 51.01003344: 0.129466422, 50.71906355: 0.1287955767, 50.42809365: 0.1281235287, 50.13712375: 0.1274502862, 49.84615385: 0.1267758576, 49.55518395: 0.126100251, 49.26421405: 0.1254234749, 48.97324415: 0.1247455372, 48.68227425: 0.1240664463, 48.39130435: 0.1233862101, 48.10033445: 0.1227048369, 47.80936455: 0.1220223347, 47.51839465: 0.1213387114, 47.22742475: 0.1206539752, 46.93645485: 0.119968134, 46.64548495: 0.1192811957, 46.35451505: 0.1185931683, 46.06354515: 0.1179040595, 45.77257525: 0.1172138774, 45.48160535: 0.1165226297, 45.19063545: 0.1158303242, 44.89966555: 0.1151369687, 44.60869565: 0.114442571, 44.31772575: 0.1137471387, 44.02675585: 0.1130506796, 43.73578595: 0.1123532013, 43.44481605: 0.1116547116, 43.15384615: 0.110955218, 42.86287625: 0.1102547281, 42.57190635: 0.1095532495, 42.28093645: 0.1088507897, 41.98996656: 0.1081473564, 41.69899666: 0.1074429569, 41.40802676: 0.1067375989, 41.11705686: 0.1060312897, 40.82608696: 0.1053240368, 40.53511706: 0.1046158476, 40.24414716: 0.1039067296, 39.95317726: 0.1031966901, 39.66220736: 0.1024857365, 39.37123746: 0.1017738761, 39.08026756: 0.1010611162, 38.78929766: 0.1003474643, 38.49832776: 0.09963292744, 38.20735786: 0.09891751303, 37.91638796: 0.0982012283, 37.62541806: 0.09748408048, 37.33444816: 0.09676607677, 37.04347826: 0.0960472244, 36.75250836: 0.09532753053, 36.46153846: 0.09460700233, 36.17056856: 0.09388564696, 35.87959866: 0.09316347155, 35.58862876: 0.09244048322, 35.29765886: 0.09171668906, 35.00668896: 0.09099209618, 34.71571906: 0.09026671163, 34.42474916: 0.08954054248, 34.13377926: 0.08881359578, 33.84280936: 0.08808587854, 33.55183946: 0.08735739779, 33.26086957: 0.08662816053, 32.96989967: 0.08589817375, 32.67892977: 0.08516744441, 32.38795987: 0.08443597949, 32.09698997: 0.08370378592, 31.80602007: 0.08297087065, 31.51505017: 0.0822372406, 31.22408027: 0.08150290268, 30.93311037: 0.08076786379, 30.64214047: 0.08003213081, 30.35117057: 0.07929571062, 30.06020067: 0.07855861009, 29.76923077: 0.07782083606, 29.47826087: 0.07708239538, 29.18729097: 0.07634329488, 28.89632107: 0.07560354139, 28.60535117: 0.07486314169, 28.31438127: 0.07412210261, 28.02341137: 0.07338043092, 27.73244147: 0.07263813341, 27.44147157: 0.07189521684, 27.15050167: 0.07115168798, 26.85953177: 0.07040755356, 26.56856187: 0.06966282035, 26.27759197: 0.06891749505, 25.98662207: 0.06817158441, 25.69565217: 0.06742509512, 25.40468227: 0.0666780339, 25.11371237: 0.06593040744, 24.82274247: 0.06518222243, 24.53177258: 0.06443348555, 24.24080268: 0.06368420347, 23.94983278: 0.06293438286, 23.65886288: 0.06218403036, 23.36789298: 0.06143315264, 23.07692308: 0.06068175632, 22.78595318: 0.05992984804, 22.49498328: 0.05917743443, 22.20401338: 0.05842452211, 21.91304348: 0.05767111769, 21.62207358: 0.05691722777, 21.33110368: 0.05616285895, 21.04013378: 0.05540801783, 20.74916388: 0.05465271098, 20.45819398: 0.053896945, 20.16722408: 0.05314072644, 19.87625418: 0.05238406188, 19.58528428: 0.05162695787, 19.29431438: 0.05086942098, 19.00334448: 0.05011145775, 18.71237458: 0.04935307472, 18.42140468: 0.04859427843, 18.13043478: 0.04783507542, 17.83946488: 0.0470754722, 17.54849498: 0.04631547531, 17.25752508: 0.04555509126, 16.96655518: 0.04479432656, 16.67558528: 0.04403318772, 16.38461538: 0.04327168123, 16.09364548: 0.04250981361, 15.80267559: 0.04174759133, 15.51170569: 0.04098502089, 15.22073579: 0.04022210878, 14.92976589: 0.03945886146, 14.63879599: 0.03869528542, 14.34782609: 0.03793138713, 14.05685619: 0.03716717306, 13.76588629: 0.03640264967, 13.47491639: 0.03563782342, 13.18394649: 0.03487270076, 12.89297659: 0.03410728815, 12.60200669: 0.03334159204, 12.31103679: 0.03257561887, 12.02006689: 0.03180937508, 11.72909699: 0.03104286712, 11.43812709: 0.03027610142, 11.14715719: 0.02950908441, 10.85618729: 0.02874182252, 10.56521739: 0.02797432218, 10.27424749: 0.02720658981, 9.983277592: 0.02643863184, 9.692307692: 0.02567045467, 9.401337793: 0.02490206474, 9.110367893: 0.02413346845, 8.819397993: 0.02336467221, 8.528428094: 0.02259568243, 8.237458194: 0.02182650553, 7.946488294: 0.02105714789, 7.655518395: 0.02028761593, 7.364548495: 0.01951791604, 7.073578595: 0.01874805463, 6.782608696: 0.01797803808, 6.491638796: 0.0172078728, 6.200668896: 0.01643756518, 5.909698997: 0.01566712161, 5.618729097: 0.01489654847, 5.327759197: 0.01412585216, 5.036789298: 0.01335503907, 4.745819398: 0.01258411557, 4.454849498: 0.01181308806, 4.163879599: 0.01104196291, 3.872909699: 0.01027074652, 3.581939799: 0.009499445249, 3.2909699: 0.008728065492, 3.0: 0.007956613623}[float(f'{e:.10g}')]
        ex.append(1 / smin)
        dm.append({90.0: 4.619814502, 89.7090301: 4.627968457, 89.4180602: 4.636212104, 89.1270903: 4.644545651, 88.8361204: 4.652969313, 88.5451505: 4.661483318, 88.2541806: 4.670087905, 87.9632107: 4.678783321, 87.6722408: 4.687569827, 87.3812709: 4.696447692, 87.090301: 4.705417195, 86.7993311: 4.714478628, 86.5083612: 4.72363229, 86.2173913: 4.732878494, 85.9264214: 4.742217559, 85.63545151: 4.751649817, 85.34448161: 4.761175611, 85.05351171: 4.77079529, 84.76254181: 4.780509218, 84.47157191: 4.790317767, 84.18060201: 4.800221317, 83.88963211: 4.810220261, 83.59866221: 4.820315001, 83.30769231: 4.830505949, 83.01672241: 4.840793527, 82.72575251: 4.851178166, 82.43478261: 4.86166031, 82.14381271: 4.872240409, 81.85284281: 4.882918925, 81.56187291: 4.893696331, 81.27090301: 4.904573107, 80.97993311: 4.915549746, 80.68896321: 4.926626748, 80.39799331: 4.937804626, 80.10702341: 4.949083899, 79.81605351: 4.960465101, 79.52508361: 4.971948771, 79.23411371: 4.983535461, 78.94314381: 4.995225731, 78.65217391: 5.007020152, 78.36120401: 5.018919306, 78.07023411: 5.030923782, 77.77926421: 5.043034182, 77.48829431: 5.055251115, 77.19732441: 5.067575202, 76.90635452: 5.080007074, 76.61538462: 5.09254737, 76.32441472: 5.105196742, 76.03344482: 5.117955848, 75.74247492: 5.13082536, 75.45150502: 5.143805958, 75.16053512: 5.15689833, 74.86956522: 5.170103179, 74.57859532: 5.183421213, 74.28762542: 5.196853152, 73.99665552: 5.210399728, 73.70568562: 5.224061679, 73.41471572: 5.237839756, 73.12374582: 5.251734718, 72.83277592: 5.265747337, 72.54180602: 5.279878391, 72.25083612: 5.294128671, 71.95986622: 5.308498976, 71.66889632: 5.322990118, 71.37792642: 5.337602915, 71.08695652: 5.352338199, 70.79598662: 5.367196807, 70.50501672: 5.382179591, 70.21404682: 5.397287411, 69.92307692: 5.412521135, 69.63210702: 5.427881643, 69.34113712: 5.443369826, 69.05016722: 5.458986581, 68.75919732: 5.474732818, 68.46822742: 5.490609456, 68.17725753: 5.506617424, 67.88628763: 5.522757658, 67.59531773: 5.539031108, 67.30434783: 5.55543873, 67.01337793: 5.571981492, 66.72240803: 5.588660368, 66.43143813: 5.605476346, 66.14046823: 5.62243042, 65.84949833: 5.639523594, 65.55852843: 5.65675688, 65.26755853: 5.674131302, 64.97658863: 5.69164789, 64.68561873: 5.709307684, 64.39464883: 5.727111733, 64.10367893: 5.745061093, 63.81270903: 5.763156829, 63.52173913: 5.781400016, 63.23076923: 5.799791735, 62.93979933: 5.818333075, 62.64882943: 5.837025133, 62.35785953: 5.855869014, 62.06688963: 5.87486583, 61.77591973: 5.8940167, 61.48494983: 5.913322748, 61.19397993: 5.932785108, 60.90301003: 5.952404919, 60.61204013: 5.972183323, 60.32107023: 5.992121471, 60.03010033: 6.01222052, 59.73913043: 6.032481628, 59.44816054: 6.052905961, 59.15719064: 6.073494688, 58.86622074: 6.094248983, 58.57525084: 6.115170021, 58.28428094: 6.136258981, 57.99331104: 6.157517046, 57.70234114: 6.178945399, 57.41137124: 6.200545225, 57.12040134: 6.22231771, 56.82943144: 6.24426404, 56.53846154: 6.2663854, 56.24749164: 6.288682974, 55.95652174: 6.311157947, 55.66555184: 6.333811496, 55.37458194: 6.3566448, 55.08361204: 6.37965903, 54.79264214: 6.402855355, 54.50167224: 6.426234936, 54.21070234: 6.449798927, 53.91973244: 6.473548476, 53.62876254: 6.49748472, 53.33779264: 6.521608788, 53.04682274: 6.545921795, 52.75585284: 6.570424845, 52.46488294: 6.595119029, 52.17391304: 6.620005422, 51.88294314: 6.645085083, 51.59197324: 6.670359051, 51.30100334: 6.695828348, 51.01003344: 6.721493974, 50.71906355: 6.747356903, 50.42809365: 6.77341809, 50.13712375: 6.799678457, 49.84615385: 6.826138901, 49.55518395: 6.852800287, 49.26421405: 6.879663448, 48.97324415: 6.906729179, 48.68227425: 6.933998241, 48.39130435: 6.961471351, 48.10033445: 6.989149184, 47.80936455: 7.017032371, 47.51839465: 7.045121491, 47.22742475: 7.073417074, 46.93645485: 7.101919593, 46.64548495: 7.130629465, 46.35451505: 7.159547041, 46.06354515: 7.188672609, 45.77257525: 7.218006388, 45.48160535: 7.24754852, 45.19063545: 7.277299073, 44.89966555: 7.30725803, 44.60869565: 7.337425287, 44.31772575: 7.367800648, 44.02675585: 7.39838382, 43.73578595: 7.429174409, 43.44481605: 7.460171909, 43.15384615: 7.491375703, 42.86287625: 7.522785052, 42.57190635: 7.554399091, 42.28093645: 7.586216819, 41.98996656: 7.618237099, 41.69899666: 7.65045864, 41.40802676: 7.682879998, 41.11705686: 7.715499566, 40.82608696: 7.748315561, 40.53511706: 7.781326021, 40.24414716: 7.814528791, 39.95317726: 7.847921516, 39.66220736: 7.881501629, 39.37123746: 7.915266343, 39.08026756: 7.949212637, 38.78929766: 7.983337244, 38.49832776: 8.017636641, 38.20735786: 8.052107035, 37.91638796: 8.08674435, 37.62541806: 8.121544211, 37.33444816: 8.156501934, 37.04347826: 8.191612503, 36.75250836: 8.226870563, 36.46153846: 8.262270396, 36.17056856: 8.297805907, 35.87959866: 8.333470605, 35.58862876: 8.369257584, 35.29765886: 8.405159501, 35.00668896: 8.441168559, 34.71571906: 8.477276483, 34.42474916: 8.513474496, 34.13377926: 8.549753297, 33.84280936: 8.586103036, 33.55183946: 8.62251329, 33.26086957: 8.65897303, 32.96989967: 8.695470601, 32.67892977: 8.731993685, 32.38795987: 8.768529277, 32.09698997: 8.805063648, 31.80602007: 8.841582317, 31.51505017: 8.87807001, 31.22408027: 8.914510632, 30.93311037: 8.950887221, 30.64214047: 8.987181917, 30.35117057: 9.023375918, 30.06020067: 9.059449436, 29.76923077: 9.095381658, 29.47826087: 9.131150698, 29.18729097: 9.166733552, 28.89632107: 9.202106047, 28.60535117: 9.237242792, 28.31438127: 9.272117128, 28.02341137: 9.306701071, 27.73244147: 9.340965256, 27.44147157: 9.374878884, 27.15050167: 9.40840966, 26.85953177: 9.441523729, 26.56856187: 9.474185621, 26.27759197: 9.506358177, 25.98662207: 9.53800249, 25.69565217: 9.569077834, 25.40468227: 9.599541595, 25.11371237: 9.629349198, 24.82274247: 9.658454039, 24.53177258: 9.686807406, 24.24080268: 9.714358406, 23.94983278: 9.74105389, 23.65886288: 9.766838372, 23.36789298: 9.791653957, 23.07692308: 9.815440258, 22.78595318: 9.83813432, 22.49498328: 9.859670541, 22.20401338: 9.879980594, 21.91304348: 9.898993353, 21.62207358: 9.916634812, 21.33110368: 9.932828018, 21.04013378: 9.947492994, 20.74916388: 9.960546674, 20.45819398: 9.971902839, 20.16722408: 9.981472052, 19.87625418: 9.989161607, 19.58528428: 9.994875477, 19.29431438: 9.998514273, 19.00334448: 9.99997521, 18.71237458: 9.999152075, 18.42140468: 9.995935221, 18.13043478: 9.990211554, 17.83946488: 9.981864543, 17.54849498: 9.970774248, 17.25752508: 9.956817347, 16.96655518: 9.939867202, 16.67558528: 9.919793923, 16.38461538: 9.896464468, 16.09364548: 9.869742757, 15.80267559: 9.83948981, 15.51170569: 9.805563917, 15.22073579: 9.767820831, 14.92976589: 9.726113991, 14.63879599: 9.680294777, 14.34782609: 9.630212802, 14.05685619: 9.575716236, 13.76588629: 9.516652169, 13.47491639: 9.452867012, 13.18394649: 9.384206941, 12.89297659: 9.310518385, 12.60200669: 9.231648548, 12.31103679: 9.147445987, 12.02006689: 9.057761232, 11.72909699: 8.962447444, 11.43812709: 8.861361128, 11.14715719: 8.754362887, 10.85618729: 8.641318218, 10.56521739: 8.522098349, 10.27424749: 8.396581125, 9.983277592: 8.264651915, 9.692307692: 8.126204569, 9.401337793: 7.981142385, 9.110367893: 7.82937912, 8.819397993: 7.670839999, 8.528428094: 7.505462745, 8.237458194: 7.333198609, 7.946488294: 7.15401339, 7.655518395: 6.967888446, 7.364548495: 6.774821673, 7.073578595: 6.574828448, 6.782608696: 6.367942525, 6.491638796: 6.154216875, 6.200668896: 5.933724445, 5.909698997: 5.706558838, 5.618729097: 5.472834897, 5.327759197: 5.232689175, 5.036789298: 4.986280293, 4.745819398: 4.733789155, 4.454849498: 4.475419035, 4.163879599: 4.2113955, 3.872909699: 3.941966186, 3.581939799: 3.667400407, 3.2909699: 3.387988591, 3.0: 3.104041556}[float(f'{e:.10g}')])
    ax.plot(el, ex, color=V4['veto'], lw=2.5)
    ax.plot(el, dm, color=V4['nervous'], lw=2.5)
    ax.axhline(10.0, color=V4['nervous'], lw=1.2, ls='--')
    ax.axhline(2.175, color=V4['plant'], lw=1.2, ls=':')
    ax.text(72, 10.0 * 1.08, 'damped ceiling 10 rad/s', color=V4['nervous'], fontsize=FS_S, ha='left')
    ax.text(88, 2.175 * 0.8, 'joint limit 2.175 rad/s', color=V4['plant'], fontsize=FS_S, ha='left', va='top')
    ax.text(22, 40, 'exact inverse', color=V4['veto'], fontsize=FS, fontweight='bold', ha='right')
    ax.text(62, 3.1, 'damped least squares', color=V4['nervous'], fontsize=FS, fontweight='bold', ha='left', va='center')
    g, b = (90, 10)
    for e in (g, b):
        ax.plot([e], [{90: 4.896751281, 10: 37.76039566}[e]], 'o', color=V4['veto'], ms=7)
        ax.plot([e], [{90: 4.619814502, 10: 8.272409621}[e]], 'o', color=V4['nervous'], ms=7)
    ax.annotate('37.8', (b, 37.76039566), xytext=(8, -2), textcoords='offset points', fontsize=FS, color=V4['veto'], fontweight='bold')
    ax.annotate('8.3', (b, 8.272409621), xytext=(-6, -16), ha='right', textcoords='offset points', fontsize=FS, color=V4['nervous'], fontweight='bold')
    ax.annotate('4.9 / 4.6', (g, 4.896751281), xytext=(6, 9), textcoords='offset points', fontsize=FS_S, color=V4['text'], ha='left')
    ax.set_yscale('log')
    ax.set_ylim(1.5, 150)
    ax.set_yticks([2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(['2', '5', '10', '20', '50', '100'])
    ax.minorticks_off()
    ax.set_xlim(92, 0)
    ax.set_xlabel('Elbow angle (degrees; 0 is straight)', fontsize=FS)
    ax.set_ylabel('Joint rate (rad/s)', fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, 'dls.svg')
dls()

def box(ax, x, y, w, h, title, lines, ec, fc, alpha=1.0, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01,rounding_size=0.015', ec=ec, fc=fc, lw=1.8, alpha=alpha, ls='--' if dashed else '-'))
    ax.text(x + 0.015, y + h - 0.03, title, fontsize=FS, fontweight='bold', color=ec, va='top', alpha=alpha)
    for i, ln in enumerate(lines):
        ax.text(x + 0.015, y + h - 0.12 - i * 0.082, ln, fontsize=FS_S, color=V4['text'], va='top', alpha=alpha)

def record(step):
    fig, ax = plt.subplots(figsize=cm(14.6, 5.6))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    hide_axis(ax)
    box(ax, 0.01, 0.52, 0.3, 0.46, 'Header', ['frame: robot base', 'evidence time', 'parent lease digest'], V4['navy'], '#EEF2F8')
    box(ax, 0.01, 0.02, 0.3, 0.46, 'Motion', ['16 setpoints, 25 ms apart', 'seam state: position,', 'velocity, acceleration'], V4['proposal'], V4['proposal_l'], dashed=True)
    a = 1 if step >= 2 else 0
    box(ax, 0.34, 0.52, 0.32, 0.46, 'Stop (latched)', ['S-curve: 5 m/s², 66.7 m/s³', '137.5 mm in 275 ms', 'runs without the proposer'], V4['veto'], V4['veto_l'], alpha=a)
    box(ax, 0.34, 0.02, 0.32, 0.46, 'Deadlines (MCU clock)', ['chunk lease: 60 ms', 'commitment: 300 ms', 'nominal end: 400 ms', 'at rest: 575 ms'], V4['lag'], V4['lag_l'], alpha=a)
    b = 1 if step >= 3 else 0
    box(ax, 0.69, 0.52, 0.3, 0.46, 'Sidecar', ['payload at most 3 kg', 'obstacle-map epoch', 'inside the ODD'], V4['plant'], V4['plant_l'], alpha=b)
    box(ax, 0.69, 0.02, 0.3, 0.46, 'Read by', ['tracking controller:', '  setpoints, feedforward', 'permission path:', '  every field, every tick'], V4['nervous'], V4['nervous_l'], alpha=b)
    save_slot(fig, f'record_step{step}.svg')
for s in (1, 2, 3):
    record(s)

def repl(step):
    fig, ax = plt.subplots(figsize=cm(14.6, 5.4))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.92, bottom=0.2)
    y0 = 3.0
    ax.barh(y0, 51.6, color=V4['lag'], height=0.6)
    ax.text(51.6 / 2, y0, '51.6', ha='center', va='center', fontsize=FS_S, color='white', fontweight='bold')
    ax.text(51.6 / 2, y0 - 0.38, 'capture to\ndispatch', ha='center', va='top', fontsize=FS_S - 1, color=V4['lag'])
    a2 = 1 if step >= 2 else 0
    segs = [('inference', 40.0, V4['proposal'], '40'), ('check', 18.0, V4['nervous'], '18'), ('', 63.31072 - 40.0 - 18.0, V4['gray'], '')]
    x = 51.6
    for lab, w, col, txt in segs:
        ax.barh(y0, w, left=x, color=col, height=0.6, alpha=a2)
        if lab:
            ax.text(x + w / 2, y0, txt, ha='center', va='center', fontsize=FS_S, color='white', fontweight='bold', alpha=a2)
            ax.text(x + w / 2, y0 - (0.75 if lab == 'check' else 0.38), lab, ha='center', va='top', fontsize=FS_S - 1, color=col, alpha=a2)
        x += w
    ax.text(x + 2, y0, '114.9 ms', va='center', fontsize=FS + 1, fontweight='bold', color=V4['text'], alpha=a2)
    ax.text(-3, y0, 'One replacement', ha='right', va='center', fontsize=FS, fontweight='bold')
    for xv, lab, ls, ha in ((50.0, 'renewal 50 ', '--', 'right'), (60.0, ' lease 60', ':', 'left')):
        ax.plot([xv, xv], [2.55, 3.55], color=V4['veto'], lw=1.5, ls=ls)
        ax.text(xv, 3.58, lab, color=V4['veto'], fontsize=FS_S, ha=ha, va='bottom')
    a3 = 1 if step >= 3 else 0
    for i in range(3):
        st = i * 50.0
        y = 1.6 - i * 0.62
        ax.barh(y - 0.35, 114.91072, left=st, color=V4['proposal_l'], ec=V4['proposal'], height=0.45, alpha=a3)
        ax.plot([st + 114.91072], [y - 0.35], 'v', color=V4['nervous'], ms=8, alpha=a3)
        ax.text(-3, y - 0.35, f'request {i + 1}', ha='right', va='center', fontsize=FS_S, alpha=a3)
    ax.text(114.91072 + 2 * 50.0 + 6, 1.6 - 2 * 0.62 - 0.35, 'admitted', color=V4['nervous'], fontsize=FS_S, va='center', alpha=a3)
    ax.set_xlim(0, 300)
    ax.set_ylim(-0.45, 3.9)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Milliseconds from capture (allocations, illustrative)', fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, f'repl_step{step}.svg')
for s in (1, 2, 3):
    repl(s)

def chunks():
    fig, ax = plt.subplots(figsize=cm(14.2, 4.4))
    fig.subplots_adjust(left=0.34, right=0.97, top=0.95, bottom=0.27)
    for i, (n, h, e) in enumerate([('Diffusion Policy, Push-T', 16, 6), ('pi0, 50 Hz robots', 50, 25), ('Running machine', 16, 2)]):
        y = 2 - i
        ax.barh(y, h, color=V4['proposal_l'], ec=V4['proposal'], height=0.6, ls='--')
        ax.barh(y, e, color=V4['nervous'], height=0.6)
        ax.text(h + 0.8, y, f'executes {e} of {h}', va='center', fontsize=FS, color=V4['text'])
        ax.text(-1, y, n, ha='right', va='center', fontsize=FS, fontweight='bold')
    ax.set_xlim(0, 80)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Steps per chunk; green: executed before replacement', fontsize=FS_S)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, 'chunks.svg')
chunks()

def stale():
    fig, ax = plt.subplots(figsize=cm(8.4, 5.6))
    fig.subplots_adjust(left=0.2, right=0.96, top=0.93, bottom=0.21)
    t = np.linspace(-20, 130, 100)
    q = 0.5 * t / 1000
    ax.plot(t, q, color=V4['nervous'], lw=2.5)
    ax.text(125, 0.5 * 0.125, 'active chunk', color=V4['nervous'], fontsize=FS_S, ha='right', va='bottom', fontweight='bold')
    t_s = 1000 * 0.08
    ax.plot([0], [0], 'o', color=V4['veto'], ms=7)
    ax.plot([t_s], [0.04], 'o', color=V4['nervous'], ms=8)
    ax.plot([t_s, t_s], [0.04, 0], color=V4['veto'], lw=2, ls='--')
    ax.plot([0, t_s], [0, 0], color=V4['veto'], lw=1, ls=':')
    ax.text(8, -0.0105, 'request: stale anchor', color=V4['veto'], fontsize=FS_S, ha='left', va='bottom')
    ax.text(t_s + 4, 0.04 / 2, '0.040 rad\nstep back', color=V4['veto'], fontsize=FS_S, va='center')
    ax.text(t_s - 4, 0.04 + 0.004, 'seam: plan\nfrom here', color=V4['nervous'], fontsize=FS_S, ha='right', va='bottom')
    ax.set_xlabel('Time after the request (ms)', fontsize=FS)
    ax.set_ylabel('Joint angle (rad)', fontsize=FS)
    ax.set_ylim(-0.012, 0.07)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, 'stale.svg')
stale()

def overshoot():
    fig, ax = plt.subplots(figsize=cm(8.4, 5.0))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.95, bottom=0.18)
    t = np.linspace(0, 1, 100)
    vb = 0.95
    v = np.where(t < 0.5, vb + 0.6 * t, vb + 0.6 * (1 - t))
    ax.plot(t, v, color=V4['veto'], lw=2.5)
    ax.plot([0, 1], [vb, vb], 'o', color=V4['proposal'], ms=9, zorder=5)
    ax.axhline(1.0, color=V4['plant'], lw=1.4, ls='--')
    ax.axhline(0.7, color=V4['nervous'], lw=1.4, ls=':')
    ax.text(0.5, 0.97, 'limit', fontsize=FS_S, color=V4['plant'], va='top', ha='center')
    ax.text(0.5, 0.66, 'contracted limit', fontsize=FS_S, color=V4['nervous'], va='top', ha='center')
    ax.text(0.5, 1.3, 'overspeed between\ntwo legal samples', fontsize=FS_S, color=V4['veto'], ha='center', va='bottom')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['sample k', 'sample k + 1'])
    ax.set_xlim(-0.12, 1.12)
    ax.set_yticks([])
    ax.set_ylim(0.45, 1.6)
    ax.set_ylabel('Joint speed', fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, 'overshoot.svg')
overshoot()

def seam(step):
    fig, axes = plt.subplots(2, 2, figsize=cm(14.6, 5.9), sharex='col', gridspec_kw=dict(height_ratios=[1, 1.2]))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.17, hspace=0.25, wspace=0.18)
    t = np.linspace(-10, 60, 701)
    v_step = np.where(t < 0, 0, np.where(t < 1.0, 0.2 * t / 1.0, 0.2))
    tau_step = np.where((t >= 0) & (t < 1.0), 36.7, 0)
    tb = np.clip(t / 50.0, 0, 1)
    v_br = 0.2 * (3 * tb ** 2 - 2 * tb ** 3)
    acc_br = np.where((t >= 0) & (t <= 50.0), 6 * 0.2 * tb * (1 - tb) / (50.0 / 1000), 0)
    tau_br = 0.1835 * acc_br
    assert abs(tau_br.max() - 1.101) < 0.001 * 1.101, 'plotted bridge peak disagrees with MLSysIM'
    cols = [(v_step, tau_step, 'One servo tick', V4['veto']), (v_br, tau_br, '50 ms quintic bridge', V4['nervous'])]
    for c, (v, tau, title, col) in enumerate(cols):
        a = 1 if c == 0 or step >= 2 else 0
        axv, axt = (axes[0, c], axes[1, c])
        axv.plot(t, v, color=col, lw=2.5, alpha=a)
        axv.set_title(title, fontsize=FS, fontweight='bold', color=col, alpha=a)
        axt.fill_between(t, 0, tau, color=col, alpha=0.35 * a, lw=0)
        axt.plot(t, tau, color=col, lw=2, alpha=a)
        for yv, lab in ((51.0, 'rated 51'), (127.0, 'peak 127')):
            axt.axhline(yv, color=V4['plant'], lw=1, ls='--')
            axt.text(58, yv + 3, lab, fontsize=FS_S - 1, color=V4['plant'], ha='right', va='bottom')
        pk = 36.7 if c == 0 else 1.101
        axt.text(5 if c == 0 else 25, 70 if c == 0 else pk + 12, f"{ {0: '36.7', 1: '1.1'}[c]} N·m", fontsize=FS, color=col, fontweight='bold', ha='left' if c == 0 else 'center', alpha=a)
        axv.set_ylim(-0.03, 0.26)
        axt.set_ylim(0, 150)
        axt.set_xlim(-10, 60)
        axv.tick_params(labelsize=FS_S - 1)
        axt.tick_params(labelsize=FS_S - 1)
        axt.set_xlabel('ms from the seam', fontsize=FS_S)
        if c == 0:
            axv.set_ylabel('rad/s', fontsize=FS_S)
            axt.set_ylabel('N·m', fontsize=FS_S)
        if c == 1 and step < 2:
            for ax_ in (axv, axt):
                for s_ in ax_.spines.values():
                    s_.set_alpha(0)
                ax_.tick_params(colors='white')
                ax_.xaxis.label.set_color('white')
    save_slot(fig, f'seam_step{step}.svg')
for s in (1, 2):
    seam(s)

def scurve_profile(dq, a, j):
    """Rest-to-rest jerk-limited swing, no cruise: (position, speed) samples."""
    ramp = a / j
    t1 = (-ramp + math.sqrt(ramp ** 2 + 4 * dq / a)) / 2
    T = 2 * (t1 + ramp)
    tt = np.linspace(0, T, 4000)
    acc = np.zeros_like(tt)
    h = T / 2
    for i, x in enumerate(tt):
        s = x if x <= h else T - x
        sign = 1 if x <= h else -1
        if s < ramp:
            acc[i] = sign * j * s
        elif s < h - ramp:
            acc[i] = sign * a
        else:
            acc[i] = sign * j * (h - s)
    v = np.concatenate([[0], np.cumsum((acc[1:] + acc[:-1]) / 2 * np.diff(tt))])
    p = np.concatenate([[0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(tt))])
    return (p, v)

def phase():
    fig, ax = plt.subplots(figsize=cm(14.2, 5.2))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.22)
    s = np.linspace(0, 0.5, 300)
    up, dn = (40.16089205, 65.07043544)
    ax.plot(s, np.sqrt(2 * up * s), color=V4['gray'], lw=1.3, ls='--')
    ax.plot(s, np.sqrt(2 * dn * (0.5 - s)), color=V4['gray'], lw=1.3, ls='--')
    ax.text(0.03, 3.3, 'torque limit alone', fontsize=FS_S, color=V4['gray'])
    a = 7.5
    bang = np.minimum(np.sqrt(2 * a * s), np.sqrt(2 * a * (0.5 - s)))
    ax.plot(s, bang, color=V4['veto'], lw=4.5)
    p, v = scurve_profile(0.5, a, 604.7777442)
    ax.plot(p, v, color=V4['nervous'], lw=1.8)
    ax.axhline(2.175, color=V4['plant'], lw=1.2, ls=':')
    ax.text(0.1, 2.175 + 0.06, 'velocity limit 2.175', fontsize=FS_S, color=V4['plant'], ha='left', va='bottom')
    ax.plot([0.5 / 2], [1.936491673], 'o', color=V4['veto'], ms=7)
    ax.annotate('switch point: 24.8 N·m torque step', (0.5 / 2, 1.936491673), xytext=(0.25, 2.9), fontsize=FS_S, ha='center', color=V4['veto'], arrowprops=dict(arrowstyle='-', color=V4['veto'], lw=1))
    ax.text(0.13, 0.75, 'bang-bang 516.4 ms', fontsize=FS, color=V4['veto'], fontweight='bold', ha='left')
    ax.text(0.13, 0.42, 'jerk-limited 528.9 ms', fontsize=FS, color=V4['nervous'], fontweight='bold', ha='left')
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 3.6)
    ax.set_xlabel('Shoulder position along the swing (rad)', fontsize=FS)
    ax.set_ylabel('Speed (rad/s)', fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, 'phase.svg')
phase()

def ruckig_arm():
    d = RK['arm']
    fig, (av, aa) = plt.subplots(2, 1, figsize=cm(9.6, 6.0), sharex=True)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.17, hspace=0.12)
    t = 1000 * d['t']
    tc = np.array([0, 200.0])
    av.plot(tc, [1.0, 0], color=V4['gray'], lw=1.6, ls='--')
    av.plot(t, d['v'], color=V4['veto'], lw=2.6)
    av.text(8, 0.06, 'constant 5 m/s²:\n200 ms, 100 mm', fontsize=FS_S, color=V4['gray'])
    av.text(1000 * 0.275 - 4, 0.55, f"Ruckig: {1000 * d['t'][-1]:.0f} ms,\n{1000 * d['p'][-1]:.1f} mm", fontsize=FS, color=V4['veto'], fontweight='bold', ha='right')
    av.set_ylabel('m/s', fontsize=FS_S)
    aa.plot([0, 0, tc[1], tc[1]], [0, -5.0, -5.0, 0], color=V4['gray'], lw=1.6, ls='--')
    aa.plot(t, d['a'], color=V4['veto'], lw=2.6)
    aa.text(85, -2.6, 'constant: steps,\ninfinite jerk', fontsize=FS_S, color=V4['gray'])
    aa.set_ylabel('m/s²', fontsize=FS_S)
    aa.set_xlabel('ms from stop onset', fontsize=FS_S)
    for ax in (av, aa):
        ax.tick_params(labelsize=FS_S - 1)
    aa.set_xlim(-5, 290)
    save_slot(fig, 'ruckig_arm.svg')
ruckig_arm()

def runway(step):
    fig, ax = plt.subplots(figsize=cm(14.2, 3.9))
    fig.subplots_adjust(left=0.21, right=0.97, top=0.93, bottom=0.3)
    rows = [('Empty gripper', 137.5, 219.5, '5 m/s²', '137.5', '219.5'), ('Open mug', 280.4219586, 362.4219586, '2.45 m/s²', '280.4', '362.4')]
    for i, (lab, stop, tot, acc, stop_txt, tot_txt) in enumerate(rows):
        y = 1 - i
        a = 1 if i < step else 0
        ax.barh(y, 82.0, color=V4['lag'], height=0.55, alpha=a)
        ax.barh(y, stop, left=82.0, color=V4['veto'], height=0.55, alpha=a)
        ax.text(82.0 / 2, y, '82', ha='center', va='center', color='white', fontsize=FS, fontweight='bold', alpha=a)
        ax.text(82.0 + stop / 2, y, f'stop {stop_txt} ({acc})', ha='center', va='center', color='white', fontsize=FS_S, fontweight='bold', alpha=a)
        ax.text(tot + 4, y, tot_txt, va='center', fontsize=FS + 1, fontweight='bold', color=V4['veto'] if tot > 300.0 else V4['nervous'], alpha=a)
        ax.text(-5, y, lab, ha='right', va='center', fontsize=FS, fontweight='bold')
    ax.axvline(300.0, color=V4['plant'], lw=2, ls='--')
    ax.text(300.0 + 4, 1.45, 'clearance 300', ha='left', fontsize=FS_S, color=V4['plant'])
    ax.text(82.0 / 2, 1.45, 'lease path', ha='center', fontsize=FS_S, color=V4['lag'])
    ax.set_xlim(0, 410)
    ax.set_ylim(-0.5, 1.75)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Travel after the last admission at 1 m/s (mm)', fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, f'runway_step{step}.svg')
for s in (1, 2):
    runway(s)

def timeline(step):
    fig, ax = plt.subplots(figsize=cm(14.6, 4.8))
    fig.subplots_adjust(left=0.19, right=0.97, top=0.93, bottom=0.25)
    h = 0.42

    def seg(y, x0, x1, col, lab=None, a=1.0, hatch=None, fc=None):
        ax.barh(y, x1 - x0, left=x0, height=h, color=fc or col, ec=col, hatch=hatch, alpha=a, lw=1.2)
        if lab:
            ax.text((x0 + x1) / 2, y, lab, ha='center', va='center', fontsize=FS_S, color='white' if fc is None else col, fontweight='bold', alpha=a)
    y1, y2 = (1.0, 0.0)
    seg(y1, 0, 60.0, V4['nervous'], 'track')
    seg(y1, 60.0, 82.0, V4['lag'])
    seg(y1, 82.0, 82.0 + 1000 * 0.275, V4['veto'], 'S-curve stop')
    ax.text(82.0 + 1000 * 0.275 + 6, y1, '219.5 of 300 mm', va='center', fontsize=FS, color=V4['text'])
    ax.text(60.0, y1 + 0.26, 'lease lapses 60', ha='center', fontsize=FS_S, color=V4['lag'])
    ax.text(-8, y1, 'Running lease', ha='right', va='center', fontsize=FS, fontweight='bold')
    a = 1 if step >= 2 else 0
    seg(y2, 0, 300.0, V4['nervous'], 'track', a=a)
    ax.plot([300.0, 16 * 25.0], [y2 - 0.48, y2 - 0.48], color=V4['proposal'], lw=2, ls='--', alpha=a)
    ax.text(16 * 25.0 + 6, y2 - 0.48, 'setpoints to 400, unused', color=V4['proposal'], fontsize=FS_S - 1, va='center', alpha=a)
    seg(y2, 300.0, 575.0, V4['veto'], a=a)
    ax.text((300.0 + 575.0) / 2, y2, 'stop', ha='center', va='center', fontsize=FS_S, color='white', fontweight='bold', alpha=a)
    for xv, lab, ha in ((240.0, 'preferred 240 ', 'right'), (278.0, 'cutoff 278 ', 'below'), (300.0, ' commit 300', 'left')):
        ax.plot([xv, xv], [y2 - 0.22, y2 + 0.32], color=V4['text'], lw=1.5, alpha=a)
        below = ha == 'below'
        ax.text(xv, y2 - 0.24 if below else y2 + 0.34, lab, ha='right' if below else ha, fontsize=FS_S, color=V4['text'], alpha=a, va='top' if below else 'bottom')
    ax.text(575.0 + 6, y2, '197.5 of 300 mm', va='center', fontsize=FS, color=V4['text'], alpha=a)
    ax.text(-8, y2, 'Longer lease', ha='right', va='center', fontsize=FS, fontweight='bold', alpha=a)
    ax.set_xlim(0, 760)
    ax.set_ylim(-0.75, 1.5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Milliseconds after tracking starts', fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, f'timeline_step{step}.svg')
for s in (1, 2):
    timeline(s)

def base_speed():
    fig, av = plt.subplots(figsize=cm(11.0, 5.3))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.92, bottom=0.21)
    d = RK['base']
    tc = 750.0
    av.plot([0, tc], [1.5, 0], color=V4['gray'], lw=2, ls='--')
    av.plot(1000 * d['t'], d['v'], color=V4['veto'], lw=2.6)
    tp = np.linspace(0, 1, 200)
    av.plot(1125.0 * tp, 1.5 * (1 - 3 * tp ** 2 + 2 * tp ** 3), color=V4['lag'], lw=2)
    for i, (lab, col) in enumerate((('constant 750 ms, 562.5 mm', V4['gray']), ('S-curve 1,031.25 ms, 773.44 mm', V4['veto']), ('polynomial 1,125 ms, 843.75 mm', V4['lag']))):
        av.text(1290, 1.55 - i * 0.2, lab, color=col, fontsize=FS_S - 1, ha='right', va='top', fontweight='bold')
    av.set_xlabel('ms from stop onset', fontsize=FS_S)
    av.set_ylabel('Base speed (m/s)', fontsize=FS_S)
    av.set_xlim(0, 1300)
    av.set_ylim(0, 1.65)
    av.tick_params(labelsize=FS_S - 1)
    save_slot(fig, 'base_speed.svg')
base_speed()

def gen_timing():
    rows = [('Ruckig, mean', 0.0198, 'nervous', '19.8 µs'), ('Ruckig, worst case', 0.123, 'nervous', '123 µs'), ('Servo tick', 1.0, 'neutral', '1 ms'), ('Plan validation', 18.0, 'neutral', '18 ms'), ('Chunk policy pass', 40.0, 'memory', '40 ms'), ('Capture to admission', 114.91072, 'danger', '114.9 ms')]
    fig, ax = plt.subplots(figsize=cm(14.2, 5.2))
    fig.subplots_adjust(left=0.31, right=0.97, top=0.97, bottom=0.24)
    n = len(rows)
    cmap = {'nervous': V4['nervous'], 'neutral': V4['gray'], 'memory': V4['proposal'], 'danger': V4['veto']}
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, v, color=cmap[ck], height=0.62)
        ax.text(v * 1.2, y, txt, va='center', fontsize=FS, fontweight='bold', color=cmap[ck])
        ax.text(0.0075, y, lab, ha='right', va='center', fontsize=FS, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(0.01, 3000)
    ax.set_xticks([0.01, 0.1, 1, 10, 100, 1000])
    ax.set_xticklabels(['0.01', '0.1', '1', '10', '100', '1,000'])
    ax.minorticks_off()
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Milliseconds (log scale)', fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, 'gen_timing.svg')
gen_timing()

def base_view():
    fig, axes = plt.subplots(1, 2, figsize=cm(9.6, 4.6))
    fig.subplots_adjust(0.02, 0.02, 0.98, 0.98, wspace=0.08)

    def body(ax, x, y, th, a=1.0):
        c, s = (math.cos(th), math.sin(th))
        pts = np.array([[-0.3, -0.22], [0.3, -0.22], [0.3, 0.22], [-0.3, 0.22]])
        R = np.array([[c, -s], [s, c]])
        P = pts @ R.T + [x, y]
        ax.add_patch(Polygon(P, closed=True, fc=V4['plant_l'], ec=V4['plant'], lw=1.8, alpha=a))
        for wy in (-0.26, 0.26):
            w = np.array([[-0.12, wy - 0.04], [0.12, wy - 0.04], [0.12, wy + 0.04], [-0.12, wy + 0.04]])
            ax.add_patch(Polygon(w @ R.T + [x, y], closed=True, fc=V4['plant'], ec=V4['plant'], alpha=a))
    ax = axes[0]
    body(ax, 0, 0, 0)
    ax.annotate('', xy=(0.75, 0), xytext=(0.32, 0), arrowprops=dict(arrowstyle='-|>', color=V4['nervous'], lw=2.5))
    ax.annotate('', xy=(0, 0.75), xytext=(0, 0.32), arrowprops=dict(arrowstyle='-|>', color=V4['veto'], lw=2, ls='--'))
    ax.plot([-0.12, 0.12], [0.45, 0.62], color=V4['veto'], lw=3)
    ax.plot([-0.12, 0.12], [0.62, 0.45], color=V4['veto'], lw=3)
    ax.text(0.52, -0.12, 'drive', fontsize=FS_S, color=V4['nervous'], ha='center', va='top')
    ax.text(0.16, 0.78, 'no sideways step', fontsize=FS_S, color=V4['veto'], ha='left', va='center')
    ax.set_xlim(-0.6, 1.1)
    ax.set_ylim(-0.6, 0.95)
    ax.set_aspect('equal')
    hide_axis(ax)
    ax = axes[1]
    body(ax, 0, 0, 0, a=0.4)
    body(ax, 0, 0.9, 0)
    ax.plot([0, 0], [0.25, 0.65], color=V4['nervous'], lw=2.5)
    for xy, n in (((0.38, 0.05), '1 turn'), ((0.12, 0.45), '2 drive'), ((0.38, 0.95), '3 turn')):
        ax.text(*xy, n, fontsize=FS_S, color=V4['nervous'], fontweight='bold', ha='left')
    ax.set_xlim(-0.6, 1.1)
    ax.set_ylim(-0.45, 1.3)
    ax.set_aspect('equal')
    hide_axis(ax)
    save_slot(fig, 'base_view.svg')
base_view()
print('figures written')
