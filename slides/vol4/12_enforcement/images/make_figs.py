#!/usr/bin/env python3
"""Figures for Lecture 12 (Safety Enforcement).

Every value comes from calcs.py (imported as K), which reads MLSysIM: the
running machine and its aisle record, the finished stopping budgets, the
loaded arm joint and its kinetic barrier, the two-joint projection, the
permission check's stage budget, Literature.ContactForceMeasurements
(Kirschner et al. 2022, Table II, hand, B1), and cited literature
(Cruise, Therac-25). Labels print the strings calcs.py recorded, through
deckvals.val; geometry uses the full-precision values. "SE" is the rendered
chapter, Safety Enforcement (Volume IV, Chapter 12).
Diagram colors follow the Volume IV causal-boundary palette
(rules/volumes/vol4/palette.md); red only for veto, hazard, and limits.

Run from this folder: python3 make_figs.py
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
sys.path.insert(0, str(HERE))
import slideplot as sp
from slideplot import plt
C = sp.C
NAVY, BLUE, GREEN, RED, AMBER, SLATE, PETROL = ('#1F407A', '#2563EB', '#059669', '#DC2626', '#D97706', '#334155', '#0F766E')
GREEN_F, BLUE_F, RED_F, SLATE_F = ('#DCFCE7', '#EFF6FF', '#FEE2E2', '#F8FAFC')
TXT = '#1A1A2E'
FONT = 'Helvetica, Arial, sans-serif'

def save(fig, name):
    fig.savefig(HERE / f'{name}.svg', format='svg', bbox_inches='tight', pad_inches=0.08, facecolor='white')
    plt.close(fig)
DEFS = f'<defs>\n<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{SLATE}"/></marker>\n<marker id="ahg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{GREEN}"/></marker>\n<marker id="ahb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{BLUE}"/></marker>\n<marker id="ahr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{RED}"/></marker>\n</defs>'

def svg(name, w, h, body):
    (HERE / f'{name}.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">\n{DEFS}\n<rect width="{w}" height="{h}" fill="white"/>\n' + '\n'.join(body) + '\n</svg>\n')

def box(x, y, w, h, fill, stroke, dash=None, rx=8, sw=2.5):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'

def text(x, y, s, size=22, fill=TXT, anchor='middle', weight='normal', italic=False):
    st = ' font-style="italic"' if italic else ''
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{st}>{s}</text>'

def lines(x, y, rows, size=22, fill=TXT, anchor='middle', weight='normal', gap=None):
    gap = gap or size * 1.25
    return [text(x, y + i * gap, r, size, fill, anchor, weight) for i, r in enumerate(rows)]

def arrow(x1, y1, x2, y2, color=SLATE, sw=3, dash=None, marker='ah'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#{marker})"/>'

def therac(name):
    b = []
    rows = [('Therac-20', 40, True), ('Therac-25', 200, False)]
    for title, y0, hw in rows:
        b.append(text(10, y0 + 2, title, 26, TXT, anchor='start', weight='bold'))
        y = y0 + 18
        xs = [10, 240, 470, 700]
        b.append(box(xs[0], y, 190, 92, BLUE_F, NAVY))
        b += lines(xs[0] + 95, y + 40, ['Operator', 'console'], 24)
        b.append(box(xs[1], y, 190, 92, GREEN_F, GREEN))
        b += lines(xs[1] + 95, y + 40, ['Software', 'check'], 24)
        if hw:
            b.append(box(xs[2], y, 190, 92, RED_F, RED))
            b += lines(xs[2] + 95, y + 40, ['Turntable', 'microswitch'], 24, '#991B1B', weight='bold')
        else:
            b.append(box(xs[2], y, 190, 92, 'white', '#94A3B8', dash='8,6'))
            b += lines(xs[2] + 95, y + 40, ['removed'], 24, '#64748B')
        b.append(box(xs[3], y, 190, 92, SLATE_F, SLATE, rx=0))
        b += lines(xs[3] + 95, y + 40, ['Beam', 'fires'], 24)
        for i in range(3):
            b.append(arrow(xs[i] + 192, y + 46, xs[i + 1] - 2, y + 46))
    svg(name, 900, 320, b)
therac('therac_hook')

def gatekeeper(step):
    b = []
    b.append(box(10, 20, 230, 120, BLUE_F, BLUE, dash='8,5'))
    b += lines(125, 60, ['Learned', 'proposer'], 25, NAVY, weight='bold')
    b.append(text(125, 122, 'proposes only', 22, BLUE))
    b.append('<line x1="262" y1="5" x2="262" y2="390" stroke="#475569" stroke-width="2" stroke-dasharray="6,4"/>')
    b.append(text(20, 175, 'proposal', 22, '#475569', anchor='start'))
    b.append(text(20, 200, 'boundary', 22, '#475569', anchor='start'))
    b.append(arrow(240, 80, 288, 80, BLUE, marker='ahb', dash='8,5'))
    b.append(box(290, 20, 300, 120, GREEN_F, GREEN))
    b += lines(440, 58, ['Enforcer on the MCU', 'every 1 ms tick'], 25, '#047857', weight='bold')
    b.append(text(440, 124, 'admit, correct, fall back', 22, '#047857'))
    b.append(arrow(590, 80, 638, 80, GREEN, marker='ahg'))
    b.append(box(640, 20, 250, 120, SLATE_F, SLATE, rx=0))
    b += lines(765, 72, ['Drives and', 'brakes'], 25, weight='bold')
    if step >= 2:
        b.append(box(290, 245, 300, 120, RED_F, RED))
        b += lines(440, 290, ['Certified floor:', 'scanner, STO, brakes'], 24, '#991B1B', weight='bold')
        b.append(f'<path d="M590,305 L765,305 L765,143" fill="none" stroke="{RED}" stroke-width="3" marker-end="url(#ahr)"/>')
        b.append(text(612, 345, 'hardware path', 22, RED, anchor='start'))
        b.append(arrow(440, 243, 440, 143, RED, dash='6,4', marker='ahr'))
        b.append(text(450, 200, 'if the MCU fails', 22, RED, anchor='start'))
    svg(f'gate_step{step}', 900, 375, b)
for s in (1, 2):
    gatekeeper(s)
fig, ax = plt.subplots(figsize=(4.4, 2.2))
t = np.linspace(0, 1.03125, 400)
ax.plot([0, 0.75], [1.5, 0], color=C['neutral'], lw=3, ls='--')
v = []
tj = 2.0 / 7.111111111
for ti in t:
    if ti < tj:
        v.append(1.5 - 0.5 * 7.111111111 * ti ** 2)
    elif ti < 1.03125 - tj:
        v.append(1.5 - 0.5 * 2.0 * tj - 2.0 * (ti - tj))
    else:
        tr = 1.03125 - ti
        v.append(0.5 * 7.111111111 * tr ** 2)
ax.plot(t, v, color=GREEN, lw=3.5)
ax.set_xticks([])
ax.set_yticks([])
save(fig, 'recall_stop')
fig, ax = plt.subplots(figsize=(5.9, 2.7))
vs = np.linspace(0.02, 1.7, 300)
fixed = 0.19 * 1000.0
ax.plot(vs, [193.7326602, 194.9603481, 196.2436793, 197.5773509, 198.9573403, 200.3804597, 201.8441017, 203.3460816, 204.8845346, 206.457844, 208.0645911, 209.7035173, 211.3734966, 213.0735134, 214.8026457, 216.5600517, 218.3449584, 220.1566533, 221.9944761, 223.8578136, 225.7460935, 227.6587804, 229.5953721, 231.5553961, 233.5384069, 235.5439834, 237.5717269, 239.6212592, 241.6922207, 243.784269, 245.8970777, 248.0303351, 250.1837432, 252.3570164, 254.5498813, 256.7620751, 258.9933456, 261.2434501, 263.5121548, 265.7992346, 268.1044724, 270.4276585, 272.7685902, 275.1270718, 277.5029137, 279.8959324, 282.3059502, 284.7327945, 287.1762983, 289.6362991, 292.1126392, 294.6051653, 297.1137285, 299.6381837, 302.1783897, 304.7342093, 307.3055084, 309.8921565, 312.4940265, 315.1109941, 317.7429382, 320.3897405, 323.0512854, 325.7274602, 328.4181544, 331.1232602, 333.8426721, 336.5762869, 339.3240036, 342.0857233, 344.8613492, 347.6507864, 350.453942, 353.2707249, 356.1010458, 358.9448172, 361.8019531, 364.6723693, 367.5559831, 370.4527134, 373.3624804, 376.285206, 379.2208132, 382.1692266, 385.130372, 388.1041764, 391.0905683, 394.089477, 397.1008334, 400.1245691, 403.1606173, 406.2089119, 409.269388, 412.3419816, 415.4266299, 418.523271, 421.6318437, 424.7526849, 427.8887113, 431.0405228, 434.2081194, 437.3915011, 440.5906678, 443.8056195, 447.0363563, 450.2828782, 453.5451851, 456.8232771, 460.1171542, 463.4268163, 466.7522634, 470.0934957, 473.450513, 476.8233153, 480.2119027, 483.6162751, 487.0364327, 490.4723752, 493.9241029, 497.3916156, 500.8749133, 504.3739961, 507.888864, 511.4195169, 514.9659549, 518.5281779, 522.106186, 525.6999792, 529.3095574, 532.9349207, 536.576069, 540.2330024, 543.9057208, 547.5942243, 551.2985129, 555.0185865, 558.7544452, 562.506089, 566.2735178, 570.0567316, 573.8557305, 577.6705145, 581.5010835, 585.3474376, 589.2095768, 593.087501, 596.9812102, 600.8907046, 604.8159839, 608.7570484, 612.7138979, 616.6865324, 620.6749521, 624.6791567, 628.6991465, 632.7349212, 636.7864811, 640.853826, 644.936956, 649.035871, 653.1505711, 657.2810562, 661.4273264, 665.5893817, 669.767222, 673.9608474, 678.1702578, 682.3954533, 686.6364338, 690.8931994, 695.1657501, 699.4540858, 703.7582066, 708.0781124, 712.4138033, 716.7652793, 721.1325403, 725.5155864, 729.9144175, 734.3290337, 738.759435, 743.2056213, 747.6675926, 752.1453491, 756.6388905, 761.1482171, 765.6733287, 770.2142253, 774.770907, 779.3433738, 783.9316256, 788.5356625, 793.1554845, 797.7910915, 802.4424836, 807.1096607, 811.7926229, 816.4913701, 821.2059024, 825.9362198, 830.6823222, 835.4442096, 840.2218822, 845.0153398, 849.8245824, 854.6496101, 859.4904229, 864.3470207, 869.2194036, 874.1075715, 879.0115245, 883.9312626, 888.8667857, 893.8180939, 898.7851871, 903.7680654, 908.7667287, 913.7811772, 918.8114106, 923.8574291, 928.9192327, 933.9968214, 939.0901951, 944.1993538, 949.3242976, 954.4650265, 959.6215404, 964.7938394, 969.9819235, 975.1857926, 980.4054467, 985.640886, 990.8921103, 996.1591196, 1001.441914, 1006.740493, 1012.054858, 1017.385008, 1022.730942, 1028.092662, 1033.470167, 1038.863456, 1044.272531, 1049.697391, 1055.138036, 1060.594466, 1066.066681, 1071.554681, 1077.058467, 1082.578037, 1088.113392, 1093.664533, 1099.231458, 1104.814169, 1110.412664, 1116.026945, 1121.657011, 1127.302861, 1132.964497, 1138.641918, 1144.335124, 1150.044115, 1155.768891, 1161.509452, 1167.265798, 1173.037929, 1178.825846, 1184.629547, 1190.449033, 1196.284305, 1202.135361, 1208.002203, 1213.884829, 1219.783241, 1225.697438, 1231.62742, 1237.573186, 1243.534738, 1249.512075, 1255.505197, 1261.514104, 1267.538797, 1273.579274, 1279.635536, 1285.707583, 1291.795416, 1297.899033, 1304.018436, 1310.153623, 1316.304596, 1322.471353, 1328.653896, 1334.852224, 1341.066337, 1347.296235, 1353.541918, 1359.803386, 1366.080639, 1372.373677, 1378.6825], color=SLATE, lw=3)
ax.plot(vs, [193.7326602, 194.9603481, 196.2436793, 197.5773509, 198.9573403, 200.3804597, 201.8441017, 203.3460816, 204.8845346, 206.457844, 208.0645911, 209.7035173, 211.3734966, 213.0735134, 214.8026457, 216.5600517, 218.3449584, 220.1566533, 221.9944761, 223.8578136, 225.7460935, 227.6587804, 229.5953721, 231.5553961, 233.5384069, 235.5439834, 237.5717269, 239.6212592, 241.6922207, 243.784269, 245.8970777, 248.0303351, 250.1864915, 252.3694413, 254.5792182, 256.8158223, 259.0792535, 261.3695118, 263.6865972, 266.0305098, 268.4012495, 270.7988164, 273.2232103, 275.6744315, 278.1524797, 280.6573551, 283.1890576, 285.7475872, 288.332944, 290.9451279, 293.584139, 296.2499771, 298.9426425, 301.6621349, 304.4084545, 307.1816012, 309.981575, 312.808376, 315.6620041, 318.5424594, 321.4497417, 324.3838512, 327.3447879, 330.3325517, 333.3471426, 336.3885606, 339.4568058, 342.5518781, 345.6737775, 348.8225041, 351.9980578, 355.2004387, 358.4296466, 361.6856817, 364.968544, 368.2782334, 371.6147499, 374.9780935, 378.3682643, 381.7852622, 385.2290872, 388.6997394, 392.1972187, 395.7215251, 399.2726587, 402.8506194, 406.4554073, 410.0870222, 413.7454643, 417.4307336, 421.1428299, 424.8817535, 428.6475041, 432.4400819, 436.2594868, 440.1057188, 443.978778, 447.8786643, 451.8053777, 455.7589183, 459.739286, 463.7464808, 467.7805028, 471.8413519, 475.9290281, 480.0435315, 484.184862, 488.3530196, 492.5480044, 496.7698163, 501.0184553, 505.2939215, 509.5962148, 513.9253352, 518.2812828, 522.6640575, 527.0736593, 531.5100883, 535.9733444, 540.4634276, 544.980338, 549.5240755, 554.0946401, 558.6920319, 563.3162508, 567.9672968, 572.64517, 577.3498703, 582.0813977, 586.8397523, 591.624934, 596.4369428, 601.2757788, 606.1414419, 611.0339321, 615.9532495, 620.899394, 625.8723656, 630.8721644, 635.8987903, 640.9522433, 646.0325235, 651.1396308, 656.2735652, 661.4343268, 666.6219155, 671.8363313, 677.0775742, 682.3456443, 687.6405416, 692.9622659, 698.3108174, 703.6861961, 709.0884018, 714.5174347, 719.9732947, 725.4559819, 730.9654962, 736.5018376, 742.0650062, 747.6550019, 753.2718247, 758.9154747, 764.5859518, 770.283256, 776.0073873, 781.7583458, 787.5361315, 793.3407442, 799.1721841, 805.0304511, 810.9155453, 816.8274666, 822.766215, 828.7317906, 834.7241933, 840.7434231, 846.78948, 852.8623641, 858.9620754, 865.0886137, 871.2419792, 877.4221718, 883.6291916, 889.8630385, 896.1237125, 902.4112137, 908.7255419, 915.0666974, 921.4346799, 927.8294896, 934.2511264, 940.6995904, 947.1748815, 953.6769997, 960.2059451, 966.7617175, 973.3443172, 979.9537439, 986.5899978, 993.2530788, 999.942987, 1006.659722, 1013.403285, 1020.173674, 1026.970891, 1033.794935, 1040.645806, 1047.523504, 1054.428029, 1061.359381, 1068.317561, 1075.302567, 1082.314401, 1089.353062, 1096.41855, 1103.510865, 1110.630008, 1117.775977, 1124.948774, 1132.148397, 1139.374848, 1146.628126, 1153.908231, 1161.215163, 1168.548923, 1175.909509, 1183.296923, 1190.711163, 1198.152231, 1205.620126, 1213.114849, 1220.636398, 1228.184774, 1235.759978, 1243.362008, 1250.990866, 1258.646551, 1266.329063, 1274.038402, 1281.774569, 1289.537562, 1297.327383, 1305.144031, 1312.987506, 1320.857808, 1328.754937, 1336.678893, 1344.629676, 1352.607287, 1360.611725, 1368.642989, 1376.701081, 1384.786, 1392.897747, 1401.03632, 1409.201721, 1417.393948, 1425.613003, 1433.858885, 1442.131594, 1450.43113, 1458.757493, 1467.110684, 1475.490701, 1483.897546, 1492.331218, 1500.791717, 1509.279043, 1517.793196, 1526.334176, 1534.901984, 1543.496618, 1552.11808, 1560.766369, 1569.441485, 1578.143428, 1586.872198, 1595.627796, 1604.41022, 1613.219472, 1622.055551, 1630.918457, 1639.80819, 1648.72475, 1657.668137, 1666.638352, 1675.635393, 1684.659262, 1693.709958, 1702.787481, 1711.891831, 1721.023008, 1730.181013, 1739.365844, 1748.577503, 1757.815989, 1767.081302, 1776.373442, 1785.692409], color=AMBER, lw=3, ls='--')
vw = vs[vs < 0.6]
ax.plot(vw, [577.1982877, 600.7905694, 622.0931411, 641.7129349, 660.030663, 677.3036211, 693.7152654, 709.401804, 724.4676194, 738.9947709, 753.0491357, 766.6845378, 779.9456142, 792.8698623, 805.4891354, 817.8307593, 829.91838, 841.7726184, 853.4115822, 864.8512722, 876.105908, 887.1881915, 898.1095231, 908.8801787, 919.5094588, 930.0058114, 940.3769369, 950.6298764, 960.7710875, 970.806509, 980.7416169, 990.5814724, 1000.330765, 1009.993847, 1019.574769, 1029.077307, 1038.504984, 1047.861099, 1057.148742, 1066.370811, 1075.530031, 1084.628968, 1093.670036, 1102.655518, 1111.587567, 1120.468221, 1129.299411, 1138.082964, 1146.820619, 1155.514022, 1164.164744, 1172.774275, 1181.344037, 1189.875386, 1198.369613, 1206.827954, 1215.251589, 1223.641645, 1231.999203, 1240.325296, 1248.620916, 1256.887012, 1265.124497, 1273.334245, 1281.517098, 1289.673864, 1297.80532, 1305.912215, 1313.995269, 1322.055175, 1330.092601, 1338.108194, 1346.102574, 1354.076342, 1362.030076, 1369.964337, 1377.879666, 1385.776585, 1393.655599, 1401.517199, 1409.361857, 1417.190031, 1425.002166, 1432.798692, 1440.580025, 1448.346569, 1456.098717, 1463.836849, 1471.561334, 1479.272529, 1486.970783, 1494.656434, 1502.329809, 1509.991228, 1517.641002, 1525.27943, 1532.906808, 1540.526063, 1548.157073, 1555.803867, 1563.466447, 1571.144812, 1578.838962, 1586.548897], color=RED, lw=3, ls=':')
ax.axhline(1.1 * 1000.0, color=C['neutral'], lw=1.5, ls='--')
ax.text(0.62, 1130, 'clear distance 1,100 mm', ha='left', fontsize=10.5, color=C['neutral'])
for vv, key, col, lab, y in [(1.436694177, 'ceiling', SLATE, 'dry floor', 440), (1.230866736, 'ceiling_inspected', AMBER, 'floor at friction 0.12', 300), (0.2599412823, 'v_walk', RED, 'person walking toward the base', 160)]:
    ax.plot([vv], [1.1 * 1000.0], 'o', ms=9, color=col, mec='white', mew=1.5, zorder=5)
    ax.text(0.9, y, f"{lab}: { {1.436694177: '1.44', 1.230866736: '1.23', 0.2599412823: '0.26'}[vv]} m/s", color=col, fontsize=11, fontweight='bold')
ax.set_xlim(0, 1.7)
ax.set_ylim(0, 1600)
ax.set_xlabel('Speed when the hazard is seen (m/s)', fontsize=11)
ax.set_ylabel('Stop needs (mm)', fontsize=11)
ax.tick_params(labelsize=10)
save(fig, 'ceilings')

def portrait(step):
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    D = np.linspace(0, 2.0, 200)
    ax.plot(D, np.sqrt(2 * 2.0 * D), color=C['neutral'], lw=2.5, ls='--')
    ax.text(0.33, 1.9, 'braking only', color=C['neutral'], fontsize=12)
    if step >= 2:
        inset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.005880701141, 0.0505098122, 0.08636193079, 0.1184292146, 0.1480905708, 0.1760104889, 0.2025786204, 0.2280498309, 0.2526030649, 0.2763704074, 0.2994530645, 0.3219308645, 0.3438682518, 0.3653182497, 0.3863251819, 0.4069266017, 0.4271546951, 0.447037326, 0.4665988298, 0.4858606252, 0.5048416953, 0.5235589688, 0.5420276269, 0.5602613529, 0.5782445543, 0.5959457461, 0.61337728, 0.6305511169, 0.6474783585, 0.6641693317, 0.6806336622, 0.6968803392, 0.712917773, 0.7287538458, 0.7443959571, 0.7598510639, 0.7751257171, 0.7902260937, 0.8051580263, 0.8199270292, 0.8345383221, 0.8489968519, 0.8633073124, 0.8774741614, 0.8915016378, 0.9053937757, 0.9191544185, 0.9327872309, 0.9462957105, 0.9596831984, 0.9729528885, 0.9861078367, 0.999150969, 1.012085089, 1.024912885, 1.037636938, 1.050259723, 1.062783622, 1.075210922, 1.087543825, 1.099784449, 1.111934836, 1.123996953, 1.135972695, 1.147863892, 1.15967231, 1.171399653, 1.18304757, 1.194617652, 1.206111439, 1.217530422, 1.228876041, 1.240149692, 1.251352729, 1.262486462, 1.273552161, 1.284551058, 1.295484349, 1.306353194, 1.31715872, 1.327902019, 1.338584156, 1.349206162, 1.359769043, 1.370273774, 1.380721305, 1.391112562, 1.401448442, 1.411729823, 1.421957558, 1.432132478, 1.442255391, 1.452327089, 1.46234834, 1.472319895, 1.482242487, 1.492116828, 1.501943617, 1.511723534, 1.521457243, 1.531145393, 1.540788619, 1.550387538, 1.559942756, 1.569454866, 1.578924444, 1.588352057, 1.597738257, 1.607083586, 1.616388573, 1.625653735, 1.63487958, 1.644066604, 1.653215294, 1.662326124, 1.671399562, 1.680436063, 1.689436075, 1.698400037, 1.707328379, 1.716221522, 1.725079878, 1.733903853, 1.742693844, 1.751450242, 1.760173427, 1.768863775, 1.777521654, 1.786147426, 1.794741445, 1.803304058, 1.811835609, 1.820336432, 1.828806857, 1.837247209, 1.845657804, 1.854038956, 1.862390971, 1.870714151, 1.879008794, 1.88727519, 1.895513626, 1.903724384, 1.911907742, 1.92006397, 1.928193339, 1.93629611, 1.944372544, 1.952422896, 1.960447417, 1.968446353, 1.976419948, 1.984368442, 1.99229207, 2.000191064, 2.008065652, 2.015916058, 2.023742506, 2.031545212, 2.039324391, 2.047080254, 2.054813011, 2.062522867, 2.070210023, 2.077874679, 2.085517032, 2.093137275, 2.100735599, 2.108312192, 2.115867238, 2.123400923, 2.130913424, 2.13840492, 2.145875587, 2.153325596, 2.160755119, 2.168164323, 2.175553375, 2.182922438, 2.190271672, 2.197601238]
        ax.fill_between(D, 0, inset, color=GREEN_F)
        ax.plot(D, inset, color=GREEN, lw=3)
        ax.text(1.6, 0.35, 'stopping\nenvelope', color='#047857', fontsize=13, fontweight='bold', ha='center')
        ax.plot([1.1], [1.436694177], 'o', color=GREEN, ms=9, mec='white', mew=1.5, zorder=5)
        ax.annotate('ceiling 1.44 m/s', (1.1, 1.436694177), xytext=(1.22, 0.98), ha='left', fontsize=12, color='#047857', fontweight='bold', arrowprops=dict(arrowstyle='-', color='#047857'))
    if step >= 3:
        ax.plot([1.1], [1.3], 's', color=NAVY, ms=8, zorder=6)
        ax.text(1.22, 1.2, 'aisle 1.3 m/s', color=NAVY, fontsize=12, fontweight='bold')
        ax.plot([0.6], [1.3], 'X', color=RED, ms=11, zorder=6)
        ax.annotate('outside:\nSTOP', (0.6, 1.3), xytext=(0.03, 1.66), color=RED, fontsize=12, fontweight='bold', ha='left', arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 2.0)
    ax.set_xlabel('Clear distance ahead (m)', fontsize=12)
    ax.set_ylabel('Speed (m/s)', fontsize=12)
    ax.tick_params(labelsize=11)
    save(fig, f'portrait_step{step}')
for s in (1, 2, 3):
    portrait(s)
fig, ax = plt.subplots(figsize=(4.6, 2.4))
data = [('Ascending', 11.34975369, 'stop_up', GREEN), ('No bias', 19.86206897, 'stop_free', SLATE), ('Descending', 79.44827586, 'stop_down', RED)]
for i, (lab, w, key, col) in enumerate(data):
    ax.barh(i, w, color=col, height=0.6)
    ax.text(w + 1.5, i, f"{ {'stop_up': '11.35', 'stop_free': '19.86', 'stop_down': '79.45'}[key]} mm", va='center', fontsize=13, color=col, fontweight='bold')
    ax.text(-1.5, i, lab, ha='right', va='center', fontsize=13, fontweight='bold', color=TXT)
ax.set_xlim(0, 105)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Tool travel before the joint stops (mm)', fontsize=12)
ax.tick_params(labelsize=11)
save(fig, 'gravity')

def reldeg(step):
    b = []
    xs = [20, 250, 480, 710]
    labs = [['Motor', 'torque'], ['Accel-', 'eration'], ['Velocity'], ['Position']]
    for i, (x, lab) in enumerate(zip(xs, labs)):
        b.append(box(x, 40, 170, 90, GREEN_F if i == 0 else SLATE_F, GREEN if i == 0 else SLATE))
        b += lines(x + 85, 78 if len(lab) == 2 else 92, lab, 23, weight='bold')
    b.append(arrow(190, 85, 248, 85))
    b.append(text(219, 72, '÷ m', 19))
    b.append(arrow(420, 85, 478, 85))
    b.append(text(449, 72, '∫ dt', 19))
    b.append(arrow(650, 85, 708, 85))
    b.append(text(679, 72, '∫ dt', 19))
    b.append(box(560, 180, 320, 110, RED_F, RED))
    b += lines(720, 218, ['Barrier on distance alone:', 'its rate is −velocity'], 21, '#991B1B', weight='bold')
    b.append(text(720, 276, 'torque has no term in it', 20, '#991B1B'))
    b.append(arrow(795, 130, 760, 178, RED, marker='ahr'))
    if step >= 2:
        b.append(box(20, 180, 500, 110, GREEN_F, GREEN))
        b += lines(270, 218, ['Kinetic barrier: clearance minus', 'the distance needed to stop'], 21, '#047857', weight='bold')
        b.append(text(270, 276, 'its rate contains torque', 20, '#047857'))
        b.append(arrow(565, 130, 400, 178, GREEN, marker='ahg'))
    svg(f'reldeg_step{step}', 900, 300, b)
for s in (1, 2):
    reldeg(s)
fig, ax = plt.subplots(figsize=(5.9, 2.7))
cl = np.linspace(0.06, 0.14, 200)
dem = [91.40625, 91.3151696, 91.2240892, 91.13300879, 91.04192839, 90.95084799, 90.85976759, 90.76868719, 90.67760678, 90.58652638, 90.49544598, 90.40436558, 90.31328518, 90.22220477, 90.13112437, 90.04004397, 89.94896357, 89.85788317, 89.76680276, 89.67572236, 89.58464196, 89.49356156, 89.40248116, 89.31140075, 89.22032035, 89.12923995, 89.03815955, 88.94707915, 88.85599874, 88.76491834, 88.67383794, 88.58275754, 88.49167714, 88.40059673, 88.30951633, 88.21843593, 88.12735553, 88.03627513, 87.94519472, 87.85411432, 87.76303392, 87.67195352, 87.58087312, 87.48979271, 87.39871231, 87.30763191, 87.21655151, 87.12547111, 87.0343907, 86.9433103, 86.8522299, 86.7611495, 86.6700691, 86.57898869, 86.48790829, 86.39682789, 86.30574749, 86.21466709, 86.12358668, 86.03250628, 85.94142588, 85.85034548, 85.75926508, 85.66818467, 85.57710427, 85.48602387, 85.39494347, 85.30386307, 85.21278266, 85.12170226, 85.03062186, 84.93954146, 84.84846106, 84.75738065, 84.66630025, 84.57521985, 84.48413945, 84.39305905, 84.30197864, 84.21089824, 84.11981784, 84.02873744, 83.93765704, 83.84657663, 83.75549623, 83.66441583, 83.57333543, 83.48225503, 83.39117462, 83.30009422, 83.20901382, 83.11793342, 83.02685302, 82.93577261, 82.84469221, 82.75361181, 82.66253141, 82.57145101, 82.4803706, 82.3892902, 82.2982098, 82.2071294, 82.11604899, 82.02496859, 81.93388819, 81.84280779, 81.75172739, 81.66064698, 81.56956658, 81.47848618, 81.38740578, 81.29632538, 81.20524497, 81.11416457, 81.02308417, 80.93200377, 80.84092337, 80.74984296, 80.65876256, 80.56768216, 80.47660176, 80.38552136, 80.29444095, 80.20336055, 80.11228015, 80.02119975, 79.93011935, 79.83903894, 79.74795854, 79.65687814, 79.56579774, 79.47471734, 79.38363693, 79.29255653, 79.20147613, 79.11039573, 79.01931533, 78.92823492, 78.83715452, 78.74607412, 78.65499372, 78.56391332, 78.47283291, 78.38175251, 78.29067211, 78.19959171, 78.10851131, 78.0174309, 77.9263505, 77.8352701, 77.7441897, 77.6531093, 77.56202889, 77.47094849, 77.37986809, 77.28878769, 77.19770729, 77.10662688, 77.01554648, 76.92446608, 76.83338568, 76.74230528, 76.65122487, 76.56014447, 76.46906407, 76.37798367, 76.28690327, 76.19582286, 76.10474246, 76.01366206, 75.92258166, 75.83150126, 75.74042085, 75.64934045, 75.55826005, 75.46717965, 75.37609925, 75.28501884, 75.19393844, 75.10285804, 75.01177764, 74.92069724, 74.82961683, 74.73853643, 74.64745603, 74.55637563, 74.46529523, 74.37421482, 74.28313442, 74.19205402, 74.10097362, 74.00989322, 73.91881281, 73.82773241, 73.73665201, 73.64557161, 73.55449121, 73.4634108, 73.3723304, 73.28125]
ax.axhspan(87.0, 100, color=RED_F)
ax.axhline(87.0, color=RED, ls='--', lw=2)
ax.text(138, 87.0 + 1.6, 'joint peak 87 N·m', color=RED, ha='right', fontsize=11.5, fontweight='bold')
ax.plot(cl * 1000.0, dem, color=GREEN, lw=3)
for c, ck, tk, col, dx, dy in [(0.11, 'clear_far', 'brake_far', GREEN, -10, 12), (79.44827586 / 1000.0, 'stop_down', 'brake_zero', NAVY, 20, -60), (0.07, 'clear_near', 'brake_near', RED, -6, 12)]:
    tq = {0.11: 80.078125, 0.07944827586: 87.0, 0.07: 89.140625}[c]
    ax.plot([c * 1000.0], [tq], 'o', color=col, ms=9, mec='white', mew=1.5, zorder=5)
    ax.annotate(f"{ {0.11: '110', 0.07944827586: '79.45', 0.07: '70'}[c]} mm: { {0.11: '80.1', 0.07944827586: '87.0', 0.07: '89.1'}[c]} N·m", (c * 1000.0, tq), xytext=(dx, dy), textcoords='offset points', fontsize=11.5, color=col, fontweight='bold')
ax.text(64, 70, 'gravity alone\nasks 65.25 N·m', fontsize=10.5, color=C['neutral'])
ax.axhline(65.25, color=C['neutral'], lw=1.2, ls=':')
ax.set_xlim(60, 140)
ax.set_ylim(60, 95)
ax.set_xlabel('Tool clearance to the packing station (mm)', fontsize=11)
ax.set_ylabel('Braking torque\nthe barrier asks (N·m)', fontsize=11)
ax.tick_params(labelsize=10)
save(fig, 'barrier_torque')

def projection(step):
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    x = np.linspace(-1, 5, 50)
    yb = 5.0 - 2.0 * x
    ax.fill_between(x, yb, -1, color=GREEN_F)
    ax.plot(x, yb, color=GREEN, lw=2.5)
    ax.text(0.62, 0.05, 'admissible', color='#047857', fontsize=12, fontweight='bold')
    ax.text(0.95, 4.05, '2τ₁ + τ₂ = 5', color='#047857', fontsize=11.5)
    ax.plot([0, 3.5], [0, 3.0], color=BLUE, lw=2, ls='--')
    ax.plot([3.5], [3.0], 'o', color=BLUE, ms=10, zorder=5)
    ax.text(3.5 + 0.12, 3.0 + 0.12, 'proposal (3.5, 3)', color=BLUE, fontsize=12, fontweight='bold')
    if step >= 2:
        ax.plot([2.0], [3.0], 's', color=RED, ms=9, zorder=5)
        ax.annotate('', (2.0, 3.0), (3.5, 3.0), arrowprops=dict(arrowstyle='->', color=RED, lw=2))
        ax.text(2.0 - 0.15, 3.0 + 0.35, 'clamp (2, 3)\n2·2 + 3 = 7', color=RED, fontsize=11.5, fontweight='bold', ha='center')
    if step >= 3:
        ax.plot([1.5], [2.0], 'D', color=GREEN, ms=9, zorder=5)
        ax.annotate('', (1.5, 2.0), (3.5, 3.0), arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.5))
        ax.text(0.1, 1.15, 'projection\n(1.5, 2)', color='#047857', fontsize=12, fontweight='bold', ha='left')
    ax.set_xlim(-0.2, 4.6)
    ax.set_ylim(-0.2, 4.4)
    ax.set_aspect('equal')
    ax.set_xlabel('Joint 1 torque τ₁ (N·m)', fontsize=12)
    ax.set_ylabel('Joint 2 torque τ₂ (N·m)', fontsize=12)
    ax.tick_params(labelsize=11)
    save(fig, f'proj_step{step}')
for s in (1, 2, 3):
    projection(s)

def latch(step):
    fig, ax = plt.subplots(figsize=(5.9, 2.7))
    v = 0.03
    lo = {'f_at_brake_onset': Q_(39.0, 'N'), 'x_remaining': Q_(0.1525, 'mm'), 'a_arm_needed': Q_(2.950819672, 'm / s ** 2'), 't_arm_stop': Q_(10.16666667, 'ms'), 'df_dt': Q_(12000.0, 'N / s')}
    rate = lo['df_dt'].to('N/s').magnitude
    f_on = lo['f_at_brake_onset'].to('N').magnitude
    acc = lo['a_arm_needed'].to('m/s**2').magnitude
    tstop = lo['t_arm_stop'].to('s').magnitude
    t_trip = 15.0 / rate
    t_b = t_trip + 0.002
    tt = np.linspace(0, t_b, 50)
    ax.plot(tt * 1000.0, rate * tt, color=GREEN, lw=3)
    t2 = np.linspace(0, tstop, 60)
    f2 = f_on + 400000.0 * (v * t2 - 0.5 * acc * t2 ** 2)
    ax.plot((t_b + t2) * 1000.0, f2, color=GREEN, lw=3)
    ax.plot([(t_b + tstop) * 1000.0, 30], [f2[-1], f2[-1]], color=GREEN, lw=3)
    tn = np.linspace(t_b, t_b + 0.006, 20)
    ax.plot(tn * 1000.0, rate * tn, color=RED, lw=2, ls='--')
    ax.text((t_b + 0.0062) * 1000.0, 104, 'no brake', color=RED, fontsize=11, fontweight='bold')
    ax.axvspan(t_trip * 1000.0, t_b * 1000.0, color=AMBER, alpha=0.18)
    ax.axhline(15.0, color=C['neutral'], ls=':', lw=1.5)
    ax.text(29.5, 15.0 + 3, 'tripwire 15 N', ha='right', fontsize=10.5, color=C['neutral'])
    ax.axhline(100.0, color=RED, ls='--', lw=2)
    ax.axhspan(100.0, 150, color=RED_F, alpha=0.6)
    ax.text(29.5, 100.0 + 4, 'latch limit 100 N', ha='right', fontsize=11, color=RED, fontweight='bold')
    ax.plot([t_b * 1000.0], [f_on], 'o', color=GREEN, ms=9, mec='white', mew=1.5, zorder=5)
    ax.text(t_b * 1000.0 + 0.6, f_on - 14, '0.03 m/s: brake at 39 N', color='#047857', fontsize=11.5, fontweight='bold')
    if step >= 2:
        r2 = 40000.0
        fo2 = 95.0
        tt2 = np.linspace(0, 15.0 / r2 + 0.002, 30)
        ax.plot(tt2 * 1000.0, r2 * tt2, color=BLUE, lw=3)
        ax.plot([tt2[-1] * 1000.0], [fo2], 'o', color=BLUE, ms=9, mec='white', mew=1.5, zorder=5)
        ax.annotate('0.10 m/s: 95 N at brake,\nstop needs 400 m/s²', (tt2[-1] * 1000.0, fo2), xytext=(0.3, 132), color=BLUE, fontsize=11.5, fontweight='bold', va='center', arrowprops=dict(arrowstyle='-', color=BLUE))
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 150)
    ax.set_xlabel('Time since first contact (ms)', fontsize=11)
    ax.set_ylabel('Force on plate (N)', fontsize=11)
    ax.tick_params(labelsize=10)
    save(fig, f'latch_step{step}')
for s in (1, 2):
    latch(s)
b = []
b.append(box(200, 30, 190, 280, 'white', GREEN, rx=4))
b.append(f'<rect x="202" y="130" width="186" height="178" fill="{GREEN_F}"/>')
b.append(f'<line x1="200" y1="255" x2="390" y2="255" stroke="{RED}" stroke-width="3" stroke-dasharray="8,5"/>')
b.append(text(295, 290, 'reserve', 25, RED, weight='bold'))
b.append(text(295, 190, 'tank', 26, '#047857', weight='bold'))
b.append(arrow(110, 115, 196, 160, GREEN, marker='ahg'))
b += lines(10, 50, ['credit:', 'damping'], 25, '#047857', anchor='start', weight='bold')
b.append(arrow(394, 160, 480, 115, AMBER))
b += lines(440, 50, ['debit:', 'active work'], 25, AMBER, anchor='start', weight='bold')
b += lines(420, 205, ['at the reserve,', 'no active work'], 24, RED, anchor='start', weight='bold')
b.append(box(140, 340, 400, 62, SLATE_F, SLATE))
b.append(text(340, 380, 'force: its own sensor', 25))
svg('tank', 640, 410, b)
fig, ax = plt.subplots(figsize=(3.6, 1.3))
for i, (lab, w, key, col) in enumerate([('Clamped', 94.86832981, 'f_clamp', RED), ('Free', 20.70196678, 'f_free', GREEN)]):
    ax.barh(i, w, color=col, height=0.62)
    ax.text(w + 1.5, i, f"{ {'f_clamp': '94.9', 'f_free': '20.7'}[key]} N", va='center', fontsize=12, color=col, fontweight='bold')
    ax.text(-1.5, i, lab, ha='right', va='center', fontsize=12, fontweight='bold')
ax.axvline(50.0, color=RED, ls='--', lw=2)
ax.text(50.0 + 1.5, 1.5, '50 N criterion', color=RED, fontsize=12, fontweight='bold')
ax.set_xlim(0, 125)
ax.set_ylim(-0.5, 1.95)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Peak force at 0.10 m/s (N)', fontsize=12)
ax.tick_params(labelsize=11)
save(fig, 'hand_force')
fig, ax = plt.subplots(figsize=(5.4, 2.7))
names = list({'UR10e': (0.28, 316.0, 230.0), 'Franka (FE)': (0.4, 289.0, 228.0), 'KUKA LBR iiwa 14': (0.36, 368.0, 279.0), 'Techman TM5': (0.28, 427.0, 292.0)})
short = {'UR10e': 'UR10e', 'Franka (FE)': 'Franka', 'KUKA LBR iiwa 14': 'KUKA iiwa', 'Techman TM5': 'TM5'}
xs = np.arange(len(names))
cvals = [316.0, 289.0, 368.0, 427.0]
nvals = [230.0, 228.0, 279.0, 292.0]
ax.bar(xs - 0.2, cvals, width=0.38, color=NAVY)
ax.bar(xs + 0.2, nvals, width=0.38, color='#93C5FD')
for x, c, n in zip(xs, cvals, nvals):
    ax.text(x - 0.2, c + 6, f'{c:.0f}', ha='center', fontsize=11, color=NAVY, fontweight='bold')
    ax.text(x + 0.2, n + 6, f'{n:.0f}', ha='center', fontsize=11, color='#1D4ED8', fontweight='bold')
ax.axhline(280.0, color=RED, ls='--', lw=2)
ax.text(3.42, 280.0 + 12, 'set for\n280 N', color=RED, ha='left', va='bottom', fontsize=11, fontweight='bold')
ax.set_xticks(xs)
ax.set_xticklabels(['UR10e\n0.28 m/s', 'Franka\n0.40 m/s', 'KUKA iiwa\n0.36 m/s', 'TM5\n0.28 m/s'], fontsize=10.5)
ax.set_xlim(-0.5, 4.4)
ax.set_ylim(0, 480)
ax.set_ylabel('Measured force (N)', fontsize=11)
ax.tick_params(axis='y', labelsize=10)
save(fig, 'kirschner')
fig, ax = plt.subplots(figsize=(5.9, 1.8))
parts = [('person walks', 478.5875862, RED), ('robot reacts', 128.256, AMBER), ('robot stops', 79.44827586, SLATE)]
x = 0
for lab, w, col in parts:
    ax.barh(0, w, left=x, color=col, height=0.6, edgecolor='white')
    if w > 200:
        ax.text(x + w / 2, 0, f'{lab}\n{w:.0f} mm', ha='center', va='center', fontsize=10.5, color='white', fontweight='bold')
    else:
        ax.text(x + w / 2, 0.42 if lab.endswith('reacts') else -0.42, f'{lab} {w:.0f} mm', ha='center', va='bottom' if lab.endswith('reacts') else 'top', fontsize=10.5, color=col, fontweight='bold')
    x += w
ax.text(x + 10, 0, f'{x:.0f} mm', va='center', fontsize=12, fontweight='bold', color=TXT)
ax.set_xlim(0, 800)
ax.set_ylim(-0.9, 0.9)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Separation the arm must keep, before sensor and position margins (mm)', fontsize=10.5)
ax.tick_params(labelsize=10)
save(fig, 'ssm')
b = []
rows = [('Model fidelity', 'observer: predicted vs measured'), ('Bounded push', 'observer against the force bound'), ('Torque available', 'current shunts, saturation flag'), ('Solver on time', 'watchdog and cycle counter'), ('Short tick', 'bus timestamps; 1.501 mm drift')]
b.append(text(155, 30, 'Premise', 25, '#047857', weight='bold'))
b.append(text(615, 30, 'Detector, every tick', 25, RED, weight='bold'))
for i, (p, d) in enumerate(rows):
    y = 48 + i * 72
    b.append(box(10, y, 290, 58, GREEN_F, GREEN))
    b.append(text(155, y + 38, p, 24, weight='bold'))
    b.append(arrow(302, y + 29, 338, y + 29))
    b.append(box(340, y, 550, 58, 'white', '#94A3B8'))
    b.append(text(615, y + 38, d, 24))
svg('premises', 900, 415, b)
fig, ax = plt.subplots(figsize=(5.9, 1.75))
x = 0
cols = [SLATE, '#64748B', '#94A3B8', AMBER, '#64748B', '#CBD5E1']
for (lab, w), col in zip([('state acquisition', 30.0), ('proposal and lease check', 15.0), ('tracking check', 10.0), ('QP solve', 150.0), ('output write', 15.0), ('reserve', 30.0)], cols):
    ax.barh(0, w, left=x, color=col, height=0.55, edgecolor='white')
    if w >= 100:
        ax.text(x + w / 2, 0, f'{lab} {w:.0f}', ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    x += w
ax.text(0, 0.36, 'state in 30, lease 15, tracking 10', fontsize=10, color=SLATE, va='bottom')
ax.text(205, 0.36, 'output 15, reserve 30', fontsize=10, color=SLATE, va='bottom')
ax.axvline(400.0, color=RED, ls='--', lw=2)
ax.text(400.0 - 4, -0.42, 'deadline 400 µs', color=RED, ha='right', fontsize=11, fontweight='bold', va='center')
ax.plot([135.0], [-0.42], '^', color=NAVY, ms=8)
ax.text(135.0 + 7, -0.42, 'unloaded run 135', color=NAVY, fontsize=10, va='center')
ax.text(250.0 + 6, 0, 'declared WCET 250', fontsize=11, fontweight='bold', va='center')
ax.set_xlim(0, 420)
ax.set_ylim(-0.65, 0.65)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Microseconds within one 1 ms tick', fontsize=11)
ax.tick_params(labelsize=10)
save(fig, 'cycle')
RUNGS = [('PROJECT', ['fresh proposal,', 'correction exists'], 'least change', GREEN, GREEN_F), ('HOLD', ['lease lost,', 'standing still'], 'powered hold', GREEN, GREEN_F), ('STOP', ['lease lost or', 'breach, moving'], 'resident stop', AMBER, '#FEF3C7'), ('INHIBIT', ['MCU fails', '(watchdog)'], 'STO + brakes', RED, RED_F)]

def ladder(step):
    b = []
    b.append(text(155, 28, 'Failed check, state', 23, TXT, weight='bold'))
    b.append(text(640, 28, 'Response', 23, TXT, weight='bold'))
    for i, (name, cond, resp, col, fill) in enumerate(RUNGS):
        if i >= step:
            continue
        y = 44 + i * 90
        b.append(box(10, y, 290, 74, 'white', '#94A3B8'))
        b += lines(155, y + 31, cond, 22, gap=26)
        mk = 'ahr' if col == RED else 'ahg' if col == GREEN else 'ah'
        b.append(arrow(302, y + 37, 346, y + 37, col, marker=mk))
        b.append(box(348, y, 170, 74, fill, col))
        b.append(text(433, y + 46, name, 25, col if col != GREEN else '#047857', weight='bold'))
        b.append(text(535, y + 46, resp, 23, anchor='start'))
    if step >= 4:
        b.append(f'<line x1="5" y1="{44 + 3 * 90 - 8}" x2="795" y2="{44 + 3 * 90 - 8}" stroke="{RED}" stroke-width="2" stroke-dasharray="6,4"/>')
        b.append(text(795, 44 + 3 * 90 + 96, 'below the line: hardware, not firmware', 21, RED, anchor='end'))
    svg(f'ladder_step{step}', 800, 420, b)
for s in (1, 2, 3, 4):
    ladder(s)
fig, ax = plt.subplots(figsize=(5.9, 2.3))
T = 0.93125
tj = 2.0 / 7.111111111
tt = np.linspace(0, T, 300)
vel = []
for ti in tt:
    if ti < tj:
        vel.append(1.3 - 0.5 * 7.111111111 * ti ** 2)
    elif ti < T - tj:
        vel.append(1.3 - 0.5 * 2.0 * tj - 2.0 * (ti - tj))
    else:
        vel.append(0.5 * 7.111111111 * (T - ti) ** 2)
ax.plot(tt * 1000.0, vel, color=GREEN, lw=3.5)
ax.text(560, 0.95, 'STOP: 931.25 ms, 605.31 mm', color='#047857', fontsize=12, fontweight='bold')
ax.plot([0, 1200], [1.3, 1.3], color=RED, lw=2.5, ls='--')
ax.text(1190, 1.38, 'STO alone: no braking torque, the base coasts', color=RED, fontsize=11.5, fontweight='bold', ha='right')
ax.axvspan(*(30.0, 80.0), color=AMBER, alpha=0.25)
ax.text(80.0 + 8, 0.1, 'spring brakes set\n30 to 80 ms after STO', fontsize=11.5, color='#92400E')
ax.set_xlim(0, 1200)
ax.set_ylim(0, 1.6)
ax.set_xlabel('Time after the fault (ms)', fontsize=11)
ax.set_ylabel('Base speed (m/s)', fontsize=11)
ax.tick_params(labelsize=10)
save(fig, 'stop_sto')
b = []
ev = [('Another car strikes a pedestrian', SLATE), ('Robotaxi hits her; emergency stop', SLATE), ('Classifier: side collision', AMBER), ('Pullover; lane read as clear', RED), ('Dragged 6.1 m, up to 3.4 m/s', RED)]
b.append(f'<line x1="30" y1="30" x2="30" y2="{30 + 4 * 66}" stroke="{SLATE}" stroke-width="3"/>')
for i, (lab, col) in enumerate(ev):
    y = 30 + i * 66
    b.append(f'<circle cx="30" cy="{y}" r="13" fill="{col}"/>')
    b.append(text(60, y + 9, lab, 26, col if col != SLATE else TXT, anchor='start', weight='bold' if col == RED else 'normal'))
svg('cruise', 520, 320, b)
b = []
b.append(text(120, 32, 'Therac-25', 27, weight='bold'))
b.append(box(20, 60, 200, 70, GREEN_F, GREEN))
b.append(text(120, 104, 'software', 25))
b.append(arrow(120, 130, 120, 250))
b.append(text(132, 198, 'nothing', 24, RED, anchor='start', weight='bold'))
b.append(text(132, 224, 'else', 24, RED, anchor='start', weight='bold'))
b.append(box(20, 252, 200, 66, SLATE_F, SLATE, rx=0))
b.append(text(120, 294, 'beam', 25))
b.append('<line x1="260" y1="15" x2="260" y2="325" stroke="#CBD5E1" stroke-width="2"/>')
b.append(text(450, 32, 'Running machine', 27, weight='bold'))
b.append(box(285, 60, 170, 70, GREEN_F, GREEN))
b.append(text(370, 104, 'enforcer', 25))
b.append(box(470, 60, 175, 70, RED_F, RED))
b.append(text(557, 104, 'floor', 25, '#991B1B'))
b.append(arrow(370, 130, 430, 250, GREEN, marker='ahg'))
b.append(arrow(557, 130, 500, 250, RED, marker='ahr'))
b.append(box(365, 252, 200, 66, SLATE_F, SLATE, rx=0))
b.append(text(465, 294, 'drives', 25))
svg('synthesis', 660, 330, b)
print('\nfigures written')
