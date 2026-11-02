#!/usr/bin/env python3
"""Data plots for Lecture 4b (Data Engineering, Part 2: Delivery and Drift).

Every plotted value and every number in a label comes from calcs.py, which
reads MLSysIM (the delivery and drift Scenarios.*, Systems.Storage, Ops,
Literature); labels print the recorded strings with val(). Bars are drawn at
the scenario's exact value, so a label and its bar cannot disagree.

Typed here, because MLSysIM does not yet hold them (the chapter types them
too): the latency ladder's DRAM 100 ns, remote SSD 2 ms, and object store
20 ms against a 1 ms line (the chapter's Sec. 4.10 margin figure); the tier
bandwidth ranges for the in-memory cache (20 to 50 GB/s), the object store's
high end (500 MB/s per connection), and archive (10 to 50 MB/s) from table 8;
and the straggler batch times, seven workers at 12 ms and one at 180 ms
(Sec. 4.10.5). Axis limits and label positions are layout.

Diagrams (drift types) are SVG written by this script.

Run from this folder: python3 make_figs.py
"""
from types import SimpleNamespace
from mlsysim.core.series_loader import load_series
import mlsysim
import sys
from math import erf, sqrt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
import numpy as np
from mlsysim.core.series_loader import load_series
out = Path(__file__).parent
C = sp.C
plt = sp.plt

def bars_lines(path, rows, step=None, lines=(), log_x=True, xlim=None, xlabel=None, size=sp.SIZE, note=None, xticks=None, fs=13, tick_fs=None):
    """Horizontal bars with direct value labels and vertical reference lines.

    rows: (label, value, color, value_text[, "in"]); a value may be a
    (low, high) pair, drawn as a range bar.
    lines: (x, text, color, text_y[, visible]); hidden lines keep their
    label in the layout (alpha 0) so the figure size does not change.
    fs sets bar and value labels; tick_fs, when given, sets the tick labels,
    the axis label, and the reference-line labels (for plots drawn small)."""
    fig, ax = plt.subplots(figsize=size)
    n = len(rows)
    for i, r in enumerate(rows):
        label, v, ck, txt = r[:4]
        inside = len(r) > 4 and r[4] == 'in'
        lo, hi = v if isinstance(v, tuple) else (xlim[0] if log_x else 0, v)
        y = n - 1 - i
        shown = step is None or i < step
        ax.barh(y, hi - lo, left=lo, color=C[ck], height=0.62, alpha=1.0 if shown else 0.0)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=fs, color=C['text'], fontweight='bold')
        if inside:
            ax.text(hi / 1.25 if log_x else hi * 0.97, y, txt, va='center', ha='right', fontsize=fs, color='white', fontweight='bold', alpha=1.0 if shown else 0.0)
        else:
            ax.text(hi * 1.18 if log_x else hi + (xlim[1] if xlim else hi) * 0.015, y, txt, va='center', fontsize=fs, color=C[ck], fontweight='bold', alpha=1.0 if shown else 0.0)
    for ln in lines:
        x, text, ck, ty = ln[:4]
        vis = ln[4] if len(ln) > 4 else True
        ax.axvline(x, color=C[ck], lw=2, ls='--', alpha=1.0 if vis else 0.0)
        ax.text(x * (1.12 if log_x else 1.0) if log_x else x + xlim[1] * 0.01, ty, text, color=C[ck], fontsize=tick_fs or 11, fontweight='bold', va='center', transform=ax.get_xaxis_transform(), alpha=1.0 if vis else 0.0)
    ax.set_yticks([])
    if log_x:
        ax.set_xscale('log')
        ax.minorticks_off()
    if xlim:
        ax.set_xlim(*xlim)
    if xticks:
        ax.set_xticks([t for t, _ in xticks])
        ax.set_xticklabels([s for _, s in xticks])
    ax.set_ylim(-0.6, n - 0.4)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=tick_fs) if tick_fs else ax.set_xlabel(xlabel)
    if tick_fs:
        ax.tick_params(axis='x', labelsize=tick_fs)
    if note:
        ax.text(1.0, -0.42, note, transform=ax.transAxes, ha='right', fontsize=9, color=C['neutral'])
    sp.save(fig, path)

def steps(stem, rows, **kw):
    for k in range(1, len(rows) + 1):
        bars_lines(out / f'{stem}_step{k}.svg', rows, step=k, **kw)
bars_lines(out / 'transfer_time.svg', [('10 Gb/s link', 222.2222222, 'danger', '9.3 days'), ('100 Gb/s link', 22.22222222, 'neutral', '22 h')], lines=[(20, '20 h training run', 'data', 1.06)], log_x=False, xlim=(0, 330), size=(4.0, 1.35), xlabel='Time to move 1 PB (hours)')

def price_trend():
    rows = load_series('vol1/04_data_engineering/data/storage_price_per_tb.csv').entries
    series = [('dram', 'DRAM', C['danger']), ('ssd', 'Solid-state drive', C['data']), ('hdd', 'Hard disk', C['memory'])]
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.75))
        label_y = {'dram': 10000.0, 'ssd': 150.0, 'hdd': 1.5}
        for key, name, col in series:
            pts = [(r['year'], r[key]) for r in rows if r[key] is not None]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], '-', lw=2.4, color=col)
            ax.text(pts[-1][0] + 1.5, label_y[key], name, color=col, fontsize=11, fontweight='bold', va='center', ha='left')
        first = next((r for r in rows if r['hdd'] is not None))
        last = rows[-1]
        a = 1.0 if k == 2 else 0.0
        ax.annotate('\\$88B per TB, 1956', xy=(first['year'], first['hdd']), xytext=(1960, 300000.0), fontsize=11, fontweight='bold', color=C['memory'], alpha=a, arrowprops=dict(arrowstyle='-', color=C['memory'], lw=1, alpha=a))
        ax.annotate('\\$10.6 per TB, 2023', xy=(last['year'], last['hdd']), xytext=(1988, 0.1), fontsize=11, fontweight='bold', color=C['memory'], alpha=a, arrowprops=dict(arrowstyle='-', color=C['memory'], lw=1, alpha=a))
        ax.text(1957, 300.0, '2023: DRAM 42× SSD,\nSSD 2.4× hard disk', fontsize=11, fontweight='bold', color=C['text'], alpha=a, va='center')
        ax.set_yscale('log')
        ax.set_xlim(1955, 2048)
        ax.set_ylim(0.01, 1e+17)
        ax.set_xticks([1960, 1980, 2000, 2020])
        ax.set_yticks([1, 100000.0, 10000000000.0, 1000000000000000.0])
        ax.set_yticklabels(['\\$1', '\\$100K', '\\$10B', '\\$1,000T'])
        ax.minorticks_off()
        ax.grid(axis='y', color=C['grid'], linewidth=0.6)
        ax.set_ylabel('Price per TB\n(2020 US\\$, log)', fontsize=11)
        ax.text(1.0, -0.2, 'Data: Our World in Data (McCallum price records)', transform=ax.transAxes, ha='right', fontsize=9, color=C['neutral'])
        sp.save(fig, out / f'price_trend_step{k}.svg')
price_trend()
bars_lines(out / 'latency_ladder.svg', [('DRAM', 100.0 * 1e-06, 'memory', '100 ns'), ('Local NVMe', 0.1, 'memory', '100 µs', 'in'), ('Remote SSD', 2, 'danger', '2 ms'), ('Object store', 20, 'danger', '20 ms')], lines=[(1, '1 ms', 'danger', 1.07)], xlim=(3e-05, 2000.0), size=(1.75, 1.45), xlabel='Access latency (log)', fs=12, tick_fs=12, xticks=[(0.0001, '100 ns'), (100, '100 ms')])
bars_lines(out / 'tier_bandwidth.svg', [('Memory cache', (20000, 50000), 'memory', '20–50 GB/s'), ('Local NVMe', (5000.0, 7000.0), 'memory', '5–7 GB/s'), ('Object store', (100, 500), 'danger', f'{0.1:.1f}–0.5 GB/s'), ('Archive', (10, 50), 'danger', '10–50 MB/s')], xlim=(5, 3000000.0), size=(1.55, 1.45), xlabel='Bandwidth (MB/s, log)', fs=12, tick_fs=12, xticks=[(10, '10'), (10000.0, '10,000')])
bars_lines(out / 'appetite.svg', [('Arithmetic ceiling', 12682, 'compute', '12,682 img/s'), ('Sustained (chapter)', 3200.0, 'compute', '3,200 img/s  (25%)'), ('Measured (NVIDIA)', 2470.0, 'neutral', '2,470 img/s  (19%)')], log_x=False, xlim=(0, 19000), size=(5.6, 1.75), xlabel='Images per second')
steps('decoded_bytes', [('JPEG, 150 KB', 480.0, 'compute', '480 MB/s  (48% lost)'), ('Decoded FP32, 602 KB', 1926.7584, 'danger', '1.9 GB/s  (87% lost)')], lines=[(250, 'Supply: 250 MB/s', 'neutral', 1.07)], log_x=False, xlim=(0, 3000), size=(6.4, 1.3), fs=15, tick_fs=14, xlabel='Storage bandwidth needed at 3,200 img/s (MB/s)')
supply_rows = [('Needed at 3,200 img/s', 480.0, 'compute', '480 MB/s'), ('Object store, 1 connection', 100, 'neutral', '100 MB/s'), ('Attached block volume', 250, 'neutral', '250 MB/s'), ('SATA SSD', 500, 'memory', '500 MB/s'), ('Local NVMe SSD', 5000.0, 'memory', '5–7 GB/s', 'in'), ('Needed at the ceiling', 1902.3, 'danger', '1.9 GB/s'), ('Needed, eight A100s', 3840.0, 'danger', '3.8 GB/s', 'in')]
for k in (1, 2, 3):
    bars_lines(out / f'storage_supply_step{k}.svg', supply_rows, step=4 + k, lines=[(15414.0672, 'decoded:\n15.4 GB/s', 'danger', 1.06, k == 3)], xlim=(30, 100000.0), size=(5.8, 3.0), xlabel='Sequential read bandwidth (MB/s, log scale)', xticks=[(100, '100'), (1000, '1,000'), (10000, '10,000')])

def choke():
    per_worker = 375
    rate = 3200.0
    crossing, whole = (8.533333333, 9)
    w = np.arange(1, 17)
    loader = np.minimum(per_worker * w, 3500)
    train = np.minimum(per_worker * w, rate)
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    ax.plot(np.concatenate(([crossing], w[w >= whole])), np.concatenate(([rate], loader[w >= whole])), '--', color=C['neutral'], lw=1.3)
    ax.text(12.2, 3680, 'loader capacity', color=C['neutral'], fontsize=11, fontweight='bold', va='center')
    ax.plot(w, train, 'o-', color=C['memory'], lw=2.4, ms=6, zorder=4)
    ax.axhline(rate, color=C['danger'], lw=2, ls='--', zorder=3)
    ax.text(16.8, 2750, 'Accelerator\n3,200 img/s', color=C['danger'], fontsize=11, fontweight='bold', va='center')
    ax.fill_between(w, 0, train, color=C['memory_l'], alpha=0.5)
    ax.axvline(crossing, color=C['text'], lw=1, ls=':')
    ax.text(8.75, 600, f'about {whole} workers', fontsize=11, fontweight='bold', color=C['text'])
    ax.text(1.0, 2600, 'Host decode binds', fontsize=11, va='center', fontweight='bold', color=C['memory'])
    ax.text(4.5, 1150, '375 img/s\nper worker', fontsize=11, va='center', fontweight='bold', color=C['memory'])
    ax.text(11, 2300, 'Accelerator binds', fontsize=11, fontweight='bold', color=C['danger'])
    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(0, 4000)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_yticks([0, 1000, 2000, 3000])
    ax.set_yticklabels(['0', '1,000', '2,000', '3,000'])
    ax.set_xlabel('Loader worker processes')
    ax.set_ylabel('Images per second', fontsize=11)
    ax.grid(color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / 'dataloader.svg')
choke()
steps('format_bytes', [('Row format (CSV)', 100, 'neutral', 'reads 100 of 100 columns'), ('Columnar (Parquet)', 100 * 0.2, 'data', 'reads 20 columns: 5× less')], log_x=False, xlim=(0, 175), size=(6.0, 1.1), fs=15, tick_fs=14, xlabel='Share of the table read from disk (%)')

def profiler_bars():
    rows = [('Default', {'step': 132827.0, 'DataLoader': 77528.0, 'dl_pct': float(58.37)}), ('4 workers', {'step': 76170.0, 'DataLoader': 768.0, 'dl_pct': float(1.01)})]
    fig, ax = plt.subplots(figsize=(4.6, 1.1))
    for i, (lab, v) in enumerate(rows):
        y = 1 - i
        dl, rest = (v['DataLoader'] / 1000, (v['step'] - v['DataLoader']) / 1000)
        ax.barh(y, dl, color=C['danger'], height=0.6)
        ax.barh(y, rest, left=dl, color=C['neutral_l'], height=0.6)
        ax.text(-3, y, lab, ha='right', va='center', fontsize=15, color=C['text'])
        ax.text(dl + rest + 2, y, f"{v['step'] / 1000:.1f} ms", va='center', fontsize=15, fontweight='bold', color=C['text'])
        if dl > 20:
            ax.text(dl / 2, y, f"DataLoader {v['dl_pct']:.1f}%", ha='center', va='center', fontsize=14, fontweight='bold', color='white')
            ax.text(dl + rest / 2, y, 'rest of step', ha='center', va='center', fontsize=14, color=C['text'])
        else:
            ax.text(dl + 2, y, f"DataLoader {v['dl_pct']:.1f}%", ha='left', va='center', fontsize=14, color=C['text'])
    ax.set_xlim(0, 160)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([0, 50, 100, 150])
    ax.tick_params(axis='x', labelsize=14)
    ax.set_xlabel('Average step time (ms)', fontsize=14)
    sp.save(fig, out / 'profiler_steps.svg')
profiler_bars()
bars_lines(out / 'codec_minutes.svg', [('gzip', 13.88888889, 'danger', '13.9 min'), ('Snappy', 3.333333333, 'data', '3.3 min')], log_x=False, xlim=(0, 24), size=(3.4, 1.4), xlabel='CPU decompression, 100 GB (min)')
bars_lines(out / 'stragglers.svg', [('Workers 1 to 7', 12, 'data', '12 ms each'), ('Worker 8', 180, 'danger', '180 ms')], log_x=False, xlim=(0, 260), size=(3.6, 1.4), xlabel='Time to load one batch (ms)')
xs = np.linspace(-50, -10, 400)
train = np.array([0.0004290603332, 0.000455559267, 0.0004835680694, 0.0005131645068, 0.0005444298425, 0.000577448964, 0.0006123105141, 0.0006491070242, 0.0006879350518, 0.0007288953197, 0.0007720928588, 0.0008176371539, 0.000865642292, 0.0009162271136, 0.0009695153663, 0.001025635862, 0.001084722636, 0.001146915107, 0.001212358245, 0.001281202733, 0.001353605138, 0.001429728081, 0.001509740409, 0.001593817371, 0.001682140788, 0.001774899239, 0.001872288229, 0.001974510377, 0.00208177559, 0.002194301247, 0.00231231238, 0.002436041851, 0.002565730538, 0.002701627514, 0.002843990226, 0.002993084674, 0.003149185589, 0.00331257661, 0.003483550458, 0.003662409107, 0.003849463953, 0.004045035983, 0.004249455935, 0.004463064459, 0.004686212274, 0.004919260316, 0.005162579884, 0.005416552782, 0.005681571454, 0.005958039112, 0.006246369855, 0.006546988789, 0.006860332129, 0.007186847302, 0.007526993035, 0.007881239435, 0.008250068064, 0.008633971998, 0.00903345588, 0.009449035959, 0.009881240116, 0.01033060788, 0.01079769045, 0.01128305064, 0.01178726292, 0.01231091331, 0.01285459939, 0.01341893019, 0.01400452612, 0.01461201889, 0.01524205135, 0.01589527742, 0.01657236188, 0.01727398024, 0.01800081854, 0.01875357315, 0.01953295054, 0.02033966704, 0.02117444859, 0.02203803042, 0.02293115681, 0.02385458068, 0.02480906332, 0.025795374, 0.02681428956, 0.02786659403, 0.02895307817, 0.03007453906, 0.03123177958, 0.03242560793, 0.03365683714, 0.03492628448, 0.03623477092, 0.03758312057, 0.03897216003, 0.04040271782, 0.04187562365, 0.04339170786, 0.04495180062, 0.04655673131, 0.04820732773, 0.04990441541, 0.05164881678, 0.05344135043, 0.05528283031, 0.05717406487, 0.05911585628, 0.06110899954, 0.06315428162, 0.06525248061, 0.06740436479, 0.06961069175, 0.07187220747, 0.0741896454, 0.07656372551, 0.07899515337, 0.08148461919, 0.08403279688, 0.08664034305, 0.0893078961, 0.09203607525, 0.09482547955, 0.09767668691, 0.1005902532, 0.1035667112, 0.1066065698, 0.1097103129, 0.1128783984, 0.1161112577, 0.1194092942, 0.1227728829, 0.1262023691, 0.1296980678, 0.1332602629, 0.136889206, 0.1405851159, 0.1443481777, 0.1481785422, 0.1520763246, 0.1560416046, 0.1600744249, 0.1641747912, 0.1683426711, 0.1725779936, 0.1768806489, 0.1812504873, 0.1856873189, 0.1901909135, 0.1947609997, 0.1993972647, 0.204099354, 0.208866871, 0.2136993768, 0.2185963902, 0.223557387, 0.2285818004, 0.2336690207, 0.2388183953, 0.2440292287, 0.2493007826, 0.2546322758, 0.260022885, 0.2654717442, 0.2709779455, 0.2765405393, 0.2821585345, 0.2878308992, 0.2935565608, 0.2993344071, 0.3051632862, 0.3110420074, 0.3169693419, 0.3229440238, 0.32896475, 0.3350301821, 0.3411389463, 0.3472896349, 0.3534808071, 0.3597109898, 0.3659786788, 0.37228234, 0.3786204101, 0.3849912983, 0.3913933869, 0.397825033, 0.4042845695, 0.4107703066, 0.4172805329, 0.4238135166, 0.4303675076, 0.4369407381, 0.4435314244, 0.4501377683, 0.4567579587, 0.4633901728, 0.4700325779, 0.4766833328, 0.4833405891, 0.4900024933, 0.4966671876, 0.5033328124, 0.5099975067, 0.5166594109, 0.5233166672, 0.5299674221, 0.5366098272, 0.5432420413, 0.5498622317, 0.5564685756, 0.5630592619, 0.5696324924, 0.5761864834, 0.5827194671, 0.5892296934, 0.5957154305, 0.602174967, 0.6086066131, 0.6150087017, 0.6213795899, 0.62771766, 0.6340213212, 0.6402890102, 0.6465191929, 0.6527103651, 0.6588610537, 0.6649698179, 0.67103525, 0.6770559762, 0.6830306581, 0.6889579926, 0.6948367138, 0.7006655929, 0.7064434392, 0.7121691008, 0.7178414655, 0.7234594607, 0.7290220545, 0.7345282558, 0.739977115, 0.7453677242, 0.7506992174, 0.7559707713, 0.7611816047, 0.7663309793, 0.7714181996, 0.776442613, 0.7814036098, 0.7863006232, 0.791133129, 0.795900646, 0.8006027353, 0.8052390003, 0.8098090865, 0.8143126811, 0.8187495127, 0.8231193511, 0.8274220064, 0.8316573289, 0.8358252088, 0.8399255751, 0.8439583954, 0.8479236754, 0.8518214578, 0.8556518223, 0.8594148841, 0.863110794, 0.8667397371, 0.8703019322, 0.8737976309, 0.8772271171, 0.8805907058, 0.8838887423, 0.8871216016, 0.8902896871, 0.8933934302, 0.8964332888, 0.8994097468, 0.9023233131, 0.9051745205, 0.9079639247, 0.9106921039, 0.913359657, 0.9159672031, 0.9185153808, 0.9210048466, 0.9234362745, 0.9258103546, 0.9281277925, 0.9303893082, 0.9325956352, 0.9347475194, 0.9368457184, 0.9388910005, 0.9408841437, 0.9428259351, 0.9447171697, 0.9465586496, 0.9483511832, 0.9500955846, 0.9517926723, 0.9534432687, 0.9550481994, 0.9566082921, 0.9581243763, 0.9595972822, 0.96102784, 0.9624168794, 0.9637652291, 0.9650737155, 0.9663431629, 0.9675743921, 0.9687682204, 0.9699254609, 0.9710469218, 0.972133406, 0.9731857104, 0.974204626, 0.9751909367, 0.9761454193, 0.9770688432, 0.9779619696, 0.9788255514, 0.979660333, 0.9804670495, 0.9812464269, 0.9819991815, 0.9827260198, 0.9834276381, 0.9841047226, 0.9847579486, 0.9853879811, 0.9859954739, 0.9865810698, 0.9871454006, 0.9876890867, 0.9882127371, 0.9887169494, 0.9892023096, 0.9896693921, 0.9901187599, 0.990550964, 0.9909665441, 0.991366028, 0.9917499319, 0.9921187606, 0.992473007, 0.9928131527, 0.9931396679, 0.9934530112, 0.9937536301, 0.9940419609, 0.9943184285, 0.9945834472, 0.9948374201, 0.9950807397, 0.9953137877, 0.9955369355, 0.9957505441, 0.995954964, 0.996150536, 0.9963375909, 0.9965164495, 0.9966874234, 0.9968508144, 0.9970069153, 0.9971560098, 0.9972983725, 0.9974342695, 0.9975639581, 0.9976876876, 0.9978056988, 0.9979182244, 0.9980254896, 0.9981277118, 0.9982251008, 0.9983178592, 0.9984061826, 0.9984902596, 0.9985702719, 0.9986463949, 0.9987187973, 0.9987876418, 0.9988530849, 0.9989152774, 0.9989743641, 0.9990304846, 0.9990837729, 0.9991343577, 0.9991823628, 0.9992279071, 0.9992711047, 0.9993120649, 0.999350893, 0.9993876895, 0.999422551, 0.9994555702, 0.9994868355, 0.9995164319, 0.9995444407, 0.9995709397])
recent = np.array([0.0008641652091, 0.0009146708645, 0.0009678761658, 0.001023909771, 0.001082905555, 0.001145002774, 0.001210346229, 0.001279086428, 0.001351379759, 0.001427388661, 0.00150728179, 0.001591234201, 0.001679427515, 0.001772050105, 0.001869297267, 0.001971371402, 0.002078482199, 0.002190846807, 0.002308690027, 0.002432244483, 0.002561750811, 0.002697457835, 0.002839622748, 0.00298851129, 0.003144397929, 0.003307566034, 0.003478308051, 0.003656925672, 0.00384373001, 0.004039041761, 0.004243191369, 0.004456519184, 0.00467937562, 0.004912121304, 0.005155127224, 0.005408774866, 0.005673456352, 0.00594957457, 0.00623754329, 0.006537787287, 0.006850742441, 0.007176855842, 0.007516585875, 0.007870402305, 0.008238786351, 0.008622230741, 0.009021239772, 0.009436329342, 0.009868026984, 0.01031687188, 0.01078341487, 0.01126821844, 0.01177185668, 0.0122949153, 0.0128379915, 0.01340169399, 0.01398664283, 0.01459346939, 0.0152228162, 0.01587533684, 0.01655169578, 0.01725256819, 0.01797863981, 0.0187306067, 0.01950917503, 0.02031506085, 0.02114898979, 0.02201169684, 0.02290392597, 0.02382642988, 0.02477996961, 0.02576531418, 0.02678324022, 0.02783453155, 0.02891997873, 0.03004037867, 0.03119653406, 0.03238925297, 0.03361934828, 0.03488763715, 0.03619494044, 0.03754208217, 0.0389298889, 0.04035918907, 0.04183081238, 0.04334558915, 0.04490434959, 0.04650792308, 0.04815713751, 0.04985281846, 0.05159578849, 0.05338686631, 0.055226866, 0.05711659622, 0.05905685932, 0.06104845053, 0.06309215708, 0.06518875733, 0.06733901988, 0.06954370264, 0.07180355196, 0.07411930168, 0.07649167218, 0.07892136949, 0.08140908429, 0.08395549097, 0.0865612467, 0.08922699045, 0.091953342, 0.09474090102, 0.09759024609, 0.1005019337, 0.1034764975, 0.1065144469, 0.1096162666, 0.1127824155, 0.1160133256, 0.1193094012, 0.1226710181, 0.1260985226, 0.1295922307, 0.133152427, 0.1367793643, 0.1404732624, 0.1442343073, 0.1480626508, 0.1519584094, 0.1559216636, 0.1599524574, 0.1640507976, 0.1682166529, 0.1724499536, 0.1767505909, 0.1811184163, 0.1855532413, 0.1900548368, 0.1946229326, 0.1992572171, 0.2039573372, 0.2087228976, 0.2135534605, 0.2184485461, 0.2234076314, 0.228430151, 0.2335154964, 0.2386630163, 0.2438720164, 0.2491417598, 0.2544714667, 0.2598603148, 0.2653074395, 0.2708119341, 0.2763728503, 0.2819891983, 0.2876599473, 0.2933840261, 0.2991603234, 0.3049876886, 0.3108649322, 0.3167908266, 0.3227641068, 0.328783471, 0.3348475817, 0.3409550663, 0.3471045179, 0.3532944969, 0.359523531, 0.365790117, 0.3720927216, 0.3784297825, 0.3847997095, 0.3912008859, 0.3976316694, 0.4040903937, 0.4105753696, 0.4170848862, 0.4236172127, 0.4301705991, 0.4367432782, 0.4433334669, 0.4499393674, 0.4565591689, 0.4631910489, 0.4698331751, 0.4764837063, 0.4831407945, 0.4898025862, 0.496467224, 0.5031328479, 0.5097975972, 0.5164596121, 0.5231170349, 0.5297680117, 0.5364106941, 0.5430432407, 0.5496638184, 0.5562706041, 0.5628617864, 0.5694355665, 0.5759901605, 0.5825238001, 0.5890347343, 0.5955212311, 0.6019815783, 0.6084140855, 0.6148170849, 0.6211889327, 0.6275280107, 0.633832727, 0.6401015177, 0.6463328476, 0.6525252117, 0.6586771359, 0.6647871785, 0.6708539307, 0.6768760178, 0.6828521001, 0.6887808737, 0.6946610714, 0.7004914632, 0.7062708574, 0.711998101, 0.7176720804, 0.7232917219, 0.7288559923, 0.7343638995, 0.7398144923, 0.7452068618, 0.7505401408, 0.7558135044, 0.7610261703, 0.766177399, 0.7712664937, 0.7762928004, 0.7812557082, 0.7861546489, 0.7909890972, 0.7957585704, 0.8004626285, 0.8051008736, 0.8096729499, 0.8141785434, 0.8186173815, 0.8229892326, 0.8272939057, 0.8315312501, 0.8357011545, 0.8398035468, 0.8438383936, 0.8478056994, 0.8517055058, 0.8555378912, 0.8593029701, 0.8630008921, 0.8666318412, 0.8701960352, 0.8736937248, 0.877125193, 0.8804907537, 0.8837907515, 0.8870255603, 0.8901955829, 0.8933012497, 0.896343018, 0.8993213707, 0.9022368161, 0.9050898863, 0.9078811364, 0.9106111437, 0.9132805066, 0.9158898438, 0.9184397931, 0.9209310105, 0.9233641696, 0.9257399599, 0.9280590867, 0.9303222695, 0.9325302413, 0.9346837478, 0.9367835462, 0.9388304045, 0.9408251005, 0.942768421, 0.9446611606, 0.9465041214, 0.9482981116, 0.9500439451, 0.9517424402, 0.9533944194, 0.955000708, 0.9565621338, 0.9580795262, 0.9595537153, 0.9609855315, 0.9623758044, 0.9637253627, 0.9650350331, 0.9663056396, 0.9675380034, 0.968732942, 0.9698912683, 0.9710137909, 0.9721013127, 0.973154631, 0.9741745368, 0.9751618143, 0.9761172406, 0.9770415851, 0.9779356094, 0.9788000667, 0.9796357015, 0.9804432494, 0.9812234364, 0.9819769794, 0.982704585, 0.9834069499, 0.9840847605, 0.9847386926, 0.9853694113, 0.9859775709, 0.9865638145, 0.9871287741, 0.9876730706, 0.9881973133, 0.9887021002, 0.9891880175, 0.9896556402, 0.9901055313, 0.9905382425, 0.9909543135, 0.9913542727, 0.9917386367, 0.9921079103, 0.9924625871, 0.992803149, 0.9931300663, 0.9934437983, 0.9937447925, 0.9940334857, 0.9943103031, 0.9945756594, 0.9948299579, 0.9950735914, 0.9953069422, 0.9955303817, 0.9957442712, 0.9959489618, 0.9961447944, 0.9963321001, 0.9965112, 0.996682406, 0.9968460202, 0.9970023356, 0.9971516362, 0.997294197, 0.9974302841, 0.9975601554, 0.9976840601, 0.9978022394, 0.9979149263, 0.9980223461, 0.9981247165, 0.9982222475, 0.9983151419, 0.9984035956, 0.9984877973, 0.998567929, 0.9986441661, 0.9987166778, 0.9987856267, 0.9988511696, 0.9989134575, 0.9989726354, 0.9990288429, 0.9990822142, 0.9991328783, 0.9991809591, 0.9992265755, 0.9992698418, 0.9993108677, 0.9993497582, 0.9993866142, 0.9994215324, 0.9994546055, 0.9994859222, 0.9995155675, 0.9995436228, 0.9995701659, 0.9995952714, 0.9996190105, 0.9996414512, 0.9996626587, 0.9996826952, 0.99970162, 0.9997194898, 0.9997363587, 0.9997522784, 0.999767298, 0.9997814645, 0.9997948226])
i = int(np.argmax(recent - train))
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    ax.plot(xs, train, color=C['memory'], lw=3)
    ax.text(-25.5, 0.45, 'Training clips', color=C['memory'], fontsize=12, fontweight='bold', ha='left')
    if k >= 2:
        ax.plot(xs, recent, color=C['danger'], lw=3)
        ax.text(-47.5, 0.42, 'Recent clips', color=C['danger'], fontsize=12, fontweight='bold', ha='left')
    if k >= 3:
        ax.plot([xs[i], xs[i]], [train[i], recent[i]], color=C['text'], lw=3.5, solid_capstyle='butt', zorder=6)
        ax.annotate('D = 0.08, the largest gap', xy=(xs[i], (train[i] + recent[i]) / 2), xytext=(-24, 0.12), fontsize=12, fontweight='bold', color=C['text'], arrowprops=dict(arrowstyle='-', color=C['text'], lw=1))
    ax.set_xlim(-50, -10)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Clip loudness (dBFS)')
    ax.set_ylabel('Share of clips\nat most this loud', fontsize=11)
    sp.save(fig, out / f'ks_cdf_step{k}.svg')

def ks_vs_n():
    n = np.logspace(2, 5, 200)
    crit = 1.36 * np.sqrt(2 / n)
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    ax.plot(n, crit, color=C['memory'], lw=3)
    ax.axhline(0.08, color=C['danger'], lw=2, ls='--')
    ax.text(14000.0, 0.088, 'observed D = 0.08', color=C['danger'], fontsize=11, fontweight='bold')
    for sc, key, tx, ty in ((SimpleNamespace(alert_gate=None, coefficient=1.36, critical_value=0.06082104899, flags=True, n_current=1000, n_reference=1000, name='K-S drift test on clip loudness, 1,000 training against 1,000 recent clips', observed_statistic=0.08, scale=0.04472135955, significance=None), '', 1400, 0.091), (SimpleNamespace(alert_gate=None, coefficient=1.36, critical_value=0.006082104899, flags=True, n_current=100000, n_reference=100000, name='K-S drift test on clip loudness, 100,000 clips per sample', observed_statistic=0.08, scale=0.004472135955, significance=None), '_big', 9000, 0.04)):
        nn, c = ({1400: 1000, 9000: 100000}[tx], {1400: 0.06082104899, 9000: 0.006082104899}[tx])
        lab = f"{ {1400: '0.061', 9000: '0.006'}[tx]} at n = { {1400: '1,000', 9000: '100,000'}[tx]}"
        ax.plot([nn], [c], 'o', color=C['text'], ms=7, zorder=5)
        ax.annotate(lab, xy=(nn, c), xytext=(tx, ty), fontsize=11, fontweight='bold', color=C['text'], arrowprops=dict(arrowstyle='-', color=C['text'], lw=1))
    ax.text(300.0, 0.17, 'Flag when D is above the curve', color=C['memory'], fontsize=11, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(100, 120000.0)
    ax.set_ylim(0, 0.2)
    ax.set_xticks([100, 1000, 10000, 100000])
    ax.set_xticklabels(['100', '1,000', '10,000', '100,000'])
    ax.minorticks_off()
    ax.set_xlabel('Clips in each sample, n')
    ax.set_ylabel('Critical value', fontsize=11)
    ax.grid(color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / 'ks_vs_n.svg')
ks_vs_n()

def psi_bins():
    base, rec, terms = ([0.2, 0.2, 0.2, 0.2, 0.2], [0.3, 0.25, 0.2, 0.15, 0.1], [0.04054651081, 0.01115717757, 0.0, 0.01438410362, 0.06931471806])
    names = ['Quietest', '2', '3', '4', 'Loudest']
    x = np.arange(5)
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.7))
        ax.bar(x - 0.2, [100 * b for b in base], 0.38, color=C['memory'])
        ax.bar(x + 0.2, [100 * r for r in rec], 0.38, color=C['danger'])
        ax.text(4.55, 21.5, 'Training', color=C['memory'], fontsize=11, fontweight='bold')
        ax.text(4.55, 15.5, 'Recent', color=C['danger'], fontsize=11, fontweight='bold')
        for j, t in enumerate(terms):
            ax.text(j, 33.5, f'{t:.3f}', ha='center', fontsize=12, fontweight='bold', color=C['text'], alpha=1.0 if k >= 2 else 0.0)
        ax.text(4.55, 33.5, 'PSI terms', ha='left', fontsize=11, color=C['neutral'], alpha=1.0 if k >= 2 else 0.0)
        ax.text(2, 40.5, 'PSI = 0.135: moderate band (0.1 to 0.25)', ha='center', fontsize=12.5, fontweight='bold', color=C['danger'], alpha=1.0 if k >= 3 else 0.0)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlim(-0.8, 5.6)
        ax.set_ylim(0, 44)
        ax.set_yticks([0, 10, 20, 30])
        ax.set_ylabel('Share of clips (%)', fontsize=11)
        ax.set_xlabel('Loudness bin (training quintiles)')
        ax.grid(axis='y', color=C['grid'], linewidth=0.6)
        ax.set_axisbelow(True)
        sp.save(fig, out / f'psi_bins_step{k}.svg')
psi_bins()
FONT = 'font-family="Helvetica, Arial, sans-serif"'
HEX = {'text': '#1A1A2E'}

def svg(path, body, w=900, h=500):
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n<rect width="{w}" height="{h}" fill="white"/>\n' + body + '\n</svg>\n')

def txt(x, y, s, size=20, bold=False, color='#1A1A2E', anchor='middle'):
    b = ' font-weight="bold"' if bold else ''
    return f'<text x="{x}" y="{y}" {FONT} font-size="{size}"{b} fill="{color}" text-anchor="{anchor}" dominant-baseline="central">{s}</text>'

def box(x, y, w, h, fill, stroke, rx=8):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>'

def drift(k):
    panels = [('Covariate shift', 'p(x) changes', 'p(y | x) stays', 'New microphone,\nsame words', 'Compare input\nfeatures', C['memory'], C['memory_l']), ('Label shift', 'p(y) changes', 'p(x | y) stays', 'Flu season: more\npositive cases', 'Compare the\nprediction mix', C['compute'], C['compute_l']), ('Concept drift', 'p(y | x) changes', 'the mapping moves', 'Fraudsters change\ntactics', 'Needs fresh\nlabels', C['danger'], C['danger_l'])]
    s = ''
    for j, (name, what, keep, ex, det, col, light) in enumerate(panels):
        x = 10 + j * 297
        if j >= k:
            s += box(x, 10, 285, 380, '#FAFAFA', '#DDDDDD')
            continue
        s += box(x, 10, 285, 380, 'white', col)
        s += f'<rect x="{x}" y="10" width="285" height="58" rx="8" fill="{col}"/>'
        s += txt(x + 142, 39, name, 27, True, 'white')
        s += txt(x + 142, 100, what, 27, True, col)
        s += txt(x + 142, 135, keep, 22, False, '#555555')
        for m, ln in enumerate(ex.split('\n')):
            s += txt(x + 142, 185 + m * 30, ln, 23)
        s += box(x + 18, 270, 249, 100, light, col)
        dl = det.split('\n')
        for m, ln in enumerate(dl):
            s += txt(x + 142, 320 - (len(dl) - 1) * 15 + m * 30, ln, 23, True, col)
    svg(out / f'drift_step{k}.svg', s, h=400)
for k in (1, 2, 3):
    drift(k)
print('figures written')
