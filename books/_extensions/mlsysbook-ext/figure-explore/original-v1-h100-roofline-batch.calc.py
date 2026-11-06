#!/usr/bin/env python3
"""Lesson data for vol1:fig-hw-acceleration-h100-roofline-lighthouses (batch walk of the classifier head).

Every value comes from MLSysIM, through the same objects the chapter's LEGO cells use:
- H100 FP16 dense peak, INT8 peak, and HBM bandwidth: Hardware.Cloud.H100, as in
  ResNet50Int8Crossing (fp16_peak, bw, ridge_fp16 = fp16_peak / bw).
- Classifier intensity at batch B: Scenarios.ResNet50ClassifierHeadBatch32.variant(batch_size=B)
  .traffic_intensity, i.e. @eq-batch-ai with 2,048 inputs, 1,000 outputs, FP16, one read of
  input and weights and one write of output. The batch-32 stop is asserted equal to
  DenseLayerAnalysisCalc.dense_ai (ResNet50ClassifierHeadH100Roofline.point_intensity), which is
  the value the static figure plots.
- The ridge-crossing batch is the smallest integer B whose intensity clears ridge_fp16,
  found with MLSysIM's batch_for_traffic_intensity and confirmed on both sides.

Pixel geometry: the axes frame of the verified PNG (3315 x 2476) is measured from the raster
(spine centres) and the printed classifier marker's centroid is asserted to coincide with the
batch-32 position computed from the figure's own log axes (x 0.5..4000, y 1..6000).

Run (repository root):
  PYTHONPATH=mlsysim ../MLSysBook/.venv/bin/python <this file>           print lesson data (JSON)
  ... <this file> --check    also assert the factory's embedded D equals it
  ... <this file> --write    rewrite the factory's embedded D block
"""
import json
import math
import re
import sys
from pathlib import Path

from mlsysim import Hardware
from mlsysim.engine.scenarios import Scenarios

ART = Path('/tmp/figures-animation-art-current/assets/vol1--fig-hw-acceleration-h100-roofline-lighthouses-d4348c048b6e.png')
ART_SHA1 = 'd4348c048b6e857332e172a12bf2b40445b174e9'
W, H = 3315, 2476
X_LIM, Y_LIM = (0.5, 4000.0), (1.0, 6000.0)     # ax.set_xlim / ax.set_ylim in the figure cell
FRAME = dict(x0=373.0, x1=3256.0, y0=59.0, y1=2138.0)  # spine centres, re-measured below when the art is present
BATCHES = [32, 1, 256, 512, None, 1024]          # None = the computed first batch past the FP16 ridge


def measure_frame():
    """Re-measure the axes spines and the printed classifier dot from the verified PNG."""
    import hashlib
    import numpy as np
    from PIL import Image
    raw = ART.read_bytes()
    assert hashlib.sha1(raw).hexdigest() == ART_SHA1, 'verified artwork changed; review the mapping'
    a = np.array(Image.open(ART).convert('RGB')).astype(int)
    assert a.shape[:2] == (H, W)
    dark = a.sum(2) < 150
    cols = [x for x in range(W) if dark[:, x].sum() > 1500]
    rows = [y for y in range(H) if dark[y, :].sum() > 2000]
    left, right = [c for c in cols if c < W / 2], [c for c in cols if c > W / 2]
    top, bottom = [r for r in rows if r < H / 2], [r for r in rows if r > H / 2]
    measured = dict(x0=(left[0] + left[-1]) / 2, x1=(right[0] + right[-1]) / 2,
                    y0=(top[0] + top[-1]) / 2, y1=(bottom[0] + bottom[-1]) / 2)
    assert measured == FRAME, measured
    return a


def px(ai, value):
    lx = (math.log10(ai) - math.log10(X_LIM[0])) / (math.log10(X_LIM[1]) - math.log10(X_LIM[0]))
    ly = (math.log10(value) - math.log10(Y_LIM[0])) / (math.log10(Y_LIM[1]) - math.log10(Y_LIM[0]))
    return [round(FRAME['x0'] + lx * (FRAME['x1'] - FRAME['x0']), 2), round(FRAME['y1'] - ly * (FRAME['y1'] - FRAME['y0']), 2)]


def compute():
    h100 = Hardware.Cloud.H100
    fp16_peak = h100.compute.precision_flops['fp16_dense']
    bw = h100.memory.bandwidth
    peak = fp16_peak.to('TFLOP/s').magnitude           # 989
    bw_tb = bw.to('TB/s').magnitude                     # 3.35; TB/s x FLOP/byte = TFLOP/s
    ridge = (fp16_peak / bw).to('flop/byte').magnitude  # ResNet50Int8Crossing.r16
    ridge8 = (h100.compute.precision_flops['int8'] / bw).to('flop/byte').magnitude
    # DenseLayerAnalysisCalc uses compute.peak_flops; the figure uses fp16_dense. They must agree.
    assert h100.compute.peak_flops.to('TFLOP/s').magnitude == peak

    head = Scenarios.ResNet50ClassifierHeadBatch32
    printed = Scenarios.ResNet50ClassifierHeadH100Roofline
    dense_ai = printed.point_intensity.to('flop/byte').magnitude  # DenseLayerAnalysisCalc.dense_ai
    assert head.batch_size == 32 and head.in_features == 2048 and head.out_features == 1000
    assert round(dense_ai, 1) == 30.5 and round(printed.peak_fraction * 100, 1) == 10.3

    cross = math.ceil(head.batch_for_traffic_intensity(ridge))
    ai = lambda b: head.variant(batch_size=b).traffic_intensity
    assert ai(cross - 1) < ridge <= ai(cross), (cross, ai(cross - 1), ai(cross))
    limit = head.in_features * head.out_features / (head.in_features + head.out_features)  # eq-batch-ai as B -> inf

    states = []
    for b in [cross if b is None else b for b in BATCHES]:
        s = head.variant(batch_size=b)
        i = s.traffic_intensity
        att = min(peak, bw_tb * i)
        wmb = s.weight_bytes.to('MB').magnitude
        amb = (s.input_bytes + s.output_bytes).to('MB').magnitude
        assert abs((s.flops.to('flop').magnitude / s.traffic_bytes.to('byte').magnitude) - i) < 1e-9
        states.append(dict(
            batch=b, ai=i, attainable=att, pct=att / peak * 100, bound='compute' if i >= ridge else 'bandwidth',
            mflop=s.flops.to('flop').magnitude / 1e6, weight_mb=wmb, act_mb=amb, xy=px(i, att),
            text=dict(batch=f'{b:,}', ai=f'{i:.1f}', attainable=f'{att:.1f}', pct=f'{att / peak * 100:.1f}',
                      mflop=f'{s.flops.to("flop").magnitude / 1e6:,.1f}', weight_mb=f'{wmb:.2f}', act_mb=f'{amb:.2f}',
                      total_mb=f'{wmb + amb:.2f}')))

    # Endpoints the lesson depends on.
    s32, s1, s256, s512, sx, s1024 = states
    assert s32['ai'] == dense_ai and s32['text']['ai'] == '30.5' and s32['text']['pct'] == '10.3'
    assert s1['text']['ai'] == '1.0' and s1['bound'] == 'bandwidth'
    assert s256['ai'] < ridge and s512['ai'] < ridge and s512['ai'] < 512 / 1.7  # AI is well short of B here
    assert sx['batch'] == 527 and sx['bound'] == 'compute' and sx['attainable'] == peak
    assert s1024['attainable'] == peak and s1024['ai'] > sx['ai'] and s1024['ai'] < ridge8 < limit
    assert [s['bound'] for s in states] == ['bandwidth'] * 4 + ['compute'] * 2
    assert all(abs(s['weight_mb'] - s32['weight_mb']) < 1e-12 for s in states), 'weight bytes do not change with batch'

    return dict(
        art=dict(width=W, height=H, sha1=ART_SHA1, frame=FRAME, x_lim=X_LIM, y_lim=Y_LIM),
        h100=dict(peak=peak, bw=bw_tb, ridge=ridge, ridge_int8=ridge8,
                  text=dict(peak=f'{peak:,.0f}', bw=f'{bw_tb:.2f}', ridge=f'{ridge:.1f}', ridge_int8=f'{ridge8:.1f}')),
        head=dict(inputs=head.in_features, outputs=head.out_features, limit=limit, limit_text=f'{limit:.1f}',
                  printed_ai=dense_ai, cross=cross, below=cross - 1, below_ai_text=f'{ai(cross - 1):.1f}'),
        ridge_xy=px(ridge, peak), slope_start_xy=px(X_LIM[0], bw_tb * X_LIM[0]), ceiling_end_xy=px(X_LIM[1], peak),
        states=states,
        ledgers=[dict(batch=s['batch'], ai=s['text']['ai'], attainable=s['text']['attainable'], pct=s['text']['pct'],
                      bound=s['bound'], weight_mb=s['text']['weight_mb'], act_mb=s['text']['act_mb']) for s in states],
        expected_final=dict(batch=1024, ai=states[-1]['text']['ai'], attainable=states[-1]['text']['attainable'],
                            pct='100.0', bound='compute'),
    )


def check_printed_marker(data):
    """The batch-32 stop must sit on the printed filled classifier marker."""
    import numpy as np
    a = measure_frame()
    x, y = data['states'][0]['xy']
    r = 45
    ys, xs = np.mgrid[int(y) - r:int(y) + r, int(x) - r:int(x) + r]
    patch = a[int(y) - r:int(y) + r, int(x) - r:int(x) + r]
    # Marker fill is near-black grey; the blue slope is excluded by its blue channel.
    m = (patch.sum(2) < 200) & (patch[:, :, 2] < 90)
    cx, cy = xs[m].mean(), ys[m].mean()
    assert m.sum() > 1500 and abs(cx - x) < 2.5 and abs(cy - y) < 2.5, (m.sum(), cx, cy, x, y)
    return [round(float(cx), 1), round(float(cy), 1)]


MODULE = Path(__file__).with_name('original-v1-h100-roofline-batch.js')
PATTERN = re.compile(r'(// BEGIN calc\.py output\nconst D=)(\{.*\})(;\n// END calc\.py output)', re.S)

if __name__ == '__main__':
    data = compute()
    if ART.exists():
        print('printed classifier centroid', check_printed_marker(data), 'vs', data['states'][0]['xy'], file=sys.stderr)
    text = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    if '--write' in sys.argv:
        source = MODULE.read_text()
        assert PATTERN.search(source), 'embedded data block not found'
        MODULE.write_text(PATTERN.sub(lambda m: m.group(1) + text + m.group(3), source))
    if '--check' in sys.argv or '--write' in sys.argv:
        embedded = json.loads(PATTERN.search(MODULE.read_text()).group(2))
        assert embedded == json.loads(text), 'factory data differs from calc.py; rerun with --write'
        print('embedded lesson data matches calc.py', file=sys.stderr)
    print(json.dumps(data, ensure_ascii=False, indent=1))
