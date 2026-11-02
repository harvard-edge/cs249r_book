#!/usr/bin/env python3
"""Lesson data for vol1:fig-mixed-precision, computed with IEEE binary16 (numpy.float16).

The chapter states FP16's limits in prose (sec-model-training-loss-scaling-f9f5 and
tbl-precision-comparison): minimum normal 2^-14 ~ 6.1e-5, maximum finite 65,504,
subnormals toward ~6e-8. No LEGO cell or MLSysIM constant owns these format limits,
so they are taken from numpy.finfo and asserted against those prose literals.
Loss-scale values 2^10 and 2^15 are the prose's examples; 2^14 is one dynamic-scaler
halving of 2^15. Weight, loss, gradients, and learning rate are declared toy values.

Run:  python3 <this file>            print the lesson data (JSON)
      python3 <this file> --check    also assert the factory's embedded D equals it
      python3 <this file> --write    rewrite the factory's embedded D block
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

F16, F32 = np.float16, np.float32
SUP = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')


def bf16(x):
    """Round an FP32 value to bfloat16 (round to nearest even), returned as FP32."""
    u = np.array([x], dtype=F32).view(np.uint32)[0].item()
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return float(np.array([u], dtype=np.uint32).view(F32)[0])


def sci(x, digits=3):
    """Readable scientific text such as 4.88 × 10⁻⁴; plain text for 1e-3 < |x| < 1e6."""
    if x == 0:
        return '0'
    if not np.isfinite(x):
        return 'inf'
    if 1e-3 <= abs(x) < 1e6:
        return (f'{x:,.{digits}g}' if abs(x) < 1000 else f'{x:,.0f}').replace('-', '−')
    mantissa, exponent = f'{x:.{digits - 1}e}'.split('e')
    return f'{mantissa} × 10{str(int(exponent)).translate(SUP)}'.replace('-', '−')


def pow2(x):
    """'2⁻¹¹' when x is an exact power of two, else None."""
    if x <= 0 or not np.isfinite(x):
        return None
    m, e = np.frexp(x)
    return f'2{str(int(e) - 1).translate(SUP)}' if m == 0.5 else None


def label(x):
    p = pow2(x)
    return f'{p} ≈ {sci(x)}' if p and not (1 <= x < 1e6 and float(x).is_integer()) else sci(x)


def classify(x16):
    x = float(x16)
    if np.isinf(x):
        return 'overflow'
    if x == 0:
        return 'underflow'
    return 'subnormal' if abs(x) < float(np.finfo(F16).tiny) else 'normal'


def bits16(x16):
    return '0x%04X' % np.array([x16], dtype=F16).view(np.uint16)[0]


def compute():
    fi = np.finfo(F16)
    min_normal, max_finite = float(fi.tiny), float(fi.max)
    min_sub = float(np.array([1], dtype=np.uint16).view(F16)[0])
    # Prose literals: 6.1 x 10^-5, 65,504, and subnormals toward 6 x 10^-8.
    assert min_normal == 2.0 ** -14 and round(min_normal, 6) == 6.1e-05
    assert max_finite == 65504.0
    assert min_sub == 2.0 ** -24 and round(min_sub * 1e8) == 6

    # Declared toy values (labeled as toy in the UI).
    w32 = F32(0.1)
    loss = F16(0.75)
    g_small, g_large, eta = F32(2.0 ** -26), F32(3.0), F32(0.5)
    assert float(g_small) < min_sub / 2, 'the tracked gradient must round to zero, not to a subnormal'

    w16 = F16(w32)
    trials = []
    for exp in (0, 10, 15, 14):
        S = F32(2.0 ** exp)
        scaled_loss = F32(S * F32(loss))
        with np.errstate(over='ignore'):  # binary16 overflow to inf is the lesson, not an error
            s_small, s_large = F16(F32(S * g_small)), F16(F32(S * g_large))
        finite = bool(np.isfinite(s_small) and np.isfinite(s_large))
        trials.append(dict(
            exponent=exp, S=float(S), S_plain=f'{int(S):,}', S_pow=f'2{str(exp).translate(SUP)}' if exp else '1',
            S_text=f'2{str(exp).translate(SUP)} = {int(S):,}' if exp else '1 (no scaling)',
            scaled_loss=float(scaled_loss), scaled_loss_text=sci(float(scaled_loss)),
            exact_small=float(F32(S * g_small)), exact_small_text=label(float(F32(S * g_small))),
            exact_large=float(F32(S * g_large)), exact_large_text=sci(float(F32(S * g_large))),
            small=float(s_small), small_text=label(float(s_small)), small_class=classify(s_small), small_bits=bits16(s_small),
            small_ftz=float(s_small) if classify(s_small) == 'normal' else 0.0,
            large=float(s_large), large_text=sci(float(s_large)), large_class=classify(s_large), large_bits=bits16(s_large),
            finite=finite))

    s1, s10, s15, s14 = trials
    # Asserted endpoints of the storyboard.
    assert s1['small'] == 0.0 and s1['small_class'] == 'underflow' and s1['large'] == 3.0
    assert s10['small'] == 2.0 ** -16 and s10['small_class'] == 'subnormal' and s10['small_ftz'] == 0.0
    assert s15['small'] == 2.0 ** -11 and s15['small_class'] == 'normal'
    assert s15['large_class'] == 'overflow' and not s15['finite'] and s15['exact_large'] > max_finite
    assert s14['small'] == 2.0 ** -12 and s14['small_class'] == 'normal' and s14['large'] == 49152.0 and s14['finite']
    assert s14['S'] == s15['S'] / 2

    # Copy & unscale at S = 2^15: the tracked value is exact, but inf / S stays inf. Dynamic scalers
    # find the nonfinite entry in these unscaled FP32 gradients and skip only the update (step 6).
    S15 = F32(s15['S'])
    with np.errstate(over='ignore'):
        ov_small, ov_large = F32(F32(F16(s15['small'])) / S15), F32(F32(F16(s15['large'])) / S15)
    assert float(ov_small) == float(g_small) == 2.0 ** -11 / 2.0 ** 15 and np.isinf(ov_large)

    # Copy & unscale (FP32), then the FP32 master-weight update with the toy rate.
    S = F32(s14['S'])
    un_small, un_large = F32(F32(s14['small']) / S), F32(F32(s14['large']) / S)
    assert float(un_small) == float(g_small) and float(un_large) == float(g_large), 'power-of-two unscale is exact here'
    delta = F32(eta * un_small)
    w_new = F32(w32 - delta)
    ulp32 = float(np.spacing(w32))
    ulp16 = float(np.spacing(w16))
    assert w_new != w32 and float(w32 - w_new) == ulp32, 'the toy update moves the FP32 master by one ulp'
    assert F16(float(w16) - float(delta)) == w16, 'the same change is lost in an FP16 copy'

    bf = dict(small=bf16(float(g_small)), large=bf16(float(g_large)), w=bf16(float(w32)), loss=bf16(float(loss)))
    assert bf['small'] == float(g_small) and bf['large'] == 3.0 and bf['loss'] == 0.75

    return dict(
        fp16=dict(min_normal=min_normal, min_normal_text=label(min_normal), max=max_finite, max_text=f'{max_finite:,.0f}',
                  min_subnormal=min_sub, min_subnormal_text=label(min_sub)),
        toy=dict(w32=float(w32), w32_text=f'{float(w32):.9f}', loss=float(loss), loss_text=sci(float(loss)),
                 g_small=float(g_small), g_small_text=label(float(g_small)), g_large=float(g_large), g_large_text=sci(float(g_large)),
                 eta=float(eta), eta_text=sci(float(eta))),
        cast=dict(w16=float(w16), w16_text=f'{float(w16):.13g}', w16_bits=bits16(w16),
                  error=float(w16) - float(w32), error_text=sci(float(w16) - float(w32)), w32_text=f'{float(w32):.9f}',
                  w16_token=f'{float(w16):.4g}'),
        trials=trials,
        unscale_overflow=dict(S_pow=s15['S_pow'], S_plain=s15['S_plain'], small=float(ov_small), small_text=label(float(ov_small)),
                              large='inf', large_text='inf', finite=False),
        unscaled=dict(small=float(un_small), small_text=label(float(un_small)), large=float(un_large), large_text=sci(float(un_large))),
        update=dict(eta_text=sci(float(eta)), delta=float(delta), delta_text=label(float(delta)), w_new=float(w_new), w_new_text=f'{float(w_new):.9f}',
                    ulp32=ulp32, ulp32_text=label(ulp32), ulp16=ulp16, ulp16_text=label(ulp16)),
        bf16=dict(small=bf['small'], small_text=label(bf['small']), large=bf['large'], large_text=sci(bf['large']),
                  w=bf['w'], w_text=f"{bf['w']:.9g}", error_text=sci(bf['w'] - float(w32)),
                  loss=bf['loss'], loss_text=sci(bf['loss'])),
    )


MODULE = Path(__file__).with_name('original-v1-mixed-precision.js')
PATTERN = re.compile(r'(// BEGIN calc\.py output\nconst D=)(\{.*\})(;\n// END calc\.py output)', re.S)

if __name__ == '__main__':
    data = compute()
    text = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    if '--write' in sys.argv:
        source = MODULE.read_text()
        assert PATTERN.search(source), 'embedded data block not found'
        MODULE.write_text(PATTERN.sub(lambda m: m.group(1) + text + m.group(3), source))
    if '--check' in sys.argv or '--write' in sys.argv:
        embedded = json.loads(PATTERN.search(MODULE.read_text()).group(2))
        assert embedded == data, 'factory data differs from calc.py; rerun with --write'
        print('embedded lesson data matches calc.py', file=sys.stderr)
    print(json.dumps(data, ensure_ascii=False, indent=1))
