#!/usr/bin/env python3
"""Lesson data for vol1:fig-rnn, from the chapter's RNNCompute LEGO cell (MLSysIM).

RNNCompute in 06_nn_architectures.qmd loads Scenarios.ElmanRnnStep: 100 inputs, a
128-wide state, FP32, and a 1,000-step sequence. The caption states 16,384 recurrent
plus 12,800 input MACs per step; this script reads the same scenario, asserts those
caption literals, and derives every per-state count the lesson displays:

- the figure's three unrolled steps are treated as a complete toy sequence (T = 3);
- retained hidden-state values = (states held after the step commits) x hidden_dim;
- cumulative MACs = completed steps x the scenario's per-step MACs;
- the comparison at the scenario's 1,000 steps uses training_states and state_bytes.

Run (MLSysIM needs numpy; the book's venv has it):
  PYTHONPATH=mlsysim ../MLSysBook/.venv/bin/python <this file>           print the lesson data (JSON)
  PYTHONPATH=mlsysim ../MLSysBook/.venv/bin/python <this file> --check   also assert the factory's embedded D equals it
  PYTHONPATH=mlsysim ../MLSysBook/.venv/bin/python <this file> --write   rewrite the factory's embedded D block
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'mlsysim'))
from mlsysim import Scenarios  # noqa: E402

FACTORY = HERE / 'original-v1-rnn-state-lifetime.js'
CHAPTER = ROOT / 'books/vol1/06_nn_architectures/06_nn_architectures.qmd'
STEPS_DRAWN = 3  # h_{t-1}, h_t, h_{t+1} in the unrolled panel


def n(x):
    return f'{x:,}'


def compute():
    step = Scenarios.ElmanRnnStep  # the scenario RNNCompute loads
    H, I = step.hidden_dim, step.input_dim
    rec, inp, per = step.recurrent_macs, step.input_macs, step.macs_per_step
    T = step.training_states  # one retained state per step, at the scenario's sequence length
    state_bytes = int(step.state_bytes.to('byte').magnitude)

    # Caption and prose literals must match the LEGO scenario.
    source = CHAPTER.read_text()
    caption = re.search(r'::: \{#fig-rnn .*?fig-cap="(.*?)" fig-alt=', source).group(1)
    assert '128-dimensional hidden state' in caption and '100-dimensional input' in caption
    assert f'{n(rec)} MACs for recurrent' in caption and f'plus {n(inp)} for input' in caption
    assert (H, I, rec, inp, per) == (128, 100, 16384, 12800, 29184)
    assert step.gates == 1 and T == step.steps == 1000 and '\\SequenceLength= 1{,}000$ therefore requires $1{,}000$ ordered state updates' in source
    assert state_bytes == H * int(step.bytes_per_value.to('byte').magnitude) == 512

    # Per-state ledgers. nodes: h_{t-1}, h_t, h_{t+1}; inputs: x_{t-1}, x_t, x_{t+1}.
    states = []

    def add(mode, nodes, inputs, done, ahead=False, active=None, phase='forward', grad=None):
        held = nodes.count('live')
        recurrent_done = done
        input_done = STEPS_DRAWN if ahead else done
        states.append({
            'mode': mode, 'phase': phase, 'nodes': list(nodes), 'inputs': list(inputs),
            'active': active, 'grad': grad,
            'retained': held, 'retainedValues': held * H,
            'recurrentSteps': recurrent_done, 'recurrentMacs': recurrent_done * rec,
            'inputSteps': input_done, 'inputMacs': input_done * inp,
            'totalMacs': recurrent_done * rec + input_done * inp,
            'whhReads': recurrent_done,
        })

    # Inference: streaming, one input arrives per step, the previous state is freed after its last reader.
    add('inference', ['next', 'blocked', 'blocked'], ['pending'] * 3, 0)
    add('inference', ['live', 'next', 'blocked'], ['current', 'pending', 'pending'], 1, active=0)
    add('inference', ['freed', 'live', 'next'], ['discarded', 'current', 'pending'], 2, active=1)
    add('inference', ['freed', 'freed', 'live'], ['discarded', 'discarded', 'current'], 3, active=2)
    # Training (BPTT): the whole sequence is known, so input projections may run ahead.
    add('training', ['next', 'blocked', 'blocked'], ['present'] * 3, 0, ahead=True)
    add('training', ['live', 'next', 'blocked'], ['present'] * 3, 1, ahead=True, active=0)
    add('training', ['live', 'live', 'next'], ['present'] * 3, 2, ahead=True, active=1)
    add('training', ['live', 'live', 'live'], ['present'] * 3, 3, ahead=True, active=2)
    add('training', ['live', 'live', 'freed'], ['present'] * 3, 3, ahead=True, active=2, phase='backward', grad=2)
    add('training', ['live', 'freed', 'freed'], ['present'] * 3, 3, ahead=True, active=1, phase='backward', grad=1)
    add('training', ['freed', 'freed', 'freed'], ['present'] * 3, 3, ahead=True, active=0, phase='backward', grad=0)
    add('compare', ['idle'] * 3, ['present'] * 3, 3, ahead=True, phase='compare')

    inference_peak = max(s['retainedValues'] for s in states if s['mode'] == 'inference')
    training_peak = max(s['retainedValues'] for s in states if s['mode'] == 'training')
    for s in states:
        s['peakValues'] = inference_peak if s['mode'] == 'inference' else training_peak if s['mode'] == 'training' else None
        s['retainedText'] = f"{s['retained']} × {n(H)} = {n(s['retainedValues'])} values" if s['retained'] else '0 values'
        s['recurrentText'] = f"{s['recurrentSteps']} × {n(rec)} = {n(s['recurrentMacs'])}" if s['recurrentSteps'] else '0'
        s['inputText'] = f"{s['inputSteps']} × {n(inp)} = {n(s['inputMacs'])}" if s['inputSteps'] else '0'
        s['totalText'] = n(s['totalMacs'])
    # Running peak within each regime, as committed so far (the compare state reports both).
    run = {}
    for s in states:
        if s['mode'] in ('inference', 'training'):
            run[s['mode']] = max(run.get(s['mode'], 0), s['retainedValues'])
            s['peakSoFar'] = run[s['mode']]
            s['peakSoFarText'] = n(s['peakSoFar'])

    # Endpoint and invariant checks.
    inf = [s for s in states if s['mode'] == 'inference']
    tr = [s for s in states if s['mode'] == 'training']
    assert [s['retainedValues'] for s in inf] == [0, H, H, H]
    assert [s['retainedValues'] for s in tr] == [0, H, 2 * H, 3 * H, 2 * H, H, 0] == [0, 128, 256, 384, 256, 128, 0]
    assert inference_peak == H and training_peak == STEPS_DRAWN * H == 384
    assert all(s['nodes'].count('live') <= 1 for s in inf), 'inference holds one committed state'
    # Training releases right to left, each only after its own backward step has run.
    for s in tr:
        if s['phase'] == 'backward':
            assert all(s['nodes'][k] == 'freed' for k in range(s['grad'], 3))
            assert all(s['nodes'][k] == 'live' for k in range(s['grad']))
    assert inf[-1]['totalMacs'] == tr[-1]['totalMacs'] == STEPS_DRAWN * per == 87552
    assert tr[0]['inputMacs'] == STEPS_DRAWN * inp == 38400 and tr[0]['recurrentMacs'] == 0
    assert inf[-1]['whhReads'] == tr[-1]['whhReads'] == STEPS_DRAWN

    big = {'T': T, 'T_text': n(T), 'inference_values': H, 'training_values': T * H,
           'inference_bytes': state_bytes, 'training_bytes': T * state_bytes}
    assert big['training_values'] == 128000 and big['training_bytes'] == 512000
    for k in list(big):
        if k != 'T_text':
            big[k + '_text'] = n(big[k])

    return {
        'hidden': H, 'hidden_text': n(H), 'input': I, 'input_text': n(I),
        'recurrent_macs': rec, 'recurrent_text': n(rec), 'input_macs': inp, 'input_macs_text': n(inp),
        'per_step': per, 'per_step_text': n(per), 'steps_drawn': STEPS_DRAWN,
        'inference_peak': inference_peak, 'inference_peak_text': n(inference_peak),
        'training_peak': training_peak, 'training_peak_text': n(training_peak),
        'input_ahead_text': n(STEPS_DRAWN * inp), 'total_text': n(STEPS_DRAWN * per),
        'chapter': big, 'states': states,
    }


BEGIN, END = '// BEGIN calc.py output\n', '// END calc.py output'


def embedded():
    text = FACTORY.read_text()
    block = text.split(BEGIN, 1)[1].split(END, 1)[0].strip()
    return json.loads(block.removeprefix('const D=').removesuffix(';'))


if __name__ == '__main__':
    data = compute()
    if '--write' in sys.argv:
        text = FACTORY.read_text()
        head, rest = text.split(BEGIN, 1)
        tail = rest.split(END, 1)[1]
        FACTORY.write_text(head + BEGIN + 'const D=' + json.dumps(data, ensure_ascii=False, separators=(',', ':')) + ';\n' + END + tail)
    if '--check' in sys.argv:
        assert embedded() == data, 'factory D block is stale; rerun with --write'
    print(json.dumps(data, ensure_ascii=False))
