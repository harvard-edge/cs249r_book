(() => {
  'use strict';
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons['original-v1-python-tax'] = async (root, { outer, verifiedArtwork }) => {
    if (!verifiedArtwork || !window.MLSysOriginalPlayer) {
      throw new Error('Verified original Python timeline required');
    }
    root.classList.add('original-v1-python-tax');
    const svg = verifiedArtwork.cloneNode(true);
    svg.removeAttribute('width');
    svg.removeAttribute('height');
    if (svg.viewBox.baseVal.width !== 3613 || svg.viewBox.baseVal.height !== 2033) {
      throw new Error('Python timeline pixel geometry changed');
    }

    const data = { n_ops: 5, dispatch_us: 10, kernel_us: 1, launch_us: 5, fused_compute_us: 4.0 };
    const steps = [
      {
        label: 'Eager sequence waits for host',
        description: 'At 0 μs on the Eager Execution row, input x₀ is ready, but Python has not dispatched operation 1 yet and the GPU sits idle.'
      },
      {
        label: 'Dispatch operation 1',
        description: 'From 0 to 10 μs, Python resolves types and dispatches operation 1 (10 μs red Python dispatch bar). The GPU waits until host submission finishes at 10 μs.'
      },
      {
        label: 'Execute operation 1',
        description: 'From 10 to 11 μs (1 μs blue GPU kernel bar), kernel 1 reads x₀, computes operation 1, and writes intermediate x₁ to global HBM (1 of 5 ops complete at 11 μs).'
      },
      {
        label: 'Dispatch operation 2',
        description: 'From 11 to 21 μs, a second 10 μs red Python dispatch bar elapses on the host while intermediate x₁ waits in global memory.'
      },
      {
        label: 'Execute operation 2',
        description: 'From 21 to 22 μs (1 μs blue GPU kernel bar), kernel 2 reads x₁ from global memory and writes intermediate x₂ back to global memory (2 of 5 ops complete at 22 μs).'
      },
      {
        label: 'Dispatch operation 3',
        description: 'From 22 to 32 μs, the third 10 μs red Python dispatch gap leaves the GPU idle waiting on host overhead.'
      },
      {
        label: 'Execute operation 3',
        description: 'From 32 to 33 μs (1 μs blue GPU kernel bar), kernel 3 reads x₂ and writes intermediate x₃ to global memory (3 of 5 ops complete at 33 μs).'
      },
      {
        label: 'Dispatch operation 4',
        description: 'From 33 to 43 μs, the fourth 10 μs red Python dispatch bar adds another 10 μs of host serialization before kernel 4 can start.'
      },
      {
        label: 'Execute operation 4',
        description: 'From 43 to 44 μs (1 μs blue GPU kernel bar), kernel 4 reads x₃ and writes intermediate x₄ to global memory (4 of 5 ops complete at 44 μs).'
      },
      {
        label: 'Dispatch operation 5',
        description: 'From 44 to 54 μs, the fifth 10 μs red Python dispatch bar brings cumulative host dispatch overhead to 50 μs across the 5 eager operations.'
      },
      {
        label: 'Execute operation 5',
        description: 'From 54 to 55 μs (1 μs blue GPU kernel bar), kernel 5 reads x₄ and writes final output x₅. Eager execution finishes at 55 μs (50 μs host dispatch + 5 μs GPU work).'
      },
      {
        label: 'Start compiled/fused alternative',
        description: 'Reset to 0 μs on the Compiled/Fused row with the same input x₀ to run the same 5 operations as a single compiled graph region.'
      },
      {
        label: 'Submit the fused region',
        description: 'From 0 to 5 μs, one compiled runtime call (5 μs orange kernel-launch bar) submits the entire 5-operation fused region at once, eliminating per-op Python dispatch.'
      },
      {
        label: 'Execute fused region and write output',
        description: 'From 5 to 9 μs (4 μs blue GPU kernel bar), 1 fused kernel runs all 5 operations in registers/SRAM (keeping x₁–x₄ on-chip) and writes x₅ once, finishing in 9 μs vs. 55 μs.'
      }
    ];

    const ns = 'http://www.w3.org/2000/svg';
    const add = (tag, attrs, txt) => {
      const n = document.createElementNS(ns, tag);
      for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
      if (txt != null) n.textContent = txt;
      return n;
    };

    const layer = add('g', { class: 'vb-overlay' });
    const masks = [
      add('rect', { y: 132, height: 438, fill: '#ffffff', 'fill-opacity': '0.86' }),
      add('rect', { y: 1212, height: 438, fill: '#ffffff', 'fill-opacity': '0.86' })
    ];
    const rowFrame = add('rect', {
      fill: 'none',
      stroke: '#006395',
      'stroke-width': 9,
      'stroke-dasharray': '22 12',
      rx: 14,
      'data-mlsys-focus': ''
    });
    const cursor = add('line', { stroke: '#E67817', 'stroke-width': 13, 'stroke-linecap': 'round' });
    const timePillBg = add('rect', {
      width: 470,
      height: 88,
      rx: 14,
      fill: '#ffffff',
      'fill-opacity': '0.96',
      stroke: '#E67817',
      'stroke-width': 4
    });
    const timePillTxt = add('text', {
      'text-anchor': 'middle',
      'dominant-baseline': 'middle',
      'font-family': 'system-ui, -apple-system, sans-serif',
      'font-size': 42,
      'font-weight': '700',
      fill: '#142832'
    });
    layer.append(masks[0], masks[1], rowFrame, cursor, timePillBg, timePillTxt);
    svg.append(layer);

    const viewport = document.createElement('div');
    viewport.className = 'vb-diagram';
    viewport.append(svg);

    const readout = document.createElement('p');
    readout.className = 'vb-readout';
    const ledger = document.createElement('div');
    ledger.className = 'vb-ledger';
    const makeCard = (role, title) => {
      const card = document.createElement('div');
      card.className = 'vb-card';
      card.dataset.role = role;
      const t = document.createElement('span');
      t.className = 'vb-card-title';
      t.textContent = title;
      const v = document.createElement('span');
      v.className = 'vb-card-val';
      card.append(t, v);
      ledger.append(card);
      return { t, v };
    };
    const cards = {
      host: makeCard('host', 'Host Python / Runtime'),
      queue: makeCard('queue', 'Submitted Queue'),
      device: makeCard('device', 'GPU Execution'),
      storage: makeCard('storage', 'Intermediates')
    };

    // Exact pixel mapping measured on HEAD 3613x2033 PNG: x(0 us) = 670, x(55 us) = 3416
    const x = t => 670 + ((3416 - 670) * t) / (data.n_ops * (data.dispatch_us + data.kernel_us));

    function time(i) {
      if (i <= 10) return Math.floor(i / 2) * (data.dispatch_us + data.kernel_us) + (i % 2 ? data.dispatch_us : 0);
      return i === 11 ? 0 : i === 12 ? data.launch_us : data.launch_us + data.fused_compute_us;
    }

    function draw(i, t) {
      const row = i <= 10 ? 0 : 1;
      for (let r = 0; r < 2; r += 1) {
        const end = r === 0 ? 3420 : 1122;
        const start = r === row ? x(t) : row === 1 && r === 0 ? end : 670;
        masks[r].setAttribute('x', start.toFixed(1));
        masks[r].setAttribute('width', Math.max(0, end - start).toFixed(1));
      }
      const frameStroke = row === 0 ? (i % 2 === 1 ? '#CB202D' : '#006395') : (i === 12 ? '#E67817' : '#008F45');
      rowFrame.setAttribute('x', '662');
      rowFrame.setAttribute('y', String(row === 0 ? 118 : 1198));
      rowFrame.setAttribute('width', String(row === 0 ? 2768 : 1100));
      rowFrame.setAttribute('height', '466');
      rowFrame.setAttribute('stroke', frameStroke);

      const cx = x(t);
      cursor.setAttribute('x1', cx.toFixed(1));
      cursor.setAttribute('x2', cx.toFixed(1));
      cursor.setAttribute('y1', String(row === 0 ? 118 : 1198));
      cursor.setAttribute('y2', String(row === 0 ? 584 : 1664));
      cursor.setAttribute('stroke', frameStroke);

      const px = Math.max(680, Math.min(3030, cx - 235));
      const py = row === 0 ? 610 : 1084;
      timePillBg.setAttribute('x', px.toFixed(1));
      timePillBg.setAttribute('y', String(py));
      timePillBg.setAttribute('stroke', frameStroke);
      timePillTxt.setAttribute('x', (px + 235).toFixed(1));
      timePillTxt.setAttribute('y', String(py + 46));
      timePillTxt.textContent = `t = ${t.toFixed(t % 1 ? 1 : 0)} μs`;
    }

    function updateLedger(i, n, eager) {
      cards.storage.t.textContent = eager ? 'Global HBM Intermediates' : 'On-Chip SRAM + Output';
      cards.host.v.textContent = eager
        ? (n < 5 ? `dispatch op ${Math.min(5, Math.ceil((i + 1) / 2))} (10 μs)` : '5 dispatches done (50 μs)')
        : (i < 12 ? 'compiled graph ready' : '1 launch done (5 μs)');
      cards.queue.v.textContent = eager
        ? (i % 2 ? `op ${Math.ceil(i / 2)} queued` : 'empty (waiting on Python)')
        : (i === 12 ? '1 fused kernel queued' : 'empty');
      cards.device.v.textContent = eager
        ? (n ? `${n} / 5 kernels run (${n} μs)` : 'idle')
        : (i >= 13 ? '5 fused ops run (4 μs)' : 'idle');
      cards.storage.v.textContent = eager
        ? (n === 0 ? 'x₀ ready in HBM' : n === 5 ? 'x₅ written (5 HBM round-trips)' : `x${n} spilled to HBM`)
        : (i >= 13 ? 'x₁–x₄ in SRAM; x₅ written once' : 'x₀ ready in HBM');
    }

    function render(i, { stage, details }) {
      if (!viewport.isConnected) stage.replaceChildren(viewport);
      if (details && !ledger.isConnected) details.replaceChildren(readout, ledger);
      const t = time(i);
      draw(i, t);
      const eager = i <= 10;
      const n = eager ? Math.floor(i / 2) : i >= 13 ? 5 : 0;
      updateLedger(i, n, eager);
      root.dataset.endpoint = String(i);
      root.dataset.operations = String(n);
      root.dataset.elapsedUs = String(t);
      root.dataset.alternative = eager ? 'eager' : 'fused';
      readout.textContent = eager
        ? `Eager state at ${t} μs: ${n} of 5 GPU kernels completed (${i === 0 ? 'awaiting first Python dispatch' : i % 2 ? 'host dispatch complete, kernel ready' : n === 5 ? 'all 55 μs elapsed: 50 μs Python dispatch + 5 μs GPU compute' : 'GPU idle again until next 10 μs Python dispatch'}).`
        : `Compiled/Fused state at ${t} μs: ${i === 11 ? 'ready to launch fused 5-op kernel' : i === 12 ? 'single 5 μs runtime launch complete' : 'all 5 operations finished at 9 μs (6.1× faster than 55 μs eager)'}.`;
    }

    return window.MLSysOriginalPlayer.create({
      root,
      outer,
      layout: 'compact',
      intro: 'Compare 5 eager operations taking 55 μs (10 μs Python dispatch plus 1 μs kernel each) against 1 compiled fused kernel finishing in 9 μs.',
      title: 'What keeps the next kernel from starting?',
      scope: 'Canonical Ops dispatch (10 μs) and kernel-launch (5 μs) values paired with the chapter scenario of five 1 μs kernels (4 μs when fused in SRAM). Eager and compiled/fused rows are alternatives for the same five operations.',
      steps,
      render,
      transition: ({ from, to, onFrame }) => {
        if (to === 11) return;
        const t0 = time(from);
        const t1 = time(to);
        onFrame(p => draw(to, t0 + (t1 - t0) * p));
      }
    });
  };
})();
