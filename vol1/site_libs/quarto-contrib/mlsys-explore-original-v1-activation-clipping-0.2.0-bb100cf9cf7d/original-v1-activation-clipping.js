(function () {
  'use strict';
  if (!window.MLSysOriginalPlayer) return;

  const NS = 'http://www.w3.org/2000/svg';
  const PROFILE = {"total":32904484,"markers":{"p9999":1.1221,"entropy":1.394,"max":2.0922},"axis":{"png_w":3315,"png_h":2234,"plot_x0":372,"plot_x1":3255,"plot_y0":59,"plot_y1":1908,"xlim":[-0.03,2.25],"ylim":[0.42,11000000.0]},"presets":[{"beta":2.0922,"scale":0.008205,"clipped":0,"clipped_pct":0.0,"rmse_quant":0.002412,"rmse_clip":0.0,"rmse_total":0.002412},{"beta":1.394,"scale":0.005467,"clipped":113,"clipped_pct":0.0003,"rmse_quant":0.001564,"rmse_clip":0.000239,"rmse_total":0.001582},{"beta":1.1221,"scale":0.0044,"clipped":1351,"clipped_pct":0.0041,"rmse_quant":0.00124,"rmse_clip":0.001023,"rmse_total":0.001608},{"beta":0.8,"scale":0.003137,"clipped":196826,"clipped_pct":0.5982,"rmse_quant":0.000855,"rmse_clip":0.008969,"rmse_total":0.00901},{"beta":0.5,"scale":0.001961,"clipped":1150113,"clipped_pct":3.4953,"rmse_quant":0.000528,"rmse_clip":0.038992,"rmse_total":0.038996}],"bins":[[0.0105,1522332],[0.0314,1839242],[0.0523,2468286],[0.0732,3243595],[0.0941,3602915],[0.1151,3757552],[0.136,3679421],[0.1569,3045441],[0.1778,2086371],[0.1988,1459683],[0.2197,979207],[0.2406,670835],[0.2615,532397],[0.2824,450020],[0.3034,396715],[0.3243,349725],[0.3452,308300],[0.3661,271782],[0.3871,229729],[0.408,211210],[0.4289,186192],[0.4498,167623],[0.4707,154111],[0.4917,141687],[0.5126,124904],[0.5335,110110],[0.5544,91137],[0.5753,85570],[0.5963,78672],[0.6172,77036],[0.6381,66499],[0.659,59867],[0.68,53896],[0.7009,49552],[0.7218,45557],[0.7427,41014],[0.7636,36923],[0.7846,32550],[0.8055,30561],[0.8264,28694],[0.8473,22773],[0.8682,19249],[0.8892,16969],[0.9101,17329],[0.931,14343],[0.9519,14045],[0.9729,10915],[0.9938,8846],[1.0147,7964],[1.0356,1514],[1.0565,915],[1.0775,552],[1.0984,403],[1.1193,403],[1.1402,210],[1.1612,181],[1.1821,147],[1.203,114],[1.2239,117],[1.2448,101],[1.2658,78],[1.2867,75],[1.3076,56],[1.3285,47],[1.3494,51],[1.3704,37],[1.3913,24],[1.4122,23],[1.4331,25],[1.4541,9],[1.475,10],[1.4959,15],[1.5168,11],[1.5377,3],[1.5587,3],[1.5796,3],[1.6005,4],[1.6214,1],[1.6424,1],[1.6842,1],[1.7051,1],[1.726,1],[1.747,1],[2.0817,1]]};

  const STEP_BETAS = [2.0922, 2.0922, 1.3940, 1.1221, 0.8000, 0.5000];

  const STEPS = [
    {
      label: 'Inspect the post-ReLU tail',
      description: 'Across 32,904,484 post-ReLU activations in ResNet-50 layer1.0.conv2, 99.9959% sit at or below 1.1221 while a sparse tail stretches to 2.0922.'
    },
    {
      label: 'Anchor UINT8 at the max (2.09)',
      description: 'Setting β = 2.0922 clips 0 activations (0.0000%), but spreads 256 UINT8 codes at Δ = 0.008205 and leaves total RMSE at 0.002412 because nearly half the code space serves sparse outliers.'
    },
    {
      label: 'Tighten to KL-entropy (1.39)',
      description: 'Moving β to the KL-entropy threshold 1.3940 clips only 113 activations (0.0003%), shrinks step width Δ to 0.005467, and lowers total RMSE by 34% to its minimum of 0.001582.'
    },
    {
      label: 'Tighten to 99.99% percentile (1.12)',
      description: 'At the 99.99% percentile β = 1.1221, step width tightens further to Δ = 0.004400 while 1,351 tail activations saturate (0.0041%), holding total RMSE near optimum at 0.001608.'
    },
    {
      label: 'Cut into the upper shoulder (0.80)',
      description: 'Pushing β down to 0.8000 narrows Δ to 0.003137, but saturates 196,826 activations (0.5982%) and drives clipping RMSE to 0.008969 (total RMSE 0.009010).'
    },
    {
      label: 'Clip deep into the bulk (0.50)',
      description: 'At β = 0.5000, step width reaches Δ = 0.001961, yet 1,150,113 bulk activations saturate (3.4953%) and total RMSE surges to 0.038996—over 24× worse than KL-entropy calibration.'
    }
  ];

  function svgEl(tag, attrs, text) {
    const el = document.createElementNS(NS, tag);
    if (attrs) {
      Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, String(v)));
    }
    if (text != null) el.textContent = text;
    return el;
  }

  function htmlEl(tag, cls, text) {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text != null) el.textContent = text;
    return el;
  }

  function evaluateBeta(beta) {
    const clamped = Math.max(0.40, Math.min(PROFILE.markers.max, Number(beta)));
    const scale = clamped / 255;
    let clipped = 0;
    let sqClip = 0;
    let sqQuant = 0;
    for (let i = 0; i < PROFILE.bins.length; i += 1) {
      const x = PROFILE.bins[i][0];
      const cnt = PROFILE.bins[i][1];
      if (x > clamped) {
        clipped += cnt;
        const diff = x - clamped;
        sqClip += cnt * diff * diff;
      } else {
        const q = Math.min(255, Math.max(0, Math.round(x / scale)));
        const diff = x - q * scale;
        sqQuant += cnt * diff * diff;
      }
    }
    const total = PROFILE.total;
    return {
      beta: clamped,
      scale,
      clipped,
      clippedPct: (100 * clipped) / total,
      rmseQuant: Math.sqrt(sqQuant / total),
      rmseClip: Math.sqrt(sqClip / total),
      rmseTotal: Math.sqrt((sqQuant + sqClip) / total)
    };
  }

  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons['original-v1-activation-clipping'] = function (root, context) {
    const { outer, verifiedArtwork } = context;
    if (!verifiedArtwork || verifiedArtwork.tagName.toLowerCase() !== 'svg') {
      throw new Error('Activation clipping requires verified SVG-wrapped figure artwork.');
    }
    root.classList.add('original-v1-activation-clipping');

    const vbParts = (verifiedArtwork.getAttribute('viewBox') || '').trim().split(/\s+/).map(Number);
    if (vbParts.length !== 4 || vbParts.some(Number.isNaN)) {
      throw new Error('Activation clipping artwork is missing a valid viewBox.');
    }
    const W = vbParts[2];
    const H = vbParts[3];
    const sx = W / PROFILE.axis.png_w;
    const sy = H / PROFILE.axis.png_h;
    const X0 = PROFILE.axis.plot_x0 * sx;
    const X1 = PROFILE.axis.plot_x1 * sx;
    const Y0 = PROFILE.axis.plot_y0 * sy;
    const Y1 = PROFILE.axis.plot_y1 * sy;
    const [VMIN, VMAX] = PROFILE.axis.xlim;

    const xOf = v => X0 + ((v - VMIN) / (VMAX - VMIN)) * (X1 - X0);

    // Interactive threshold slider placed in viewControls
    const toolbar = htmlEl('div', 'ac-toolbar');
    const sliderLabel = htmlEl('label', 'ac-slider');
    sliderLabel.append(htmlEl('span', '', 'Clip threshold β'));
    const sliderInput = document.createElement('input');
    sliderInput.type = 'range';
    sliderInput.min = '0.40';
    sliderInput.max = '2.0922';
    sliderInput.step = '0.005';
    sliderInput.value = '2.0922';
    sliderInput.setAttribute('aria-label', 'Clip threshold beta');
    const sliderVal = htmlEl('span', 'ac-slider-val', 'β = 2.09');
    sliderLabel.append(sliderInput, sliderVal);
    toolbar.append(sliderLabel);

    let ui = null;
    let currentStepIndex = 0;

    function ensureScene(stage, details) {
      if (ui) return ui;
      stage.replaceChildren();
      const wrap = htmlEl('div', 'ac-source');
      const svg = verifiedArtwork.cloneNode(true);
      svg.removeAttribute('width');
      svg.removeAttribute('height');
      svg.setAttribute('role', 'img');
      svg.setAttribute('aria-label', 'Post-ReLU activation histogram with interactive UINT8 clipping threshold');

      const overlay = svgEl('g', { class: 'ac-overlay' });
      const tailCurtain = svgEl('rect', {
        y: (Y0 + 18 * sy).toFixed(1),
        height: (Y1 - Y0 - 30 * sy).toFixed(1),
        rx: (6 * sx).toFixed(1),
        fill: '#FAD9DC',
        'fill-opacity': '0.38',
        stroke: '#CB202D',
        'stroke-width': (3 * sx).toFixed(1),
        'stroke-dasharray': `${(10 * sx).toFixed(1)} ${(7 * sx).toFixed(1)}`
      });
      const rulerGroup = svgEl('g', { class: 'ac-ruler' });
      const rulerTrack = svgEl('rect', {
        y: (Y0 + 26 * sy).toFixed(1),
        height: (36 * sy).toFixed(1),
        rx: (6 * sx).toFixed(1),
        fill: '#D1E6F3',
        'fill-opacity': '0.88',
        stroke: '#006395',
        'stroke-width': (3 * sx).toFixed(1)
      });
      const rulerTicks = [];
      for (let t = 1; t < 16; t += 1) {
        const tick = svgEl('line', {
          y1: (Y0 + 28 * sy).toFixed(1),
          y2: (Y0 + 60 * sy).toFixed(1),
          stroke: '#006395',
          'stroke-opacity': '0.55',
          'stroke-width': (2 * sx).toFixed(1)
        });
        rulerTicks.push(tick);
      }
      rulerGroup.append(rulerTrack, ...rulerTicks);

      const cursorLine = svgEl('line', {
        y1: (Y0 + 16 * sy).toFixed(1),
        y2: (Y1 - 12 * sy).toFixed(1),
        stroke: '#006395',
        'stroke-width': (6.5 * sx).toFixed(1)
      });
      const cursorDot = svgEl('circle', {
        cy: (Y0 + 44 * sy).toFixed(1),
        r: (13 * sx).toFixed(1),
        fill: '#006395',
        stroke: '#ffffff',
        'stroke-width': (4 * sx).toFixed(1)
      });

      const badgeGroup = svgEl('g', { class: 'ac-badge' });
      const badgeBg = svgEl('rect', {
        width: (690 * sx).toFixed(1),
        height: (136 * sy).toFixed(1),
        rx: (14 * sx).toFixed(1),
        fill: '#ffffff',
        'fill-opacity': '0.96',
        stroke: '#006395',
        'stroke-width': (3.5 * sx).toFixed(1)
      });
      const badgeTitle = svgEl('text', {
        'font-family': 'system-ui, -apple-system, sans-serif',
        'font-size': (42 * sx).toFixed(1),
        'font-weight': '700',
        fill: '#006395'
      });
      const badgeSub = svgEl('text', {
        'font-family': 'system-ui, -apple-system, sans-serif',
        'font-size': (35 * sx).toFixed(1),
        'font-weight': '600',
        fill: '#334155'
      });
      badgeGroup.append(badgeBg, badgeTitle, badgeSub);

      const focusBox = svgEl('rect', {
        fill: 'none',
        stroke: 'none',
        'pointer-events': 'none',
        'data-mlsys-focus': ''
      });

      overlay.append(tailCurtain, rulerGroup, cursorLine, cursorDot, badgeGroup, focusBox);
      svg.append(overlay);
      wrap.append(svg);
      stage.append(wrap);

      // Details cards + preset table
      details.replaceChildren();
      const readout = htmlEl('div', 'ac-readout');
      const cardSpecs = [
        ['beta', 'Clip threshold β'],
        ['scale', 'UINT8 step Δ = β/255'],
        ['share', 'Clipped activations'],
        ['quant', 'Rounding RMSE'],
        ['total', 'Total RMSE']
      ];
      const cards = {};
      cardSpecs.forEach(([kind, title]) => {
        const c = htmlEl('div', 'ac-card');
        c.dataset.kind = kind;
        const nm = htmlEl('span', 'ac-name', title);
        const vl = htmlEl('span', 'ac-value');
        const nt = htmlEl('span', 'ac-note');
        c.append(nm, vl, nt);
        readout.append(c);
        cards[kind] = { vl, nt };
      });

      const cap = htmlEl('p', 'ac-cap', 'Calibration comparison across the 32,904,484-activation layer profile:');
      const table = htmlEl('div', 'ac-compare');
      const head = htmlEl('div', 'ac-row ac-head');
      ['β', 'Δ (UINT8)', 'Clipped %', 'Total RMSE'].forEach(label => head.append(htmlEl('span', '', label)));
      table.append(head);
      const presetRows = PROFILE.presets.map(p => {
        const r = htmlEl('div', 'ac-row ac-data');
        r.dataset.beta = String(p.beta);
        r.append(
          htmlEl('span', '', p.beta.toFixed(2)),
          htmlEl('span', '', p.scale.toFixed(6)),
          htmlEl('span', '', p.clipped_pct.toFixed(4) + '%'),
          htmlEl('span', '', p.rmse_total.toFixed(6))
        );
        table.append(r);
        return r;
      });
      details.append(readout, cap, table);

      ui = {
        svg, tailCurtain, rulerTrack, rulerTicks, cursorLine, cursorDot,
        badgeBg, badgeTitle, badgeSub, focusBox, cards, presetRows
      };
      return ui;
    }

    function paintBeta(beta, modeStep) {
      if (!ui) return;
      const stats = evaluateBeta(beta);
      const xZero = xOf(0);
      const xBeta = xOf(stats.beta);
      const xMax = xOf(PROFILE.markers.max);
      const isTailSevere = stats.rmseClip > stats.rmseQuant;
      const accent = modeStep === 0 ? '#006395' : (isTailSevere ? '#CB202D' : (Math.abs(stats.beta - 1.394) < 0.03 ? '#7E317B' : '#008F45'));

      sliderInput.value = stats.beta.toFixed(4);
      sliderVal.textContent = 'β = ' + stats.beta.toFixed(2);
      root.dataset.beta = stats.beta.toFixed(4);

      const tailW = Math.max(0, xMax - xBeta);
      ui.tailCurtain.setAttribute('x', xBeta.toFixed(1));
      ui.tailCurtain.setAttribute('width', tailW.toFixed(1));
      ui.tailCurtain.setAttribute('visibility', tailW > 2 * sx ? 'visible' : 'hidden');

      const spanW = Math.max(8 * sx, xBeta - xZero);
      ui.rulerTrack.setAttribute('x', xZero.toFixed(1));
      ui.rulerTrack.setAttribute('width', spanW.toFixed(1));
      ui.rulerTicks.forEach((tick, idx) => {
        const tx = xZero + (spanW * (idx + 1)) / 16;
        tick.setAttribute('x1', tx.toFixed(1));
        tick.setAttribute('x2', tx.toFixed(1));
      });

      ui.cursorLine.setAttribute('x1', xBeta.toFixed(1));
      ui.cursorLine.setAttribute('x2', xBeta.toFixed(1));
      ui.cursorLine.setAttribute('stroke', accent);
      ui.cursorDot.setAttribute('cx', xBeta.toFixed(1));
      ui.cursorDot.setAttribute('fill', accent);

      const bw = 690 * sx;
      const bh = 136 * sy;
      const bx = stats.beta > 1.52 ? (xBeta - bw - 26 * sx) : Math.min(X1 - bw - 20 * sx, xBeta + 26 * sx);
      const by = Y0 + 720 * sy;
      ui.badgeBg.setAttribute('x', bx.toFixed(1));
      ui.badgeBg.setAttribute('y', by.toFixed(1));
      ui.badgeBg.setAttribute('stroke', accent);
      ui.badgeTitle.setAttribute('x', (bx + 24 * sx).toFixed(1));
      ui.badgeTitle.setAttribute('y', (by + 56 * sy).toFixed(1));
      ui.badgeTitle.setAttribute('fill', accent);
      ui.badgeTitle.textContent = `β = ${stats.beta.toFixed(2)} · Δ = ${stats.scale.toFixed(5)}`;
      ui.badgeSub.setAttribute('x', (bx + 24 * sx).toFixed(1));
      ui.badgeSub.setAttribute('y', (by + 108 * sy).toFixed(1));
      ui.badgeSub.textContent = `RMSE ${stats.rmseTotal.toFixed(5)} (${stats.clippedPct.toFixed(4)}% clipped)`;

      // Keep interior focus rectangle strictly inside the outer SVG frame (>= 16 px margin)
      const fLeft = Math.max(X0 + 18 * sx, Math.min(bx, xBeta - 36 * sx));
      const fRight = Math.min(X1 - 18 * sx, Math.max(bx + bw, xBeta + 36 * sx));
      ui.focusBox.setAttribute('x', fLeft.toFixed(1));
      ui.focusBox.setAttribute('y', (Y0 + 24 * sy).toFixed(1));
      ui.focusBox.setAttribute('width', Math.max(80 * sx, fRight - fLeft).toFixed(1));
      ui.focusBox.setAttribute('height', (Y1 - Y0 - 48 * sy).toFixed(1));

      ui.cards.beta.vl.textContent = stats.beta.toFixed(4);
      ui.cards.beta.nt.textContent = '256 UINT8 levels on [0, β]';
      ui.cards.scale.vl.textContent = stats.scale.toFixed(6);
      ui.cards.scale.nt.textContent = 'step width β / 255';
      ui.cards.share.vl.textContent = stats.clipped.toLocaleString('en-US');
      ui.cards.share.nt.textContent = stats.clippedPct.toFixed(4) + '% of 32.90M';
      ui.cards.quant.vl.textContent = stats.rmseQuant.toFixed(6);
      ui.cards.quant.nt.textContent = 'in-range grid rounding';
      ui.cards.total.vl.textContent = stats.rmseTotal.toFixed(6);
      ui.cards.total.nt.textContent = 'clip RMSE ' + stats.rmseClip.toFixed(6);

      ui.presetRows.forEach(row => {
        const rb = Number(row.dataset.beta);
        row.classList.toggle('is-current', Math.abs(rb - stats.beta) < 0.025);
      });
    }

    sliderInput.addEventListener('input', () => {
      if (player && player.state && player.state.playing) player.pause();
      paintBeta(Number(sliderInput.value), currentStepIndex);
    });

    const player = window.MLSysOriginalPlayer.create({
      root,
      outer,
      layout: 'compact',
      title: 'How does activation clipping trade grid resolution against tail saturation?',
      intro: 'Move the clip threshold from 2.09 down to 1.39, 1.12, and 0.50, and watch quantization step width trade off against clipped-tail error.',
      scope: 'Computed from the digitized 100-bin ResNet-50 layer1.0.conv2 post-ReLU profile (Wu et al., 2020; 32,904,484 activations) with asymmetric UINT8 quantization on [0, β] (zero-point 0, 256 levels).',
      viewControls: toolbar,
      steps: STEPS,
      render(index, { stage, details }) {
        currentStepIndex = index;
        ensureScene(stage, details);
        paintBeta(STEP_BETAS[index], index);
      },
      transition({ from, to, stage, details, onFrame }) {
        currentStepIndex = to;
        ensureScene(stage, details);
        const start = STEP_BETAS[from];
        const end = STEP_BETAS[to];
        onFrame(t => {
          const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
          paintBeta(start + (end - start) * eased, to);
        });
      }
    });

    return player;
  };
})();
