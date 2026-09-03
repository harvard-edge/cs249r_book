const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');
const rough = require('roughjs');
const { execSync } = require('child_process');

const dom = new JSDOM('<!DOCTYPE html><html><body><svg id="svg" width="960" height="600" viewBox="0 0 960 600" xmlns="http://www.w3.org/2000/svg"></svg></body></html>');
const document = dom.window.document;
const svg = document.getElementById('svg');
const rc = rough.svg(svg);

// Background
svg.appendChild(rc.rectangle(10, 10, 940, 580, {
  fill: '#FDFBF7',
  fillStyle: 'solid',
  roughness: 0.8,
  stroke: '#D3D1C7',
  strokeWidth: 1.5
}));

function addText(text, x, y, options = {}) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  el.setAttribute('x', x);
  el.setAttribute('y', y);
  el.setAttribute('font-family', options.fontFamily || 'Helvetica Neue, Arial, sans-serif');
  el.setAttribute('font-size', options.fontSize || '13px');
  el.setAttribute('font-weight', options.fontWeight || 'normal');
  el.setAttribute('fill', options.fill || '#2A2A2A');
  if (options.textAnchor) el.setAttribute('text-anchor', options.textAnchor);
  if (options.fontStyle) el.setAttribute('font-style', options.fontStyle);
  el.textContent = text;
  svg.appendChild(el);
  return el;
}

// Title
addText('THE PHYSICAL AI CURRICULUM BLUEPRINT & ROADMAP', 480, 44, {
  fontSize: '18px',
  fontWeight: 'bold',
  fill: '#1F407A',
  textAnchor: 'middle'
});
addText('Systematic Deconstruction: From Causal Foundations to Pipeline Organs & Real-World Assurance', 480, 64, {
  fontSize: '12.5px',
  fontStyle: 'italic',
  fill: '#666666',
  textAnchor: 'middle'
});

// PART 1
svg.appendChild(rc.rectangle(35, 80, 890, 135, {
  fill: '#FDF8F0',
  fillStyle: 'hachure',
  fillWeight: 1.0,
  hachureGap: 10,
  hachureAngle: -20,
  roughness: 1.2,
  stroke: '#B87333',
  strokeWidth: 1.8
}));
addText('PART 1: THE FOUNDATIONAL TRIAD (THE REALM OF PHYSICS & LATENCY)', 50, 102, {
  fontSize: '13.5px',
  fontWeight: 'bold',
  fill: '#B87333'
});

// Part 1 Cards
const p1Cards = [
  { ch: 'CHAPTER 1', name: 'Physical Causality', d1: 'Causal boundary, 3 criteria,', d2: 'irreversible mutation (Wt → Wt+1)', x: 50 },
  { ch: 'CHAPTER 2', name: 'Time & Latency Metrology', d1: '7-stage latency ledger, tail P99,', d2: 'freshness wall & stopping distance', x: 340 },
  { ch: 'CHAPTER 3', name: 'The Agent Workflow', d1: 'Great Tug-of-War, 9-station lifecycle,', d2: '3 cadences & multi-rate contracts', x: 630 }
];
p1Cards.forEach(c => {
  svg.appendChild(rc.rectangle(c.x, 118, 270, 85, {
    fill: '#FFFFFF',
    fillStyle: 'solid',
    roughness: 1.1,
    stroke: '#B87333',
    strokeWidth: 1.2
  }));
  addText(c.ch, c.x + 12, 138, { fontSize: '11px', fontWeight: 'bold', fill: '#B87333' });
  addText(c.name, c.x + 12, 156, { fontSize: '12px', fontWeight: 'bold', fill: '#1F407A' });
  addText(c.d1, c.x + 12, 174, { fontSize: '11px', fill: '#4A5568' });
  addText(c.d2, c.x + 12, 190, { fontSize: '11px', fill: '#4A5568' });
});

// PART 2
svg.appendChild(rc.rectangle(35, 230, 890, 175, {
  fill: '#F0F4FA',
  fillStyle: 'hachure',
  fillWeight: 1.0,
  hachureGap: 10,
  hachureAngle: 25,
  roughness: 1.2,
  stroke: '#1F407A',
  strokeWidth: 1.8
}));
addText('PART 2: THE 7 CANONICAL PIPELINE ORGANS (BUILDING THE WORKFLOW)', 50, 252, {
  fontSize: '13.5px',
  fontWeight: 'bold',
  fill: '#1F407A'
});

const p2Cards = [
  { ch: 'CHAPTER 4', name: 'Perception', d1: 'MIPI DMA stream', d2: 'ViT / DINOv2 tokens', d3: '3D SE(3) affordances', tag: '[Stations 2 & 3]', col: '#1F407A', x: 50 },
  { ch: 'CHAPTER 5', name: 'World Models', d1: 'Latent JEPAs', d2: 'Coordinate trees', d3: 'Uncertainty bounds', tag: '[Station 4]', col: '#1F407A', x: 225 },
  { ch: 'CHAPTER 6', name: 'Semantic Intent', d1: '1 Hz VLMs', d2: 'Open-world goals', d3: 'Expiring leases (TTL)', tag: '[Stations 1 & 5]', col: '#1F407A', x: 400 },
  { ch: 'CHAPTER 7', name: 'Action Chunking', d1: 'Diffusion / ACT', d2: 'Delay amortization', d3: 'C² jerk continuous', tag: '[Station 6]', col: '#1F407A', x: 575 },
  { ch: 'CHAPTER 8', name: 'Real-Time Reflex', d1: '1 kHz CBF safety QP', d2: 'Proposal-permission', d3: 'IEC dynamic halts', tag: '[Station 7]', col: '#A51C30', x: 750 }
];

p2Cards.forEach(c => {
  svg.appendChild(rc.rectangle(c.x, 268, 165, 122, {
    fill: '#FFFFFF',
    fillStyle: 'solid',
    roughness: 1.1,
    stroke: c.col,
    strokeWidth: 1.2
  }));
  addText(c.ch, c.x + 10, 286, { fontSize: '11px', fontWeight: 'bold', fill: c.col });
  addText(c.name, c.x + 10, 304, { fontSize: '12px', fontWeight: 'bold', fill: '#1F407A' });
  addText(c.d1, c.x + 10, 322, { fontSize: '11px', fill: '#4A5568' });
  addText(c.d2, c.x + 10, 338, { fontSize: '11px', fill: '#4A5568' });
  addText(c.d3, c.x + 10, 354, { fontSize: '11px', fill: '#4A5568' });
  addText(c.tag, c.x + 10, 374, { fontSize: '11px', fontWeight: 'bold', fill: c.col === '#A51C30' ? '#A51C30' : '#007A87' });
});

// PART 3
svg.appendChild(rc.rectangle(35, 420, 890, 145, {
  fill: '#FBF2F3',
  fillStyle: 'hachure',
  fillWeight: 1.0,
  hachureGap: 10,
  hachureAngle: -35,
  roughness: 1.2,
  stroke: '#A51C30',
  strokeWidth: 1.8
}));
addText('PART 3: INTEGRATION, GOVERNANCE & DEPLOYMENT (SYSTEM QUALIFICATION)', 50, 442, {
  fontSize: '13.5px',
  fontWeight: 'bold',
  fill: '#A51C30'
});

const p3Cards = [
  { ch: 'CHAPTER 9', name: 'Workload Placement', d1: 'MPU vs MCU vs NPU', d2: 'UMA contention & IPC', x: 50 },
  { ch: 'CHAPTER 10', name: 'Human Governance', d1: 'Bumpless manual takeover', d2: 'Intervention tagging', x: 270 },
  { ch: 'CHAPTER 11', name: 'Assurance & Release', d1: 'STPA hazard mitigation', d2: 'Claim-Argument-Evidence', x: 490 },
  { ch: 'CHAPTER 12', name: 'Capstone Deployment', d1: 'Full dual-brain release', d2: 'Arduino UNO Q bench run', x: 710 }
];

p3Cards.forEach(c => {
  svg.appendChild(rc.rectangle(c.x, 458, 205, 92, {
    fill: '#FFFFFF',
    fillStyle: 'solid',
    roughness: 1.1,
    stroke: '#A51C30',
    strokeWidth: 1.2
  }));
  addText(c.ch, c.x + 10, 476, { fontSize: '11px', fontWeight: 'bold', fill: '#A51C30' });
  addText(c.name, c.x + 10, 494, { fontSize: '12px', fontWeight: 'bold', fill: '#1F407A' });
  addText(c.d1, c.x + 10, 514, { fontSize: '11px', fill: '#4A5568' });
  addText(c.d2, c.x + 10, 530, { fontSize: '11px', fill: '#4A5568' });
});

const outSvg = path.resolve(__dirname, 'demo_rough_blueprint.svg');
const outPng = path.resolve(__dirname, 'demo_rough_blueprint.png');
fs.writeFileSync(outSvg, svg.outerHTML);
execSync(`/opt/homebrew/bin/rsvg-convert -w 1600 -f png -o ${outPng} ${outSvg}`);
console.log('Saved Rough Blueprint to:', outPng);
