const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');
const rough = require('roughjs');

const dom = new JSDOM('<!DOCTYPE html><html><body><svg id="svg" width="960" height="620" viewBox="0 0 960 620" xmlns="http://www.w3.org/2000/svg"></svg></body></html>');
const document = dom.window.document;
const svg = document.getElementById('svg');
const rc = rough.svg(svg);

// Background
svg.appendChild(rc.rectangle(10, 10, 940, 600, {
  fill: '#FDFBF7',
  fillStyle: 'solid',
  roughness: 0.8,
  stroke: '#D3D1C7',
  strokeWidth: 1.5
}));

// Helper to create text
function addText(text, x, y, options = {}) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  el.setAttribute('x', x);
  el.setAttribute('y', y);
  el.setAttribute('font-family', options.fontFamily || 'Helvetica Neue, Arial, sans-serif');
  el.setAttribute('font-size', options.fontSize || '14px');
  el.setAttribute('font-weight', options.fontWeight || 'normal');
  el.setAttribute('fill', options.fill || '#2A2A2A');
  if (options.textAnchor) el.setAttribute('text-anchor', options.textAnchor);
  if (options.fontStyle) el.setAttribute('font-style', options.fontStyle);
  el.textContent = text;
  svg.appendChild(el);
  return el;
}

// Title
addText('THE PHYSICAL AI DUAL-BRAIN ARCHITECTURE', 480, 50, {
  fontSize: '20px',
  fontWeight: 'bold',
  fill: '#1F407A',
  textAnchor: 'middle'
});
addText('Proposal–Permission Privilege Split on Heterogeneous Silicon', 480, 75, {
  fontSize: '13px',
  fontStyle: 'italic',
  fill: '#666666',
  textAnchor: 'middle'
});

// 1. MPU Card (Untrusted Proposal Engine)
svg.appendChild(rc.rectangle(40, 105, 880, 140, {
  fill: '#F0F4FA',
  fillStyle: 'hachure',
  fillWeight: 1.2,
  hachureGap: 8,
  hachureAngle: -25,
  roughness: 1.2,
  stroke: '#1F407A',
  strokeWidth: 2
}));

addText('UNTRUSTED PROPOSAL ENGINE (Linux MPU / Edge NPU)', 65, 135, {
  fontSize: '15px',
  fontWeight: 'bold',
  fill: '#1F407A'
});
addText('• Multi-Modal Foundation Models (VLMs), Vision Encoders (ViTs / DINOv2), Action Chunk Decoders (Diffusion / ACT)', 65, 162, {
  fontSize: '13px',
  fill: '#333333'
});
addText('• Asynchronous Deliberation: 1 Hz Semantic Intent → 20–50 Hz Action Chunking (Amortizes Inference Delay)', 65, 187, {
  fontSize: '13px',
  fill: '#333333'
});
addText('• Emits: Expiring Intent Leases  pt = ⟨SE(3) Target, Workspace Bounding Volume, t_expire, Monotonic Counter⟩', 65, 212, {
  fontSize: '12.5px',
  fontWeight: 'bold',
  fill: '#007A87'
});

// Arrow 1: Proposal Dataflow
svg.appendChild(rc.line(480, 245, 480, 295, {
  roughness: 1.5,
  stroke: '#007A87',
  strokeWidth: 2.5
}));
svg.appendChild(rc.polygon([[474, 285], [480, 298], [486, 285]], {
  fill: '#007A87',
  fillStyle: 'solid',
  stroke: '#007A87',
  roughness: 1.0
}));
addText('Shared SRAM Mailbox · Expiring Proposal  pt (TTL ≤ 100 ms)', 495, 275, {
  fontSize: '12px',
  fontWeight: 'bold',
  fill: '#007A87'
});

// 2. MCU Card (Trusted Permission Authority)
svg.appendChild(rc.rectangle(40, 305, 880, 140, {
  fill: '#FBF2F3',
  fillStyle: 'hachure',
  fillWeight: 1.2,
  hachureGap: 8,
  hachureAngle: 35,
  roughness: 1.2,
  stroke: '#A51C30',
  strokeWidth: 2
}));

addText('TRUSTED PERMISSION AUTHORITY (Real-Time Bare-Metal MCU)', 65, 335, {
  fontSize: '15px',
  fontWeight: 'bold',
  fill: '#A51C30'
});
addText('• Dedicated 1000 Hz Timing Loop: Zero Dynamic Memory Heap (Static SRAM buffers only)', 65, 362, {
  fontSize: '13px',
  fill: '#333333'
});
addText('• Minimal-Intervention Control Barrier Functions (CBF: h(x) ≥ 0) + Dynamic Stopping Clearance (d_stop ≤ d_clear)', 65, 387, {
  fontSize: '13px',
  fill: '#333333'
});
addText('• Two-Stage Emergency Interlock (IEC 60204-1): Heartbeat Watchdog Timeout (> 20 ms) → SS1 Dynamic Halt → Safe Torque Off', 65, 412, {
  fontSize: '12.5px',
  fontWeight: 'bold',
  fill: '#A51C30'
});

// Arrow 2: Permitted Control Dataflow
svg.appendChild(rc.line(480, 445, 480, 495, {
  roughness: 1.5,
  stroke: '#A51C30',
  strokeWidth: 2.5
}));
svg.appendChild(rc.polygon([[474, 485], [480, 498], [486, 485]], {
  fill: '#A51C30',
  fillStyle: 'solid',
  stroke: '#A51C30',
  roughness: 1.0
}));
addText('Permitted Phase Currents  ut = permit(pt) (Space Vector PWM @ 20 kHz)', 495, 475, {
  fontSize: '12px',
  fontWeight: 'bold',
  fill: '#A51C30'
});

// 3. Physical World Card
svg.appendChild(rc.rectangle(40, 505, 880, 95, {
  fill: '#FDF8F0',
  fillStyle: 'cross-hatch',
  fillWeight: 0.8,
  hachureGap: 10,
  roughness: 1.1,
  stroke: '#B87333',
  strokeWidth: 2
}));

addText('THE PHYSICAL WORLD (Irreversible State Mutation:  Wt ──► Wt+1)', 65, 535, {
  fontSize: '15px',
  fontWeight: 'bold',
  fill: '#B87333'
});
addText('• 3-Phase MOSFET Inverters · Stator Magnetic Flux · Kinetic Momentum (p = mv) · Joule Heating (I²R)', 65, 560, {
  fontSize: '13px',
  fill: '#333333'
});
addText('• Endogenous Feedback: Physical Mutation Wt+1 Instantly Shapes Next Sensory Observation Ot+1', 65, 582, {
  fontSize: '12.5px',
  fontStyle: 'italic',
  fill: '#555555'
});

const outPath = path.resolve(__dirname, 'demo_roughjs.svg');
fs.writeFileSync(outPath, svg.outerHTML);
console.log('Saved Rough.js SVG to:', outPath);
