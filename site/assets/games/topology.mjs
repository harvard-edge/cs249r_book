import { mountPixiOnCanvas, burst, flash, floatText, shake, tween } from "/assets/games/runtime.mjs";
import * as P from "/assets/games/vendor/pixi.min.mjs";

window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};
window.MLSP.games.topology = function(canvas, opts) { return mountTopology(canvas, opts); };

export async function mountTopology(canvas, opts = {}) {
  const { app, stage, width: W, height: H, onTick } = await mountPixiOnCanvas(canvas, { bg: 0x11151c });

  const COL = {
    bg: 0x11151c,
    node: 0x2a3b4c,
    nodeSelect: 0xeeeeee,
    ib: 0x4a90c4, // Slow InfiniBand
    nvlink: 0x3d9e5a, // Fast NVLink
    packet: 0xeeeeee,
    jam: 0xc44444,
    text: 0xeeeeee
  };

  const state = {
    phase: 'build', // 'build' | 'run' | 'over'
    time: 20, // 20s build, 30s run
    score: 0,
    nvlinksUsed: 0,
    maxNvlinks: 6,
    selectedNode: null
  };

  const nodes = [];
  const links = new Map(); // key "i,j" -> type (1=IB, 2=NVLink)
  const linkGraphics = new Map(); // key "i,j" -> P.Graphics
  const packets = [];
  
  const gameLayer = new P.Container();
  const linkLayer = new P.Container();
  const packetLayer = new P.Container();
  stage.addChild(linkLayer, packetLayer, gameLayer);

  const packetPool = Array.from({ length: 200 }, () => {
    const g = new P.Graphics();
    g.circle(0, 0, 4).fill({ color: COL.packet });
    g.visible = false;
    packetLayer.addChild(g);
    return g;
  });

  const pulseLayer = new P.Graphics();
  pulseLayer.rect(0, 0, W, H).fill({ color: COL.ib });
  pulseLayer.alpha = 0;
  stage.addChild(pulseLayer);

  // Layout 8 nodes (2 rows of 4)
  const cols = 4, rows = 2;
  const paddingX = 100, paddingY = 80;
  const startX = W / 2 - (cols - 1) * paddingX / 2;
  const startY = H / 2 - (rows - 1) * paddingY / 2;

  for (let i = 0; i < 8; i++) {
    const c = i % cols;
    const r = Math.floor(i / cols);
    const x = startX + c * paddingX;
    const y = startY + r * paddingY;

    const nodeContainer = new P.Container();
    nodeContainer.position.set(x, y);

    const box = new P.Graphics();
    box.roundRect(-20, -20, 40, 40, 6).fill({ color: COL.node });
    
    const label = new P.Text({
      text: i.toString(),
      style: { fill: 0xffffff, fontSize: 16, fontWeight: "bold" }
    });
    label.anchor.set(0.5);

    nodeContainer.addChild(box, label);
    gameLayer.addChild(nodeContainer);

    box.eventMode = 'static';
    box.cursor = 'pointer';
    box.on('pointerdown', () => handleNodeClick(i));
    
    // Also allow clicking the text label
    label.eventMode = 'static';
    label.cursor = 'pointer';
    label.on('pointerdown', () => handleNodeClick(i));

    nodes.push({ x, y, box, container: nodeContainer });
  }

  function getLinkKey(u, v) { return u < v ? `${u},${v}` : `${v},${u}`; }

  function handleNodeClick(i) {
    if (state.phase !== 'build') return;

    if (state.selectedNode === null) {
      state.selectedNode = i;
      nodes[i].box.tint = COL.nodeSelect;
    } else if (state.selectedNode === i) {
      nodes[i].box.tint = 0xffffff;
      state.selectedNode = null;
    } else {
      const u = state.selectedNode;
      const v = i;
      const key = getLinkKey(u, v);
      let type = links.get(key) || 0;

      if (type === 0) {
        type = 1; // IB
      } else if (type === 1) {
        if (state.nvlinksUsed < state.maxNvlinks) {
          type = 2; // NVLink
          state.nvlinksUsed++;
        } else {
          type = 0; // Skip NVLink if full
        }
      } else if (type === 2) {
        type = 0; // None
        state.nvlinksUsed--;
      }
      
      if (type === 0) links.delete(key);
      else links.set(key, type);

      nodes[u].box.tint = 0xffffff;
      state.selectedNode = null;
      drawLinks();
      updateHUD();
    }
  }

  function drawLinks() {
    for (const g of linkGraphics.values()) g.destroy();
    linkGraphics.clear();
    for (const [key, type] of links.entries()) {
      const [u, v] = key.split(',').map(Number);
      const width = type === 1 ? 2 : 6;
      const g = new P.Graphics();
      g.moveTo(nodes[u].x, nodes[u].y);
      g.lineTo(nodes[v].x, nodes[v].y);
      g.stroke({ width, color: 0xffffff });
      g.tint = type === 1 ? COL.ib : COL.nvlink;
      g.alpha = 0.8;
      linkLayer.addChild(g);
      linkGraphics.set(key, g);
    }
  }

  // BFS to find path
  function findPath(start, end) {
    const q = [[start]];
    const visited = new Set([start]);
    while (q.length > 0) {
      const path = q.shift();
      const curr = path[path.length - 1];
      if (curr === end) return path;

      for (let next = 0; next < 8; next++) {
        if (!visited.has(next) && links.has(getLinkKey(curr, next))) {
          visited.add(next);
          q.push([...path, next]);
        }
      }
    }
    return null;
  }

  let spawnTimer = 0;
  
  function updateHUD() {
    if (opts.onScoreChange) {
      opts.onScoreChange({
        phase: state.phase,
        time: Math.ceil(state.time),
        score: Math.floor(state.score),
        nvlinks: state.maxNvlinks - state.nvlinksUsed
      });
    }
  }

  onTick((dt) => {
    if (state.phase === 'over') return;

    state.time -= dt / 1000;
    if (state.time <= 0) {
      if (state.phase === 'build') {
        state.phase = 'run';
        state.time = 30; // 30s run phase
        if (state.selectedNode !== null) {
          nodes[state.selectedNode].box.tint = 0xffffff;
          state.selectedNode = null;
        }
      } else if (state.phase === 'run') {
        state.phase = 'over';
        if (opts.onGameOver) opts.onGameOver({ won: true, score: Math.floor(state.score) });
      }
    }
    updateHUD();

    if (state.phase === 'run') {
      spawnTimer -= dt;
      // Spawn packet more frequently as time goes on, causing bottlenecks
      const spawnRate = 200 + (state.time / 30) * 800; // 1000ms -> 200ms
      if (spawnTimer <= 0) {
        spawnTimer = spawnRate;
        const u = Math.floor(Math.random() * 8);
        let v = Math.floor(Math.random() * 8);
        while (v === u) v = Math.floor(Math.random() * 8);
        
        const path = findPath(u, v);
        if (path) {
          const pkt = packetPool.find(p => !p.visible);
          if (pkt) {
            pkt.visible = true;
            pkt.tint = 0xffffff;
            pkt.position.set(nodes[u].x, nodes[u].y);
            
            packets.push({
              sprite: pkt,
              path: path,
              pathIndex: 0,
              progress: 0,
              speed: 0 // set per link
            });
          }
        }
      }

      // Count traffic on links to detect jams
      const traffic = new Map();

      for (let i = packets.length - 1; i >= 0; i--) {
        const p = packets[i];
        const u = p.path[p.pathIndex];
        const v = p.path[p.pathIndex + 1];
        if (v === undefined) {
          // Reached destination
          burst(gameLayer, nodes[u].x, nodes[u].y, COL.nvlink, 5);
          p.sprite.visible = false;
          packets.splice(i, 1);
          state.score += 10;
          
          // Compute blue pulse for high throughput
          if (Math.random() > 0.8) {
             pulseLayer.alpha = 0.05;
             setTimeout(() => pulseLayer.alpha = 0, 50);
          }
          continue;
        }

        const linkKey = getLinkKey(u, v);
        const type = links.get(linkKey);
        
        // Traffic jam penalty
        let currentTraffic = traffic.get(linkKey) || 0;
        traffic.set(linkKey, currentTraffic + 1);
        
        let capacity = type === 2 ? 8 : 2;
        let baseSpeed = type === 2 ? 0.05 : 0.005; // dt multiplier. NVLink is 10x faster.
        
        let actualSpeed = baseSpeed;
        if (currentTraffic > capacity) {
          actualSpeed = baseSpeed * 0.2; // Slow down drastically
          p.sprite.tint = COL.jam;
        } else {
          p.sprite.tint = 0xffffff;
        }

        p.progress += actualSpeed * dt;
        if (p.progress >= 1) {
          p.progress = 0;
          p.pathIndex++;
        } else {
          const nx = nodes[u].x + (nodes[v].x - nodes[u].x) * p.progress;
          const ny = nodes[u].y + (nodes[v].y - nodes[u].y) * p.progress;
          p.sprite.position.set(nx, ny);
        }
      }

      // Draw red lines for congested links (optional visual juice)
      // Done implicitly via packet tint, but we could tint the line too.
      for (const [key, type] of links.entries()) {
        const tr = traffic.get(key) || 0;
        const cap = type === 2 ? 8 : 2;
        const congested = tr > cap;
        
        const g = linkGraphics.get(key);
        if (g) {
          g.tint = congested ? COL.jam : (type === 1 ? COL.ib : COL.nvlink);
          g.alpha = congested ? 1.0 : 0.8;
        }
      }
    }
  });

  return {
    id: "topology",
    ahaLabel: "You just experienced",
    ahaText: "Bisection Bandwidth bottlenecks. A cluster's throughput isn't just about raw GPU speed; it's about the fabric connecting them. Fast NVLinks handle huge traffic without dropping speeds, while slow InfiniBand links quickly turn into traffic jams under load."
  };
}