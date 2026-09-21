/* ============================================================
   Safety Gate — Volume IV Decision Game
   Teaches: Safe set projection & control gate enforcement.
   ============================================================ */

import { shell, button } from "./arcade.mjs";

const hazardKeys = new Set(["1,3", "2,2", "3,3", "4,1"]);
const directions = [
  ["North", 0, -1, "↑"], ["East", 1, 0, "→"], ["South", 0, 1, "↓"], ["West", -1, 0, "←"]
];

function safe(x, y) {
  return x >= 0 && x < 6 && y >= 0 && y < 5 && !hazardKeys.has(`${x},${y}`);
}

function shortestStep(x, y) {
  const queue = [[x, y, null]], seen = new Set([`${x},${y}`]);
  for (let i = 0; i < queue.length; i++) {
    const [cx, cy, first] = queue[i];
    if (cx === 5 && cy === 0) return first;
    for (const dir of directions) {
      const nx = cx + dir[1], ny = cy + dir[2], key = `${nx},${ny}`;
      if (safe(nx, ny) && !seen.has(key)) { seen.add(key); queue.push([nx, ny, first || dir]); }
    }
  }
  return null;
}

export function mountSafetyGate(root) {
  let x = 0, y = 4, tick = 0, score = 0, over = false, message = "";
  const reset = () => { x = 0; y = 4; tick = 0; score = 0; over = false; message = ""; };
  const proposal = () => {
    const route = shortestStep(x, y);
    if (tick % 3 === 1) {
      const unsafe = directions.find(d => !safe(x + d[1], y + d[2]));
      if (unsafe) return unsafe;
    }
    return route;
  };
  const render = () => {
    const direction = proposal();
    const proposedX = direction ? x + direction[1] : x;
    const proposedY = direction ? y + direction[2] : y;
    const unsafe = !safe(proposedX, proposedY);
    const won = x === 5 && y === 0;
    shell(root, {
      volume: "IV", title: "Safety Gate", lead: "Guide a rover to the dock. The learned policy proposes; the safety gate decides what may move.",
      stats: `<strong>Tick ${tick}/12</strong><span>Safe decisions ${score}</span><span>Dock at (5, 0)</span>`,
      body: `<div class="mlsp-safety-layout"><div class="mlsp-safety-grid" role="img" aria-label="Rover at column ${x + 1}, row ${y + 1}. Red cells are obstacles. Dock at top right.">
        ${Array.from({ length: 30 }, (_, n) => {
          const cx = n % 6, cy = Math.floor(n / 6);
          const classes = [hazardKeys.has(`${cx},${cy}`) ? "hazard" : "", cx === 5 && cy === 0 ? "dock" : "", cx === x && cy === y ? "rover" : "", !over && cx === proposedX && cy === proposedY && safe(cx, cy) ? "proposal" : ""].join(" ");
          return `<span class="${classes}">${cx === x && cy === y ? '<b class="mlsp-rover-sprite" aria-hidden="true"><i></i></b>' : cx === 5 && cy === 0 ? "★" : hazardKeys.has(`${cx},${cy}`) ? "×" : ""}</span>`;
        }).join("")}</div>
        <div class="mlsp-safety-panel"><span>POLICY PROPOSAL</span><strong>${over ? won ? "Dock reached" : "Run ended" : `${direction?.[3] || "■"} ${direction?.[0] || "Hold"}`}</strong>
        <p>Red cells block motion. Project an unsafe proposal to a safe route. Projecting a safe one adds a wasted tick.</p></div></div>`,
      feedback: message || (over ? won ? "The rover reached the dock without crossing an obstacle." : "Try again and keep the physical boundary intact." : "Inspect the proposed destination, then allow, project, or brake."),
      actions: over ? button("Try again", "restart") : `${button("Allow proposal", "allow")}${button("Project to safe route", "project")}${button("Brake", "brake")}`
    });
  };
  root.addEventListener("click", event => {
    const target = event.target.closest("button[data-action]");
    if (!target || !root.contains(target)) return;
    const action = target.dataset.action;
    if (action === "restart") { reset(); render(); return; }
    if (over) return;
    const direction = proposal();
    const unsafe = !safe(x + direction[1], y + direction[2]);
    tick++;
    if (action === "allow") {
      if (unsafe) { over = true; message = "Unsafe command crossed the gate. The rover hit a boundary."; }
      else { x += direction[1]; y += direction[2]; score++; message = "Safe proposal allowed."; }
    } else if (action === "project") {
      const move = unsafe ? shortestStep(x, y) : direction;
      x += move[1]; y += move[2];
      if (unsafe) { score++; message = "Unsafe proposal rejected; the command was projected onto a safe route."; }
      else { tick++; message = "The proposal was already safe. Unnecessary projection spent an extra tick."; }
    } else if (action === "brake") {
      message = "Rover held position. Braking preserves safety but spends one control tick.";
    }
    if (tick > 12) { over = true; message = "Deadline exceeded. Extra interventions and holds used the control budget."; }
    else if (x === 5 && y === 0) { over = true; message = "Dock reached! Every physical move stayed within the safe set."; }
    else if (tick >= 12) { over = true; message = "Deadline reached before the dock. Too many holds or extra interventions consumed the control budget."; }
    render();
  });
  render();
}

if (typeof window !== "undefined") {
  window.MLSP = window.MLSP || {};
  window.MLSP.games = window.MLSP.games || {};
  window.MLSP.games.safetyGate = mountSafetyGate;
  window.MLSP.games["safety-gate"] = mountSafetyGate;
}
