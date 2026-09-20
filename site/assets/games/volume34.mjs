/* Four short, self-contained systems games for Volumes III and IV.
   Timing, cost, and quality values are deliberately toy units. */

function shell(root, { volume, title, lead, stats, body, feedback = "", actions = "" }) {
  root.innerHTML = `
    <div class="mlsp-arcade-head">
      <span class="mlsp-arcade-volume">Volume ${volume}</span>
      ${volume === "IV" ? '<span class="mlsp-robot-mark" aria-hidden="true"><i></i></span>' : ""}
      <h2>${title}</h2>
      <p>${lead}</p>
    </div>
    <div class="mlsp-arcade-stats">${stats}</div>
    <div class="mlsp-arcade-body">${body}</div>
    <p class="mlsp-arcade-feedback" role="status" aria-live="polite">${feedback}</p>
    <div class="mlsp-arcade-actions">${actions}</div>`;
}

function button(label, action, extra = "") {
  return `<button type="button" data-action="${action}" ${extra}>${label}</button>`;
}

function progressDots(count, current, outcomes) {
  return `<div class="mlsp-round-dots" aria-label="Round ${Math.min(current + 1, count)} of ${count}">
    ${Array.from({ length: count }, (_, i) => `<span class="${i < current ? (outcomes[i] ? "won" : "lost") : i === current ? "current" : ""}"></span>`).join("")}
  </div>`;
}

function agentTrack(step, moving = false) {
  return `<div class="mlsp-agent-track" role="img" aria-label="Small agent at station ${step + 1} of 4">
    <span class="mlsp-agent-rail"></span>
    ${Array.from({ length: 4 }, (_, i) => `<span class="mlsp-agent-station ${i <= step ? "visited" : ""}" style="--station:${i}">${i + 1}</span>`).join("")}
    <span class="mlsp-agent-sprite ${moving ? "moving" : ""}" style="--agent-step:${step};--agent-from:${Math.max(0, step - 1)}" aria-hidden="true">
      <i class="mlsp-agent-antenna"></i><i class="mlsp-agent-head"><b></b></i><i class="mlsp-agent-body"></i>
    </span>
  </div>`;
}

const contextRounds = [
  {
    task: "A customer asks for the exact refund amount. Keep the facts the next tool call needs.",
    cards: [
      ["order", "Order total", "$42.00 paid", 2],
      ["policy", "Refund rule", "Full refund within 14 days", 3],
      ["greeting", "Old greeting", "The chat opened with hello", 1],
      ["shipping", "Shipping label", "Courier code Z17", 2],
      ["theme", "Theme preference", "Customer likes dark mode", 1]
    ],
    needed: ["order", "policy"],
    why: "The amount and refund rule together support the answer. The other notes use context without resolving it."
  },
  {
    task: "A deployment alarm fires. Keep what a recovery tool needs to choose a known-good build.",
    cards: [
      ["error", "Current error", "New build fails health checks", 2],
      ["hash", "Last good build", "Build 8c4 passed checks", 3],
      ["weather", "Weather", "Rain in the region", 1],
      ["old", "Old log", "A month-old run succeeded", 2],
      ["tone", "Tone guide", "Write concise updates", 1]
    ],
    needed: ["error", "hash"],
    why: "The live failure and verified build identify the recovery target; a stale success log does not."
  },
  {
    task: "The agent must cite a specific claim. Keep the evidence needed to verify the sentence.",
    cards: [
      ["source", "Source link", "Document URL and title", 2],
      ["span", "Evidence span", "Relevant passage and page", 3],
      ["style", "Style guide", "Prefer short sentences", 1],
      ["draft", "Old draft", "Unverified paraphrase", 2],
      ["avatar", "Avatar", "Profile icon name", 1]
    ],
    needed: ["source", "span"],
    why: "A citation needs both a retrievable source and the passage supporting this claim."
  },
  {
    task: "A paused agent resumes a tool workflow. Keep the authorization and the latest observed state.",
    cards: [
      ["scope", "Approved scope", "Only update the draft", 3],
      ["result", "Tool result", "Draft saved; no publish call", 2],
      ["banter", "Small talk", "A joke from the opening", 1],
      ["old", "Old draft", "Version before edits", 2],
      ["color", "Color choice", "Blue accent requested", 1]
    ],
    needed: ["scope", "result"],
    why: "The resumed step needs its authority boundary and current tool result, not just conversation history."
  }
];

export function mountContextCache(root) {
  let round = 0, score = 0, selected = new Set(), revealed = false, outcomes = [];
  let renderedRound = 0;
  const budget = 5;
  const render = () => {
    if (round >= contextRounds.length) {
      shell(root, {
        volume: "III", title: "Context Cache", lead: "Run complete",
        stats: `<strong>${score}/${contextRounds.length}</strong> next steps had the evidence they needed`,
        body: `<div class="mlsp-result-emblem">${score >= 3 ? "✓" : "↻"}</div><p>${score >= 3 ? "Your working set kept the useful facts close." : "Try again: preserve the facts the next action depends on."}</p>`,
        feedback: "Context selection is a working-set decision; more notes are not always more useful.",
        actions: button("Play again", "restart")
      });
      return;
    }
    const data = contextRounds[round];
    const moving = round !== renderedRound;
    renderedRound = round;
    const used = data.cards.filter(c => selected.has(c[0])).reduce((sum, c) => sum + c[3], 0);
    shell(root, {
      volume: "III", title: "Context Cache", lead: data.task,
      stats: `<strong>Mission ${round + 1}/${contextRounds.length}</strong><span>Ready ${score}</span>${progressDots(contextRounds.length, round, outcomes)}`,
      body: `${agentTrack(round, moving)}<div class="mlsp-meter-label"><span>Context used</span><strong>${used}/${budget} slots</strong></div>
        <div class="mlsp-meter"><span style="width:${used / budget * 100}%"></span></div>
        <div class="mlsp-choice-grid">${data.cards.map(c => `
          <button type="button" class="mlsp-clue ${selected.has(c[0]) ? "selected" : ""} ${revealed && data.needed.includes(c[0]) ? "needed" : ""}"
            data-action="toggle" data-id="${c[0]}" aria-pressed="${selected.has(c[0])}" ${revealed ? "disabled" : ""}>
            <span><strong>${c[1]}</strong><small>${c[3]} slot${c[3] === 1 ? "" : "s"}</small></span><span>${c[2]}</span>
          </button>`).join("")}</div>`,
      feedback: revealed ? `${outcomes[round] ? "Next step succeeded. " : "Missing evidence. "}${data.why}` : "Select notes within the budget, then run the next step.",
      actions: revealed ? button(round === contextRounds.length - 1 ? "See result" : "Next mission", "next") : button("Run next step", "check", selected.size ? "" : "disabled")
    });
  };
  root.addEventListener("click", event => {
    const target = event.target.closest("button[data-action]");
    if (!target || !root.contains(target)) return;
    if (target.dataset.action === "toggle" && !revealed) {
      const id = target.dataset.id;
      const cost = contextRounds[round].cards.find(c => c[0] === id)[3];
      const used = contextRounds[round].cards.filter(c => selected.has(c[0])).reduce((sum, c) => sum + c[3], 0);
      if (selected.has(id)) selected.delete(id);
      else if (used + cost <= budget) selected.add(id);
      else { root.querySelector(".mlsp-arcade-feedback").textContent = "Not enough slots. Evict a note first."; return; }
    } else if (target.dataset.action === "check" && selected.size) {
      const good = contextRounds[round].needed.every(id => selected.has(id));
      outcomes[round] = good;
      if (good) score++;
      revealed = true;
    } else if (target.dataset.action === "next") {
      round++; selected = new Set(); revealed = false;
    } else if (target.dataset.action === "restart") {
      round = 0; score = 0; selected = new Set(); revealed = false; outcomes = []; renderedRound = 0;
    }
    render();
  });
  render();
}

const toolRounds = [
  {
    title: "Read-only lookup timed out",
    detail: "The tool only reads a catalog. No external state changed.",
    best: "retry",
    why: "A read-only call can be retried without duplicating an external effect."
  },
  {
    title: "Ticket creation reply vanished",
    detail: "The request may already have created a ticket. No idempotency key was sent.",
    best: "stop",
    why: "Stop and reconcile the remote state before any retry. An uncertain write can duplicate a ticket."
  },
  {
    title: "Second step failed",
    detail: "The first step updated a remote record. A registered compensating action can undo that update.",
    best: "rollback",
    why: "Use the compensating action to restore the external state before resuming."
  },
  {
    title: "Tool contract changed",
    detail: "The response has an unexpected schema. No side effect is known, but the next write would use unverified fields.",
    best: "stop",
    why: "Stop at the contract boundary and validate the new schema before an effectful action."
  }
];

export function mountToolTrail(root) {
  let round = 0, score = 0, integrity = 3, choice = "", outcomes = [];
  let renderedRound = 0;
  const render = () => {
    if (round >= toolRounds.length || (integrity <= 0 && !choice)) {
      shell(root, {
        volume: "III", title: "Tool Trail", lead: integrity ? "Trace complete" : "The trace became unsafe",
        stats: `<strong>${score}/${toolRounds.length}</strong> decisions preserved the trajectory <span>Integrity ${integrity}/3</span>`,
        body: `<div class="mlsp-result-emblem">${integrity && score >= 3 ? "✓" : "↻"}</div><p>${integrity && score >= 3 ? "The agent can resume from a known state." : "Try another route through the tool failures."}</p>`,
        feedback: "Retries, compensating actions, and stops have different effects on external state.",
        actions: button("Run trail again", "restart")
      });
      return;
    }
    const data = toolRounds[round];
    const moving = round !== renderedRound;
    renderedRound = round;
    shell(root, {
      volume: "III", title: "Tool Trail", lead: "Choose the recovery action that preserves a trustworthy trajectory.",
      stats: `<strong>Event ${round + 1}/${toolRounds.length}</strong><span>Integrity ${integrity}/3</span>${progressDots(toolRounds.length, round, outcomes)}`,
      body: `${agentTrack(round, moving)}<div class="mlsp-event-card"><span>TOOL EVENT</span><h3>${data.title}</h3><p>${data.detail}</p></div>
        <div class="mlsp-trail-actions">
          ${[["retry", "↻ Retry", "Run the same call again"], ["rollback", "↶ Roll back", "Compensate for a prior effect"], ["stop", "■ Stop & inspect", "Pause before another effect"]].map(a => `
            <button type="button" data-action="choose" data-choice="${a[0]}" class="${choice === a[0] ? (choice === data.best ? "selected-good" : "selected-bad") : ""}" ${choice ? "disabled" : ""}><strong>${a[1]}</strong><small>${a[2]}</small></button>`).join("")}
        </div>`,
      feedback: choice ? `${choice === data.best ? "Trace preserved. " : "Integrity lost. "}${data.why}` : "Read the tool event, then choose one action.",
      actions: choice ? button(round === toolRounds.length - 1 || integrity <= 0 ? "See result" : "Next event", "next") : ""
    });
  };
  root.addEventListener("click", event => {
    const target = event.target.closest("button[data-action]");
    if (!target || !root.contains(target)) return;
    if (target.dataset.action === "choose" && !choice) {
      choice = target.dataset.choice;
      const good = choice === toolRounds[round].best;
      outcomes[round] = good;
      if (good) score++; else integrity--;
    } else if (target.dataset.action === "next") {
      round++; choice = "";
    } else if (target.dataset.action === "restart") {
      round = 0; score = 0; integrity = 3; choice = ""; outcomes = []; renderedRound = 0;
    }
    render();
  });
  render();
}

const latencyStages = [
  { name: "Perception", options: [["Quick", 8, 1], ["Balanced", 16, 2], ["Careful", 28, 3]] },
  { name: "Planning", options: [["Quick", 7, 1], ["Balanced", 14, 2], ["Careful", 24, 3]] },
  { name: "Control link", options: [["Quick", 4, 1], ["Reliable", 9, 2], ["Redundant", 15, 3]] }
];
const latencyRounds = [
  ["Sorting arm", 35, 5],
  ["Warehouse rover", 42, 6],
  ["Mobile manipulator", 48, 7],
  ["Fast conveyor", 56, 8]
];

export function mountLatencyLine(root) {
  let round = 0, score = 0, selection = [0, 0, 0], dispatched = false, outcomes = [];
  const render = () => {
    if (round >= latencyRounds.length) {
      shell(root, {
        volume: "IV", title: "Latency Line", lead: "Shift complete",
        stats: `<strong>${score}/${latencyRounds.length}</strong> actions met both limits`,
        body: `<div class="mlsp-result-emblem">${score >= 3 ? "✓" : "↻"}</div><p>${score >= 3 ? "You kept useful actions inside their deadlines." : "Try a different mix of speed and quality."}</p>`,
        feedback: "The deadline measures end-to-end age when the command reaches the actuator.",
        actions: button("Run another shift", "restart")
      });
      return;
    }
    const [scene, deadline, minimum] = latencyRounds[round];
    const parts = latencyStages.map((stage, i) => stage.options[selection[i]][1]);
    const total = parts.reduce((a, b) => a + b, 0);
    const quality = latencyStages.reduce((sum, stage, i) => sum + stage.options[selection[i]][2], 0);
    const success = total <= deadline && quality >= minimum;
    const scale = Math.max(total, deadline);
    shell(root, {
      volume: "IV", title: "Latency Line", lead: `${scene}: configure one sense–plan–act cycle before dispatch. Timing and quality are toy units.`,
      stats: `<strong>Frame ${round + 1}/${latencyRounds.length}</strong><span>On time ${score}</span>${progressDots(latencyRounds.length, round, outcomes)}`,
      body: `<div class="mlsp-arcade-targets"><span>Deadline <strong>${deadline} ms</strong></span><span>Needed quality <strong>${minimum}</strong></span></div>
        <div class="mlsp-latency-plot" style="--deadline-pct:${deadline / scale * 100}%" aria-label="Estimated latency ${total} milliseconds, deadline ${deadline} milliseconds">
          ${latencyStages.map((stage, i) => `<div style="width:${parts[i] / scale * 100}%" title="${stage.name}: ${parts[i]} ms">${["Sense", "Plan", "Link"][i]}</div>`).join("")}
        </div>
        <div class="mlsp-meter-label"><span>Estimated age <strong class="${total > deadline ? "over" : ""}">${total}/${deadline} ms</strong></span><span>Quality <strong class="${quality < minimum ? "over" : ""}">${quality}/${minimum}</strong></span></div>
        <div class="mlsp-stage-list">${latencyStages.map((stage, i) => `<div class="mlsp-stage"><strong>${stage.name}</strong><div>${stage.options.map((option, j) => `
          <button type="button" data-action="select" data-stage="${i}" data-option="${j}" class="${selection[i] === j ? "selected" : ""}" aria-pressed="${selection[i] === j}" ${dispatched ? "disabled" : ""}>
            ${option[0]}<small>${option[1]} ms · quality ${option[2]}</small></button>`).join("")}</div></div>`).join("")}</div>`,
      feedback: dispatched ? success ? "On time with enough information. The action crosses the boundary." : total > deadline ? "Missed the deadline: even a good command arrived too late." : "On time, but the command used too little information for this task." : "Choose a profile for each stage. The strip shows where the time goes.",
      actions: dispatched ? button(round === latencyRounds.length - 1 ? "See result" : "Next frame", "next") : button("Dispatch action", "dispatch")
    });
  };
  root.addEventListener("click", event => {
    const target = event.target.closest("button[data-action]");
    if (!target || !root.contains(target)) return;
    if (target.dataset.action === "select" && !dispatched) {
      selection[Number(target.dataset.stage)] = Number(target.dataset.option);
    } else if (target.dataset.action === "dispatch" && !dispatched) {
      const total = latencyStages.reduce((sum, stage, i) => sum + stage.options[selection[i]][1], 0);
      const quality = latencyStages.reduce((sum, stage, i) => sum + stage.options[selection[i]][2], 0);
      outcomes[round] = total <= latencyRounds[round][1] && quality >= latencyRounds[round][2];
      if (outcomes[round]) score++;
      dispatched = true;
    } else if (target.dataset.action === "next") {
      round++; selection = [0, 0, 0]; dispatched = false;
    } else if (target.dataset.action === "restart") {
      round = 0; score = 0; selection = [0, 0, 0]; dispatched = false; outcomes = [];
    }
    render();
  });
  render();
}

const hazardKeys = new Set(["1,3", "2,2", "3,3", "4,1"]);
const directions = [
  ["North", 0, -1, "↑"], ["East", 1, 0, "→"], ["South", 0, 1, "↓"], ["West", -1, 0, "←"]
];
function safe(x, y) { return x >= 0 && x < 6 && y >= 0 && y < 5 && !hazardKeys.has(`${x},${y}`); }
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
