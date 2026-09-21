/* ============================================================
   Latency Line — Volume IV Decision Game
   Teaches: End-to-end sense-plan-act deadline budgets.
   ============================================================ */

import { shell, button, progressDots } from "./arcade.mjs";

export const latencyStages = [
  { name: "Perception", options: [["Quick", 8, 1], ["Balanced", 16, 2], ["Careful", 28, 3]] },
  { name: "Planning", options: [["Quick", 7, 1], ["Balanced", 14, 2], ["Careful", 24, 3]] },
  { name: "Control link", options: [["Quick", 4, 1], ["Reliable", 9, 2], ["Redundant", 15, 3]] }
];

export const latencyRounds = [
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

if (typeof window !== "undefined") {
  window.MLSP = window.MLSP || {};
  window.MLSP.games = window.MLSP.games || {};
  window.MLSP.games.latencyLine = mountLatencyLine;
  window.MLSP.games["latency-line"] = mountLatencyLine;
}
