/* ============================================================
   MLSysBook Arcade — Shared DOM UI primitives
   Lightweight, accessible card & track layouts for decision games.
   ============================================================ */

export function shell(root, { volume, title, lead, stats, body, feedback = "", actions = "" }) {
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

export function button(label, action, extra = "") {
  return `<button type="button" data-action="${action}" ${extra}>${label}</button>`;
}

export function progressDots(count, current, outcomes) {
  return `<div class="mlsp-round-dots" aria-label="Round ${Math.min(current + 1, count)} of ${count}">
    ${Array.from({ length: count }, (_, i) => `<span class="${i < current ? (outcomes[i] ? "won" : "lost") : i === current ? "current" : ""}"></span>`).join("")}
  </div>`;
}

export function agentTrack(step, moving = false) {
  return `<div class="mlsp-agent-track" role="img" aria-label="Small agent at station ${step + 1} of 4">
    <span class="mlsp-agent-rail"></span>
    ${Array.from({ length: 4 }, (_, i) => `<span class="mlsp-agent-station ${i <= step ? "visited" : ""}" style="--station:${i}">${i + 1}</span>`).join("")}
    <span class="mlsp-agent-sprite ${moving ? "moving" : ""}" style="--agent-step:${step};--agent-from:${Math.max(0, step - 1)}" aria-hidden="true">
      <i class="mlsp-agent-antenna"></i><i class="mlsp-agent-head"><b></b></i><i class="mlsp-agent-body"></i>
    </span>
  </div>`;
}
