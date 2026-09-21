/* ============================================================
   Tool Trail — Volume III Decision Game
   Teaches: Tool actuation, retries, and trajectory recovery.
   ============================================================ */

import { shell, button, progressDots, agentTrack } from "./arcade.mjs";

export const toolRounds = [
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

if (typeof window !== "undefined") {
  window.MLSP = window.MLSP || {};
  window.MLSP.games = window.MLSP.games || {};
  window.MLSP.games.toolTrail = mountToolTrail;
  window.MLSP.games["tool-trail"] = mountToolTrail;
}
