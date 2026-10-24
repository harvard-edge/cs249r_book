/* ============================================================
   Context Cache — Volume III Decision Game
   Teaches: Context working sets & compaction.
   ============================================================ */

import { shell, button, progressDots, agentTrack } from "./arcade.mjs";

export const contextRounds = [
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

if (typeof window !== "undefined") {
  window.MLSP = window.MLSP || {};
  window.MLSP.games = window.MLSP.games || {};
  window.MLSP.games.contextCache = mountContextCache;
  window.MLSP.games["context-cache"] = mountContextCache;
}
