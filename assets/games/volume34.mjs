/* ============================================================
   MLSysBook Arcade — Backward-compatibility bridge
   Re-exports Volume III & IV games and shared primitives from
   their dedicated modules.
   ============================================================ */

export { shell, button, progressDots, agentTrack } from "./arcade.mjs";
export { contextRounds, mountContextCache } from "./context-cache.mjs";
export { toolRounds, mountToolTrail } from "./tool-trail.mjs";
export { latencyStages, latencyRounds, mountLatencyLine } from "./latency-line.mjs";
export { mountSafetyGate } from "./safety-gate.mjs";
