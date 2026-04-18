/**
 * Meditation Timer — circular volume-clock with breathing pulse + interval reminders
 */

const MODAL_ID     = 'socratiq-meditation-modal';
const NUDGE_ID     = 'socratiq-meditation-nudge';
const MAX_MINUTES  = 60;
const MAX_INTERVAL = 120; // max interval reminder: 2 hours

// Module-level interval state so it persists across modal open/close
let _reminderIntervalId  = null;
let _reminderMinutes     = 0;
let _shadowRootRef       = null; // kept for nudge → timer re-open

export function openMeditationTimer(shadowRoot) {
  if (shadowRoot.querySelector('#' + MODAL_ID)) return; // already open
  _shadowRootRef = shadowRoot;

  const modal = document.createElement('div');
  modal.id = MODAL_ID;
  modal.innerHTML = _buildHTML();

  // Inject scoped styles
  const style = document.createElement('style');
  style.textContent = _buildCSS();
  modal.appendChild(style);

  shadowRoot.appendChild(modal);
  _wire(shadowRoot, modal);
}

export function stopReminderInterval() {
  if (_reminderIntervalId) {
    clearInterval(_reminderIntervalId);
    _reminderIntervalId = null;
    _reminderMinutes = 0;
  }
}

export function closeMeditationTimer(shadowRoot) {
  shadowRoot.querySelector('#' + MODAL_ID)?.remove();
}

// ─── HTML ─────────────────────────────────────────────────────────────────────

function _buildHTML() {
  return `
  <div id="med-backdrop" style="
    position:fixed;inset:0;z-index:99998;
    background:rgba(0,0,0,0.6);backdrop-filter:blur(4px);
    display:flex;align-items:center;justify-content:center;
  ">
    <div id="med-card" style="
      background:var(--socratiq-bg,#0f172a);
      color:var(--socratiq-text,#e2e8f0);
      border-radius:24px;
      padding:36px 32px 28px;
      width:min(380px,92vw);
      box-shadow:0 24px 64px rgba(0,0,0,0.5);
      position:relative;
      text-align:center;
    ">
      <button id="med-close" style="
        position:absolute;top:14px;right:16px;
        background:none;border:none;cursor:pointer;
        color:#64748b;font-size:1.1rem;line-height:1;padding:4px;border-radius:6px;
      ">✕</button>

      <div style="font-size:1rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:#94a3b8;margin-bottom:6px;">
        Meditation Timer
      </div>
      <div id="med-status" style="font-size:0.78rem;color:#64748b;min-height:1.2em;margin-bottom:20px;">
        Drag the dial to set duration
      </div>

      <!-- Circular dial -->
      <div style="position:relative;width:220px;height:220px;margin:0 auto 24px;">
        <svg id="med-svg" viewBox="0 0 220 220" width="220" height="220" style="position:absolute;inset:0;cursor:pointer;touch-action:none;">
          <!-- Track -->
          <circle cx="110" cy="110" r="90" fill="none" stroke="#1e293b" stroke-width="18"/>
          <!-- Progress arc -->
          <circle id="med-arc" cx="110" cy="110" r="90" fill="none"
            stroke="#6366f1" stroke-width="18"
            stroke-linecap="round"
            stroke-dasharray="565.48" stroke-dashoffset="565.48"
            transform="rotate(-90 110 110)"
            style="transition:stroke 0.3s;"/>
          <!-- Thumb dot -->
          <circle id="med-thumb" cx="110" cy="20" r="10" fill="#6366f1"/>
        </svg>

        <!-- Centre display -->
        <div style="
          position:absolute;inset:0;display:flex;flex-direction:column;
          align-items:center;justify-content:center;pointer-events:none;
        ">
          <div id="med-time-display" style="font-size:2.6rem;font-weight:700;font-variant-numeric:tabular-nums;line-height:1;">
            0:00
          </div>
          <div style="font-size:0.7rem;color:#64748b;margin-top:4px;letter-spacing:.08em;text-transform:uppercase;">
            minutes
          </div>
        </div>
      </div>

      <!-- Controls -->
      <div style="display:flex;gap:12px;justify-content:center;align-items:center;margin-bottom:16px;">
        <button id="med-play" style="
          background:#6366f1;color:#fff;border:none;cursor:pointer;
          border-radius:50%;width:52px;height:52px;font-size:1.3rem;
          display:flex;align-items:center;justify-content:center;
          transition:background 0.2s;box-shadow:0 4px 14px rgba(99,102,241,0.4);
        ">▶</button>
        <button id="med-reset" style="
          background:#1e293b;color:#94a3b8;border:none;cursor:pointer;
          border-radius:50%;width:38px;height:38px;font-size:0.85rem;
          display:flex;align-items:center;justify-content:center;
        ">↺</button>
      </div>

      <!-- Interval reminder section -->
      <div style="
        margin-top:20px;border-top:1px solid #1e293b;padding-top:16px;
        display:flex;flex-direction:column;gap:10px;
      ">
        <div style="font-size:0.72rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#64748b;">
          Break Reminders
        </div>
        <div style="display:flex;align-items:center;gap:10px;justify-content:center;flex-wrap:wrap;">
          <span style="font-size:0.78rem;color:#94a3b8;">Every</span>
          <div style="position:relative;display:flex;align-items:center;">
            <input id="med-interval-input" type="number" min="1" max="120" value="30"
              style="
                width:60px;padding:4px 8px;border-radius:8px;
                background:#1e293b;color:#e2e8f0;border:1px solid #334155;
                font-size:0.82rem;text-align:center;outline:none;
              "/>
          </div>
          <span style="font-size:0.78rem;color:#94a3b8;">min</span>
          <button id="med-interval-toggle" style="
            background:#1e293b;color:#94a3b8;border:1px solid #334155;
            border-radius:8px;padding:4px 12px;font-size:0.78rem;cursor:pointer;
            transition:all 0.2s;white-space:nowrap;
          ">Enable</button>
        </div>
        <div id="med-interval-status" style="font-size:0.7rem;color:#475569;min-height:1em;"></div>
      </div>

      <div style="font-size:0.72rem;color:#475569;margin-top:12px;">
        Close anytime with <kbd style="background:#1e293b;padding:1px 5px;border-radius:4px;font-family:monospace;">Esc</kbd>
      </div>
    </div>
  </div>
  `;
}

// ─── CSS (breathing pulse) ────────────────────────────────────────────────────

function _buildCSS() {
  return `
  @keyframes med-breathe {
    0%,100% { transform: scale(1);   opacity: 1;   }
    50%      { transform: scale(1.045); opacity: 0.85; }
  }
  #med-card.breathing {
    animation: med-breathe 8s ease-in-out infinite;
  }
  #med-arc.breathing {
    filter: drop-shadow(0 0 8px #6366f1);
  }
  #med-play:hover { background: #4f46e5 !important; }
  #med-reset:hover { background: #334155 !important; }
  `;
}

// ─── Wiring ───────────────────────────────────────────────────────────────────

function _wire(shadowRoot, modal) {
  const svg       = modal.querySelector('#med-svg');
  const arc       = modal.querySelector('#med-arc');
  const thumb     = modal.querySelector('#med-thumb');
  const display   = modal.querySelector('#med-time-display');
  const playBtn   = modal.querySelector('#med-play');
  const resetBtn  = modal.querySelector('#med-reset');
  const statusEl  = modal.querySelector('#med-status');
  const card      = modal.querySelector('#med-card');

  const CX = 110, CY = 110, R = 90;
  const CIRCUMFERENCE = 2 * Math.PI * R; // 565.48

  let totalSeconds = 0;
  let remainingSeconds = 0;
  let running = false;
  let intervalId = null;
  let dragging = false;

  // ── Dial interaction ──

  function _angleToMinutes(angle) {
    // angle: 0 = top, clockwise. Map 0-360 → 0-MAX_MINUTES
    return Math.round((angle / 360) * MAX_MINUTES);
  }

  function _getAngle(e) {
    const svgRect = svg.getBoundingClientRect();
    const cx = svgRect.left + svgRect.width / 2;
    const cy = svgRect.top  + svgRect.height / 2;
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    let angle = Math.atan2(clientY - cy, clientX - cx) * (180 / Math.PI) + 90;
    if (angle < 0) angle += 360;
    return angle;
  }

  function _setMinutes(mins) {
    mins = Math.max(0, Math.min(MAX_MINUTES, mins));
    totalSeconds = mins * 60;
    remainingSeconds = totalSeconds;
    _updateDial(mins / MAX_MINUTES);
    display.textContent = _formatTime(totalSeconds);
    statusEl.textContent = mins > 0 ? `${mins} min session` : 'Drag the dial to set duration';
  }

  function _updateDial(fraction) {
    // fraction: 0 = empty, 1 = full
    const offset = CIRCUMFERENCE * (1 - fraction);
    arc.setAttribute('stroke-dashoffset', offset);

    // Move thumb dot along circle
    const angle = fraction * 360 - 90; // -90 = start at top
    const rad = angle * Math.PI / 180;
    const tx = CX + R * Math.cos(rad);
    const ty = CY + R * Math.sin(rad);
    thumb.setAttribute('cx', tx);
    thumb.setAttribute('cy', ty);

    // Colour shifts: indigo → teal as time depletes
    const hue = Math.round(245 - fraction * 60); // 245(indigo) → 185(teal)
    arc.setAttribute('stroke', `hsl(${hue},80%,60%)`);
    thumb.setAttribute('fill', `hsl(${hue},80%,60%)`);
  }

  svg.addEventListener('pointerdown', (e) => {
    if (running) return;
    dragging = true;
    svg.setPointerCapture(e.pointerId);
    _setMinutes(_angleToMinutes(_getAngle(e)));
  });
  svg.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    _setMinutes(_angleToMinutes(_getAngle(e)));
  });
  svg.addEventListener('pointerup', () => { dragging = false; });

  // ── Playback ──

  function _start() {
    if (totalSeconds === 0) return;
    if (running) { _pause(); return; }
    running = true;
    playBtn.textContent = '⏸';
    statusEl.textContent = 'Breathe…';
    card.classList.add('breathing');
    arc.classList.add('breathing');

    intervalId = setInterval(() => {
      remainingSeconds--;
      _updateDial(remainingSeconds / totalSeconds);
      display.textContent = _formatTime(remainingSeconds);

      if (remainingSeconds <= 0) {
        _finish();
      }
    }, 1000);
  }

  function _pause() {
    running = false;
    playBtn.textContent = '▶';
    statusEl.textContent = 'Paused';
    card.classList.remove('breathing');
    arc.classList.remove('breathing');
    clearInterval(intervalId);
  }

  function _reset() {
    _pause();
    remainingSeconds = totalSeconds;
    _updateDial(totalSeconds > 0 ? 1 : 0);
    display.textContent = _formatTime(totalSeconds);
    statusEl.textContent = totalSeconds > 0 ? `${Math.round(totalSeconds/60)} min session` : 'Drag the dial to set duration';
  }

  function _finish() {
    _pause();
    statusEl.textContent = '🎉 Session complete';
    display.textContent = '0:00';
    _updateDial(0);
    // Gentle bell via Web Audio
    _playBell();
  }

  playBtn.addEventListener('click', _start);
  resetBtn.addEventListener('click', _reset);

  // ── Close ──

  modal.querySelector('#med-close').addEventListener('click', () => {
    _pause();
    closeMeditationTimer(shadowRoot);
  });
  modal.querySelector('#med-backdrop').addEventListener('click', (e) => {
    if (e.target === modal.querySelector('#med-backdrop')) {
      _pause();
      closeMeditationTimer(shadowRoot);
    }
  });

  const _escHandler = (e) => {
    if (e.key === 'Escape') { _pause(); closeMeditationTimer(shadowRoot); document.removeEventListener('keydown', _escHandler); }
  };
  document.addEventListener('keydown', _escHandler);
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function _formatTime(totalSec) {
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function _playBell() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = 'sine';
    osc.frequency.setValueAtTime(528, ctx.currentTime);        // 528 Hz — solfeggio tone
    osc.frequency.exponentialRampToValueAtTime(264, ctx.currentTime + 2);
    gain.gain.setValueAtTime(0.4, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 2.5);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 2.5);
  } catch (_) { /* AudioContext not available */ }
}
