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
    background:rgba(0,0,0,0.65);backdrop-filter:blur(6px);
    display:flex;align-items:center;justify-content:center;
    overflow-y:auto;padding:16px 0;
  ">
    <div id="med-card" style="
      background:var(--socratiq-bg,#ffffff);
      color:var(--socratiq-text,#1f2328);
      border-radius:24px;
      padding:32px 28px 24px;
      width:min(400px,94vw);
      box-shadow:0 24px 64px rgba(0,0,0,0.3);
      border:1px solid var(--socratiq-border,#e5e7eb);
      position:relative;
      text-align:center;
    ">
      <!-- Close button -->
      <button id="med-close" title="Close (Esc)" style="
        position:absolute;top:12px;right:12px;
        background:var(--socratiq-input-bg,#f3f4f6);border:1px solid var(--socratiq-border,#e5e7eb);cursor:pointer;
        color:var(--socratiq-text,#1f2328);font-size:0.8rem;line-height:1;
        padding:5px 9px;border-radius:8px;
        display:flex;align-items:center;gap:5px;
      ">
        <span style="font-size:0.9rem;">✕</span>
        <kbd style="font-size:0.65rem;background:var(--socratiq-border,#e5e7eb);padding:1px 4px;border-radius:4px;font-family:monospace;color:var(--socratiq-text-muted,#6b7280);">Esc</kbd>
      </button>

      <div style="font-size:0.9rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:var(--socratiq-text-muted,#6b7280);margin-bottom:4px;">
        🧘 Meditation Timer
      </div>
      <div id="med-status" style="font-size:0.75rem;color:var(--socratiq-text-muted,#6b7280);min-height:1.2em;margin-bottom:18px;">
        Drag the dial to set duration
      </div>

      <!-- Circular dial -->
      <div style="position:relative;width:210px;height:210px;margin:0 auto 20px;">
        <svg id="med-svg" viewBox="0 0 220 220" width="210" height="210" style="position:absolute;inset:0;cursor:pointer;touch-action:none;">
          <circle cx="110" cy="110" r="90" fill="none" stroke="var(--socratiq-border,#e5e7eb)" stroke-width="18"/>
          <circle id="med-arc" cx="110" cy="110" r="90" fill="none"
            stroke="#6366f1" stroke-width="18"
            stroke-linecap="round"
            stroke-dasharray="565.48" stroke-dashoffset="565.48"
            transform="rotate(-90 110 110)"
            style="transition:stroke 0.3s;"/>
          <circle id="med-thumb" cx="110" cy="20" r="10" fill="#6366f1"/>
        </svg>
        <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;pointer-events:none;">
          <div id="med-time-display" style="font-size:2.4rem;font-weight:700;font-variant-numeric:tabular-nums;line-height:1;">0:00</div>
          <div style="font-size:0.65rem;color:var(--socratiq-text-muted,#6b7280);margin-top:4px;letter-spacing:.08em;text-transform:uppercase;">minutes</div>
        </div>
      </div>

      <!-- Play / Reset -->
      <div style="display:flex;gap:12px;justify-content:center;align-items:center;margin-bottom:20px;">
        <button id="med-play" style="
          background:#6366f1;color:#fff;border:none;cursor:pointer;
          border-radius:50%;width:52px;height:52px;font-size:1.3rem;
          display:flex;align-items:center;justify-content:center;
          transition:background 0.2s;box-shadow:0 4px 14px rgba(99,102,241,0.4);
        ">▶</button>
        <button id="med-reset" style="
          background:var(--socratiq-input-bg,#f3f4f6);color:var(--socratiq-text-muted,#6b7280);border:1px solid var(--socratiq-border,#e5e7eb);cursor:pointer;
          border-radius:50%;width:38px;height:38px;font-size:0.95rem;
          display:flex;align-items:center;justify-content:center;
        ">↺</button>
        <button id="med-pulse-toggle" title="Toggle breathing pulse" style="
          background:var(--socratiq-input-bg,#f3f4f6);color:var(--socratiq-text-muted,#6b7280);border:1px solid var(--socratiq-border,#e5e7eb);cursor:pointer;
          border-radius:20px;padding:0 12px;height:38px;font-size:0.72rem;
          display:flex;align-items:center;gap:5px;white-space:nowrap;
          transition:all 0.2s;
        "><span id="med-pulse-icon" style="font-size:0.8rem;">〇</span><span id="med-pulse-label">Pulse: on</span></button>
      </div>

      <!-- Sound picker -->
      <div style="border-top:1px solid var(--socratiq-border,#e5e7eb);padding-top:16px;margin-bottom:16px;">
        <div style="font-size:0.68rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--socratiq-text-muted,#6b7280);margin-bottom:10px;">
          Completion Sound
        </div>
        <div style="display:flex;gap:6px;justify-content:center;flex-wrap:wrap;" id="med-sound-picker">
          <button class="med-sound-opt active" data-sound="bell"    style="padding:5px 12px;border-radius:8px;font-size:0.75rem;cursor:pointer;border:1px solid var(--socratiq-border,#e5e7eb);background:var(--socratiq-input-bg,#f3f4f6);color:var(--socratiq-text-muted,#6b7280);">🔔 Bell</button>
          <button class="med-sound-opt"        data-sound="bowl"    style="padding:5px 12px;border-radius:8px;font-size:0.75rem;cursor:pointer;border:1px solid var(--socratiq-border,#e5e7eb);background:var(--socratiq-input-bg,#f3f4f6);color:var(--socratiq-text-muted,#6b7280);">🎵 Bowl</button>
          <button class="med-sound-opt"        data-sound="chime"   style="padding:5px 12px;border-radius:8px;font-size:0.75rem;cursor:pointer;border:1px solid var(--socratiq-border,#e5e7eb);background:var(--socratiq-input-bg,#f3f4f6);color:var(--socratiq-text-muted,#6b7280);">🎶 Chime</button>
          <button class="med-sound-opt"        data-sound="silence" style="padding:5px 12px;border-radius:8px;font-size:0.75rem;cursor:pointer;border:1px solid var(--socratiq-border,#e5e7eb);background:var(--socratiq-input-bg,#f3f4f6);color:var(--socratiq-text-muted,#6b7280);">🔇 None</button>
        </div>
        <button id="med-preview-sound" style="
          margin-top:8px;background:none;border:none;color:#6366f1;
          font-size:0.72rem;cursor:pointer;text-decoration:underline;
        ">▶ Preview sound</button>
      </div>

      <!-- Interval reminder section -->
      <div style="border-top:1px solid var(--socratiq-border,#e5e7eb);padding-top:16px;display:flex;flex-direction:column;gap:10px;">
        <div style="display:flex;align-items:center;justify-content:center;gap:6px;">
          <span style="font-size:0.68rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--socratiq-text-muted,#6b7280);">Break Reminders</span>
        </div>
        <div style="font-size:0.7rem;color:var(--socratiq-text-muted,#6b7280);padding:0 4px;line-height:1.5;">
          Get a gentle nudge sound + notification to step away and meditate at a set interval — even when this modal is closed.
        </div>
        <div style="display:flex;align-items:center;gap:8px;justify-content:center;flex-wrap:wrap;">
          <span style="font-size:0.78rem;color:var(--socratiq-text-muted,#6b7280);">Every</span>
          <input id="med-interval-input" type="number" min="1" max="120" value="30"
            style="width:58px;padding:4px 8px;border-radius:8px;background:var(--socratiq-input-bg,#f3f4f6);color:var(--socratiq-text,#1f2328);border:1px solid var(--socratiq-border,#e5e7eb);font-size:0.82rem;text-align:center;outline:none;"/>
          <span style="font-size:0.78rem;color:var(--socratiq-text-muted,#6b7280);">min</span>
          <button id="med-interval-toggle" style="
            background:var(--socratiq-input-bg,#f3f4f6);color:var(--socratiq-text-muted,#6b7280);border:1px solid var(--socratiq-border,#e5e7eb);
            border-radius:8px;padding:5px 12px;font-size:0.78rem;cursor:pointer;
            transition:all 0.2s;white-space:nowrap;
          ">Enable</button>
        </div>
        <div id="med-interval-status" style="font-size:0.7rem;color:var(--socratiq-text-muted,#6b7280);min-height:1em;"></div>
      </div>
    </div>
  </div>
  `;
}

// ─── CSS (breathing pulse) ────────────────────────────────────────────────────

function _buildCSS() {
  return `
  @keyframes med-breathe {
    0%,100% { transform: scale(1);     opacity: 1;    }
    50%      { transform: scale(1.035); opacity: 0.88; }
  }
  #med-card.breathing {
    animation: med-breathe 16s ease-in-out infinite;
  }
  #med-arc.breathing {
    filter: drop-shadow(0 0 8px #6366f1);
  }
  #med-play:hover { background: #4f46e5 !important; }
  #med-reset:hover { opacity: 0.8; }
  #med-interval-toggle.active {
    background: #6366f1 !important;
    color: #fff !important;
    border-color: #6366f1 !important;
  }
  #med-interval-toggle:hover { background: rgba(99,102,241,0.08) !important; color: #6366f1 !important; }
  #med-interval-input:focus { border-color: #6366f1 !important; }
  .med-sound-opt.active {
    background: #6366f1 !important;
    color: #fff !important;
    border-color: #6366f1 !important;
  }
  .med-sound-opt:hover { background: rgba(99,102,241,0.08) !important; color: #6366f1 !important; }
  #med-close:hover { opacity: 0.8; }
  #med-preview-sound:hover { color: #818cf8 !important; }

  @keyframes med-nudge-in {
    from { opacity:0; transform:translateY(-16px) scale(0.96); }
    to   { opacity:1; transform:translateY(0)     scale(1); }
  }
  #socratiq-meditation-nudge { animation: med-nudge-in 0.22s ease-out; }
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
  let selectedSound = 'bell'; // default

  // ── Sound picker ──
  const soundOpts = modal.querySelectorAll('.med-sound-opt');
  soundOpts.forEach(btn => {
    btn.addEventListener('click', () => {
      soundOpts.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      selectedSound = btn.dataset.sound;
    });
  });
  modal.querySelector('#med-preview-sound').addEventListener('click', () => {
    _playSound(selectedSound);
  });

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
    _playSound(selectedSound);
  }

  // ── Pulse toggle ──
  const pulseToggle = modal.querySelector('#med-pulse-toggle');
  const pulseIcon   = modal.querySelector('#med-pulse-icon');
  const pulseLabel  = modal.querySelector('#med-pulse-label');
  let pulseEnabled = true;

  function _applyPulseState() {
    if (pulseEnabled) {
      pulseToggle.style.color = '#6366f1';
      pulseToggle.style.borderColor = '#6366f1';
      pulseToggle.style.background = 'rgba(99,102,241,0.08)';
      pulseIcon.textContent = '◉';
      pulseLabel.textContent = 'Pulse: on';
      if (running) { card.classList.add('breathing'); arc.classList.add('breathing'); }
    } else {
      pulseToggle.style.color = '';
      pulseToggle.style.borderColor = '';
      pulseToggle.style.background = '';
      pulseIcon.textContent = '〇';
      pulseLabel.textContent = 'Pulse: off';
      card.classList.remove('breathing');
      arc.classList.remove('breathing');
    }
  }

  pulseToggle.addEventListener('click', () => {
    pulseEnabled = !pulseEnabled;
    _applyPulseState();
  });

  // Wire play through a single handler that respects pulseEnabled
  playBtn.addEventListener('click', () => {
    _start();
    if (!pulseEnabled) {
      card.classList.remove('breathing');
      arc.classList.remove('breathing');
    }
  });
  resetBtn.addEventListener('click', _reset);

  // ── Interval reminder wiring ──

  const intervalToggle = modal.querySelector('#med-interval-toggle');
  const intervalInput  = modal.querySelector('#med-interval-input');
  const intervalStatus = modal.querySelector('#med-interval-status');

  // Sync UI to current module state when modal reopens
  if (_reminderIntervalId) {
    intervalToggle.textContent = 'Disable';
    intervalToggle.classList.add('active');
    intervalInput.value = _reminderMinutes;
    intervalStatus.textContent = `Reminding every ${_reminderMinutes} min`;
  }

  intervalToggle.addEventListener('click', () => {
    if (_reminderIntervalId) {
      // Disable
      clearInterval(_reminderIntervalId);
      _reminderIntervalId = null;
      _reminderMinutes = 0;
      intervalToggle.textContent = 'Enable';
      intervalToggle.classList.remove('active');
      intervalStatus.textContent = '';
    } else {
      // Enable
      const mins = Math.max(1, Math.min(MAX_INTERVAL, parseInt(intervalInput.value) || 30));
      intervalInput.value = mins;
      _reminderMinutes = mins;
      intervalStatus.textContent = `Reminding every ${mins} min`;
      intervalToggle.textContent = 'Disable';
      intervalToggle.classList.add('active');

      _reminderIntervalId = setInterval(() => {
        _showNudge(shadowRoot, selectedSound);
      }, mins * 60 * 1000);
    }
  });

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

// ─── Nudge modal ─────────────────────────────────────────────────────────────

function _showNudge(shadowRoot, sound = 'bell') {
  // Don't stack nudges
  if (document.getElementById(NUDGE_ID)) return;
  _playSound(sound);

  const nudge = document.createElement('div');
  nudge.id = NUDGE_ID;
  nudge.style.cssText = `
    position:fixed;top:24px;left:50%;transform:translateX(-50%);
    z-index:99999;width:min(340px,92vw);
    background:#0f172a;color:#e2e8f0;
    border-radius:18px;padding:24px 22px 20px;
    box-shadow:0 16px 48px rgba(0,0,0,0.55);
    border:1px solid #6366f155;
    display:flex;flex-direction:column;gap:14px;
  `;
  nudge.innerHTML = `
    <div style="display:flex;align-items:center;gap:10px;">
      <div style="font-size:1.6rem;line-height:1;">🧘</div>
      <div>
        <div style="font-size:0.95rem;font-weight:700;margin-bottom:2px;">Time for a meditation break</div>
        <div style="font-size:0.75rem;color:#64748b;">A few minutes of calm can improve focus and clarity.</div>
      </div>
      <button id="med-nudge-close" style="
        margin-left:auto;background:none;border:none;cursor:pointer;
        color:#475569;font-size:1rem;padding:2px;line-height:1;flex-shrink:0;
      ">✕</button>
    </div>
    <div style="display:flex;gap:8px;">
      <button id="med-nudge-start" style="
        flex:1;background:#6366f1;color:#fff;border:none;border-radius:10px;
        padding:9px 14px;font-size:0.82rem;font-weight:600;cursor:pointer;
        transition:background 0.2s;
      ">Start Meditation</button>
      <button id="med-nudge-dismiss" style="
        flex:1;background:#1e293b;color:#94a3b8;border:none;border-radius:10px;
        padding:9px 14px;font-size:0.82rem;cursor:pointer;
        transition:background 0.2s;
      ">Maybe Later</button>
    </div>
  `;

  // Inject animation style into document head once
  if (!document.getElementById('socratiq-nudge-style')) {
    const s = document.createElement('style');
    s.id = 'socratiq-nudge-style';
    s.textContent = `
      @keyframes med-nudge-in {
        from { opacity:0; transform:translateX(-50%) translateY(-16px) scale(0.96); }
        to   { opacity:1; transform:translateX(-50%) translateY(0)      scale(1);   }
      }
      #${NUDGE_ID} { animation: med-nudge-in 0.22s ease-out; }
      #med-nudge-start:hover { background: #4f46e5 !important; }
      #med-nudge-dismiss:hover { background: #334155 !important; color:#e2e8f0 !important; }
    `;
    document.head.appendChild(s);
  }

  document.body.appendChild(nudge);

  nudge.querySelector('#med-nudge-close').addEventListener('click', () => _dismissNudge());
  nudge.querySelector('#med-nudge-dismiss').addEventListener('click', () => _dismissNudge());
  nudge.querySelector('#med-nudge-start').addEventListener('click', () => {
    _dismissNudge();
    openMeditationTimer(shadowRoot);
  });

  // Auto-dismiss after 30s if ignored
  setTimeout(() => _dismissNudge(), 30000);
}

function _dismissNudge() {
  document.getElementById(NUDGE_ID)?.remove();
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function _formatTime(totalSec) {
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function _playSound(type = 'bell') {
  if (type === 'silence') return;
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const gain = ctx.createGain();
    gain.connect(ctx.destination);

    if (type === 'bell') {
      // Soft layered bell — fundamental + 3 harmonics, gentle attack, long tail
      const t = ctx.currentTime;
      [[440, 0.28], [880, 0.14], [1320, 0.07], [1760, 0.035]].forEach(([freq, vol]) => {
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.connect(g); g.connect(ctx.destination);
        o.type = 'sine';
        o.frequency.setValueAtTime(freq, t);
        g.gain.setValueAtTime(0, t);
        g.gain.linearRampToValueAtTime(vol, t + 0.06);   // soft attack
        g.gain.exponentialRampToValueAtTime(0.001, t + 4.5);
        o.start(t); o.stop(t + 4.5);
      });

    } else if (type === 'bowl') {
      // Singing bowl: fundamental + overtone, slow attack
      [220, 440, 660].forEach((freq, i) => {
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.connect(g); g.connect(ctx.destination);
        o.type = 'sine';
        o.frequency.setValueAtTime(freq, ctx.currentTime);
        g.gain.setValueAtTime(0, ctx.currentTime);
        g.gain.linearRampToValueAtTime(0.18 - i * 0.04, ctx.currentTime + 0.3);
        g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 4);
        o.start(ctx.currentTime);
        o.stop(ctx.currentTime + 4);
      });

    } else if (type === 'chime') {
      // Three ascending chime notes
      [523, 659, 784].forEach((freq, i) => {
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.connect(g); g.connect(ctx.destination);
        o.type = 'triangle';
        o.frequency.setValueAtTime(freq, ctx.currentTime + i * 0.35);
        g.gain.setValueAtTime(0, ctx.currentTime + i * 0.35);
        g.gain.linearRampToValueAtTime(0.3, ctx.currentTime + i * 0.35 + 0.05);
        g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + i * 0.35 + 1.8);
        o.start(ctx.currentTime + i * 0.35);
        o.stop(ctx.currentTime + i * 0.35 + 1.8);
      });
    }
  } catch (_) { /* AudioContext not available */ }
}
