import katexCss from 'katex/dist/katex.min.css?inline';

export const readerStyles = `
  :host {
    all: initial;
    --reader-bottom-clearance: 0px;
    --reader-visible-height: 100dvh;
    --reader-visible-width: 100vw;
    --reader-right-clearance: 0px;
    color-scheme: light;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }

  *, *::before, *::after { box-sizing: border-box; }

  .reader-shell {
    --reader-edge: max(16px, env(safe-area-inset-left), env(safe-area-inset-right));
    --reader-accent: #8e1537;
    --reader-accent-strong: #74132f;
    --reader-accent-text: #781632;
    --reader-accent-soft: #fbf1f4;
    --reader-accent-soft-hover: #f3e9ec;
    --reader-accent-border: #dfd2d8;
    --reader-accent-focus: #d88ea4;
    --reader-accent-dark: #E85D75;
    --reader-accent-dark-soft: #382830;
    --reader-accent-dark-border: #60404b;
    position: fixed;
    inset: 0;
    z-index: 10000;
    pointer-events: none;
    color: #272329;
    font-size: 14px;
    line-height: 1.45;
    -webkit-font-smoothing: antialiased;
  }

  .reader-shell[data-volume="vol2"] {
    --reader-accent: #1F407A;
    --reader-accent-strong: #162f59;
    --reader-accent-text: #1F407A;
    --reader-accent-soft: #eef3fa;
    --reader-accent-soft-hover: #dce7f5;
    --reader-accent-border: #c5d6ec;
    --reader-accent-focus: #6B9FD4;
    --reader-accent-dark: #6B9FD4;
    --reader-accent-dark-soft: #1a283d;
    --reader-accent-dark-border: #2f496e;
  }

  .reader-shell[data-volume="vol3"] {
    --reader-accent: #581C87;
    --reader-accent-strong: #431466;
    --reader-accent-text: #581C87;
    --reader-accent-soft: #f5effb;
    --reader-accent-soft-hover: #eadcf8;
    --reader-accent-border: #d8c2ee;
    --reader-accent-focus: #C084FC;
    --reader-accent-dark: #C084FC;
    --reader-accent-dark-soft: #2a1b3d;
    --reader-accent-dark-border: #4e3270;
  }

  .reader-shell[data-volume="vol4"] {
    --reader-accent: #065F46;
    --reader-accent-strong: #044634;
    --reader-accent-text: #065F46;
    --reader-accent-soft: #edf7f4;
    --reader-accent-soft-hover: #d6ede6;
    --reader-accent-border: #b8dfd3;
    --reader-accent-focus: #34D399;
    --reader-accent-dark: #34D399;
    --reader-accent-dark-soft: #122e26;
    --reader-accent-dark-border: #215446;
  }

  .reader-shell[data-theme="dark"] { color: #f5f0f2; }
  .reader-shell[data-theme="dark"] { color-scheme: dark; }
  .reader-shell [hidden] { display: none !important; }

  button, textarea { font: inherit; }
  button { -webkit-tap-highlight-color: transparent; }

  .reader-launcher,
  .reader-panel,
  .selection-action { pointer-events: auto; }

  .reader-launcher {
    position: fixed;
    right: calc(var(--reader-right-clearance) + var(--reader-edge));
    bottom: calc(var(--reader-bottom-clearance) + max(16px, env(safe-area-inset-bottom)));
    display: inline-flex;
    align-items: center;
    gap: 9px;
    min-height: 44px;
    padding: 0 16px;
    border: 1px solid var(--reader-accent-strong);
    border-radius: 999px;
    background: var(--reader-accent);
    color: #fff;
    box-shadow: 0 5px 18px rgb(0 0 0 / 22%);
    cursor: pointer;
    font-size: 13px;
    font-weight: 650;
    letter-spacing: .01em;
    transition: background-color .16s ease, box-shadow .16s ease, transform .16s ease;
  }

  .reader-launcher:hover { background: var(--reader-accent-strong); box-shadow: 0 7px 22px rgb(0 0 0 / 28%); }
  .reader-launcher:active { transform: translateY(1px); }
  .reader-launcher:focus-visible,
  .reader-panel button:focus-visible,
  .reader-panel textarea:focus-visible,
  .selection-action:focus-visible { outline: 3px solid var(--reader-accent-focus); outline-offset: 3px; }

  .reader-mark {
    display: grid;
    width: 21px;
    height: 21px;
    place-items: center;
    border: 1px solid rgb(255 255 255 / 62%);
    border-radius: 50%;
    font-family: Georgia, serif;
    font-size: 14px;
    font-weight: 700;
    line-height: 1;
  }

  .reader-panel {
    position: fixed;
    right: calc(var(--reader-right-clearance) + var(--reader-edge));
    bottom: calc(var(--reader-bottom-clearance) + max(68px, calc(env(safe-area-inset-bottom) + 64px)));
    display: flex;
    width: min(390px, calc(var(--reader-visible-width) - var(--reader-edge) - var(--reader-edge)));
    max-height: min(695px, calc(var(--reader-visible-height) - 82px));
    flex-direction: column;
    overflow: hidden;
    border: 1px solid #e2d9dc;
    border-radius: 17px;
    background: #fffdfd;
    box-shadow: 0 18px 58px rgb(36 22 28 / 24%), 0 2px 8px rgb(36 22 28 / 9%);
    color: #272329;
  }

  .reader-panel[hidden], .reader-launcher[hidden], .selection-action[hidden] { display: none !important; }

  .reader-shell[data-theme="dark"] .reader-panel {
    border-color: #484047;
    background: #211e22;
    color: #f5f0f2;
    box-shadow: 0 18px 58px rgb(0 0 0 / 45%), 0 2px 8px rgb(0 0 0 / 25%);
  }

  .reader-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    padding: 13px 17px 10px;
    border-bottom: 1px solid #eee7e9;
    background: linear-gradient(110deg, #fff9fa, #fffdfd 65%);
  }

  .reader-shell[data-theme="dark"] .reader-header {
    border-color: #3d373d;
    background: linear-gradient(110deg, #2c252a, #211e22 65%);
  }

  .reader-heading { min-width: 0; }
  .reader-eyebrow {
    margin: 0 0 2px;
    color: var(--reader-accent);
    font-size: 10px;
    font-weight: 750;
    letter-spacing: .12em;
    text-transform: uppercase;
  }
  .reader-shell[data-theme="dark"] .reader-eyebrow { color: var(--reader-accent-dark); }
  .reader-title { margin: 0; font-size: 17px; font-weight: 680; letter-spacing: -.02em; }
  .reader-header-note { margin: 3px 0 0; color: #766e73; font-size: 11px; }
  .reader-shell[data-theme="dark"] .reader-header-note { color: #c0b6bd; }

  .reader-icon-button {
    display: grid;
    width: 34px;
    height: 34px;
    flex: 0 0 auto;
    place-items: center;
    border: 0;
    border-radius: 9px;
    background: transparent;
    color: inherit;
    cursor: pointer;
    font-size: 21px;
    line-height: 1;
  }
  .reader-icon-button:hover { background: var(--reader-accent-soft-hover); }
  .reader-shell[data-theme="dark"] .reader-icon-button:hover { background: #393139; }

  .reader-tools { display: flex; align-items: center; gap: 10px; padding: 3px 17px; border-bottom: 1px solid #eee7e9; }
  .reader-study { min-height: 36px; flex: 0 0 auto; padding: 4px 10px; border: 0; border-radius: 7px; background: transparent; color: var(--reader-accent-text); font-size: 12px; font-weight: 650; cursor: pointer; }
  .reader-study:hover { background: var(--reader-accent-soft-hover); }
  .reader-study:disabled { opacity: .6; cursor: wait; }
  .reader-study-status { font-size: 11px; line-height: 1.4; }
  .reader-shell[data-theme="dark"] .reader-tools { border-color: #3d373d; }
  .reader-shell[data-theme="dark"] .reader-study { color: var(--reader-accent-dark); }
  .reader-shell[data-theme="dark"] .reader-study:hover { background: #393139; }

  .reader-privacy {
    margin: 0;
    padding: 7px 17px;
    border-bottom: 1px solid #f0eaec;
    background: #fbf7f8;
    color: #665b61;
    font-size: 11px;
  }
  .reader-shell[data-theme="dark"] .reader-privacy {
    border-color: #39333a;
    background: #29252a;
    color: #c9bec5;
  }

  .reader-body { min-height: 0; flex: 1 1 auto; overflow: auto; overscroll-behavior: contain; }
  .reader-body:focus-visible { outline: 2px solid var(--reader-accent-focus); outline-offset: -2px; }
  .reader-sessions { display: flex; align-items: center; gap: 8px; padding: 8px 16px; font-size: 12px; }
  .reader-session-select { min-width: 0; flex: 1; height: 44px; min-height: 44px; max-width: 100%; font: inherit; color: inherit; background-color: transparent; border: 1px solid #b9a8b0; border-radius: 8px; padding: 6px 26px 6px 6px; appearance: none; -webkit-appearance: none; background-image: linear-gradient(45deg, transparent 50%, currentColor 50%), linear-gradient(135deg, currentColor 50%, transparent 50%); background-position: calc(100% - 14px) center, calc(100% - 9px) center; background-size: 5px 5px; background-repeat: no-repeat; }
  .reader-picker { position: fixed; right: calc(var(--reader-right-clearance) + var(--reader-edge)); bottom: calc(var(--reader-bottom-clearance) + 12px); width: min(440px, calc(var(--reader-visible-width) - var(--reader-edge) - var(--reader-edge))); padding: 12px; background: #fffdfd; color: #272329; border: 1px solid #b9a8b0; border-radius: 12px; box-shadow: 0 5px 20px #0003; pointer-events: auto; }
  .reader-picker p { margin: 0 0 8px; }
  .reader-picker button, .reader-instruction button { min-height: 36px; padding: 5px 12px; font: inherit; color: var(--reader-accent-text); background: #fffdfd; border: 1px solid #b9a8b0; border-radius: 8px; cursor: pointer; }
  .reader-instruction button { margin-top: 7px; }
  .reader-shell[data-theme="dark"] .reader-picker,
  .reader-shell[data-theme="dark"] .reader-picker button,
  .reader-shell[data-theme="dark"] .reader-instruction button { background: #211e22; color: #f5f0f2; }
  .reader-picker button:focus-visible, .reader-session-select:focus-visible { outline: 3px solid var(--reader-accent-focus); outline-offset: 2px; }
  .reader-source { position: sticky; top: 0; z-index: 1; padding: 13px 16px 12px; border-bottom: 1px solid #eee7e9; background: #fffdfd; }
  .reader-shell[data-theme="dark"] .reader-source { background: #211e22; }
  .reader-shell[data-theme="dark"] .reader-source { border-color: #3d373d; }
  .reader-source-label { margin: 0 0 5px; color: #877a81; font-size: 10px; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; }
  .reader-source-link { color: var(--reader-accent-text); font-weight: 650; text-decoration: none; }
  .reader-source-link:hover { text-decoration: underline; }
  .reader-shell[data-theme="dark"] .reader-source-link { color: var(--reader-accent-dark); }
  .reader-quote { display: -webkit-box; margin: 7px 0 0; overflow: hidden; color: #60575d; font-size: 12px; line-height: 1.45; -webkit-box-orient: vertical; -webkit-line-clamp: 3; }
  .reader-shell[data-theme="dark"] .reader-quote { color: #c6bbc2; }

  .reader-caution { margin: 7px 0 0; color: #766e73; font-size: 10px; }
  .reader-shell[data-theme="dark"] .reader-caution { color: #c0b6bd; }
  .reader-learning { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 16px 14px; }
  .reader-learning button, .reader-copy {
    min-height: 36px; padding: 6px 9px; border: 1px solid var(--reader-accent-border); border-radius: 8px;
    background: transparent; color: var(--reader-accent-text); cursor: pointer; font-size: 11px; line-height: 1.3;
  }
  .reader-learning button:hover, .reader-copy:hover { background: var(--reader-accent-soft); }
  .reader-learning .reader-reset { color: #665b61; }
  .reader-learning button:disabled { opacity: .5; cursor: default; }
  .reader-copy { margin-top: 7px; min-height: 32px; }
  .reader-shell[data-theme="dark"] .reader-learning button,
  .reader-shell[data-theme="dark"] .reader-copy { border-color: var(--reader-accent-dark-border); color: var(--reader-accent-dark); }
  .reader-shell[data-theme="dark"] .reader-learning button:hover,
  .reader-shell[data-theme="dark"] .reader-copy:hover { background: var(--reader-accent-dark-soft); }

  .reader-instruction { padding: 8px 16px 10px; color: #70666c; font-size: 12.5px; text-align: center; }
  .reader-instruction strong { display: block; margin-bottom: 3px; color: #3d3339; font-size: 13.5px; }
  .reader-instruction kbd { padding: 1px 4px; border: 1px solid #d8cdd2; border-radius: 4px; background: #fff; color: #4c3a42; font: 600 11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .reader-shell[data-theme="dark"] .reader-instruction { color: #c5bac1; }
  .reader-shell[data-theme="dark"] .reader-instruction strong { color: #f3e9ee; }
  .reader-shell[data-theme="dark"] .reader-instruction kbd { border-color: #514950; background: #2b272c; color: #f5f0f2; }

  .reader-thread { display: flex; flex-direction: column; gap: 13px; padding: 15px 16px 18px; }
  .reader-turn { min-width: 0; max-width: 100%; }
  .reader-turn-user { align-self: flex-end; max-width: 88%; }
  .reader-turn-assistant { align-self: stretch; }
  .reader-turn-label { margin: 0 0 4px; color: #847980; font-size: 10px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
  .reader-turn-user .reader-turn-label { text-align: right; }
  .reader-user-text { margin: 0; padding: 9px 12px; border: 1px solid var(--reader-accent-border); border-radius: 13px 13px 4px 13px; background: var(--reader-accent-soft); color: #272329; white-space: pre-wrap; overflow-wrap: anywhere; }
  .reader-shell[data-theme="dark"] .reader-user-text { border-color: var(--reader-accent-dark-border); background: var(--reader-accent-dark-soft); color: #f5e8ed; }
  .reader-answer { color: inherit; overflow-wrap: anywhere; }
  .reader-answer > :first-child { margin-top: 0; }
  .reader-answer > :last-child { margin-bottom: 0; }
  .reader-answer p { margin: 0 0 10px; }
  .reader-answer h1, .reader-answer h2, .reader-answer h3, .reader-answer h4 { margin: 13px 0 6px; color: #3d3339; font-size: 1em; font-weight: 700; }
  .reader-shell[data-theme="dark"] .reader-answer h1,
  .reader-shell[data-theme="dark"] .reader-answer h2,
  .reader-shell[data-theme="dark"] .reader-answer h3,
  .reader-shell[data-theme="dark"] .reader-answer h4 { color: #f4e9ee; }
  .reader-answer ul, .reader-answer ol { margin: 6px 0 11px; padding-left: 21px; }
  .reader-answer li { margin: 3px 0; }
  .reader-answer a { color: var(--reader-accent-text); text-decoration: underline; text-underline-offset: 2px; }
  .reader-shell[data-theme="dark"] .reader-answer a { color: var(--reader-accent-dark); }
  .reader-answer blockquote { margin: 9px 0; padding: 2px 0 2px 11px; border-left: 3px solid var(--reader-accent); color: #675b62; }
  .reader-shell[data-theme="dark"] .reader-answer blockquote { border-left-color: var(--reader-accent-dark); color: #c6bbc2; }
  .reader-answer pre { max-width: 100%; margin: 9px 0; padding: 10px; overflow: auto; border: 1px solid #e7dfe2; border-radius: 8px; background: #f8f5f6; font-size: 11px; }
  .reader-shell[data-theme="dark"] .reader-answer pre { border-color: #403a40; background: #19171a; }
  .reader-answer code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .92em; }
  .reader-answer :not(pre) > code { padding: 1px 4px; border-radius: 4px; background: #f1ecee; }
  .reader-shell[data-theme="dark"] .reader-answer :not(pre) > code { background: #39333a; }
  .reader-answer .katex-display { max-width: 100%; overflow-x: auto; overflow-y: hidden; padding: 3px 0; }
  .reader-answer .reader-math-error { color: var(--reader-accent); }

  .reader-status { margin: 0; padding: 0 16px 10px; color: #766970; font-size: 12px; }
  .reader-shell[data-theme="dark"] .reader-status { color: #c6bbc2; }
  .reader-error { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin: 0 14px 12px; padding: 9px 10px; border: 1px solid #edc6ce; border-radius: 9px; background: #fff5f6; color: #74243a; font-size: 12px; }
  .reader-shell[data-theme="dark"] .reader-error { border-color: #74414e; background: #39272d; color: #ffdce4; }
  .reader-retry { flex: 0 0 auto; border: 0; background: none; color: inherit; cursor: pointer; font-weight: 700; text-decoration: underline; }

  .reader-composer { display: flex; align-items: flex-end; gap: 8px; padding: 10px 13px 5px; border-top: 1px solid #eee7e9; background: #fffdfd; }
  .reader-shell[data-theme="dark"] .reader-composer { border-color: #3d373d; background: #211e22; }
  .reader-footnote { margin: 0; padding: 0 13px 8px; background: #fffdfd; color: #665b61; font-size: 10.5px; line-height: 1.3; text-align: center; }
  .reader-shell[data-theme="dark"] .reader-footnote { background: #211e22; color: #c9bec5; }
  .reader-composer textarea { min-width: 0; min-height: 42px; max-height: 110px; flex: 1 1 auto; resize: vertical; padding: 10px 11px; border: 1px solid #dcd2d7; border-radius: 10px; background: #fff; color: #272329; font-size: 13px; line-height: 1.35; }
  .reader-composer textarea::placeholder { color: #8a7f85; }
  .reader-shell[data-theme="dark"] .reader-composer textarea { border-color: #514950; background: #2b272c; color: #f5f0f2; }
  .reader-send, .reader-stop { display: grid; width: 42px; height: 42px; flex: 0 0 auto; place-items: center; border: 0; border-radius: 11px; background: var(--reader-accent); color: #fff; cursor: pointer; font-size: 16px; font-weight: 700; }
  .reader-send:hover { background: var(--reader-accent-strong); }
  .reader-stop { background: #ece5e8; color: #5e4d55; font-size: 12px; }
  .reader-stop:hover { background: #e1d3d9; }
  .reader-shell[data-theme="dark"] .reader-stop { background: #40383e; color: #f3e9ee; }
  .reader-composer[aria-disabled="true"] textarea { opacity: .68; }

  .selection-action {
    position: fixed;
    z-index: 10001;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    min-height: 40px;
    padding: 0 12px;
    border: 1px solid var(--reader-accent-strong);
    border-radius: 10px;
    background: var(--reader-accent);
    color: #fff;
    box-shadow: 0 5px 18px rgb(0 0 0 / 26%);
    cursor: pointer;
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
  }
  .selection-action:hover { background: var(--reader-accent-strong); }
  .selection-action .reader-mark { width: 18px; height: 18px; font-size: 12px; }

  @media (max-width: 600px) {
    .reader-shell { --reader-edge: max(12px, env(safe-area-inset-left), env(safe-area-inset-right)); }
    .reader-launcher { bottom: calc(var(--reader-bottom-clearance) + max(12px, env(safe-area-inset-bottom))); min-height: 42px; padding-inline: 13px; }
    .reader-panel { bottom: calc(var(--reader-bottom-clearance) + max(12px, env(safe-area-inset-bottom))); width: min(440px, calc(var(--reader-visible-width) - var(--reader-edge) - var(--reader-edge))); max-height: calc(var(--reader-visible-height) - 24px); border-radius: 15px; }
    .reader-panel[data-launch-origin="launcher"] { bottom: calc(var(--reader-bottom-clearance) + max(64px, calc(env(safe-area-inset-bottom) + 60px))); max-height: calc(var(--reader-visible-height) - 80px); }
    .reader-header { padding: 14px 14px 11px; }
    .reader-privacy { padding-inline: 14px; }
    .reader-source { padding-inline: 14px; }
    .reader-thread { padding-inline: 14px; }
  }

  /* Short windows and the mobile keyboard must leave the answer scrollable. */
  .reader-shell[data-compact="true"] .reader-panel {
    bottom: calc(var(--reader-bottom-clearance) + max(12px, env(safe-area-inset-bottom)));
    max-height: calc(var(--reader-visible-height) - 24px);
  }
  .reader-shell[data-compact="true"] .reader-header { padding-block: 8px; }
  .reader-shell[data-compact="true"] .reader-header-note { display: none; }
  .reader-shell[data-compact="true"] .reader-privacy { padding-block: 5px; }
  .reader-shell[data-compact="true"] .reader-source { position: static; }
  .reader-shell[data-compact="true"] .reader-composer { padding-block: 6px 3px; }
  .reader-shell[data-compact="true"] .reader-footnote { padding-bottom: 5px; }

  @media (pointer: coarse) {
    .reader-panel button, .reader-launcher, .selection-action { min-height: 44px; min-width: 44px; }
    .reader-panel textarea { font-size: 16px; }
    .reader-session-select { font-size: 16px; }
  }

  @media (prefers-reduced-motion: reduce) {
    .reader-launcher { transition: none; }
  }
`;


const learnerStyles = `
  .reader-tools { flex-wrap: wrap; }
  .reader-preferences { margin: 6px 14px; padding: 6px 10px; border: 1px solid var(--reader-accent-border); border-radius: 10px; background: var(--reader-accent-soft); }
  .reader-preferences summary { cursor: pointer; font-weight: 650; min-height: 20px; }
  .reader-preferences summary:focus-visible, .reader-preferences select:focus-visible { outline: 3px solid var(--reader-accent-focus); outline-offset: 3px; }
  .reader-preferences-fields { display: grid; gap: 5px; padding-top: 6px; }
  .reader-preferences label { display: block; font-weight: 600; font-size: 12px; }
  .reader-preferences label span { font-weight: 400; }
  .reader-preferences select, .reader-preferences textarea { display: block; margin-top: 3px; width: 100%; padding: 5px 8px; border: 1px solid #b8a7af; border-radius: 7px; background: #fffdfd; color: inherit; font: inherit; font-size: 12px; }
  .reader-preferences textarea { resize: vertical; min-height: 36px; max-height: 110px; }
  .reader-preferences-fields p { margin: 0; font-size: 10.5px; line-height: 1.35; }
  .reader-preference-actions { display: flex; flex-wrap: wrap; gap: 8px; }
  .reader-preference-actions button, .reader-tour button, .reader-tour-replay { min-height: 30px; padding: 4px 10px; border: 1px solid #b8a7af; border-radius: 7px; background: #fffdfd; color: inherit; cursor: pointer; font: inherit; font-size: 11.5px; }
  .reader-tour-spotlight { position: fixed; z-index: 5; border: 2px solid var(--reader-accent-focus); border-radius: 8px; box-shadow: 0 0 0 9999px rgb(20 12 16 / 45%); pointer-events: none; }
  .reader-tour { position: fixed; z-index: 6; pointer-events: auto; padding: 16px; border: 1px solid var(--reader-accent-border); border-radius: 12px; background: #fffdfd; color: #272329; box-shadow: 0 8px 30px rgb(0 0 0 / 24%); overflow: auto; }
  .reader-tour p { margin: 0 0 12px; line-height: 1.5; }
  .reader-tour h3 { margin: 0 0 8px; font-size: 16px; line-height: 1.3; }
  .reader-tour h3:focus { outline: none; }
  .reader-tour-count { font-size: 12px; font-weight: 600; }
  .reader-tour-actions { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }
  .reader-tour button:focus-visible { outline: 3px solid var(--reader-accent-focus); outline-offset: 2px; }
  .reader-tour button:disabled { opacity: .5; cursor: default; }
  .reader-tour button[data-tour="next"] { background: var(--reader-accent); border-color: var(--reader-accent); color: #fff; }
  .reader-shell[data-theme="dark"] .reader-preferences { background: var(--reader-accent-dark-soft); border-color: var(--reader-accent-dark-border); }
  .reader-shell[data-theme="dark"] .reader-preferences select,
  .reader-shell[data-theme="dark"] .reader-preferences textarea,
  .reader-shell[data-theme="dark"] .reader-preference-actions button,
  .reader-shell[data-theme="dark"] .reader-tour-replay,
  .reader-shell[data-theme="dark"] .reader-tour { background: #211e22; color: #f5f0f2; border-color: var(--reader-accent-dark-border); }
  .reader-shell[data-theme="dark"] .reader-tour button { background: #3b3038; color: #f5f0f2; }
  .reader-shell[data-theme="dark"] .reader-tour button[data-tour="next"] { background: var(--reader-accent); color: #fff; }
  @media (pointer: coarse) {
    .reader-preferences summary, .reader-tour button, .reader-preference-actions button { min-height: 44px; }
    .reader-preferences select, .reader-preferences textarea { font-size: 16px; }
  }
`;

export const completeReaderStyles = `${katexCss}\n${readerStyles}\n${learnerStyles}`;
