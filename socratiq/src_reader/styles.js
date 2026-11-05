import katexCss from 'katex/dist/katex.min.css?inline';

export const readerStyles = `
  :host {
    all: initial;
    --reader-bottom-clearance: 0px;
    --reader-visible-height: 100dvh;
    color-scheme: light;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }

  *, *::before, *::after { box-sizing: border-box; }

  .reader-shell {
    position: fixed;
    inset: 0;
    z-index: 10000;
    pointer-events: none;
    color: #272329;
    font-size: 14px;
    line-height: 1.45;
    -webkit-font-smoothing: antialiased;
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
    right: max(16px, env(safe-area-inset-right));
    bottom: calc(var(--reader-bottom-clearance) + max(16px, env(safe-area-inset-bottom)));
    display: inline-flex;
    align-items: center;
    gap: 9px;
    min-height: 44px;
    padding: 0 16px;
    border: 1px solid #74132f;
    border-radius: 999px;
    background: #8e1537;
    color: #fff;
    box-shadow: 0 5px 18px rgb(47 15 27 / 22%);
    cursor: pointer;
    font-size: 13px;
    font-weight: 650;
    letter-spacing: .01em;
    transition: background-color .16s ease, box-shadow .16s ease, transform .16s ease;
  }

  .reader-launcher:hover { background: #74132f; box-shadow: 0 7px 22px rgb(47 15 27 / 28%); }
  .reader-launcher:active { transform: translateY(1px); }
  .reader-launcher:focus-visible,
  .reader-panel button:focus-visible,
  .reader-panel textarea:focus-visible,
  .selection-action:focus-visible { outline: 3px solid #d88ea4; outline-offset: 3px; }

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
    right: max(16px, env(safe-area-inset-right));
    bottom: calc(var(--reader-bottom-clearance) + max(72px, calc(env(safe-area-inset-bottom) + 68px)));
    display: flex;
    width: min(390px, calc(100vw - 32px));
    max-height: min(660px, calc(var(--reader-visible-height) - 104px));
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
    padding: 17px 17px 13px;
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
    color: #8e1537;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: .12em;
    text-transform: uppercase;
  }
  .reader-shell[data-theme="dark"] .reader-eyebrow { color: #e99bb2; }
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
  .reader-icon-button:hover { background: #f3e9ec; }
  .reader-shell[data-theme="dark"] .reader-icon-button:hover { background: #393139; }

  .reader-privacy {
    margin: 0;
    padding: 9px 17px;
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
  .reader-source { position: sticky; top: 0; z-index: 1; padding: 13px 16px 12px; border-bottom: 1px solid #eee7e9; background: #fffdfd; }
  .reader-shell[data-theme="dark"] .reader-source { background: #211e22; }
  .reader-shell[data-theme="dark"] .reader-source { border-color: #3d373d; }
  .reader-source-label { margin: 0 0 5px; color: #877a81; font-size: 10px; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; }
  .reader-source-link { color: #7d1734; font-weight: 650; text-decoration: none; }
  .reader-source-link:hover { text-decoration: underline; }
  .reader-shell[data-theme="dark"] .reader-source-link { color: #f1a2b8; }
  .reader-quote { display: -webkit-box; margin: 7px 0 0; overflow: hidden; color: #60575d; font-size: 12px; line-height: 1.45; -webkit-box-orient: vertical; -webkit-line-clamp: 3; }
  .reader-shell[data-theme="dark"] .reader-quote { color: #c6bbc2; }

  .reader-instruction { padding: 20px 20px 22px; color: #70666c; font-size: 13px; text-align: center; }
  .reader-instruction strong { display: block; margin-bottom: 5px; color: #3d3339; font-size: 14px; }
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
  .reader-user-text { margin: 0; padding: 9px 12px; border: 1px solid #ead6dd; border-radius: 13px 13px 4px 13px; background: #fbf1f4; color: #3b2b32; white-space: pre-wrap; overflow-wrap: anywhere; }
  .reader-shell[data-theme="dark"] .reader-user-text { border-color: #60404b; background: #382830; color: #f5e8ed; }
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
  .reader-answer a { color: #781632; text-decoration: underline; text-underline-offset: 2px; }
  .reader-shell[data-theme="dark"] .reader-answer a { color: #f1a2b8; }
  .reader-answer blockquote { margin: 9px 0; padding: 2px 0 2px 11px; border-left: 3px solid #ba788d; color: #675b62; }
  .reader-shell[data-theme="dark"] .reader-answer blockquote { color: #c6bbc2; }
  .reader-answer pre { max-width: 100%; margin: 9px 0; padding: 10px; overflow: auto; border: 1px solid #e7dfe2; border-radius: 8px; background: #f8f5f6; font-size: 11px; }
  .reader-shell[data-theme="dark"] .reader-answer pre { border-color: #403a40; background: #19171a; }
  .reader-answer code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .92em; }
  .reader-answer :not(pre) > code { padding: 1px 4px; border-radius: 4px; background: #f1ecee; }
  .reader-shell[data-theme="dark"] .reader-answer :not(pre) > code { background: #39333a; }
  .reader-answer .katex-display { max-width: 100%; overflow-x: auto; overflow-y: hidden; padding: 3px 0; }
  .reader-answer .reader-math-error { color: #87354e; }

  .reader-status { margin: 0; padding: 0 16px 10px; color: #766970; font-size: 12px; }
  .reader-shell[data-theme="dark"] .reader-status { color: #c6bbc2; }
  .reader-error { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin: 0 14px 12px; padding: 9px 10px; border: 1px solid #edc6ce; border-radius: 9px; background: #fff5f6; color: #74243a; font-size: 12px; }
  .reader-shell[data-theme="dark"] .reader-error { border-color: #74414e; background: #39272d; color: #ffdce4; }
  .reader-retry { flex: 0 0 auto; border: 0; background: none; color: inherit; cursor: pointer; font-weight: 700; text-decoration: underline; }

  .reader-composer { display: flex; align-items: flex-end; gap: 8px; padding: 11px 13px 13px; border-top: 1px solid #eee7e9; background: #fffdfd; }
  .reader-shell[data-theme="dark"] .reader-composer { border-color: #3d373d; background: #211e22; }
  .reader-composer textarea { min-width: 0; min-height: 42px; max-height: 110px; flex: 1 1 auto; resize: vertical; padding: 10px 11px; border: 1px solid #dcd2d7; border-radius: 10px; background: #fff; color: #272329; font-size: 13px; line-height: 1.35; }
  .reader-composer textarea::placeholder { color: #8a7f85; }
  .reader-shell[data-theme="dark"] .reader-composer textarea { border-color: #514950; background: #2b272c; color: #f5f0f2; }
  .reader-send, .reader-stop { display: grid; width: 42px; height: 42px; flex: 0 0 auto; place-items: center; border: 0; border-radius: 11px; background: #8e1537; color: #fff; cursor: pointer; font-size: 16px; font-weight: 700; }
  .reader-send:hover { background: #74132f; }
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
    border: 1px solid #74132f;
    border-radius: 10px;
    background: #8e1537;
    color: #fff;
    box-shadow: 0 5px 18px rgb(47 15 27 / 26%);
    cursor: pointer;
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
  }
  .selection-action:hover { background: #74132f; }
  .selection-action .reader-mark { width: 18px; height: 18px; font-size: 12px; }

  @media (max-width: 600px) {
    .reader-launcher { right: max(12px, env(safe-area-inset-right)); bottom: calc(var(--reader-bottom-clearance) + max(12px, env(safe-area-inset-bottom))); min-height: 42px; padding-inline: 13px; }
    .reader-panel { right: max(12px, env(safe-area-inset-right)); bottom: calc(var(--reader-bottom-clearance) + max(12px, env(safe-area-inset-bottom))); width: min(440px, calc(100vw - 24px)); max-height: calc(var(--reader-visible-height) - 24px); border-radius: 15px; }
    .reader-panel[data-launch-origin="launcher"] { bottom: calc(var(--reader-bottom-clearance) + max(64px, calc(env(safe-area-inset-bottom) + 60px))); max-height: calc(var(--reader-visible-height) - 80px); }
    .reader-header { padding: 14px 14px 11px; }
    .reader-privacy { padding-inline: 14px; }
    .reader-source { padding-inline: 14px; }
    .reader-thread { padding-inline: 14px; }
  }

  @media (prefers-reduced-motion: reduce) {
    .reader-launcher { transition: none; }
  }
`;

export const completeReaderStyles = `${katexCss}\n${readerStyles}`;
