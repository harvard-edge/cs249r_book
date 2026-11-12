const MOBILE_QUERY = '(max-width: 800px)';

/** Own page appearance and visual viewport listeners for one reader mount. */
export function createReaderEnvironment({ host, shell }) {
  updateViewportVars();
  updateTheme();
  window.addEventListener('resize', updateViewportVars, { passive: true });
  window.visualViewport?.addEventListener('resize', updateViewportVars, { passive: true });
  window.visualViewport?.addEventListener('scroll', updateViewportVars, { passive: true });
  const themeObserver = installThemeTracking();

  function positionSelectionAction(selectionButton, rect = {}) {
    const viewport = window.visualViewport;
    const leftEdge = viewport?.offsetLeft ?? 0;
    const topEdge = viewport?.offsetTop ?? 0;
    const width = viewport?.width ?? window.innerWidth;
    const height = viewport?.height ?? window.innerHeight;
    selectionButton.style.left = `${leftEdge + 8}px`;
    const barEl = selectionButton.querySelector?.('.selection-clip-bar') || selectionButton;
    const buttonWidth = barEl.offsetWidth || selectionButton.offsetWidth || 220;
    const buttonHeight = barEl.offsetHeight || selectionButton.offsetHeight || 44;
    const selectionLeft = Number(rect.left ?? rect.x ?? 0);
    const selectionTop = Number(rect.top ?? rect.y ?? 0);
    const selectionWidth = Number(rect.width ?? Math.max(0, Number(rect.right) - selectionLeft));
    const selectionBottom = Number(rect.bottom ?? (selectionTop + Number(rect.height ?? 0)));
    let left = selectionWidth < buttonWidth
      ? selectionLeft
      : selectionLeft + Math.max(0, Math.min((selectionWidth - buttonWidth) / 2, 40));
    let top = selectionTop - buttonHeight - 9;
    if (top < topEdge + 8) top = selectionBottom + 9;
    left = Math.min(Math.max(left, leftEdge + 8), Math.max(leftEdge + 8, leftEdge + width - buttonWidth - 8));
    let maxTop = topEdge + height - buttonHeight - 8;
    const launcher = shell.querySelector('.reader-launcher');
    if (launcher && !launcher.hidden) {
      const lr = launcher.getBoundingClientRect();
      if (lr.width > 0 && left + buttonWidth > lr.left - 6) {
        maxTop = Math.min(maxTop, lr.top - buttonHeight - 8);
      }
    }
    top = Math.min(Math.max(top, topEdge + 8), Math.max(topEdge + 8, maxTop));
    selectionButton.style.left = `${left}px`;
    selectionButton.style.top = `${top}px`;
    if (selectionButton.dataset) {
      selectionButton.dataset.menuPlacement = (top + buttonHeight + 250 > topEdge + height - 8) ? 'above' : 'below';
    }
  }

  function updateViewportVars() {
    const viewport = window.visualViewport;
    const visibleHeight = viewport?.height ?? window.innerHeight;
    const visibleBottom = (viewport?.offsetTop ?? 0) + visibleHeight;
    const bottomClearance = Math.max(0, window.innerHeight - visibleBottom);
    const visibleWidth = viewport?.width ?? window.innerWidth;
    const visibleRight = (viewport?.offsetLeft ?? 0) + visibleWidth;
    const mobile = (window.matchMedia?.(MOBILE_QUERY).matches ?? false) || visibleWidth <= 800;
    const compact = (mobile && visibleHeight < 600) || visibleHeight < 540;
    host.style.setProperty('--reader-visible-height', `${visibleHeight}px`);
    host.style.setProperty('--reader-visible-width', `${visibleWidth}px`);
    host.style.setProperty('--reader-right-clearance', `${Math.max(0, window.innerWidth - visibleRight)}px`);
    host.style.setProperty('--reader-bottom-clearance', `${bottomClearance}px`);
    shell.dataset.mobile = String(mobile);
    shell.dataset.compact = String(compact);
  }

  function detectTheme() {
    const override = window.SocratiqWidgetTheme;
    if (override === 'dark' || override === 'light') return override;
    if (document.documentElement.matches('.quarto-dark, [data-bs-theme="dark"]') || document.body?.classList.contains('dark-mode')) return 'dark';
    let background = [255, 255, 255];
    let hasBackground = false;
    for (const element of [document.documentElement, document.body].filter(Boolean)) {
      const color = window.getComputedStyle(element).backgroundColor;
      const rgb = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:\s*,\s*([\d.]+))?\s*\)/i);
      if (!rgb) continue;
      const [red, green, blue] = rgb.slice(1, 4).map(Number);
      const alpha = rgb[4] === undefined ? 1 : Number(rgb[4]);
      if (alpha <= 0) continue;
      hasBackground = true;
      background = [red, green, blue].map((channel, index) => channel * alpha + background[index] * (1 - alpha));
    }
    if (hasBackground) {
      const [red, green, blue] = background;
      const luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255;
      if (luminance < 0.43) return 'dark';
      if (luminance > 0.57) return 'light';
    }
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function detectVolume() {
    const normalize = value => {
      const text = String(value || '').trim().toLowerCase();
      return /^vol[1-4]$/.test(text) ? text : '';
    };
    const explicit = normalize(window.SocratiqWidgetVolume) ||
      normalize(document.documentElement?.dataset?.socratiqVolume) ||
      normalize(document.body?.dataset?.socratiqVolume) ||
      normalize(document.querySelector('script[data-volume]')?.dataset?.volume);
    if (explicit) return explicit;
    const themeColor = document.querySelector('meta[name="theme-color"]')?.content?.trim().toLowerCase();
    if (themeColor === '#1f407a') return 'vol2';
    if (themeColor === '#581c87') return 'vol3';
    if (themeColor === '#065f46') return 'vol4';
    if (themeColor === '#a51c30' || themeColor === '#8e1537') return 'vol1';
    const pathMatch = location.pathname.match(/\/(vol[1-4])(\/|$)/i);
    if (pathMatch) return pathMatch[1].toLowerCase();
    return 'vol1';
  }

  function updateTheme() {
    shell.dataset.theme = detectTheme();
    shell.dataset.volume = detectVolume();
  }

  function installThemeTracking() {
    const observer = new MutationObserver(updateTheme);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'style', 'data-bs-theme', 'data-socratiq-volume'] });
    if (document.body) observer.observe(document.body, { attributes: true, attributeFilter: ['class', 'style', 'data-socratiq-volume'] });
    document.addEventListener('socratiqThemeChanged', updateTheme);
    return observer;
  }

  return {
    updateViewport: updateViewportVars,
    positionSelectionAction,
    destroy() {
      document.removeEventListener('socratiqThemeChanged', updateTheme);
      window.removeEventListener('resize', updateViewportVars);
      window.visualViewport?.removeEventListener('resize', updateViewportVars);
      window.visualViewport?.removeEventListener('scroll', updateViewportVars);
      themeObserver.disconnect();
    },
  };
}
