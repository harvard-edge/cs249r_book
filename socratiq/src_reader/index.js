import {captureSelection, captureParagraph, revealSource} from './context.js';
import {createExplanationRequest} from './transport.mjs';
import {mountReader} from './ui.js';

const script = document.currentScript;
const config = script?.dataset || {};
let version = '';

function start() {
  if (document.getElementById('socratiq-reader')) return;
  mountReader({
    captureSelection: () => captureSelection({volume: config.volume, version}),
    captureParagraph: element => captureParagraph(element, {volume: config.volume, version}),
    onRevealSource: revealSource,
    requestExplanation: createExplanationRequest({endpoint: config.endpoint}),
  });
  // Release identity is best effort and must never block reading assistance.
  const manifest = document.querySelector('meta[name="release-manifest"]')?.content;
  if (manifest) {
    const url = new URL(manifest, location.href);
    if (url.origin === location.origin) fetch(url, {signal: AbortSignal.timeout(5000)})
      .then(response => response.ok ? response.json() : null)
      .then(data => { version = [data?.releaseId, data?.releaseHash].filter(Boolean).join(' ').slice(0, 160); })
      .catch(() => {});
  }
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, {once: true});
else start();
