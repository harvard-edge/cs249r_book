// Security-hardened Bring-Your-Own-Key (BYOK) & OpenRouter OAuth PKCE module.
// Shared across the Focused Reader (src_reader) and Study Tools (src_shadow).

export const BYOK_SESSION_KEY = 'socratiq_byok_session_v1';
export const BYOK_PERSISTED_KEY = 'socratiq_byok_persisted_v1';
export const BYOK_PKCE_STATE_KEY = 'socratiq_pkce_state_v1';

export const BYOK_PROVIDERS = Object.freeze({
  openrouter: Object.freeze({
    id: 'openrouter',
    label: 'OpenRouter (1-Click SSO — Gemini, Claude, GPT)',
    shortLabel: 'OpenRouter',
    endpoint: 'https://openrouter.ai/api/v1/chat/completions',
    defaultModel: 'google/gemini-2.5-flash',
    presetModels: Object.freeze([
      'google/gemini-2.5-flash',
      'anthropic/claude-sonnet-4',
      'openai/gpt-4.1-mini',
    ]),
    authUrl: 'https://openrouter.ai/auth',
    tokenUrl: 'https://openrouter.ai/api/v1/auth/keys',
    portalUrl: 'https://openrouter.ai/settings/keys',
    portalLabel: 'Get OpenRouter key ↗',
  }),
  'google-gemini': Object.freeze({
    id: 'google-gemini',
    label: 'Google Gemini (Direct API Key)',
    shortLabel: 'Google Gemini',
    endpoint: 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
    defaultModel: 'gemini-2.5-flash',
    presetModels: Object.freeze(['gemini-2.5-flash', 'gemini-2.5-pro']),
    portalUrl: 'https://aistudio.google.com/app/apikey',
    portalLabel: 'Get free Gemini key ↗',
  }),
  anthropic: Object.freeze({
    id: 'anthropic',
    label: 'Anthropic Claude (Direct API Key)',
    shortLabel: 'Anthropic Claude',
    endpoint: 'https://api.anthropic.com/v1/messages',
    defaultModel: 'claude-sonnet-4-20250514',
    presetModels: Object.freeze(['claude-sonnet-4-20250514', 'claude-3-5-haiku-latest']),
    portalUrl: 'https://console.anthropic.com/settings/keys',
    portalLabel: 'Get Claude key ↗',
  }),
  openai: Object.freeze({
    id: 'openai',
    label: 'OpenAI (Direct API Key)',
    shortLabel: 'OpenAI',
    endpoint: 'https://api.openai.com/v1/chat/completions',
    defaultModel: 'gpt-4.1-mini',
    presetModels: Object.freeze(['gpt-4.1-mini', 'gpt-4.1']),
    portalUrl: 'https://platform.openai.com/api-keys',
    portalLabel: 'Get OpenAI key ↗',
  }),
  groq: Object.freeze({
    id: 'groq',
    label: 'Groq (Direct API Key)',
    shortLabel: 'Groq',
    endpoint: 'https://api.groq.com/openai/v1/chat/completions',
    defaultModel: 'openai/gpt-oss-120b',
    presetModels: Object.freeze(['openai/gpt-oss-120b', 'llama-3.3-70b-versatile']),
    portalUrl: 'https://console.groq.com/keys',
    portalLabel: 'Get Groq key ↗',
  }),
  ollama: Object.freeze({
    id: 'ollama',
    label: 'Ollama (Local)',
    shortLabel: 'Ollama',
    endpoint: 'http://localhost:11434/v1/chat/completions',
    defaultModel: 'gemma3:270m',
    presetModels: Object.freeze(['gemma3:270m']),
    portalUrl: '',
    portalLabel: '',
  }),
});

const ALLOWED_HTTPS_KEY_HOSTS = new Set([
  'openrouter.ai',
  'generativelanguage.googleapis.com',
  'api.anthropic.com',
  'api.openai.com',
  'api.groq.com',
]);

const FORBIDDEN_GATEWAY_HOSTS = new Set([
  'mlsysbook.ai',
  'www.mlsysbook.ai',
  'harvard-edge.github.io',
]);

function getMemoryByokConfig() {
  try { return globalThis.__SOCRATIQ_BYOK_MEMORY__ || null; } catch { return null; }
}

function setMemoryByokConfig(val) {
  try { globalThis.__SOCRATIQ_BYOK_MEMORY__ = val || null; } catch {}
}

function getSessionStorage() {
  try { return globalThis.sessionStorage || null; } catch { return null; }
}

function getLocalStorage() {
  try { return globalThis.localStorage || null; } catch { return null; }
}

export function isSharedPublicOrigin(win = globalThis.window) {
  try {
    const hostname = String(win?.location?.hostname || '').toLowerCase();
    return hostname.endsWith('.github.io');
  } catch {
    return false;
  }
}

export function normalizeProviderId(provider) {
  const raw = String(provider || '').trim().toLowerCase();
  if (raw === 'open-router') return 'openrouter';
  if (raw === 'gemini') return 'google-gemini';
  if (raw === 'claude' || raw === 'anthropic-claude') return 'anthropic';
  return BYOK_PROVIDERS[raw] ? raw : '';
}

export function validateByokKeyFormat(provider, apiKey) {
  const rawProvider = String(provider || '').trim().toLowerCase();
  const normalized = normalizeProviderId(rawProvider) || rawProvider;
  const key = String(apiKey || '').trim();
  if (normalized === 'ollama' || normalized === 'custom' || normalized === '') {
    return { valid: true, reason: '' };
  }
  if (!key) {
    return { valid: false, reason: 'Please enter an API key before connecting.' };
  }
  if (normalized === 'google-gemini') {
    if (!key.startsWith('AIza') || key.length < 20) {
      return { valid: false, reason: 'Google Gemini API keys begin with "AIza".' };
    }
    return { valid: true, reason: '' };
  }
  if (normalized === 'anthropic') {
    if (!key.startsWith('sk-ant-') || key.length < 16) {
      return { valid: false, reason: 'Anthropic Claude API keys begin with "sk-ant-".' };
    }
    return { valid: true, reason: '' };
  }
  if (normalized === 'openrouter' || normalized === 'openai') {
    if (!key.startsWith('sk-') || key.length < 12) {
      return { valid: false, reason: 'API keys for this provider begin with "sk-".' };
    }
    return { valid: true, reason: '' };
  }
  if (normalized === 'groq') {
    if (!key.startsWith('gsk_') || key.length < 12) {
      return { valid: false, reason: 'Groq API keys begin with "gsk_".' };
    }
    return { valid: true, reason: '' };
  }
  return { valid: true, reason: '' };
}

export function redactSecrets(input) {
  if (typeof input !== 'string' || !input) return '';
  return input
    .replace(/AIza[0-9A-Za-z_-]{16,}/g, '[REDACTED_API_KEY]')
    .replace(/sk-ant-[0-9A-Za-z_-]{8,}/g, '[REDACTED_API_KEY]')
    .replace(/sk-or-v1-[0-9A-Za-z_-]{8,}/g, '[REDACTED_API_KEY]')
    .replace(/sk-or-[0-9A-Za-z_-]{8,}/g, '[REDACTED_API_KEY]')
    .replace(/sk-[0-9A-Za-z_-]{8,}/g, '[REDACTED_API_KEY]')
    .replace(/gsk_[0-9A-Za-z_-]{8,}/g, '[REDACTED_API_KEY]')
    .replace(/Bearer\s+[0-9A-Za-z._-]{12,}/gi, 'Bearer [REDACTED_API_KEY]')
    .replace(/([?&]key=)[^&\s]+/gi, '$1[REDACTED_API_KEY]');
}

/**
 * Validate that a custom provider endpoint is safe before attaching any user key.
 * User keys are NEVER sent over plaintext HTTP (except keyless localhost Ollama)
 * and NEVER sent to the textbook's shared gateway (mlsysbook.ai).
 */
export function validateByokEndpoint(endpoint, provider = '') {
  const rawProvider = String(provider || '').trim().toLowerCase();
  const normalizedProvider = normalizeProviderId(rawProvider);
  const spec = BYOK_PROVIDERS[normalizedProvider];
  const candidate = String(endpoint || spec?.endpoint || '').trim();
  let parsed;
  try {
    parsed = new URL(candidate);
  } catch {
    throw new Error('Custom provider endpoint must be a valid URL.');
  }
  const host = parsed.hostname.toLowerCase();
  if (FORBIDDEN_GATEWAY_HOSTS.has(host) || parsed.pathname.includes('/api/socratiq')) {
    throw new Error('Refusing to send a personal API key to the shared SocratiQ gateway.');
  }
  parsed.username = '';
  parsed.password = '';
  parsed.searchParams.delete('key');
  parsed.searchParams.delete('api_key');
  const allowsLocalLoopback = !rawProvider || normalizedProvider === 'ollama' || rawProvider === 'custom';
  if (normalizedProvider === 'ollama' || (allowsLocalLoopback && (host === 'localhost' || host === '127.0.0.1'))) {
    if ((host === 'localhost' || host === '127.0.0.1') && (parsed.protocol === 'http:' || parsed.protocol === 'https:')) {
      return { valid: true, url: parsed.toString(), host, isLocal: true };
    }
    throw new Error('Local Ollama endpoint must target localhost or 127.0.0.1.');
  }
  if (parsed.protocol !== 'https:') {
    throw new Error('Custom AI provider endpoints must use HTTPS.');
  }
  if (!ALLOWED_HTTPS_KEY_HOSTS.has(host)) {
    throw new Error(`Endpoint host "${host}" is not in the approved HTTPS provider allowlist.`);
  }
  return { valid: true, url: parsed.toString(), host, isLocal: false };
}

export function normalizeByokConfig(raw = {}) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return {
      enabled: false,
      provider: 'google-gemini',
      model: BYOK_PROVIDERS['google-gemini'].defaultModel,
      endpoint: BYOK_PROVIDERS['google-gemini'].endpoint,
      apiKey: '',
      saveLocally: false,
      authMethod: 'manual',
    };
  }
  const provider = normalizeProviderId(raw.provider) || 'google-gemini';
  const spec = BYOK_PROVIDERS[provider] || BYOK_PROVIDERS['google-gemini'];
  const model = typeof raw.model === 'string' && raw.model.trim()
    ? raw.model.trim().slice(0, 160)
    : spec.defaultModel;
  let endpoint = typeof raw.endpoint === 'string' && raw.endpoint.trim()
    ? raw.endpoint.trim().slice(0, 500)
    : spec.endpoint;
  // Upgrade legacy Gemini base URL or legacy Ollama /api/generate URL to OpenAI-compatible endpoint
  if (provider === 'google-gemini' && endpoint === 'https://generativelanguage.googleapis.com/v1beta') {
    endpoint = spec.endpoint;
  } else if (provider === 'ollama' && endpoint === 'http://localhost:11434/api/generate') {
    endpoint = spec.endpoint;
  }
  const apiKey = typeof raw.apiKey === 'string' ? raw.apiKey.trim().slice(0, 512) : '';
  const saveLocally = raw.saveLocally === true;
  const enabled = Boolean(raw.enabled && (provider === 'ollama' || apiKey.length > 0));
  const authMethod = raw.authMethod === 'oauth-pkce' ? 'oauth-pkce' : 'manual';
  return { enabled, provider, model, endpoint, apiKey, saveLocally, authMethod };
}

export function readActiveByokConfig(options = {}) {
  const isWinArg = options && typeof options === 'object' && 'location' in options && !('sessionStore' in options) && !('localStore' in options);
  const win = isWinArg ? options : (options?.win || globalThis.window);
  const defaultSessionStore = globalThis.__SOCRATIQ_BYOK_SESSION_STORE__ || getSessionStorage();
  const defaultLocalStore = globalThis.__SOCRATIQ_BYOK_LOCAL_STORE__ || getLocalStorage();
  const sessionStore = isWinArg ? (options.sessionStorage ?? defaultSessionStore) : (options?.sessionStore !== undefined ? options.sessionStore : defaultSessionStore);
  const localStore = isWinArg ? (options.localStorage ?? defaultLocalStore) : (options?.localStore !== undefined ? options.localStore : defaultLocalStore);
  const sharedOrigin = isSharedPublicOrigin(win);
  let sessionAccessSucceeded = false;

  try {
    if (sessionStore && typeof sessionStore.getItem === 'function') {
      const sessionRaw = sessionStore.getItem(BYOK_SESSION_KEY);
      sessionAccessSucceeded = true;
      if (sessionRaw && sessionRaw.length <= 8192) {
        const parsed = JSON.parse(sessionRaw);
        if (parsed?.version === 1) {
          const cfg = normalizeByokConfig(parsed);
          setMemoryByokConfig(cfg);
          return cfg;
        }
      }
    }
  } catch { /* Ignore blocked sessionStorage */ }

  if (!sharedOrigin) {
    try {
      if (localStore && typeof localStore.getItem === 'function') {
        const persistedRaw = localStore.getItem(BYOK_PERSISTED_KEY);
        if (persistedRaw && persistedRaw.length <= 8192) {
          const parsed = JSON.parse(persistedRaw);
          if (parsed?.version === 1 && parsed.saveLocally === true) {
            const cfg = normalizeByokConfig(parsed);
            setMemoryByokConfig(cfg);
            return cfg;
          }
        }
      }
    } catch { /* Ignore blocked localStorage */ }

    // Also check legacy Study Tools userSettings if opted in on non-shared origin
    try {
      if (localStore && typeof localStore.getItem === 'function') {
        const legacyRaw = localStore.getItem('userSettings');
        if (legacyRaw && legacyRaw.length <= 65536) {
          const legacy = JSON.parse(legacyRaw);
          if (legacy?.customAPI?.enabled && legacy?.customAPI?.saveLocally === true && legacy?.customAPI?.apiKey) {
            const cfg = normalizeByokConfig(legacy.customAPI);
            setMemoryByokConfig(cfg);
            return cfg;
          }
        }
      }
    } catch { /* Ignore damaged legacy settings */ }
  }

  if (globalThis.__SOCRATIQ_BYOK_SESSION_FALLBACK_ONLY__ === true) {
    const mem = getMemoryByokConfig();
    if (mem && mem.enabled && mem.apiKey) return { ...mem };
  }

  if (sessionAccessSucceeded) {
    setMemoryByokConfig(null);
    return { ...normalizeByokConfig({}), enabled: false };
  }

  const mem = getMemoryByokConfig();
  if (mem) return { ...mem };
  return normalizeByokConfig({});
}

export function saveByokConfig(input, {
  sessionStore = getSessionStorage(),
  localStore = getLocalStorage(),
} = {}) {
  const sharedOrigin = isSharedPublicOrigin();
  const config = normalizeByokConfig({
    ...input,
    // Never persist API keys to localStorage on shared *.github.io origins
    saveLocally: sharedOrigin ? false : input?.saveLocally === true,
  });
  setMemoryByokConfig(config);
  try {
    globalThis.__SOCRATIQ_BYOK_SESSION_STORE__ = sessionStore || null;
    globalThis.__SOCRATIQ_BYOK_LOCAL_STORE__ = localStore || null;
  } catch {}
  const payload = JSON.stringify({ version: 1, ...config });

  let sessionWriteSucceeded = false;
  try {
    if (sessionStore && typeof sessionStore.setItem === 'function') {
      sessionStore.setItem(BYOK_SESSION_KEY, payload);
      sessionWriteSucceeded = true;
    }
  } catch { /* Keep in-memory fallback */ }
  try {
    globalThis.__SOCRATIQ_BYOK_SESSION_FALLBACK_ONLY__ = !sessionWriteSucceeded;
  } catch {}

  try {
    if (config.saveLocally && config.enabled && !sharedOrigin) {
      localStore?.setItem(BYOK_PERSISTED_KEY, payload);
    } else {
      localStore?.removeItem(BYOK_PERSISTED_KEY);
    }
  } catch { /* Ignore blocked localStorage */ }

  // Keep Study Tools runtime & legacy record synchronized without leaking keys when saveLocally is false
  try {
    if (globalThis.window) {
      globalThis.window.customAPISettings = {
        enabled: config.enabled,
        provider: config.provider === 'openrouter' ? 'open-router' : config.provider,
        endpoint: config.endpoint,
        model: config.model,
        apiKey: config.apiKey,
        saveLocally: config.saveLocally,
      };
    }
    const rawLegacy = localStore?.getItem('userSettings');
    const legacy = rawLegacy && rawLegacy.length <= 65536 ? JSON.parse(rawLegacy) : null;
    if (legacy && typeof legacy === 'object' && !Array.isArray(legacy)) {
      legacy.customAPI = {
        enabled: config.enabled,
        provider: config.provider === 'openrouter' ? 'open-router' : config.provider,
        endpoint: config.endpoint,
        model: config.model,
        apiKey: config.saveLocally ? config.apiKey : '',
        saveLocally: config.saveLocally,
      };
      localStore.setItem('userSettings', JSON.stringify(legacy));
    }
  } catch { /* Ignore storage errors */ }

  globalThis.dispatchEvent?.(new Event('socratiq:byok-changed'));
  return config;
}

export function clearByokConfig({
  sessionStore = getSessionStorage(),
  localStore = getLocalStorage(),
} = {}) {
  setMemoryByokConfig(null);
  try {
    globalThis.__SOCRATIQ_BYOK_SESSION_FALLBACK_ONLY__ = false;
    const activeSession = globalThis.__SOCRATIQ_BYOK_SESSION_STORE__;
    const activeLocal = globalThis.__SOCRATIQ_BYOK_LOCAL_STORE__;
    globalThis.__SOCRATIQ_BYOK_SESSION_STORE__ = null;
    globalThis.__SOCRATIQ_BYOK_LOCAL_STORE__ = null;
    activeSession?.removeItem?.(BYOK_SESSION_KEY);
    activeSession?.removeItem?.(BYOK_PKCE_STATE_KEY);
    activeLocal?.removeItem?.(BYOK_PERSISTED_KEY);
  } catch {}
  try { sessionStore?.removeItem(BYOK_SESSION_KEY); } catch {}
  try { sessionStore?.removeItem(BYOK_PKCE_STATE_KEY); } catch {}
  try { localStore?.removeItem(BYOK_PERSISTED_KEY); } catch {}
  try {
    if (globalThis.window) globalThis.window.customAPISettings = null;
    const rawLegacy = localStore?.getItem('userSettings');
    const legacy = rawLegacy && rawLegacy.length <= 65536 ? JSON.parse(rawLegacy) : null;
    if (legacy && typeof legacy === 'object' && !Array.isArray(legacy) && legacy.customAPI) {
      legacy.customAPI = {
        provider: '',
        endpoint: '',
        model: '',
        apiKey: '',
        saveLocally: false,
        enabled: false,
      };
      localStore.setItem('userSettings', JSON.stringify(legacy));
    }
  } catch {}
  globalThis.dispatchEvent?.(new Event('socratiq:byok-changed'));
  return normalizeByokConfig({});
}

function base64UrlEncode(bytes) {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

export async function createPkcePair(cryptoImpl = globalThis.crypto) {
  if (!cryptoImpl?.getRandomValues || !cryptoImpl?.subtle?.digest) {
    throw new Error('Web Crypto API is required for OAuth PKCE.');
  }
  const randomBytes = new Uint8Array(32);
  cryptoImpl.getRandomValues(randomBytes);
  const verifier = base64UrlEncode(randomBytes);
  const digest = await cryptoImpl.subtle.digest('SHA-256', new TextEncoder().encode(verifier));
  const challenge = base64UrlEncode(new Uint8Array(digest));
  return { verifier, challenge, method: 'S256' };
}

export async function startOpenRouterPKCE({
  model = BYOK_PROVIDERS.openrouter.defaultModel,
  locationObj = globalThis.location,
  sessionStore = getSessionStorage(),
  cryptoImpl = globalThis.crypto,
  navigate = true,
} = {}) {
  const { verifier, challenge, method } = await createPkcePair(cryptoImpl);
  if (!sessionStore) {
    throw new Error('Session storage is required to complete OAuth PKCE sign-in.');
  }
  const cleanCallback = new URL(locationObj.href);
  cleanCallback.searchParams.delete('code');
  cleanCallback.searchParams.delete('error');
  sessionStore.setItem(
    BYOK_PKCE_STATE_KEY,
    JSON.stringify({
      version: 1,
      verifier,
      model: model || BYOK_PROVIDERS.openrouter.defaultModel,
      createdAt: Date.now(),
    }),
  );
  const authUrl = new URL(BYOK_PROVIDERS.openrouter.authUrl);
  authUrl.searchParams.set('callback_url', cleanCallback.toString());
  authUrl.searchParams.set('code_challenge', challenge);
  authUrl.searchParams.set('code_challenge_method', method);
  const target = authUrl.toString();
  if (navigate && typeof locationObj?.assign === 'function') {
    locationObj.assign(target);
  }
  return { authUrl: target, challenge, method };
}

export async function completeOpenRouterPKCE({
  fetchImpl = globalThis.fetch,
  locationObj = globalThis.location,
  historyObj = globalThis.history,
  sessionStore = getSessionStorage(),
  localStore = getLocalStorage(),
} = {}) {
  if (!locationObj?.href) return null;
  let currentUrl;
  try {
    currentUrl = new URL(locationObj.href);
  } catch {
    return null;
  }
  const code = currentUrl.searchParams.get('code');
  if (!code) return null;

  // Immediately scrub ?code= and ?state= from the address bar BEFORE checking rawState
  // or making any network call so authorization codes never remain in the URL bar.
  currentUrl.searchParams.delete('code');
  currentUrl.searchParams.delete('state');
  const cleanUrl = currentUrl.pathname + currentUrl.search + currentUrl.hash;
  try {
    historyObj?.replaceState?.(historyObj.state ?? {}, '', cleanUrl);
  } catch { /* Ignore history errors in test stubs */ }

  let rawState = null;
  try {
    rawState = sessionStore?.getItem?.(BYOK_PKCE_STATE_KEY) ?? null;
  } catch {}
  try {
    sessionStore?.removeItem?.(BYOK_PKCE_STATE_KEY);
  } catch {}
  if (!rawState) return null;

  let state;
  try {
    state = JSON.parse(rawState);
  } catch {
    return null;
  }
  if (!state?.verifier || typeof state.verifier !== 'string') return null;
  // Require a valid non-future createdAt timestamp within the 15-minute PKCE window
  const now = Date.now();
  if (typeof state.createdAt !== 'number' || !Number.isFinite(state.createdAt) || state.createdAt > now || now - state.createdAt > 15 * 60 * 1000) {
    return null;
  }

  const response = await fetchImpl(BYOK_PROVIDERS.openrouter.tokenUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify({
      code,
      code_verifier: state.verifier,
      code_challenge_method: 'S256',
    }),
  });
  if (!response?.ok) {
    throw new Error(`OpenRouter sign-in exchange failed (HTTP ${response?.status ?? 'unknown'}).`);
  }
  const data = await response.json();
  const apiKey = typeof data?.key === 'string' ? data.key.trim() : '';
  if (!apiKey) {
    throw new Error('OpenRouter sign-in did not return an API key.');
  }
  return saveByokConfig(
    {
      enabled: true,
      provider: 'openrouter',
      model: state.model || BYOK_PROVIDERS.openrouter.defaultModel,
      endpoint: BYOK_PROVIDERS.openrouter.endpoint,
      apiKey,
      saveLocally: false,
      authMethod: 'oauth-pkce',
    },
    { sessionStore, localStore },
  );
}

/**
 * Build provider-specific request URL, headers, and body for direct browser BYOK requests.
 */
export function buildByokRequest({ config, messages, maxTokens = 2048, stream = true }) {
  const normalized = normalizeByokConfig(config);
  if (!normalized.enabled) {
    throw new Error('Custom AI provider is not enabled.');
  }
  const check = validateByokEndpoint(normalized.endpoint, normalized.provider);
  const url = check.url;
  const isLocal = Boolean(check.isLocal || url.startsWith('http://localhost') || url.startsWith('http://127.0.0.1'));
  const provider = normalized.provider;
  const headers = {
    'Content-Type': 'application/json',
    Accept: stream ? 'text/event-stream' : 'application/json',
  };

  const systemText = messages
    .filter(m => m.role === 'system')
    .map(m => m.content)
    .join('\n\n');
  const conversation = messages.filter(m => m.role !== 'system');

  if (provider === 'anthropic') {
    if (!isLocal && normalized.apiKey) {
      headers['x-api-key'] = normalized.apiKey;
    }
    headers['anthropic-version'] = '2023-06-01';
    headers['anthropic-dangerous-direct-browser-access'] = 'true';
    return {
      url,
      provider,
      format: 'anthropic',
      headers,
      body: JSON.stringify({
        model: normalized.model || BYOK_PROVIDERS.anthropic.defaultModel,
        max_tokens: maxTokens,
        ...(systemText ? { system: systemText } : {}),
        messages: conversation.map(m => ({
          role: m.role === 'assistant' ? 'assistant' : 'user',
          content: m.content,
        })),
        stream,
      }),
    };
  }

  if (provider === 'google-gemini' && !url.includes('/openai/')) {
    // Native Gemini REST endpoint using x-goog-api-key header (never URL query param)
    if (!isLocal && normalized.apiKey) {
      headers['x-goog-api-key'] = normalized.apiKey;
    }
    const baseUrl = url.replace(/\/$/, '');
    const model = normalized.model || BYOK_PROVIDERS['google-gemini'].defaultModel;
    const action = stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
    return {
      url: `${baseUrl}/models/${encodeURIComponent(model)}:${action}`,
      provider,
      format: 'gemini-native',
      headers,
      body: JSON.stringify({
        contents: conversation.map(m => ({
          role: m.role === 'assistant' ? 'model' : 'user',
          parts: [{ text: m.content }],
        })),
        ...(systemText ? { systemInstruction: { parts: [{ text: systemText }] } } : {}),
      }),
    };
  }

  if (!isLocal && normalized.apiKey) {
    headers.Authorization = `Bearer ${normalized.apiKey}`;
  }

  return {
    url,
    provider,
    format: 'openai',
    headers,
    body: JSON.stringify({
      model: normalized.model,
      messages,
      max_tokens: maxTokens,
      stream,
    }),
  };
}

export function parseByokEvent(rawData, format = 'openai') {
  if (rawData === '[DONE]') return { done: true };
  let payload;
  try {
    payload = JSON.parse(rawData);
  } catch {
    throw new Error('The custom AI provider returned an unreadable stream.');
  }
  if (payload?.error) {
    const rawMsg = typeof payload.error === 'string'
      ? payload.error
      : (payload.error.message || 'The custom AI provider rejected the request.');
    throw new Error(redactSecrets(rawMsg));
  }

  if (format === 'anthropic') {
    if (payload.type === 'message_stop') return { done: true, finishReason: 'stop' };
    if (payload.type === 'message_delta') {
      const stopReason = payload.delta?.stop_reason;
      return {
        content: '',
        finishReason: stopReason === 'max_tokens' ? 'length' : (stopReason ? 'stop' : null),
      };
    }
    if (payload.type === 'content_block_delta' && typeof payload.delta?.text === 'string') {
      return { content: payload.delta.text, finishReason: null };
    }
    return { content: '', finishReason: null };
  }

  if (format === 'gemini-native') {
    const candidate = payload?.candidates?.[0];
    const text = candidate?.content?.parts?.map(p => p.text || '').join('') || '';
    const finish = candidate?.finishReason;
    return {
      content: text,
      finishReason: finish === 'MAX_TOKENS' ? 'length' : (finish === 'STOP' ? 'stop' : null),
      done: finish === 'STOP',
    };
  }

  const choice = payload?.choices?.[0];
  if (!choice) return { content: '', finishReason: null };
  const content = choice.delta?.content;
  if (content !== undefined && typeof content !== 'string') {
    throw new Error('The custom AI provider returned an unexpected response.');
  }
  const rawFinish = choice.finish_reason ?? null;
  const finishReason = rawFinish === 'end_turn' || rawFinish === 'STOP' ? 'stop' : rawFinish;
  return { content: content ?? '', finishReason };
}
