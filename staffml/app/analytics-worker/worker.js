/**
 * StaffML Analytics Worker — Cloudflare Workers
 *
 * Receives anonymous analytics events from StaffML clients and stores them
 * in Cloudflare KV for aggregate analysis. Email-pattern filtering is a
 * heuristic and does not guarantee that submitted fields contain no PII.
 *
 * Setup:
 *   1. For a new deployment, create KV: wrangler kv namespace create staffml-analytics-events
 *   2. Assign its ID to STAFFML_ANALYTICS; reuse existing IDs for an existing deployment
 *   3. Deploy: wrangler deploy
 *   4. Set NEXT_PUBLIC_ANALYTICS_URL in the StaffML build env
 *
 * Data model:
 *   KV key: "events:{YYYY-MM-DD}:{UUID}" → JSON array for one submitted batch
 *   Summaries are computed on read over the recent event keys.
 *
 * Security:
 *   - CORS restricted to allowed origins
 *   - Batch cap: max 100 events per request; no request-frequency limiter
 *   - Email-pattern heuristic rejects matching events
 *   - Session IDs are ephemeral UUIDs (per-tab, not per-user)
 */

const ALLOWED_ORIGINS = [
  'https://mlsysbook.ai',
  'https://harvard-edge.github.io',
  'http://localhost:3000',
  'http://localhost:3001',
];

const VALID_EVENT_TYPES = new Set([
  'session_start',
  'question_scored', 'question_skipped', 'question_reported',
  'question_thumbs', 'question_difficulty_feedback', 'question_contributed',
  'answer_response_time', 'answer_revealed',
  'gauntlet_started', 'gauntlet_completed', 'gauntlet_abandoned',
  'plan_started', 'plan_completed', 'daily_completed',
  'page_view', 'search_query',
  'star_gate_shown', 'star_gate_verified',
  'improvement_suggested', 'progress_exported', 'progress_imported',
  // Runtime error reports — captured by window.onerror + unhandledrejection.
  // Stack traces can exceed the default per-event cap, so these events are
  // sized against MAX_ERROR_EVENT_SIZE instead.
  'client_error',
]);

const MAX_EVENTS_PER_REQUEST = 100;
const MAX_EVENT_SIZE = 1024;        // bytes — default per-event cap
const MAX_ERROR_EVENT_SIZE = 8192;  // bytes — larger cap for client_error stacks

export default {
  async fetch(request, env) {
    const origin = request.headers.get('Origin') || '';
    const corsHeaders = {
      'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Access-Control-Max-Age': '86400',
    };

    // Origin contains a serialized origin, never a URL path or credentials.
    // Compare parsed origins so lookalike hosts cannot pass a prefix check.
    if (origin) {
      let parsed;
      try {
        parsed = new URL(origin);
      } catch {
        return new Response('Forbidden', { status: 403 });
      }
      if (parsed.username || parsed.password || parsed.pathname !== '/' ||
          parsed.search || parsed.hash || !ALLOWED_ORIGINS.includes(parsed.origin)) {
        return new Response('Forbidden', { status: 403 });
      }
      corsHeaders['Access-Control-Allow-Origin'] = parsed.origin;
    }

    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders });
    }

    // POST /events — receive event batch
    if (request.method === 'POST') {
      return handleEvents(request, env, corsHeaders);
    }

    // GET /summary — return aggregate stats
    if (request.method === 'GET') {
      const auth = request.headers.get('Authorization');
      const secret = env.ADMIN_SECRET;
      if (typeof secret !== 'string' || !secret.trim() || auth !== `Bearer ${secret}`) {
        return new Response('Unauthorized', { status: 401, headers: corsHeaders });
      }
      return handleSummary(env, corsHeaders);
    }

    return new Response('Method not allowed', { status: 405, headers: corsHeaders });
  },
};

async function handleEvents(request, env, corsHeaders) {
  let body;
  try {
    body = await request.json();
  } catch {
    return jsonResponse({ error: 'Invalid JSON' }, 400, corsHeaders);
  }

  const events = body?.events;
  if (!Array.isArray(events)) {
    return jsonResponse({ error: 'events must be an array' }, 400, corsHeaders);
  }

  if (events.length > MAX_EVENTS_PER_REQUEST) {
    return jsonResponse({ error: `Max ${MAX_EVENTS_PER_REQUEST} events per request` }, 400, corsHeaders);
  }

  // Validate and sanitize each event
  const validated = [];
  for (const event of events) {
    if (!event || typeof event !== 'object' || Array.isArray(event)) continue;
    // The client event contract contains scalar fields. Reject nested values
    // before string coercion or aggregation can invoke attacker-controlled keys.
    if (Object.values(event).some(value => value !== null && typeof value === 'object')) continue;
    // Check event type
    if (!event.type || !VALID_EVENT_TYPES.has(event.type)) continue;

    // Size cap: client_error gets a bigger envelope to fit stack traces.
    const sizeLimit = event.type === 'client_error' ? MAX_ERROR_EVENT_SIZE : MAX_EVENT_SIZE;
    if (JSON.stringify(event).length > sizeLimit) continue;

    // Reject if any value looks like PII (email pattern)
    const values = Object.values(event).map(String).join(' ');
    if (/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/.test(values)) continue;

    // Strip any unexpected fields — only keep known ones + metadata
    const clean = { type: event.type };
    const allowedFields = [
      'topic', 'subject', 'zone', 'level', 'track', 'score', 'questionId',
      'questionCount', 'pct', 'questionsAnswered', 'planId',
      'page', 'category', '_ts', '_sid',
      'value', 'perceived', 'seconds', 'napkinGrade',
      'hadUserAnswer', 'isReturning', 'screenWidth',
      'query', 'topicResults', 'questionResults',
      'userLevel', 'industryRole', 'yearsExperience', // IRT telemetry
      // client_error fields
      'message', 'stack', 'url', 'userAgent',
    ];
    for (const field of allowedFields) {
      if (event[field] !== undefined) clean[field] = event[field];
    }
    validated.push(clean);
  }

  if (validated.length === 0) {
    return jsonResponse({ accepted: 0 }, 200, corsHeaders);
  }

  // Store events keyed by day + uuid to prevent RMW race condition
  const today = new Date().toISOString().split('T')[0];
  const batchId = crypto.randomUUID();
  const key = `events:${today}:${batchId}`;

  try {
    // Store with 90-day TTL
    await env.STAFFML_ANALYTICS.put(key, JSON.stringify(validated), {
      expirationTtl: 90 * 24 * 60 * 60,
    });

    // We no longer update a single running counter here to avoid race conditions.
    // The summary function can aggregate keys instead.
  } catch (err) {
    return jsonResponse({ error: 'Storage error' }, 500, corsHeaders);
  }

  return jsonResponse({ accepted: validated.length }, 200, corsHeaders);
}

async function handleSummary(env, corsHeaders) {
  try {
    // Get last 7 days of events for recent summary using list()
    const days = [];
    const eventsByDay = {};
    let recentEvents = [];
    let totalEvents = 0;

    for (let i = 0; i < 7; i++) {
      const d = new Date();
      d.setDate(d.getDate() - i);
      const day = d.toISOString().split('T')[0];
      days.push(day);
      
      let dayEvents = [];
      let cursor = null;
      do {
        const listResult = await env.STAFFML_ANALYTICS.list({ prefix: `events:${day}:`, cursor });
        for (const key of listResult.keys) {
          const batch = await env.STAFFML_ANALYTICS.get(key.name, { type: 'json' }) || [];
          dayEvents.push(...batch);
        }
        cursor = listResult.cursor;
      } while (cursor);

      eventsByDay[day] = dayEvents.length;
      recentEvents = recentEvents.concat(dayEvents);
      totalEvents += dayEvents.length;
    }

    // Compute aggregates from recent events
    const sessions = new Set();
    let questionsScored = 0;
    let gauntletsCompleted = 0;
    let questionsReported = 0;
    let improvementsSuggested = 0;
    // Levels are submitted by anonymous clients, including prototype key names.
    const scoresByLevel = Object.create(null);
    // Dedup feedback: last-write-wins per (questionId, sessionId)
    const latestThumbs = new Map();
    const latestDifficulty = new Map();

    for (const event of recentEvents) {
      if (event._sid) sessions.add(event._sid);

      switch (event.type) {
        case 'question_scored':
          questionsScored++;
          if (typeof event.level === 'string' && event.level && Number.isFinite(event.score)) {
            if (!scoresByLevel[event.level]) scoresByLevel[event.level] = { total: 0, count: 0 };
            scoresByLevel[event.level].total += event.score;
            scoresByLevel[event.level].count++;
          }
          break;
        case 'gauntlet_completed':
          gauntletsCompleted++;
          break;
        case 'question_reported':
          questionsReported++;
          break;
        case 'improvement_suggested':
          improvementsSuggested++;
          break;
        case 'question_thumbs':
          if (event.questionId && event._sid) {
            latestThumbs.set(`${event.questionId}:${event._sid}`, event.value);
          }
          break;
        case 'question_difficulty_feedback':
          if (event.questionId && event._sid) {
            latestDifficulty.set(`${event.questionId}:${event._sid}`, event.perceived);
          }
          break;
      }
    }

    // Aggregate deduplicated feedback
    let thumbsUp = 0, thumbsDown = 0;
    for (const v of latestThumbs.values()) {
      if (v === 'up') thumbsUp++; else thumbsDown++;
    }
    const difficultyDistribution = { too_easy: 0, about_right: 0, too_hard: 0 };
    for (const v of latestDifficulty.values()) {
      if (Object.hasOwn(difficultyDistribution, v)) difficultyDistribution[v]++;
    }

    // Compute averages
    for (const v of Object.values(scoresByLevel)) {
      v.avg = v.count > 0 ? (v.total / v.count).toFixed(2) : 0;
    }

    return jsonResponse({
      totalEvents,
      last7Days: {
        uniqueSessions: sessions.size,
        questionsScored,
        gauntletsCompleted,
        questionsReported,
        improvementsSuggested,
        thumbsUp,
        thumbsDown,
        difficultyDistribution,
        eventsByDay,
        scoresByLevel,
      },
    }, 200, corsHeaders);
  } catch (err) {
    return jsonResponse({ error: 'Summary error' }, 500, corsHeaders);
  }
}

function jsonResponse(data, status, headers) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
  });
}
