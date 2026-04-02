/**
 * StaffML Analytics Worker — Cloudflare Workers
 *
 * Receives anonymous analytics events from StaffML clients and stores them
 * in Cloudflare KV for aggregate analysis. No PII is collected.
 *
 * Setup:
 *   1. Create a KV namespace: wrangler kv:namespace create STAFFML_ANALYTICS
 *   2. Update wrangler.toml with the namespace ID
 *   3. Deploy: wrangler deploy
 *   4. Set NEXT_PUBLIC_ANALYTICS_URL in the StaffML build env
 *
 * Data model:
 *   KV key: "events:{YYYY-MM-DD}" → JSON array of events for that day
 *   KV key: "summary:latest"      → aggregated summary (updated on read)
 *   KV key: "meta:total_events"   → running event counter
 *
 * Security:
 *   - CORS restricted to allowed origins
 *   - Rate limited: max 100 events per request
 *   - No PII validation: rejects events containing email-like patterns
 *   - Session IDs are ephemeral UUIDs (per-tab, not per-user)
 */

const ALLOWED_ORIGINS = [
  'https://mlsysbook.ai',
  'https://harvard-edge.github.io',
  'http://localhost:3000',
  'http://localhost:3001',
];

const VALID_EVENT_TYPES = new Set([
  'question_scored', 'question_skipped', 'question_reported',
  'question_thumbs', 'question_difficulty_feedback', 'question_contributed',
  'answer_response_time',
  'gauntlet_started', 'gauntlet_completed', 'gauntlet_abandoned',
  'plan_started', 'plan_completed', 'daily_completed',
  'page_view', 'answer_revealed', 'improvement_suggested',
  'progress_exported', 'progress_imported',
]);

const MAX_EVENTS_PER_REQUEST = 100;
const MAX_EVENT_SIZE = 1024; // bytes

export default {
  async fetch(request, env) {
    const origin = request.headers.get('Origin') || '';
    const corsHeaders = {
      'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Access-Control-Max-Age': '86400',
    };

    // Only allow known origins
    if (ALLOWED_ORIGINS.some(o => origin.startsWith(o))) {
      corsHeaders['Access-Control-Allow-Origin'] = origin;
    } else if (origin) {
      return new Response('Forbidden', { status: 403 });
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

  const events = body.events;
  if (!Array.isArray(events)) {
    return jsonResponse({ error: 'events must be an array' }, 400, corsHeaders);
  }

  if (events.length > MAX_EVENTS_PER_REQUEST) {
    return jsonResponse({ error: `Max ${MAX_EVENTS_PER_REQUEST} events per request` }, 400, corsHeaders);
  }

  // Validate and sanitize each event
  const validated = [];
  for (const event of events) {
    // Check event type
    if (!event.type || !VALID_EVENT_TYPES.has(event.type)) continue;

    // Reject if event is too large (prevents abuse)
    if (JSON.stringify(event).length > MAX_EVENT_SIZE) continue;

    // Reject if any value looks like PII (email pattern)
    const values = Object.values(event).map(String).join(' ');
    if (/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/.test(values)) continue;

    // Strip any unexpected fields — only keep known ones + metadata
    const clean = { type: event.type };
    const allowedFields = [
      'topic', 'zone', 'level', 'track', 'score', 'questionId',
      'questionCount', 'pct', 'questionsAnswered', 'planId',
      'page', 'category', '_ts', '_sid',
      'value', 'perceived', 'seconds', 'napkinGrade',
    ];
    for (const field of allowedFields) {
      if (event[field] !== undefined) clean[field] = event[field];
    }
    validated.push(clean);
  }

  if (validated.length === 0) {
    return jsonResponse({ accepted: 0 }, 200, corsHeaders);
  }

  // Store events keyed by day
  const today = new Date().toISOString().split('T')[0];
  const key = `events:${today}`;

  try {
    // Read existing events for today
    const existing = await env.STAFFML_ANALYTICS.get(key, { type: 'json' }) || [];
    existing.push(...validated);

    // Store with 90-day TTL
    await env.STAFFML_ANALYTICS.put(key, JSON.stringify(existing), {
      expirationTtl: 90 * 24 * 60 * 60,
    });

    // Update running counter
    const countStr = await env.STAFFML_ANALYTICS.get('meta:total_events') || '0';
    const newCount = parseInt(countStr, 10) + validated.length;
    await env.STAFFML_ANALYTICS.put('meta:total_events', String(newCount));
  } catch (err) {
    return jsonResponse({ error: 'Storage error' }, 500, corsHeaders);
  }

  return jsonResponse({ accepted: validated.length }, 200, corsHeaders);
}

async function handleSummary(env, corsHeaders) {
  try {
    const totalEvents = parseInt(
      await env.STAFFML_ANALYTICS.get('meta:total_events') || '0', 10
    );

    // Get last 7 days of events for recent summary
    const days = [];
    const eventsByDay = {};
    let recentEvents = [];

    for (let i = 0; i < 7; i++) {
      const d = new Date();
      d.setDate(d.getDate() - i);
      const day = d.toISOString().split('T')[0];
      days.push(day);
      const dayEvents = await env.STAFFML_ANALYTICS.get(`events:${day}`, { type: 'json' }) || [];
      eventsByDay[day] = dayEvents.length;
      recentEvents = recentEvents.concat(dayEvents);
    }

    // Compute aggregates from recent events
    const sessions = new Set();
    let questionsScored = 0;
    let gauntletsCompleted = 0;
    let questionsReported = 0;
    const scoresByLevel = {};

    for (const event of recentEvents) {
      if (event._sid) sessions.add(event._sid);

      switch (event.type) {
        case 'question_scored':
          questionsScored++;
          if (event.level && event.score !== undefined) {
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
      }
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
