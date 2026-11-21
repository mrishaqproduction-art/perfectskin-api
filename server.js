'use strict';

/**
 * PerfectSkin API — production backend for the Flutter app.
 *
 * The Groq key lives here only. The app sends a selfie to this service; this
 * service validates it, asks a vision model for a STRUCTURED JSON report,
 * validates every number, and returns the report. Nothing is mocked.
 *
 * Privacy: images are held in memory only for the duration of the request.
 * They are never written to disk and are not stored after analysis.
 *
 * Run:  node server.js
 * Env:  GROQ_API_KEY (required), PORT (default 8000), GROQ_MODEL (optional)
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

// ── tiny .env loader (no dependencies) ────────────────────────────────────
const envPath = path.join(__dirname, '.env');
if (fs.existsSync(envPath)) {
  for (const line of fs.readFileSync(envPath, 'utf8').split(/\r?\n/)) {
    const m = /^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/.exec(line);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2].replace(/^['"]|['"]$/g, '');
  }
}

const PORT = Number(process.env.PORT || 8000);
const GROQ_API_KEY = (process.env.GROQ_API_KEY || '').trim();
const GROQ_MODEL = (process.env.GROQ_MODEL || 'qwen/qwen3.8-27b').trim();
const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';
const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const MAX_BODY_BYTES = 12 * 1024 * 1024;
const REQUEST_TIMEOUT_MS = Number(process.env.AI_TIMEOUT_MS || 90000);

// ── metric definitions (exact keys consumed by the Flutter app) ──────────
const METRICS = [
  { key: 'acne', name: 'Acne & Breakouts', emoji: '🫧' },
  { key: 'pigmentation', name: 'Pigmentation', emoji: '🍊' },
  { key: 'pores', name: 'Open Pores', emoji: '🔍' },
  { key: 'dullness', name: 'Dullness', emoji: '✨' },
  { key: 'hydration', name: 'Dryness', emoji: '💧' },
  { key: 'oiliness', name: 'Oiliness', emoji: '💦' },
  { key: 'redness', name: 'Redness & Sensitivity', emoji: '🌡️' },
  { key: 'dark_circles', name: 'Dark Circles', emoji: '👁️' },
  { key: 'texture', name: 'Texture & Fine Lines', emoji: '🪞' },
  { key: 'sun_damage', name: 'Sun Damage', emoji: '☀️' },
];

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Device-Id, X-App-Version',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
};

// ── helpers ───────────────────────────────────────────────────────────────
function send(res, status, body) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    ...CORS,
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    'Content-Length': Buffer.byteLength(payload),
  });
  res.end(payload);
}

function fail(status, message) {
  return Object.assign(new Error(message), { status });
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', (c) => {
      size += c.length;
      if (size > MAX_BODY_BYTES) {
        reject(fail(413, 'This request is too large.'));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on('end', () => {
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')));
      } catch (_) {
        reject(fail(400, 'Invalid JSON body.'));
      }
    });
    req.on('error', reject);
  });
}

function clamp(n, lo, hi) {
  const v = Number(n);
  return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : null;
}

function severity(pct) {
  if (pct >= 66) return 'High';
  if (pct >= 40) return 'Medium';
  return 'Low';
}

function extractJson(text) {
  if (!text) throw fail(502, 'The AI service returned an empty response.');
  let s = String(text).trim();
  s = s.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
  const start = s.indexOf('{');
  const end = s.lastIndexOf('}');
  if (start < 0 || end <= start) throw fail(502, 'The AI service did not return structured data.');
  try {
    return JSON.parse(s.slice(start, end + 1));
  } catch (_) {
    throw fail(502, 'The AI service returned invalid structured data.');
  }
}

/** Basic image sanity checks: real JPEG/PNG/WebP bytes and sensible dimensions. */
function inspectImage(buf) {
  if (buf.length < 1024) throw fail(400, 'This photo is too small or invalid.');
  if (buf.length > MAX_IMAGE_BYTES) throw fail(413, 'This photo is too large. Use a smaller image.');

  let width = 0;
  let height = 0;

  // JPEG
  if (buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff) {
    let i = 2;
    while (i + 9 < buf.length) {
      if (buf[i] !== 0xff) { i++; continue; }
      const marker = buf[i + 1];
      if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) { i += 2; continue; }
      const len = buf.readUInt16BE(i + 2);
      if (len < 2) break;
      if ([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf].includes(marker)) {
        height = buf.readUInt16BE(i + 5);
        width = buf.readUInt16BE(i + 7);
        break;
      }
      i += 2 + len;
    }
  } else if (buf.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) {
    width = buf.readUInt32BE(16);
    height = buf.readUInt32BE(20);
  } else if (buf.subarray(0, 4).toString('ascii') === 'RIFF' && buf.subarray(8, 12).toString('ascii') === 'WEBP') {
    // VP8X stores 24-bit dimensions at offset 24/27.
    if (buf.subarray(12, 16).toString('ascii') === 'VP8X') {
      width = 1 + buf.readUIntLE(24, 3);
      height = 1 + buf.readUIntLE(27, 3);
    }
  } else {
    throw fail(400, 'Unsupported photo format. Use JPEG, PNG or WebP.');
  }

  if (!width || !height) throw fail(400, 'Could not read this photo. Please try another image.');
  if (width < 200 || height < 200) throw fail(400, 'Photo is too small. Use a clearer selfie.');
  if (width > 8000 || height > 8000) throw fail(413, 'Photo resolution is too high. Use a smaller image.');
  return { width, height };
}

async function groq(messages, temperature = 0.35, maxTokens = 1200, jsonMode = false) {
  if (!GROQ_API_KEY) throw fail(503, 'The AI service is not configured on the server.');

  let res;
  try {
    res = await fetch(GROQ_URL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${GROQ_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: GROQ_MODEL,
        messages,
        temperature,
        max_tokens: maxTokens,
        ...(jsonMode ? { response_format: { type: 'json_object' } } : {}),
      }),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch (err) {
    if (err && err.name === 'TimeoutError') throw fail(504, 'The AI service took too long. Please try again.');
    throw fail(502, 'Could not reach the AI service. Please try again.');
  }

  const text = await res.text();
  if (!res.ok) {
    let detail = text.slice(0, 240);
    try { detail = JSON.parse(text)?.error?.message || detail; } catch (_) {}
    if (res.status === 429) throw fail(429, 'The AI service is busy right now. Please try again shortly.');
    throw fail(502, `AI provider failed (${res.status}): ${detail}`);
  }

  let data;
  try { data = JSON.parse(text); } catch (_) { throw fail(502, 'Invalid response from the AI service.'); }
  const content = data?.choices?.[0]?.message?.content;
  if (!content || !String(content).trim()) throw fail(502, 'The AI service returned an empty response.');
  return String(content).trim();
}

// ── rate limit (anonymous per-IP; no fake accounts) ───────────────────────
const limits = new Map();
const RATE_MAX = Number(process.env.RATE_LIMIT || 30);
const RATE_WINDOW_MS = 60 * 60 * 1000;
function checkRate(req) {
  const device = String(req.headers['x-device-id'] || '').slice(0, 80);
  const ip = req.socket.remoteAddress || 'unknown';
  const key = `${ip}|${device}`;
  const now = Date.now();
  let e = limits.get(key);
  if (!e || now - e.start > RATE_WINDOW_MS) {
    e = { start: now, count: 0 };
    limits.set(key, e);
  }
  e.count += 1;
  if (e.count > RATE_MAX) throw fail(429, 'Too many scans. Please try again in an hour.');
  // Opportunistic cleanup so the map does not grow forever.
  if (limits.size > 5000) {
    for (const [k, v] of limits) if (now - v.start > RATE_WINDOW_MS) limits.delete(k);
  }
}

// ── endpoints ─────────────────────────────────────────────────────────────
async function analyzeSkin(req, res) {
  checkRate(req);
  const body = await readJson(req);

  const b64 = typeof body.image_base64 === 'string' ? body.image_base64.trim() : '';
  if (!b64) throw fail(400, 'A selfie is required.');

  const mime = ['image/jpeg', 'image/png', 'image/webp'].includes(body.mime) ? body.mime : 'image/jpeg';
  let image;
  try {
    image = Buffer.from(b64, 'base64');
  } catch (_) {
    throw fail(400, 'Could not read this photo.');
  }
  const imageInfo = inspectImage(image);

  const answers = body.answers && typeof body.answers === 'object' ? body.answers : {};
  const metricList = METRICS.map((m) => `"${m.key}"`).join(', ');

  const prompt = `You are a cosmetic skin-analysis assistant. Examine the uploaded selfie carefully.

STEP 1 — Face check
Set "face_detected" to true ONLY if a clear human face is visible. If no face, or the face is obscured/too dark/blurry, return exactly:
{"face_detected": false, "reason": "brief reason"}

STEP 2 — Photo-derived metrics (only when face_detected is true)
Estimate SEVERITY of each concern from 0-100 (0 = none visible, 100 = very apparent). Base numbers ONLY on what is visible in the photo. Do not infer from stereotypes, age, gender or ethnicity. Do not invent data.

Return ONLY a JSON object with exactly these keys:
{
  "face_detected": true,
  ${metricList},
  "skinType": "Oily" | "Dry" | "Combination" | "Sensitive" | "Normal",
  "skinAge": integer 18..60,
  "summary": "one warm, encouraging sentence, max 35 words, no medical claims",
  "notes": "2 short sentences describing what you actually see in the photo"
}

User-reported context (use only as a hint, never as the source of numbers): skin type ${answers.skinType || 'unknown'}, concerns ${JSON.stringify(answers.concerns || [])}.
Cosmetic guidance only. No medical diagnosis.`;

  const reply = await groq([{
    role: 'user',
    content: [
      { type: 'text', text: prompt },
      { type: 'image_url', image_url: { url: `data:${mime};base64,${b64}` } },
    ],
  }], 0.25, 1000, true);

  const parsed = extractJson(reply);
  if (parsed.face_detected === false) {
    throw fail(422, String(parsed.reason || 'No clear face detected. Use a well-lit selfie.').slice(0, 180));
  }

  const rawMetrics = {};
  const metrics = METRICS.map((def) => {
    const pct = clamp(parsed[def.key], 0, 100);
    if (pct === null) throw fail(502, `The AI service did not return a valid value for ${def.key}.`);
    const value = Number((pct / 100).toFixed(3));
    rawMetrics[def.key] = value;
    return {
      key: def.key,
      name: def.name,
      emoji: def.emoji,
      value,
      severity: severity(pct),
    };
  });

  // Overall score is derived from the returned photo metrics — never invented.
  const avg = metrics.reduce((s, m) => s + m.value, 0) / metrics.length;
  const score = clamp(Math.round(94 - avg * 58), 1, 100);
  const skinAge = clamp(parsed.skinAge, 18, 60) ?? Math.round(22 + avg * 16);
  const allowedTypes = ['Oily', 'Dry', 'Combination', 'Sensitive', 'Normal'];
  const rawType = String(parsed.skinType || '').toLowerCase();
  const skinType = allowedTypes.find((t) => t.toLowerCase() === rawType) ||
    (allowedTypes.includes(answers.skinType) ? answers.skinType : 'Combination');

  const summary = String(parsed.summary || '').trim().slice(0, 320) ||
    'Your selfie has been analysed. Review your stats below.';
  const notes = String(parsed.notes || '').trim().slice(0, 500);

  send(res, 200, {
    score,
    skinAge,
    skinType,
    summary,
    aiUsed: true,
    grade: score >= 85 ? 'Radiant' : score >= 72 ? 'Healthy' : score >= 58 ? 'Fair' : 'Needs care',
    metrics,
    concerns: [...metrics].sort((a, b) => b.value - a.value).slice(0, 4),
    visualNotes: notes,
    imageInfo,
  });
}

async function coachAsk(req, res) {
  checkRate(req);
  const body = await readJson(req);
  const prompt = String(body.prompt || '').trim();
  if (!prompt) throw fail(400, 'A question is required.');
  if (prompt.length > 2000) throw fail(400, 'Question is too long.');

  const reply = await groq([
    {
      role: 'system',
      content: "You are PerfectSkin's friendly AI skin coach. Answer in at most 80 words, practical and kind. Cosmetic/educational guidance only — no medical diagnosis. If a serious condition is described, suggest a dermatologist.",
    },
    { role: 'user', content: prompt },
  ], 0.7, 400);

  send(res, 200, { reply });
}

const PRIVACY = `PerfectSkin Privacy Notice
We ask for a selfie only when you choose to scan. The photo is sent over HTTPS to our backend, which uses it for a single AI analysis and then discards it. We do not write your photo to disk, do not sell it, and do not use it for advertising. Questionnaire answers are used only to personalise the report. We do not operate user accounts in this release, so there is no profile to delete. Contact the operator for any privacy request.`;

const TERMS = `PerfectSkin Terms
PerfectSkin provides cosmetic, educational guidance only. It is not a medical device and does not diagnose or treat any condition. Results are estimates from a single photo and can be wrong. Do not rely on them for health decisions; see a qualified dermatologist for medical concerns. Product links open Amazon; if an affiliate tag is configured, we may earn a commission at no extra cost to you.`;

// ── server ────────────────────────────────────────────────────────────────
const server = http.createServer(async (req, res) => {
  if (req.method === 'OPTIONS') {
    res.writeHead(204, CORS);
    return res.end();
  }
  const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  try {
    if (req.method === 'GET' && url.pathname === '/health') {
      return send(res, 200, { ok: true, groqConfigured: Boolean(GROQ_API_KEY), model: GROQ_MODEL, storage: 'none' });
    }
    if (req.method === 'GET' && url.pathname === '/legal/privacy') {
      res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8', ...CORS });
      return res.end(PRIVACY);
    }
    if (req.method === 'GET' && url.pathname === '/legal/terms') {
      res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8', ...CORS });
      return res.end(TERMS);
    }
    if (req.method === 'POST' && url.pathname === '/v1/skin/analyze') return await analyzeSkin(req, res);
    if (req.method === 'POST' && url.pathname === '/v1/coach/ask') return await coachAsk(req, res);
    return send(res, 404, { error: 'Not found' });
  } catch (err) {
    const status = err.status || 500;
    if (status >= 500) console.error('[error]', err);
    return send(res, status, { error: err.message || 'Internal server error' });
  }
});

server.listen(PORT, () => {
  console.log(`PerfectSkin API listening on http://localhost:${PORT}`);
  console.log(`Groq configured: ${Boolean(GROQ_API_KEY)} | model: ${GROQ_MODEL} | storage: none`);
});
