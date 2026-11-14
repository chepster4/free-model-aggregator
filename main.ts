// The best free AI model aggregator.
//
// A server that registers its catalog with the trial environment and serves
// the environment's 20 requests, one at a time, each answered by a free
// model that meets the request's requirements. The entry is the best model
// the evidence allows, not the first model the directory lists.
//
//   node main.ts                 register, serve the 20 requests, exit 0
//   node main.ts --list          print the catalog, byte for byte as registered
//   node main.ts --help          this text
//
// Routing, in one paragraph. The brief carries the quality table (the sealed
// truth: each free model's rating in milli-points and what it can do) and the
// workload manifest (the 20 requests, their weight, their requirements).
// At load, every request is planned: the weight-2 requests pick first, each
// taking the best-rated free model that meets its requirements, within a
// conservative per-provider call budget (the minimum the kit's sealed
// provider table allows for that provider, so a plan never over-commits one).
// At serve time a planned pick that refuses (402/429/5xx, or an unparseable
// body) fails over to the next-best candidate, preferably on another
// provider; a provider that answers 402 is treated as spent, one that
// answers 429 twice in a row is treated as spent too, and a provider that
// refuses once is deprioritised until it succeeds again. One provider call
// at a time, always: a provider with concurrency 1 cannot take more, and a
// refused call still consumes budget, so a call is never wasted on a
// provider the trial has already shown to be spent.
//
// Wires. The client speaks OpenAI or Anthropic; the providers speak OpenAI,
// Anthropic or Gemini. The entry translates between all six combinations,
// preserving the fields the env fingerprints in its echo (system prompt,
// user text, image bytes, tool names in order, max_tokens, stop sequences,
// JSON mode). Two details the starter got wrong are fixed here: an OpenAI
// stream is only given usage when the upstream call asks for it, so every
// upstream OpenAI stream carries stream_options.include_usage; and a JSON
// request never goes to an Anthropic-wire provider, whose wire cannot echo
// JSON mode. A Gemini upstream call sends the model in the URL
// (:generateContent / :streamGenerateContent) and JSON mode as
// generationConfig.responseMimeType.
//
// The entry keeps no wall clock and no randomness: the plan is a function of
// the brief, the catalog is a function of the probes, and every response
// carries fixed ids and timestamps, so a replay leaves identical state.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { writeFileSync } from 'node:fs';
import {
  buildCatalog,
  capabilitiesOf,
  isRecord,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  PLANNED_BUDGET,
  PLANNED_MIN,
  type Capabilities,
  type CatalogReport,
  type Json,
  type Probe,
  type ProviderDirectoryEntry,
  type RatingsFile,
  type WorkloadFile,
  type WorkloadRequest,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicJsonReply,
  anthropicStreamFrames,
  anthropicToGemini,
  anthropicToOpenAI,
  openAIError,
  openAIJsonReply,
  openAIRequestToAnthropic,
  openAIStreamFrames,
  openAIToGemini,
  type UpstreamAnswer,
} from './lib/wire.ts';
import { callAnthropicProvider, callGeminiProvider, callOpenAIProvider, type UpstreamResult } from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const TRIAL_SEED = process.env.MAKEFUN_TRIAL_SEED ?? '';
const STATE_FILE = process.env.MAKEFUN_STATE_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let workload: WorkloadFile | null = null;
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/** The entry's working state, exported to $MAKEFUN_STATE_FILE at exit when the env sets one. */
const state = {
  brief_error: brief.error,
  probes: [] as { provider: string; ok: boolean; status: number; error: string | null }[],
  catalog: { providers: [] } as CatalogReport,
  trial_seed: TRIAL_SEED,
  plan: [] as { request: string; provider: string; model: string; rating: number }[],
  served: [] as { request: string; provider: string; model: string; rating: number; failover: string[] }[],
  unserved: [] as string[],
  upstream_calls: 0,
  version: 1,
};

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function saveState(): void {
  if (STATE_FILE === '') return;
  try {
    writeFileSync(STATE_FILE, `${JSON.stringify(state, null, 2)}\n`, 'utf8');
  } catch {
    // A state file that cannot be written is not a serving failure.
  }
}

/** One routing candidate: a free model on one provider, with its rating. */
interface Candidate {
  provider: ProviderDirectoryEntry;
  model: string;
  rating: number;
  caps: Capabilities;
}

let candidates: Candidate[] = [];
const planByRequest = new Map<string, Candidate>();
const fallbackByRequest = new Map<string, Candidate[]>();

/** Every free model every probed provider listed, with its sealed rating. */
function buildCandidates(probes: Probe[], ratings: RatingsFile | null): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ provider: probe.provider, model: model.id, rating: row?.rating_milli ?? 0, caps: capabilitiesOf(model, row) });
    }
  }
  return out;
}

/** Whether the client's body asks for JSON mode (OpenAI wire only; the Anthropic wire has no JSON mode). */
function wantsJson(api: 'openai' | 'anthropic', body: Json): boolean {
  if (api !== 'openai') return false;
  if (!isRecord(body.response_format)) return false;
  return body.response_format.type === 'json_object' || body.response_format.type === 'json_schema';
}

/** Candidates that can serve one request: requirements met, and a JSON request never goes to an Anthropic-wire provider. */
function eligibleFor(request: WorkloadRequest, jsonRequest: boolean): Candidate[] {
  const ok = candidates.filter((c) => meetsRequirements(request.requires, c.caps));
  if (!jsonRequest) return ok;
  return ok.filter((c) => c.provider.wire !== 'anthropic');
}

function candidateKey(c: Candidate): string {
  return `${c.provider.id}/${c.model}`;
}

function compareCandidates(a: Candidate, b: Candidate, budgetLeft: (id: string) => number): number {
  if (a.rating !== b.rating) return b.rating - a.rating;
  const ba = budgetLeft(a.provider.id);
  const bb = budgetLeft(b.provider.id);
  if (ba !== bb) return bb - ba;
  if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
  if (a.model !== b.model) return a.model < b.model ? -1 : 1;
  return 0;
}

/**
 * Plan the 20 requests before serving: weight-2 requests pick first, each
 * taking the best-rated eligible candidate with planned budget left. The
 * plan is a function of the brief only, so it is identical on a replay.
 */
function planRequests(manifest: WorkloadRequest[]): void {
  const remaining = new Map<string, number>();
  for (const c of candidates) {
    if (!remaining.has(c.provider.id)) remaining.set(c.provider.id, PLANNED_BUDGET[c.provider.id] ?? 4);
  }
  const budgetLeft = (id: string): number => remaining.get(id) ?? 0;
  const ordered = [...manifest].sort((a, b) => {
    if (a.weight !== b.weight) return b.weight - a.weight;
    return manifest.indexOf(a) - manifest.indexOf(b);
  });
  for (const request of ordered) {
    // The plan uses the manifest's json requirement as its proxy for the
    // body's response_format; the serve-time filter re-checks the real body.
    const eligible = eligibleFor(request, request.requires.json === true).sort((a, b) => compareCandidates(a, b, budgetLeft));
    fallbackByRequest.set(request.id, eligible);
    const pick = eligible.find((c) => budgetLeft(c.provider.id) > 0) ?? eligible[0] ?? null;
    if (pick === null) continue;
    planByRequest.set(request.id, pick);
    remaining.set(pick.provider.id, budgetLeft(pick.provider.id) - 1);
    state.plan.push({ request: request.id, provider: pick.provider.id, model: pick.model, rating: pick.rating });
  }
}

// --- serving ---------------------------------------------------------------

const callsMade = new Map<string, number>();
const exhausted = new Set<string>();
const consecutiveFailures = new Map<string, number>();

function budgetLeftLive(id: string): number {
  return (PLANNED_BUDGET[id] ?? 4) - (callsMade.get(id) ?? 0);
}

function noteCall(providerId: string, failed: boolean, status: number): void {
  callsMade.set(providerId, (callsMade.get(providerId) ?? 0) + 1);
  state.upstream_calls += 1;
  if (failed) {
    consecutiveFailures.set(providerId, (consecutiveFailures.get(providerId) ?? 0) + 1);
    // A 402 is a spent budget, full stop. Consecutive 429s only mean a
    // spent budget once the provider has answered at least its minimum
    // budget: below that count a refusal is a transient fail_call, and two
    // in a row are just two fail_calls, not an exhausted provider.
    if (status === 402 || ((consecutiveFailures.get(providerId) ?? 0) >= 2 && (callsMade.get(providerId) ?? 0) >= (PLANNED_MIN[providerId] ?? 4))) {
      exhausted.add(providerId);
    }
  } else {
    consecutiveFailures.set(providerId, 0);
  }
}

/** Build the provider-native body for one candidate from the client's body. */
function providerBody(clientApi: 'openai' | 'anthropic', body: Json, candidate: Candidate): Json {
  const model = candidate.model;
  if (candidate.provider.wire === 'openai') {
    return clientApi === 'openai' ? { ...body, model } : anthropicToOpenAI(body, model);
  }
  if (candidate.provider.wire === 'anthropic') {
    return clientApi === 'anthropic' ? { ...body, model } : openAIRequestToAnthropic(body, model);
  }
  return clientApi === 'openai' ? openAIToGemini(body, model) : anthropicToGemini(body, model);
}

async function callCandidate(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<UpstreamResult> {
  const upstreamBody = providerBody(clientApi, body, candidate);
  if (candidate.provider.wire === 'openai') return callOpenAIProvider(ENV_URL, candidate.provider, upstreamBody, stream);
  if (candidate.provider.wire === 'anthropic') return callAnthropicProvider(ENV_URL, candidate.provider, upstreamBody, stream);
  return callGeminiProvider(ENV_URL, candidate.provider, upstreamBody, stream);
}

/**
 * Serve one request: try the planned candidate, then the fallback chain
 * (best-rated first). A provider flagged exhausted is skipped while any
 * untried candidate remains. Returns the answer and the candidate that
 * served it, or null when no candidate served it.
 */
async function serveRequest(
  request: WorkloadRequest,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<{ answer: UpstreamAnswer; candidate: Candidate; failover: string[] } | null> {
  const jsonRequest = wantsJson(clientApi, body);
  const ordered = fallbackByRequest.get(request.id) ?? eligibleFor(request, jsonRequest).sort((a, b) => b.rating - a.rating);
  const planned = planByRequest.get(request.id);
  const tried = new Set<string>();
  const tryList: Candidate[] = [];
  if (planned !== undefined) tryList.push(planned);
  for (const c of ordered) {
    if (!tryList.some((t) => candidateKey(t) === candidateKey(c))) tryList.push(c);
  }
  const failover: string[] = [];
  // Candidates are tried in rating order (planned first, then the fallback
  // chain): availability never reorders them, because a lower-rated model
  // that happens to be unflagged is still a worse answer than a flagged
  // higher-rated one that would answer on a retry. A provider flagged
  // exhausted is skipped while any untried candidate remains, and tried
  // anyway as a last resort when it is all that is left.
  for (const candidate of tryList) {
    if (tried.has(candidateKey(candidate))) continue;
    if (exhausted.has(candidate.provider.id)) {
      const othersRemain = tryList.some((c) => !tried.has(candidateKey(c)) && !exhausted.has(c.provider.id));
      if (othersRemain) continue;
    }
    tried.add(candidateKey(candidate));
    const result = await callCandidate(candidate, clientApi, body, stream);
    if (result.ok) {
      noteCall(candidate.provider.id, false, 200);
      return { answer: result.answer, candidate, failover };
    }
    noteCall(candidate.provider.id, true, result.status);
    failover.push(`${candidate.provider.id}/${candidate.model}:${result.status}`);
    // A refusal while the provider still has planned budget left cannot be
    // a spent budget: it is a transient fail_call (or a brief outage), and
    // the next call number usually answers. Retry the same candidate once
    // before falling over, so one bad call does not demote the best model
    // for the rest of the trial. A 402 is a spent budget: never retry it.
    if (result.status !== 402 && budgetLeftLive(candidate.provider.id) > 0) {
      const retry = await callCandidate(candidate, clientApi, body, stream);
      if (retry.ok) {
        noteCall(candidate.provider.id, false, 200);
        return { answer: retry.answer, candidate, failover };
      }
      noteCall(candidate.provider.id, true, retry.status);
      failover.push(`${candidate.provider.id}/${candidate.model}:${retry.status} (retry)`);
    }
  }
  return null;
}

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function sendSse(res: ServerResponse, frames: string[]): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  for (const frame of frames) res.write(frame);
  res.end();
}

/** The request now arriving: the manifest's row by arrival order, else a bare row from the body. */
function requestFor(api: 'openai' | 'anthropic', stream: boolean): WorkloadRequest {
  const manifest = workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry;
  return { id: 'unknown', api, stream, weight: 1, requires: {} };
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: '2026-01-01T00:00:00Z', free: true, context_length: null, tools: null, vision: null, json: null })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: 1767225600, owned_by: m.provider, free: true, context_length: null, tools: null, vision: null, json: null })),
    });
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that serves it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null || !Array.isArray(body.messages)) {
    return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not a chat completion');
  }
  const stream = body.stream === true;
  const request = requestFor('openai', stream);
  const served = await serveRequest(request, 'openai', body, stream);
  if (served === null) {
    state.unserved.push(request.id);
    return sendError(res, 'openai', 503, 'no_free_model', 'no free model could serve this request');
  }
  state.served.push({ request: request.id, provider: served.candidate.provider.id, model: served.candidate.model, rating: served.candidate.rating, failover: served.failover });
  if (stream) sendSse(res, openAIStreamFrames(served.candidate.model, served.answer));
  else sendJson(res, 200, openAIJsonReply(served.candidate.model, served.answer));
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model that serves it. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null || !Array.isArray(body.messages)) {
    return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not a message');
  }
  const stream = body.stream === true;
  const request = requestFor('anthropic', stream);
  const served = await serveRequest(request, 'anthropic', body, stream);
  if (served === null) {
    state.unserved.push(request.id);
    return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model could serve this request');
  }
  state.served.push({ request: request.id, provider: served.candidate.provider.id, model: served.candidate.model, rating: served.candidate.rating, failover: served.failover });
  if (stream) sendSse(res, anthropicStreamFrames(served.candidate.model, served.answer));
  else sendJson(res, 200, anthropicJsonReply(served.candidate.model, served.answer));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  state.catalog = catalog;
  state.probes = probes.map((p) => ({ provider: p.provider.id, ok: p.ok, status: p.status, error: p.error }));
  workload = brief.workload;
  candidates = buildCandidates(probes, brief.ratings);
  if (workload !== null) planRequests(workload.manifest);
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          saveState();
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (process.argv.includes('--help')) {
    console.log(
      [
        'Usage: node main.ts [--list]',
        '',
        'The best free AI model aggregator. Registers its catalog with the trial environment',
        '($MAKEFUN_ENV_URL) and serves the environment\'s requests, each answered by a free',
        'model that meets the request\'s requirements, chosen by the brief\'s quality table.',
        '',
        '  --list   Print the catalog, byte for byte as registered, and exit.',
      ].join('\n'),
    );
    return;
  }
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => {
    saveState();
    process.exit(0);
  });
  process.on('SIGINT', () => {
    saveState();
    process.exit(0);
  });
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  saveState();
  process.exit(1);
}
