// One provider call, in the provider's own wire, normalized on return.
//
// The entry speaks three upstream wires. Each call returns either a
// normalized answer (the provider's echo text plus its usage) or a failure
// with the HTTP status, so the caller can fail over to the next candidate.
// A call that answers 2xx but whose body cannot be parsed counts as a
// failure too: a malformed upstream answer can never serve a request.

import { authHeaders, joinBase, type ProviderDirectoryEntry, type Json } from './providers.ts';
import {
  parseAnthropicAnswer,
  parseAnthropicStream,
  parseGeminiAnswer,
  parseGeminiStream,
  parseOpenAIAnswer,
  parseOpenAIStream,
  type UpstreamAnswer,
} from './wire.ts';

export type UpstreamResult = { ok: true; answer: UpstreamAnswer } | { ok: false; status: number; error: string };

const CALL_TIMEOUT_MS = 15000;

async function postJson(url: string, headers: Record<string, string>, body: unknown): Promise<{ status: number; text: string }> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...headers },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
  });
  const text = await res.text();
  return { status: res.status, text };
}

/** POST an OpenAI-shaped body to an OpenAI-wire provider. */
export async function callOpenAIProvider(
  envUrl: string,
  provider: ProviderDirectoryEntry,
  body: Json,
  stream: boolean,
): Promise<UpstreamResult> {
  const base = joinBase(envUrl, provider.base);
  const url = `${base}/v1/chat/completions`;
  const headers = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  const request: Json = { ...body };
  if (stream) request.stream = true;
  // The client never sends stream_options, and without it the provider's
  // stream carries no usage chunk; the env's usage check would then fail.
  // Ask for it on every upstream stream.
  if (stream) request.stream_options = { include_usage: true };
  let res: { status: number; text: string };
  try {
    res = await postJson(url, headers, request);
  } catch (err) {
    return { ok: false, status: 0, error: `POST ${url} failed: ${(err as Error).message}` };
  }
  if (res.status < 200 || res.status >= 300) {
    return { ok: false, status: res.status, error: `POST ${url} answered HTTP ${res.status}` };
  }
  const answer = stream ? parseOpenAIStream(res.text) : parseOpenAIAnswer(res.text);
  if (answer === null) return { ok: false, status: res.status, error: `POST ${url} answered an unparseable body` };
  return { ok: true, answer };
}

/** POST an Anthropic-shaped body to an Anthropic-wire provider. */
export async function callAnthropicProvider(
  envUrl: string,
  provider: ProviderDirectoryEntry,
  body: Json,
  stream: boolean,
): Promise<UpstreamResult> {
  const base = joinBase(envUrl, provider.base);
  const url = `${base}/v1/messages`;
  const headers = { accept: 'application/json', 'anthropic-version': '2023-06-01', ...authHeaders(provider.auth, provider.key) };
  const request: Json = { ...body };
  if (stream) request.stream = true;
  let res: { status: number; text: string };
  try {
    res = await postJson(url, headers, request);
  } catch (err) {
    return { ok: false, status: 0, error: `POST ${url} failed: ${(err as Error).message}` };
  }
  if (res.status < 200 || res.status >= 300) {
    return { ok: false, status: res.status, error: `POST ${url} answered HTTP ${res.status}` };
  }
  const answer = stream ? parseAnthropicStream(res.text) : parseAnthropicAnswer(res.text);
  if (answer === null) return { ok: false, status: res.status, error: `POST ${url} answered an unparseable body` };
  return { ok: true, answer };
}

/** POST a Gemini-shaped body to a Gemini-wire provider. */
export async function callGeminiProvider(
  envUrl: string,
  provider: ProviderDirectoryEntry,
  body: Json,
  stream: boolean,
): Promise<UpstreamResult> {
  const base = joinBase(envUrl, provider.base);
  const model = typeof body.model === 'string' ? body.model : '';
  const action = stream ? ':streamGenerateContent' : ':generateContent';
  const url = `${base}/v1beta/models/${encodeURIComponent(model)}${action}${stream ? '?alt=sse' : ''}`;
  const headers = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  // The Gemini body carries its model in the URL, not in the payload.
  const { model: _model, ...request } = body;
  let res: { status: number; text: string };
  try {
    res = await postJson(url, headers, request);
  } catch (err) {
    return { ok: false, status: 0, error: `POST ${url} failed: ${(err as Error).message}` };
  }
  if (res.status < 200 || res.status >= 300) {
    return { ok: false, status: res.status, error: `POST ${url} answered HTTP ${res.status}` };
  }
  const answer = stream ? parseGeminiStream(res.text) : parseGeminiAnswer(res.text);
  if (answer === null) return { ok: false, status: res.status, error: `POST ${url} answered an unparseable body` };
  return { ok: true, answer };
}
