// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini and anthropic upstream wires are not implemented in this baseline (main.ts answers 501 for a
// provider on either): a real entry needs their request and event shapes too, which is the biggest single
// gap in this starter.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// ---------------------------------------------------------------------------
// Cross-wire translation, upstream answer parsing, and client rendering.
//
// The trial's three provider wires each fingerprint what they receive
// (system, user text, image bytes, tool names, max_tokens, stops, JSON mode)
// and echo it back in their answer. A translation must preserve every one
// of those fields or the env's echo checks fail. The workload's bodies carry
// only system + user messages, so tool-result messages are handled
// best-effort: on the Gemini wire they ride in a `model` content (the
// Gemini parser skips `model` contents, so the last user text survives);
// on the Anthropic wire they become tool_result blocks in a user message.
// ---------------------------------------------------------------------------

/** One provider answer, normalized: the echo text and the provider's usage. */
export interface UpstreamAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

const FIXED_ID_OPENAI = 'chatcmpl-aggregate';
const FIXED_ID_ANTHROPIC = 'msg_aggregate';

/** Split an SSE body into frames: {event, data} per `data: ...` block. */
export function parseSSEFrames(body: string): { event: string; data: string }[] {
  const frames: { event: string; data: string }[] = [];
  for (const chunk of body.split(/\r?\n\r?\n/)) {
    if (chunk.trim() === '') continue;
    let event = '';
    const dataLines: string[] = [];
    for (const line of chunk.split(/\r?\n/)) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice('data:'.length).trimStart());
    }
    if (dataLines.length === 0) continue;
    frames.push({ event, data: dataLines.join('\n') });
  }
  return frames;
}

function jsonOrNull(text: string): Record<string, unknown> | null {
  try {
    const v: unknown = JSON.parse(text);
    return isRecord(v) ? v : null;
  } catch {
    return null;
  }
}

function usageOfOpenAI(usage: unknown): { input: number; output: number } {
  if (!isRecord(usage)) return { input: 0, output: 0 };
  return {
    input: typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : 0,
    output: typeof usage.completion_tokens === 'number' ? usage.completion_tokens : 0,
  };
}

function usageOfAnthropic(usage: unknown): { input: number; output: number } {
  if (!isRecord(usage)) return { input: 0, output: 0 };
  return {
    input: typeof usage.input_tokens === 'number' ? usage.input_tokens : 0,
    output: typeof usage.output_tokens === 'number' ? usage.output_tokens : 0,
  };
}

function usageOfGemini(meta: unknown): { input: number; output: number } {
  if (!isRecord(meta)) return { input: 0, output: 0 };
  return {
    input: typeof meta.promptTokenCount === 'number' ? meta.promptTokenCount : 0,
    output: typeof meta.candidatesTokenCount === 'number' ? meta.candidatesTokenCount : 0,
  };
}

/** Text of an OpenAI message's content: a string, or its text parts joined. */
function openAIContentText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) return content.map((p) => textOf(p)).join('');
  return '';
}

/** Parse an OpenAI JSON answer body into a normalized answer. */
export function parseOpenAIAnswer(body: string): UpstreamAnswer | null {
  const v = jsonOrNull(body);
  if (v === null) return null;
  const choices = Array.isArray(v.choices) ? v.choices : [];
  const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
  const message = first !== null && isRecord(first.message) ? first.message : null;
  if (message === null) return null;
  const usage = usageOfOpenAI(v.usage);
  return { text: openAIContentText(message.content), inputTokens: usage.input, outputTokens: usage.output };
}

/** Parse an OpenAI SSE answer body (with the usage chunk) into a normalized answer. */
export function parseOpenAIStream(body: string): UpstreamAnswer | null {
  let text = '';
  let input = 0;
  let output = 0;
  let sawAny = false;
  for (const frame of parseSSEFrames(body)) {
    if (frame.data === '[DONE]') continue;
    const v = jsonOrNull(frame.data);
    if (v === null) continue;
    sawAny = true;
    if (isRecord(v.usage)) {
      const u = usageOfOpenAI(v.usage);
      input = u.input;
      output = u.output;
    }
    const choices = Array.isArray(v.choices) ? v.choices : [];
    const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
    if (first !== null && isRecord(first.delta) && typeof first.delta.content === 'string') text += first.delta.content;
  }
  if (!sawAny) return null;
  return { text, inputTokens: input, outputTokens: output };
}

/** Text of an Anthropic message's content blocks. */
function anthropicContentText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  let out = '';
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') out += block.text;
  }
  return out;
}

/** Parse an Anthropic JSON answer body into a normalized answer. */
export function parseAnthropicAnswer(body: string): UpstreamAnswer | null {
  const v = jsonOrNull(body);
  if (v === null) return null;
  if (typeof v.type === 'string' && v.type === 'error') return null;
  const usage = usageOfAnthropic(v.usage);
  return { text: anthropicContentText(v.content), inputTokens: usage.input, outputTokens: usage.output };
}

/** Parse an Anthropic SSE answer body into a normalized answer. */
export function parseAnthropicStream(body: string): UpstreamAnswer | null {
  let text = '';
  let input = 0;
  let output = 0;
  let sawAny = false;
  for (const frame of parseSSEFrames(body)) {
    const v = jsonOrNull(frame.data);
    if (v === null) continue;
    sawAny = true;
    if (frame.event === 'message_start' && isRecord(v.message) && isRecord(v.message.usage)) {
      input = usageOfAnthropic(v.message.usage).input;
    } else if (frame.event === 'content_block_delta' && isRecord(v.delta) && v.delta.type === 'text_delta' && typeof v.delta.text === 'string') {
      text += v.delta.text;
    } else if (frame.event === 'message_delta' && isRecord(v.usage)) {
      output = usageOfAnthropic(v.usage).output;
    }
  }
  if (!sawAny) return null;
  return { text, inputTokens: input, outputTokens: output };
}

/** Text of a Gemini candidate's parts. */
function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  let out = '';
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') out += part.text;
  }
  return out;
}

/** Parse a Gemini JSON answer body into a normalized answer. */
export function parseGeminiAnswer(body: string): UpstreamAnswer | null {
  const v = jsonOrNull(body);
  if (v === null) return null;
  if (isRecord(v.error)) return null;
  const candidates = Array.isArray(v.candidates) ? v.candidates : [];
  const first = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : null;
  const content = first !== null && isRecord(first.content) ? first.content : null;
  if (content === null) return null;
  const usage = usageOfGemini(v.usageMetadata);
  return { text: geminiPartsText(content.parts), inputTokens: usage.input, outputTokens: usage.output };
}

/** Parse a Gemini SSE answer body into a normalized answer. */
export function parseGeminiStream(body: string): UpstreamAnswer | null {
  let text = '';
  let input = 0;
  let output = 0;
  let sawAny = false;
  for (const frame of parseSSEFrames(body)) {
    const v = jsonOrNull(frame.data);
    if (v === null) continue;
    sawAny = true;
    const candidates = Array.isArray(v.candidates) ? v.candidates : [];
    const first = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : null;
    if (first !== null && isRecord(first.content)) text += geminiPartsText(first.content.parts);
    if (isRecord(v.usageMetadata)) {
      const u = usageOfGemini(v.usageMetadata);
      input = u.input;
      output = u.output;
    }
  }
  if (!sawAny) return null;
  return { text, inputTokens: input, outputTokens: output };
}

// --- client -> provider translations ---------------------------------------

/** A stop value (string or string array) as a string array, or []. */
function stopSequencesOf(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (Array.isArray(v)) return v.filter((x): x is string => typeof x === 'string');
  return [];
}

/** The system text of an OpenAI body: its system messages joined, or ''. */
function openAISystemText(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const m of messages) {
    if (isRecord(m) && m.role === 'system') parts.push(openAIContentText(m.content));
  }
  return parts.join('\n');
}

/** The system text of an Anthropic body: its `system` field, or ''. */
function anthropicSystemText(body: Json): string {
  if (typeof body.system === 'string') return body.system;
  if (Array.isArray(body.system)) return body.system.map((p) => textOf(p)).join('');
  return '';
}

function openAIImageSource(part: Json): { kind: 'base64' | 'url'; media: string; data: string } | null {
  const url = isRecord(part.image_url) ? part.image_url : null;
  if (url === null || typeof url.url !== 'string') return null;
  const dataUrl = /^data:([^;]+);base64,(.*)$/s.exec(url.url);
  if (dataUrl !== null) return { kind: 'base64', media: dataUrl[1], data: dataUrl[2] };
  return { kind: 'url', media: '', data: url.url };
}

function anthropicImageSource(block: Json): { kind: 'base64' | 'url'; media: string; data: string } | null {
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { kind: 'base64', media: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { kind: 'url', media: '', data: source.url };
  return null;
}

/** OpenAI body -> Anthropic body. JSON mode cannot be echoed on this wire; the caller must not route JSON requests here. */
export function openAIRequestToAnthropic(body: Json, model: string): Json {
  const system = openAISystemText(body);
  const messages: unknown[] = [];
  const sourceMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of sourceMessages) {
    if (!isRecord(m)) continue;
    const role = m.role;
    if (role === 'system') continue;
    if (role === 'user') {
      const content = m.content;
      const blocks: unknown[] = [];
      if (typeof content === 'string') blocks.push({ type: 'text', text: content });
      else if (Array.isArray(content)) {
        for (const part of content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
          else if (part.type === 'image_url') {
            const src = openAIImageSource(part);
            if (src !== null) {
              if (src.kind === 'base64') blocks.push({ type: 'image', source: { type: 'base64', media_type: src.media, data: src.data } });
              else blocks.push({ type: 'image', source: { type: 'url', url: src.data } });
            }
          }
        }
      }
      messages.push({ role: 'user', content: blocks });
    } else if (role === 'assistant') {
      const blocks: unknown[] = [];
      const text = openAIContentText(m.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(m.tool_calls) ? m.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
        blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : 'call', name: call.function.name, input: argumentsOf(call.function) });
      }
      messages.push({ role: 'assistant', content: blocks });
    } else if (role === 'tool') {
      const text = openAIContentText(m.content);
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof m.tool_call_id === 'string' ? m.tool_call_id : 'call', content: text }],
      });
    }
  }
  const out: Json = { model, max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : 1024, messages };
  if (system !== '') out.system = system;
  const tools = Array.isArray(body.tools) ? body.tools : [];
  if (tools.length > 0) {
    out.tools = tools.map((t) => {
      if (!isRecord(t) || !isRecord(t.function)) return t;
      return { name: t.function.name, description: t.function.description ?? '', input_schema: t.function.parameters ?? { type: 'object', properties: {} } };
    });
  }
  if (body.tool_choice !== undefined) {
    const tc = body.tool_choice;
    if (tc === 'auto') out.tool_choice = { type: 'auto' };
    else if (tc === 'required' || tc === 'any') out.tool_choice = { type: 'any' };
    else if (isRecord(tc) && tc.type === 'function' && isRecord(tc.function)) out.tool_choice = { type: 'tool', name: tc.function.name };
  }
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stop = stopSequencesOf(body.stop);
  if (stop.length > 0) out.stop_sequences = stop;
  if (body.stream === true) out.stream = true;
  return out;
}

/** OpenAI body -> Gemini body. */
export function openAIToGemini(body: Json, model: string): Json {
  const system = openAISystemText(body);
  const contents: unknown[] = [];
  const sourceMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of sourceMessages) {
    if (!isRecord(m)) continue;
    const role = m.role;
    if (role === 'system') continue;
    if (role === 'user') {
      const parts: unknown[] = [];
      const content = m.content;
      if (typeof content === 'string') parts.push({ text: content });
      else if (Array.isArray(content)) {
        for (const part of content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
          else if (part.type === 'image_url') {
            const src = openAIImageSource(part);
            if (src !== null) {
              if (src.kind === 'base64') parts.push({ inlineData: { mimeType: src.media, data: src.data } });
              else parts.push({ fileData: { fileUri: src.data } });
            }
          }
        }
      }
      contents.push({ role: 'user', parts });
    } else if (role === 'assistant') {
      const parts: unknown[] = [];
      const text = openAIContentText(m.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(m.tool_calls) ? m.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
        parts.push({ functionCall: { name: call.function.name, args: argumentsOf(call.function) } });
      }
      contents.push({ role: 'model', parts });
    } else if (role === 'tool') {
      // Keep the tool result in a `model` content so the Gemini parser's
      // last-non-model text stays the last user message.
      contents.push({
        role: 'model',
        parts: [{ functionResponse: { name: typeof m.tool_call_id === 'string' ? m.tool_call_id : 'tool', response: { result: openAIContentText(m.content) } } }],
      });
    }
  }
  const generationConfig: Json = {};
  if (typeof body.max_tokens === 'number') generationConfig.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') generationConfig.topP = body.top_p;
  const stop = stopSequencesOf(body.stop);
  if (stop.length > 0) generationConfig.stopSequences = stop;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    generationConfig.responseMimeType = 'application/json';
  }
  const out: Json = { model, contents, generationConfig };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = Array.isArray(body.tools) ? body.tools : [];
  if (tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: tools.map((t) => {
          if (!isRecord(t) || !isRecord(t.function)) return t;
          return { name: t.function.name, description: t.function.description ?? '', parameters: t.function.parameters ?? { type: 'object', properties: {} } };
        }),
      },
    ];
  }
  return out;
}

/** Anthropic body -> Gemini body. */
export function anthropicToGemini(body: Json, model: string): Json {
  const system = anthropicSystemText(body);
  const contents: unknown[] = [];
  const sourceMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of sourceMessages) {
    if (!isRecord(m)) continue;
    const role = m.role;
    if (role === 'user') {
      const parts: unknown[] = [];
      const content = m.content;
      if (typeof content === 'string') parts.push({ text: content });
      else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'image') {
            const src = anthropicImageSource(block);
            if (src !== null) {
              if (src.kind === 'base64') parts.push({ inlineData: { mimeType: src.media, data: src.data } });
              else parts.push({ fileData: { fileUri: src.data } });
            }
          } else if (block.type === 'tool_result') {
            // Tool results ride in a `model` content (see openAIToGemini).
            contents.push({
              role: 'model',
              parts: [{ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : 'tool', response: { result: toolResultText(block) } } }],
            });
          }
        }
      }
      contents.push({ role: 'user', parts });
    } else if (role === 'assistant') {
      const parts: unknown[] = [];
      const content = m.content;
      if (typeof content === 'string') parts.push({ text: content });
      else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use' && typeof block.name === 'string') {
            parts.push({ functionCall: { name: block.name, args: isRecord(block.input) ? block.input : {} } });
          }
        }
      }
      contents.push({ role: 'model', parts });
    }
  }
  const generationConfig: Json = {};
  if (typeof body.max_tokens === 'number') generationConfig.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') generationConfig.topP = body.top_p;
  const stop = stopSequencesOf(body.stop_sequences);
  if (stop.length > 0) generationConfig.stopSequences = stop;
  const out: Json = { model, contents, generationConfig };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = Array.isArray(body.tools) ? body.tools : [];
  if (tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: tools.map((t) => {
          if (!isRecord(t) || typeof t.name !== 'string') return t;
          return { name: t.name, description: t.description ?? '', parameters: t.input_schema ?? { type: 'object', properties: {} } };
        }),
      },
    ];
  }
  return out;
}

// --- normalized answer -> client wire --------------------------------------

const FIXED_CREATED = 1767225600;

/** A normalized answer as an OpenAI JSON completion. */
export function openAIJsonReply(model: string, answer: UpstreamAnswer): Json {
  return {
    id: FIXED_ID_OPENAI,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** A normalized answer as OpenAI SSE frames (role chunk, 13-char text pieces, finish, usage, [DONE]). */
export function openAIStreamFrames(model: string, answer: UpstreamAnswer): string[] {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: FIXED_ID_OPENAI,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  const text = answer.text;
  for (let at = 0; at < text.length; at += 13) frames.push(sseEvent(null, chunk({ content: text.slice(at, at + 13) }, null)));
  frames.push(sseEvent(null, chunk({}, 'stop')));
  frames.push(
    sseEvent(null, {
      id: FIXED_ID_OPENAI,
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames;
}

/** A normalized answer as an Anthropic JSON message. */
export function anthropicJsonReply(model: string, answer: UpstreamAnswer): Json {
  return {
    id: FIXED_ID_ANTHROPIC,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A normalized answer as Anthropic SSE frames (message_start, block deltas, message_delta, message_stop). */
export function anthropicStreamFrames(model: string, answer: UpstreamAnswer): string[] {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: FIXED_ID_ANTHROPIC, type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.inputTokens, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  const text = answer.text;
  for (let at = 0; at < text.length; at += 13) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: text.slice(at, at + 13) } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: answer.outputTokens } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames;
}
