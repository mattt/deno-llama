/**
 * A chat session: message history + an inference context, with `respond()`
 * returning an async iterable of tokens. Supports sampler settings, a JSON schema
 * (converted to GBNF internally), and a token budget. The assistant's reply is
 * appended to the history once generation completes.
 */

import type { LlamaContext } from "./context.ts";
import type { LlamaModel } from "./model.ts";
import { applyChatTemplate, type ChatMessage } from "./template.ts";
import { tokenize } from "./tokenize.ts";
import {
  type FinishReason,
  generate,
  type GeneratedToken,
  type GenerateOptions,
} from "./generate.ts";
import { InferenceLock } from "./inference_lock.ts";
import { jsonSchemaToGrammar } from "./grammar/json_schema_to_gbnf.ts";

export type { ChatMessage };

export interface RespondOptions extends GenerateOptions {
  /** JSON schema constraining the reply; converted to GBNF and enforced. */
  jsonSchema?: unknown;
}

export class ChatSession {
  #model: LlamaModel;
  #ctx: LlamaContext;
  #history: ChatMessage[] = [];
  #lock = new InferenceLock();

  constructor(model: LlamaModel, ctx: LlamaContext, system?: string) {
    this.#model = model;
    this.#ctx = ctx;
    if (system) this.#history.push({ role: "system", content: system });
  }

  /** The conversation so far (including the system message, if any). */
  get messages(): readonly ChatMessage[] {
    return this.#history;
  }

  /** The underlying context (advanced use). */
  get context(): LlamaContext {
    return this.#ctx;
  }

  /** Append a message without generating (e.g. seeding assistant context). */
  add(message: ChatMessage): void {
    this.#history.push(message);
  }

  /** Clear history, optionally keeping the system message. */
  reset(keepSystem = true): void {
    const sys = this.#history[0];
    this.#history = keepSystem && sys?.role === "system" ? [sys] : [];
  }

  /**
   * Stream a reply to `userText`. The user message is added to history
   * immediately; the assistant reply is appended once the stream completes.
   */
  async *respond(
    userText: string,
    opts: RespondOptions = {},
  ): AsyncGenerator<GeneratedToken> {
    // Hold the lock across the whole turn so history mutation, prompt building,
    // and generation are serialized as a unit; overlapping calls would otherwise
    // interleave (the second prompt missing the first reply). generate() runs
    // without its own lock since we already hold it.
    const release = await this.#lock.acquire(opts.signal);
    this.#history.push({ role: "user", content: userText });
    let reply = "";
    try {
      const grammar = opts.jsonSchema !== undefined
        ? jsonSchemaToGrammar(opts.jsonSchema)
        : opts.grammar;

      const raw = this.#model.raw;
      const prompt = applyChatTemplate(raw, this.#model.ptr, this.#history, true);
      // The template already emits BOS/role markers, so add_special=false and let
      // parse_special=true recognize them.
      const ids = tokenize(raw, this.#model.vocab, prompt, false, true);

      for await (
        const t of generate(raw, this.#ctx, this.#model.vocab, ids, { ...opts, grammar })
      ) {
        reply += t.text; // content tokens and the terminal marker's flushed tail
        yield t;
      }
    } finally {
      // Append the (possibly partial) assistant reply even if the consumer stops
      // iterating early or generation throws — otherwise history desyncs, leaving
      // a user turn with no matching assistant turn.
      this.#history.push({ role: "assistant", content: reply });
      release();
    }
  }

  /**
   * Run a reply to completion and return the full text. With a `jsonSchema`, the
   * returned text is guaranteed to parse: the grammar only guarantees a valid
   * *prefix*, so anything that cuts the reply short (hitting `maxTokens`, being
   * cancelled, or a `stopTokenIds` entry firing mid-value) leaves text that will
   * not parse, and this throws rather than returning it. A reply that completes
   * the JSON exactly on its last permitted token is returned normally.
   */
  async respondText(userText: string, opts: RespondOptions = {}): Promise<string> {
    let text = "";
    let finishReason: FinishReason | undefined;
    for await (const t of this.respond(userText, opts)) {
      text += t.text;
      if (t.finishReason) finishReason = t.finishReason;
    }
    // Whether the reply is whole is decided by parsing it, not by why generation
    // ended: a run that completes the JSON on its very last permitted token ends
    // with "length" and is perfectly valid, while a custom stop token can cut a
    // reply short and still report "stop". finishReason only picks the hint.
    if (opts.jsonSchema !== undefined && !isJson(text)) {
      throw new Error(
        `Schema-constrained generation ended before the JSON was complete ` +
          `(finishReason: ${finishReason}); the output is a partial prefix that ` +
          `will not parse. ` +
          (finishReason === "stop"
            ? `A stopTokenIds entry fired mid-value — remove it, as the grammar ` +
              `already terminates the reply on its own.`
            : `Increase maxTokens.`),
      );
    }
    return text;
  }

  [Symbol.dispose](): void {
    this.#ctx[Symbol.dispose]();
  }
}

function isJson(text: string): boolean {
  try {
    JSON.parse(text);
    return true;
  } catch {
    return false;
  }
}
