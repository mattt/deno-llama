/**
 * Streaming token generation: prompt token ids -> tokens, driving llama_decode
 * on Deno's nonblocking FFI path so the event loop stays responsive between
 * steps. Sampling and grammar advance through `llama_sampler_sample` (which also
 * accepts the token). The KV cache is shifted when the window fills.
 */

import type { Llama } from "@deno-llama/core";
import { Batch, type LlamaContext } from "./context.ts";
import { Detokenizer } from "./tokenize.ts";
import { buildSampler, type SamplerOptions } from "./sampler.ts";
import { InferenceLock } from "./inference_lock.ts";

type Raw = Llama["raw"];

export type FinishReason = "stop" | "length" | "cancelled";

export interface GenerateOptions extends SamplerOptions {
  /** Maximum tokens to generate (default 512). */
  maxTokens?: number;
  /** Abort generation (including while queued on the lock). */
  signal?: AbortSignal;
  /** Extra stop token ids in addition to the model's EOG tokens. */
  stopTokenIds?: number[];
  /** Initial tokens preserved across context-shift (e.g. a system prompt). */
  keepTokens?: number;
  /**
   * Serialize generation on a lock shared with other callers. Sessions already
   * serialize their own turns; use this to serialize across sessions too.
   */
  lock?: InferenceLock;
}

export interface GeneratedToken {
  /** Token id, or -1 for the terminal marker. */
  id: number;
  /** Newly-decoded text delta for this token (streaming-safe). */
  text: string;
  /** Zero-based index among generated tokens. */
  index: number;
  /** Set only on the terminal marker. */
  finishReason?: FinishReason;
}

export interface GenerateResult {
  text: string;
  tokens: number[];
  finishReason: FinishReason;
}

/**
 * Async generator over tokens. `promptIds` must already include any special/BOS
 * tokens (see `tokenize`, `applyChatTemplate`).
 */
export async function* generate(
  raw: Raw,
  ctx: LlamaContext,
  vocab: Deno.PointerValue,
  promptIds: number[],
  opts: GenerateOptions = {},
): AsyncGenerator<GeneratedToken> {
  const release = opts.lock ? await opts.lock.acquire(opts.signal) : undefined;
  try {
    yield* run(raw, ctx, vocab, promptIds, opts);
  } finally {
    release?.();
  }
}

async function* run(
  raw: Raw,
  ctx: LlamaContext,
  vocab: Deno.PointerValue,
  promptIds: number[],
  opts: GenerateOptions,
): AsyncGenerator<GeneratedToken> {
  const maxTokens = opts.maxTokens ?? 512;
  const nCtx = ctx.contextLength;
  const keep = Math.min(opts.keepTokens ?? 0, promptIds.length);
  const signal = opts.signal;

  if (promptIds.length === 0) throw new Error("empty prompt");
  if (promptIds.length >= nCtx) {
    throw new Error(
      `Prompt length ${promptIds.length} exceeds context length ${nCtx}`,
    );
  }

  const stop = new Set<number>(opts.stopTokenIds ?? []);
  using sampler = buildSampler(raw, vocab, opts);
  const detok = new Detokenizer(raw, vocab);

  // This generator decodes `promptIds` from position 0, so start from an empty
  // KV cache — otherwise llama_decode rejects a batch whose start position isn't
  // seq_pos_max + 1 (e.g. a second respond() on the same session).
  ctx.clearMemory();

  // Decode the prompt in n_batch-sized chunks; only the final token needs logits.
  const nBatch = Math.max(1, Number(raw.llama_n_batch(ctx.ptr)) || promptIds.length);
  const promptBatch = new Batch(Math.min(nBatch, promptIds.length));
  let nPast = 0;
  for (let i = 0; i < promptIds.length; i += nBatch) {
    const chunk = promptIds.slice(i, i + nBatch);
    const isLast = i + chunk.length >= promptIds.length;
    promptBatch.set(chunk, nPast, isLast);
    const rc = await raw.llama_decode_async(ctx.ptr, promptBatch.bytes);
    if (rc < 0) throw new Error(`llama_decode failed on prompt: ${rc}`);
    nPast += chunk.length;
  }

  const single = new Batch(1);
  let produced = 0;
  let finishReason: FinishReason = "length";

  for (let n = 0; n < maxTokens; n++) {
    if (signal?.aborted) {
      finishReason = "cancelled";
      break;
    }

    const id = raw.llama_sampler_sample(sampler.ptr, ctx.ptr, -1);
    if (raw.llama_vocab_is_eog(vocab, id) || stop.has(id)) {
      finishReason = "stop";
      break;
    }

    yield { id, text: detok.push(id), index: n };
    produced++;

    // Make room, then decode the accepted token to set up the next step.
    if (nPast + 1 >= nCtx) {
      const shiftable = nPast - keep;
      if (shiftable <= 0) {
        // Everything up to the window edge is pinned by keepTokens — no token can
        // be evicted, so stop cleanly rather than corrupting the KV positions.
        finishReason = "length";
        break;
      }
      const discard = Math.min(shiftable, Math.max(1, Math.floor(shiftable / 2)));
      if (!ctx.contextShift(keep, discard)) {
        // This model's memory can't drop a partial range or shift positions
        // (recurrent / hybrid architectures). Nothing was evicted, so stop here
        // instead of decoding at a position the KV cache no longer agrees with.
        finishReason = "length";
        break;
      }
      nPast -= discard;
    }
    single.set([id], nPast);
    const rc = await raw.llama_decode_async(ctx.ptr, single.bytes);
    if (rc < 0) throw new Error(`llama_decode failed: ${rc}`);
    nPast++;
  }

  yield { id: -1, text: detok.flush(), index: produced, finishReason };
}

/** Run generation to completion, returning text, token ids, and finish reason. */
export async function generateResult(
  raw: Raw,
  ctx: LlamaContext,
  vocab: Deno.PointerValue,
  promptIds: number[],
  opts: GenerateOptions = {},
): Promise<GenerateResult> {
  let text = "";
  const tokens: number[] = [];
  let finishReason: FinishReason = "length";
  for await (const t of generate(raw, ctx, vocab, promptIds, opts)) {
    if (t.finishReason) finishReason = t.finishReason;
    text += t.text;
    if (t.id >= 0) tokens.push(t.id);
  }
  return { text, tokens, finishReason };
}
