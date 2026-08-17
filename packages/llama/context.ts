/**
 * Inference context + a JS-owned decode batch.
 *
 * The batch memory (token / pos / seq_id / logits arrays) is allocated as JS
 * typed arrays rather than via `llama_batch_init`, so positions can be written
 * explicitly — which is what makes deterministic KV bookkeeping and context-shift
 * possible.
 */

import { layoutOf, type Llama, Struct } from "@deno-llama/core";

type Raw = Llama["raw"];

// enum llama_flash_attn_type: AUTO=-1, DISABLED=0, ENABLED=1
const FLASH_ATTN_DISABLED = 0;
const FLASH_ATTN_ENABLED = 1;

export interface ContextOptions {
  /** Context window (tokens). 0 or omitted uses the model's trained default. */
  contextLength?: number;
  /** Logical batch size for prompt ingestion. */
  batchSize?: number;
  /** Threads for generation and batch processing. */
  threads?: number;
  /** Force Flash Attention on/off (default: llama.cpp AUTO). */
  flashAttention?: boolean;
  /** Enable embeddings output mode. */
  embeddings?: boolean;
}

/** A loaded inference context bound to a model. */
export class LlamaContext {
  readonly ptr: Deno.PointerValue;
  readonly memory: Deno.PointerValue;
  readonly contextLength: number;
  #raw: Raw;

  private constructor(
    raw: Raw,
    ptr: Deno.PointerValue,
    memory: Deno.PointerValue,
    contextLength: number,
  ) {
    this.#raw = raw;
    this.ptr = ptr;
    this.memory = memory;
    this.contextLength = contextLength;
  }

  static create(
    raw: Raw,
    model: Deno.PointerValue,
    opts: ContextOptions = {},
  ): LlamaContext {
    const s = Struct.from(
      "llama_context_params",
      raw.llama_context_default_params() as Uint8Array,
    );
    // llama.cpp's default is 512, not "whatever the model was trained with" —
    // pass 0 so an omitted contextLength means the trained default.
    s.set("n_ctx", opts.contextLength ?? 0);
    if (opts.batchSize !== undefined) {
      s.set("n_batch", opts.batchSize);
      s.set("n_ubatch", opts.batchSize);
    }
    if (opts.threads !== undefined) {
      s.set("n_threads", opts.threads);
      s.set("n_threads_batch", opts.threads);
    }
    if (opts.flashAttention !== undefined) {
      s.set(
        "flash_attn_type",
        opts.flashAttention ? FLASH_ATTN_ENABLED : FLASH_ATTN_DISABLED,
      );
    }
    if (opts.embeddings) s.set("embeddings", true);

    const ptr = raw.llama_init_from_model(model, s.bytes);
    if (!ptr) {
      throw new Error("llama_init_from_model returned null (context creation failed)");
    }
    const memory = raw.llama_get_memory(ptr);
    const nCtx = raw.llama_n_ctx(ptr);
    return new LlamaContext(raw, ptr, memory, nCtx);
  }

  /** Decode a batch; returns llama_decode's status (0 ok, >0 warn, <0 error). */
  decode(batch: Batch): number {
    return this.#raw.llama_decode(this.ptr, batch.bytes);
  }

  /** Clear the KV cache so the next decode may start again from position 0. */
  clearMemory(): void {
    this.#raw.llama_memory_clear(this.memory, true);
  }

  /**
   * Drop `count` tokens from the front of sequence 0 (after `keep` initial
   * tokens) and shift the rest down, freeing room in the KV cache.
   *
   * Returns false, having changed nothing, when the model's memory cannot do
   * this: recurrent and hybrid models reject removal of a partial range
   * (`llama_memory_seq_rm` -> false) and some cannot shift positions at all. The
   * caller must stop rather than carry on with its own position bookkeeping,
   * which would no longer agree with the native cache.
   */
  contextShift(keep: number, count: number): boolean {
    if (!this.#raw.llama_memory_can_shift(this.memory)) return false;
    if (!this.#raw.llama_memory_seq_rm(this.memory, 0, keep, keep + count)) {
      return false;
    }
    this.#raw.llama_memory_seq_add(this.memory, 0, keep + count, -1, -count);
    return true;
  }

  [Symbol.dispose](): void {
    this.#raw.llama_free(this.ptr);
  }
}

/**
 * A reusable, JS-owned decode batch for a single sequence (seq id 0).
 *
 * The struct bytes are laid out via the generated `llama_batch` layout (field
 * offsets by name), not hardcoded — so a rebase that reshapes the struct is
 * caught by the codegen rather than silently corrupting memory.
 */
export class Batch {
  readonly capacity: number;
  #token: Int32Array;
  #pos: Int32Array;
  #nSeqId: Int32Array;
  #seqIdInner: Int32Array;
  #seqIdPtrs: BigUint64Array;
  #logits: Int8Array;
  #struct: Struct;

  constructor(capacity: number) {
    this.capacity = capacity;
    this.#token = new Int32Array(capacity);
    this.#pos = new Int32Array(capacity);
    this.#nSeqId = new Int32Array(capacity).fill(1);
    this.#seqIdInner = new Int32Array(capacity); // one seq id (0) per token
    this.#seqIdPtrs = new BigUint64Array(capacity);
    this.#logits = new Int8Array(capacity);
    this.#struct = new Struct(layoutOf("llama_batch"));

    const innerBase = Deno.UnsafePointer.value(Deno.UnsafePointer.of(this.#seqIdInner));
    for (let i = 0; i < capacity; i++) this.#seqIdPtrs[i] = innerBase + BigInt(i * 4);

    // Pointer fields are fixed for the batch's lifetime (arrays don't move).
    this.#struct.set("token", Deno.UnsafePointer.of(this.#token));
    this.#struct.set("embd", 0n); // null
    this.#struct.set("pos", Deno.UnsafePointer.of(this.#pos));
    this.#struct.set("n_seq_id", Deno.UnsafePointer.of(this.#nSeqId));
    this.#struct.set("seq_id", Deno.UnsafePointer.of(this.#seqIdPtrs));
    this.#struct.set("logits", Deno.UnsafePointer.of(this.#logits));
  }

  /**
   * Load `tokens` at positions `startPos..startPos+n-1`. When `logitsOnLast` is
   * true, requests logits only on the final token (the one we sample from);
   * intermediate prompt chunks pass false.
   */
  set(tokens: number[], startPos: number, logitsOnLast = true): Uint8Array {
    const n = tokens.length;
    if (n > this.capacity) {
      throw new Error(`batch of ${n} exceeds capacity ${this.capacity}`);
    }
    for (let i = 0; i < n; i++) {
      this.#token[i] = tokens[i];
      this.#pos[i] = startPos + i;
      this.#logits[i] = 0;
    }
    if (n > 0 && logitsOnLast) this.#logits[n - 1] = 1;
    this.#struct.set("n_tokens", n);
    return this.#struct.bytes;
  }

  /** The batch struct bytes to pass to `llama_decode`. */
  get bytes(): Uint8Array {
    return this.#struct.bytes;
  }
}
