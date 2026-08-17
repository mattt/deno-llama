/**
 * Tokenize / detokenize over a llama.cpp vocabulary.
 *
 * Detokenization is streaming-safe: `Detokenizer` feeds each token's raw bytes
 * through a streaming `TextDecoder`, so multi-byte UTF-8 sequences split across
 * token boundaries decode correctly and yield clean text deltas.
 */

import type { Llama } from "@deno-llama/core";
import { cString, ptr, readCString } from "./util.ts";

type Raw = Llama["raw"];

const encoder = new TextEncoder();

/** Tokenize `text` into token ids. */
export function tokenize(
  raw: Raw,
  vocab: Deno.PointerValue,
  text: string,
  addSpecial: boolean,
  parseSpecial = true,
): number[] {
  const textBuf = encoder.encode(text);
  const textPtr = ptr(textBuf);
  // First pass with a generous buffer; grow if llama reports it was too small.
  let cap = textBuf.length + 8;
  let tokens = new Int32Array(cap);
  let n = raw.llama_tokenize(
    vocab,
    textPtr,
    textBuf.length,
    ptr(tokens),
    cap,
    addSpecial,
    parseSpecial,
  );
  if (n < 0) {
    cap = -n;
    tokens = new Int32Array(cap);
    n = raw.llama_tokenize(
      vocab,
      textPtr,
      textBuf.length,
      ptr(tokens),
      cap,
      addSpecial,
      parseSpecial,
    );
    if (n < 0) throw new Error(`llama_tokenize failed: ${n}`);
  }
  return Array.from(tokens.subarray(0, n));
}

/** Raw UTF-8 bytes for a single token piece. */
export function tokenToPieceBytes(
  raw: Raw,
  vocab: Deno.PointerValue,
  token: number,
  special: boolean,
): Uint8Array {
  let cap = 64;
  let buf = new Uint8Array(cap);
  let n = raw.llama_token_to_piece(vocab, token, ptr(buf), cap, 0, special);
  if (n < 0) {
    cap = -n;
    buf = new Uint8Array(cap);
    n = raw.llama_token_to_piece(vocab, token, ptr(buf), cap, 0, special);
    if (n < 0) throw new Error(`llama_token_to_piece failed: ${n}`);
  }
  return buf.subarray(0, n);
}

/** Detokenize a full sequence of ids to a string (non-streaming). */
export function detokenize(
  raw: Raw,
  vocab: Deno.PointerValue,
  ids: number[],
  special = false,
): string {
  const dec = new TextDecoder();
  const parts: Uint8Array[] = [];
  let total = 0;
  for (const id of ids) {
    const b = tokenToPieceBytes(raw, vocab, id, special);
    parts.push(b);
    total += b.length;
  }
  const all = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    all.set(p, off);
    off += p.length;
  }
  return dec.decode(all);
}

/** Incremental detokenizer producing text deltas one token at a time. */
export class Detokenizer {
  #raw: Raw;
  #vocab: Deno.PointerValue;
  #dec = new TextDecoder(undefined, { fatal: false });
  #special: boolean;

  constructor(raw: Raw, vocab: Deno.PointerValue, special = false) {
    this.#raw = raw;
    this.#vocab = vocab;
    this.#special = special;
  }

  /** Append one token and return its decoded text delta (may be ""). */
  push(token: number): string {
    const bytes = tokenToPieceBytes(this.#raw, this.#vocab, token, this.#special);
    return this.#dec.decode(bytes, { stream: true });
  }

  /** Flush any buffered bytes (call once at end of stream). */
  flush(): string {
    return this.#dec.decode();
  }
}

/** Convenience: read a NUL-terminated string the vocab returned via pointer. */
export function readString(p: Deno.PointerValue): string {
  return readCString(p);
}

/** Encode a string as a NUL-terminated C string buffer. */
export function toCString(s: string): Uint8Array {
  return cString(s);
}
