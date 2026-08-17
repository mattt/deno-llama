/**
 * `LlamaModel` — load a GGUF model (from a Hub repo id or local path) and open
 * chat sessions on it. Weights are loaded once; each `session()` gets its own
 * inference context. Lifetimes are explicit (`using` / `Symbol.dispose`).
 */

import {
  type DownloadProgress,
  ensureLibrary,
  type Llama,
  openLlama,
  Struct,
} from "@deno-llama/core";
import { type GgufProgress, resolveGguf } from "./hub.ts";
import { cString, ptr, readCString } from "./util.ts";
import { ContextOptions, LlamaContext } from "./context.ts";
import { ChatSession } from "./session.ts";

type Raw = Llama["raw"];

export interface LoadOptions {
  /** Layers to offload to the GPU. Default -1 (all). 0 = CPU only. */
  gpuLayers?: number;
  /** Load only the vocabulary (no weights). */
  vocabOnly?: boolean;
  /** Quantization to prefer when the repo has several GGUFs (e.g. "Q4_K_M"). */
  quant?: string;
  /** Git revision (branch/tag/commit). Default "main". */
  revision?: string;
  /** HF token for gated repos. */
  accessToken?: string;
  /** Progress for the model download. */
  onProgress?: (p: GgufProgress) => void;
  /** Progress for the first-run llama.cpp dylib download. */
  onLibraryProgress?: (p: DownloadProgress) => void;
}

export class LlamaModel {
  readonly ptr: Deno.PointerValue;
  readonly vocab: Deno.PointerValue;
  readonly path: string;
  #llama: Llama;

  private constructor(
    llama: Llama,
    ptr: Deno.PointerValue,
    vocab: Deno.PointerValue,
    path: string,
  ) {
    this.#llama = llama;
    this.ptr = ptr;
    this.vocab = vocab;
    this.path = path;
  }

  /** Raw FFI symbol table (advanced use). */
  get raw(): Raw {
    return this.#llama.raw;
  }

  /**
   * Load a model from a Hub repo id (`owner/name`, optionally `:file.gguf`) or a
   * local `.gguf` path. Downloads the pinned llama.cpp dylib and the model on
   * first use.
   */
  static async load(source: string, opts: LoadOptions = {}): Promise<LlamaModel> {
    const path = await resolveGguf(source, {
      quant: opts.quant,
      revision: opts.revision,
      accessToken: opts.accessToken,
      onProgress: opts.onProgress,
    });
    await ensureLibrary(opts.onLibraryProgress);
    const llama = openLlama();
    const raw = llama.raw;

    const params = Struct.from(
      "llama_model_params",
      raw.llama_model_default_params() as Uint8Array,
    );
    if (opts.gpuLayers !== undefined) params.set("n_gpu_layers", opts.gpuLayers);
    if (opts.vocabOnly !== undefined) params.set("vocab_only", opts.vocabOnly);

    const pathBuf = cString(path);
    const model = raw.llama_model_load_from_file(ptr(pathBuf), params.bytes);
    if (!model) {
      throw new Error(
        `Failed to load model: ${path} (${llama.log.at(-1) ?? "unknown error"})`,
      );
    }
    const vocab = raw.llama_model_get_vocab(model);
    return new LlamaModel(llama, model, vocab, path);
  }

  /** Open a chat session with its own inference context. */
  session(opts: ContextOptions & { system?: string } = {}): ChatSession {
    const ctx = LlamaContext.create(this.raw, this.ptr, opts);
    return new ChatSession(this, ctx, opts.system);
  }

  /** Read a GGUF metadata string value (e.g. "general.name"), or undefined. */
  metadata(key: string): string | undefined {
    const keyBuf = cString(key);
    let cap = 256;
    let buf = new Uint8Array(cap);
    let n = this.raw.llama_model_meta_val_str(
      this.ptr,
      ptr(keyBuf),
      ptr(buf),
      BigInt(cap),
    );
    if (n < 0) return undefined;
    if (n >= cap) {
      cap = n + 1;
      buf = new Uint8Array(cap);
      n = this.raw.llama_model_meta_val_str(this.ptr, ptr(keyBuf), ptr(buf), BigInt(cap));
      if (n < 0) return undefined;
    }
    return new TextDecoder().decode(buf.subarray(0, n));
  }

  /** Number of tokens in the vocabulary. */
  get vocabSize(): number {
    return this.raw.llama_vocab_n_tokens(this.vocab);
  }

  /** Context length the model was trained with. */
  get trainedContextLength(): number {
    return this.raw.llama_model_n_ctx_train(this.ptr);
  }

  /** The model's built-in chat template, or undefined. */
  get chatTemplate(): string | undefined {
    const p = this.raw.llama_model_chat_template(this.ptr, null);
    const s = readCString(p);
    return s.length ? s : undefined;
  }

  [Symbol.dispose](): void {
    this.raw.llama_model_free(this.ptr);
  }
}
