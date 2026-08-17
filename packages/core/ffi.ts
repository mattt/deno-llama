/**
 * @deno-llama/core loader — open libllama with the generated symbol table,
 * install a log handler so failures carry the real llama.cpp message, and expose
 * `raw` (direct FFI) and `checked` (decode/encode status codes -> thrown errors).
 */

import { resolveLibPath } from "./resolver.ts";
import { symbols } from "./generated/symbols.ts";
import { statusReturning } from "./generated/meta.ts";

/**
 * The generated table plus a hand-added nonblocking view of `llama_decode`
 * (same C symbol, run on a threadpool so a long prompt/token decode does not
 * block the event loop). The generated `llama_decode` stays for callers that
 * want the synchronous path.
 */
const extendedSymbols = {
  ...symbols,
  llama_decode_async: {
    name: "llama_decode",
    parameters: symbols.llama_decode.parameters,
    result: symbols.llama_decode.result,
    nonblocking: true,
  },
} as const satisfies Deno.ForeignLibraryInterface;

/** Error raised when a checked llama.cpp call returns a failing status code. */
export class LlamaError extends Error {
  constructor(readonly fn: string, readonly code: number, detail?: string) {
    super(
      detail ? `${fn} failed (${code}): ${detail}` : `${fn} failed with status ${code}`,
    );
    this.name = "LlamaError";
  }
}

// llama.cpp reports detail through a global log callback; keep a short ring
// buffer of recent lines so LlamaError can quote the most relevant message.
const LOG_RING = 16;
const recentLog: string[] = [];
const verbose = !!Deno.env.get("DENO_LLAMA_VERBOSE");

function pushLog(text: string): void {
  recentLog.push(text);
  if (recentLog.length > LOG_RING) recentLog.shift();
}

/** The most recent non-empty log line (best-effort error detail). */
function lastLogLine(): string | undefined {
  for (let i = recentLog.length - 1; i >= 0; i--) {
    const t = recentLog[i].trim();
    if (t) return t;
  }
  return undefined;
}

// void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data)
// Created per-open and disposed on close(), so re-opening after close() installs
// a fresh callback rather than a dangling (freed) pointer.
type LogCallback = Deno.UnsafeCallback<{
  parameters: ["i32", "pointer", "pointer"];
  result: "void";
}>;

let logCallback: LogCallback | undefined;

function makeLogCallback(): LogCallback {
  return new Deno.UnsafeCallback(
    { parameters: ["i32", "pointer", "pointer"], result: "void" },
    (_level: number, textPtr: Deno.PointerValue) => {
      if (!textPtr) return;
      const text = new Deno.UnsafePointerView(textPtr).getCString();
      pushLog(text);
      if (verbose) Deno.stderr.writeSync(new TextEncoder().encode(text));
    },
  );
}

type Symbols = Deno.StaticForeignLibraryInterface<typeof extendedSymbols>;

export interface Llama {
  /** The libllama.dylib path that was loaded. */
  readonly path: string;
  /** Raw FFI symbols — no status checking (use for hot paths you check yourself). */
  readonly raw: Symbols;
  /** decode/encode status codes (< 0) throw {@link LlamaError}; others pass through. */
  readonly checked: Symbols;
  /** The last few log lines llama.cpp emitted (most recent last). */
  readonly log: readonly string[];
  /** Free the backend and unload the library. */
  close(): void;
}

let singleton: Llama | undefined;

/**
 * Open libllama (idempotent — repeated calls return the same handle).
 * The dylib must already be present; call `ensureLibrary()` first on first run.
 */
export function openLlama(): Llama {
  if (singleton) return singleton;

  const path = resolveLibPath();
  const lib = Deno.dlopen(path, extendedSymbols);
  const raw = lib.symbols;

  // Route llama.cpp logs through a fresh callback, then bring the backend up.
  logCallback = makeLogCallback();
  raw.llama_log_set(logCallback.pointer, null);
  raw.llama_backend_init();

  const checked = new Proxy(raw, {
    get(target, prop: string) {
      const value = Reflect.get(target, prop);
      if (typeof value !== "function" || !statusReturning.has(prop)) return value;
      return (...args: unknown[]) => {
        // deno-lint-ignore no-explicit-any
        const rc = (value as any)(...args) as number;
        if (rc < 0) throw new LlamaError(prop, rc, lastLogLine());
        return rc;
      };
    },
  }) as Symbols;

  singleton = {
    path,
    raw,
    checked,
    get log() {
      return recentLog;
    },
    close() {
      try {
        raw.llama_backend_free();
      } finally {
        lib.close();
        logCallback?.close();
        logCallback = undefined;
        singleton = undefined;
      }
    },
  };
  return singleton;
}

/** Whether libllama has been opened in this process. */
export function isOpen(): boolean {
  return singleton !== undefined;
}
