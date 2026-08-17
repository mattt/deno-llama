/**
 * Shared model selection for the examples.
 *
 * The reference/tuning target is a large agentic model (see the README's hardware
 * tiers), but the examples default to a small model so they run on any Mac and in
 * CI. Override with DENO_LLAMA_MODEL (a Hub repo id or local .gguf) and
 * DENO_LLAMA_QUANT.
 */

import { type GgufProgress, LlamaModel, type LoadOptions } from "@deno-llama/llama";

/** Small, first-party GGUF that runs on any Apple Silicon Mac. */
export const DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
export const DEFAULT_QUANT = "q4_k_m";

export function modelSource(): string {
  return Deno.env.get("DENO_LLAMA_MODEL") ?? DEFAULT_MODEL;
}

export function modelQuant(): string {
  return Deno.env.get("DENO_LLAMA_QUANT") ?? DEFAULT_QUANT;
}

/** Load the configured model, printing download progress to stderr. */
export function loadDefaultModel(opts: LoadOptions = {}): Promise<LlamaModel> {
  return LlamaModel.load(modelSource(), {
    quant: modelQuant(),
    gpuLayers: 999,
    onProgress: printProgress,
    onLibraryProgress: (p) => {
      if (p.total) bar("llama.cpp", p.received, p.total);
    },
    ...opts,
  });
}

let lastPct = -1;
function printProgress(p: GgufProgress): void {
  if (!p.total) return;
  bar(p.file.split("/").pop() ?? p.file, p.received, p.total);
}

function bar(label: string, received: number, total: number): void {
  const pct = Math.floor((received / total) * 100);
  if (pct === lastPct) return;
  lastPct = pct;
  const enc = new TextEncoder();
  Deno.stderr.writeSync(
    enc.encode(
      `\r  ${label}: ${pct}% (${(received / 1e6).toFixed(0)}/${
        (total / 1e6).toFixed(0)
      } MB)`,
    ),
  );
  if (pct >= 100) Deno.stderr.writeSync(enc.encode("\n"));
}
