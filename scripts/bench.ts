#!/usr/bin/env -S deno run -P=llama
/**
 * Lightweight reference benchmarks (advisory).
 *
 * Loads a small chat model, runs a few prompts, and reports prompt tokens/s
 * (prefill) and generation tokens/s (decode).
 *
 *   deno task bench
 *   deno task bench -- --model Qwen/Qwen2.5-0.5B-Instruct-GGUF --quant q4_k_m
 */

import { applyChatTemplate, LlamaModel, tokenize } from "../packages/llama/mod.ts";

function flag(name: string): string | undefined {
  const i = Deno.args.indexOf(name);
  return i >= 0 ? Deno.args[i + 1] : undefined;
}

if (Deno.build.os !== "darwin" || Deno.build.arch !== "aarch64") {
  console.error("bench requires Apple Silicon macOS");
  Deno.exit(1);
}

const repo = flag("--model") ?? Deno.env.get("DENO_LLAMA_MODEL") ??
  "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
const quant = flag("--quant") ?? "q4_k_m";
const maxTokens = Number(flag("--max-tokens") ?? "64");

const prompts = [
  "Name three prime numbers.",
  "Write one sentence about the ocean.",
  "What is 17 plus 26?",
];

console.log(`model: ${repo}`);
console.log(`quant: ${quant}`);

using model = await LlamaModel.load(repo, { quant });

interface Row {
  prompt: string;
  promptTokens: number;
  genTokens: number;
  prefillTps: number;
  decodeTps: number;
}

const rows: Row[] = [];

for (const prompt of prompts) {
  using session = model.session();

  // Count prompt tokens by applying the chat template and tokenizing it.
  const templated = applyChatTemplate(
    model.raw,
    model.ptr,
    [{ role: "user", content: prompt }],
    true,
  );
  const promptTokens = tokenize(model.raw, model.vocab, templated, false, true)
    .length;

  const t0 = performance.now();
  let tFirst = 0;
  let genTokens = 0;
  for await (const t of session.respond(prompt, { maxTokens })) {
    if (t.id < 0) continue; // terminal marker
    if (genTokens === 0) tFirst = performance.now();
    genTokens++;
  }
  const tEnd = performance.now();

  const prefillS = (tFirst - t0) / 1000;
  const decodeS = (tEnd - tFirst) / 1000;
  rows.push({
    prompt,
    promptTokens,
    genTokens,
    prefillTps: prefillS > 0 ? promptTokens / prefillS : 0,
    decodeTps: decodeS > 0 ? genTokens / decodeS : 0,
  });
}

// Print a small table.
const header = ["prompt", "p_tok", "g_tok", "prefill_tps", "decode_tps"];
const cells = rows.map((r) => [
  r.prompt.length > 28 ? r.prompt.slice(0, 27) + "…" : r.prompt,
  String(r.promptTokens),
  String(r.genTokens),
  r.prefillTps.toFixed(1),
  r.decodeTps.toFixed(1),
]);
const widths = header.map((h, i) => Math.max(h.length, ...cells.map((c) => c[i].length)));
const fmt = (row: string[]) => row.map((v, i) => v.padEnd(widths[i])).join("  ");

console.log("");
console.log(fmt(header));
console.log(widths.map((w) => "-".repeat(w)).join("  "));
for (const c of cells) console.log(fmt(c));

const avgPrefill = rows.reduce((s, r) => s + r.prefillTps, 0) / rows.length;
const avgDecode = rows.reduce((s, r) => s + r.decodeTps, 0) / rows.length;
console.log("");
console.log(`avg prefill_tps: ${avgPrefill.toFixed(1)}`);
console.log(`avg decode_tps:  ${avgDecode.toFixed(1)}`);
