/**
 * deno-llama CLI — chat, judge, and doctor.
 *
 *   deno task cli -- chat "Hello"
 *   deno task cli -- judge "What is 2+2?" "4" "5"
 *   deno task cli -- doctor
 *
 * Compiled form (see `deno task compile:cli`): `deno-llama <cmd> ...`.
 */

import {
  cachedLibPath,
  cacheRoot,
  ensureLibrary,
  llamaTag,
  openLlama,
} from "@deno-llama/core";
import { LlamaModel } from "@deno-llama/llama";
import {
  DEFAULT_MODEL,
  DEFAULT_QUANT,
  loadDefaultModel,
  modelSource,
} from "../shared/models.ts";

function usage(): never {
  console.log(`deno-llama ${llamaTag}

Usage:
  deno-llama chat [prompt] [--max-tokens N] [--temperature T] [--seed N] [--model REPO] [--quant Q]
  deno-llama judge <question> <reference> <candidate> [--model REPO] [--quant Q]
  deno-llama doctor

Environment:
  DENO_LLAMA_MODEL / DENO_LLAMA_QUANT   default model + quant
  DENO_LLAMA_LIB_PATH                    use a specific libllama.dylib
  DENO_LLAMA_CACHE                       override the dylib cache dir
  HF_TOKEN                               token for gated Hub repos
`);
  Deno.exit(2);
}

function requireAppleSilicon(): void {
  if (Deno.build.os !== "darwin" || Deno.build.arch !== "aarch64") {
    console.error("deno-llama requires Apple Silicon macOS (aarch64-apple-darwin).");
    Deno.exit(1);
  }
}

function parseArgs(args: string[]) {
  const positional: string[] = [];
  const flags: Record<string, string | boolean> = {};
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const next = args[i + 1];
      if (next !== undefined && !next.startsWith("--")) {
        flags[key] = next;
        i++;
      } else {
        flags[key] = true;
      }
    } else {
      positional.push(a);
    }
  }
  return { positional, flags };
}

function loadOpts(flags: Record<string, string | boolean>) {
  const opts: { quant?: string; source?: string } = {};
  if (typeof flags.model === "string") opts.source = flags.model;
  if (typeof flags.quant === "string") opts.quant = flags.quant;
  return opts;
}

async function cmdChat(args: string[]) {
  const { positional, flags } = parseArgs(args);
  const prompt = positional.join(" ") || "In one sentence, what is Deno?";
  const maxTokens = Number(flags["max-tokens"] ?? 256);
  const temperature = Number(flags.temperature ?? 0.7);
  const seed = flags.seed !== undefined ? Number(flags.seed) : undefined;
  const { source, quant } = loadOpts(flags);

  const ac = new AbortController();
  Deno.addSignalListener("SIGINT", () => ac.abort());

  using model = source
    ? await LlamaModel.load(source, { quant, gpuLayers: 999 })
    : await loadDefaultModel(quant ? { quant } : {});
  using chat = model.session({ system: "You are a concise, helpful assistant." });

  const enc = new TextEncoder();
  const t0 = performance.now();
  let n = 0;
  for await (
    const t of chat.respond(prompt, { maxTokens, temperature, seed, signal: ac.signal })
  ) {
    await Deno.stdout.write(enc.encode(t.text));
    if (t.id >= 0) n++;
  }
  const dt = (performance.now() - t0) / 1000;
  console.log(`\n\n[${n} tokens, ${(n / Math.max(dt, 1e-6)).toFixed(1)} tok/s]`);
}

async function cmdJudge(args: string[]) {
  const { positional, flags } = parseArgs(args);
  if (positional.length < 3) usage();
  const [question, reference, candidate] = positional;
  const { source, quant } = loadOpts(flags);

  const schema = {
    type: "object",
    properties: {
      verdict: { type: "string", enum: ["correct", "partially_correct", "incorrect"] },
      confidence: { type: "number" },
      reason: { type: "string" },
    },
    required: ["verdict", "confidence", "reason"],
    additionalProperties: false,
  };

  using model = source
    ? await LlamaModel.load(source, { quant, gpuLayers: 999 })
    : await loadDefaultModel(quant ? { quant } : {});
  using chat = model.session({ system: "You are a strict grading judge." });
  const out = await chat.respondText(
    `Question: ${question}\nReference answer: ${reference}\nCandidate answer: ${candidate}\n` +
      `Grade the candidate answer.`,
    { jsonSchema: schema, temperature: 0, maxTokens: 256 },
  );
  console.log(JSON.stringify(JSON.parse(out), null, 2));
}

async function cmdDoctor() {
  const target = Deno.build.os === "darwin" && Deno.build.arch === "aarch64";
  console.log("deno-llama doctor");
  console.log(`  os:            ${Deno.build.os}`);
  console.log(`  arch:          ${Deno.build.arch}`);
  console.log(`  deno:          ${Deno.version.deno}`);
  console.log(`  target ok:     ${target}`);
  console.log(`  llama.cpp pin: ${llamaTag}`);
  console.log(`  cache root:    ${cacheRoot()}`);
  console.log(`  cached dylib:  ${cachedLibPath()}`);
  try {
    const path = await ensureLibrary((p) => {
      if (p.total) {
        Deno.stderr.writeSync(
          new TextEncoder().encode(
            `\r  downloading dylib: ${Math.floor(p.received / p.total * 100)}%`,
          ),
        );
      }
    });
    console.log(`\n  libllama:      ${path}`);
    openLlama();
    console.log(`  ffi:           ok`);
  } catch (err) {
    console.log(`\n  libllama:      ERROR ${(err as Error).message}`);
  }
  console.log(`  default model: ${DEFAULT_MODEL} (${DEFAULT_QUANT})`);
  console.log(`  model source:  ${modelSource()}`);
}

async function main() {
  requireAppleSilicon();
  const args = Deno.args[0] === "--" ? Deno.args.slice(1) : Deno.args;
  const [cmd, ...rest] = args;
  if (!cmd || cmd === "-h" || cmd === "--help") usage();
  switch (cmd) {
    case "chat":
      await cmdChat(rest);
      break;
    case "judge":
      await cmdJudge(rest);
      break;
    case "doctor":
      await cmdDoctor();
      break;
    default:
      usage();
  }
}

await main();
