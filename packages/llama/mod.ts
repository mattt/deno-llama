/**
 * @deno-llama/llama — the ergonomic layer over llama.cpp.
 *
 * Load a GGUF model, open a chat session, and stream tokens. Grammars
 * (JSON-schema-constrained output), chat templates, and Hub downloads are built
 * in. Resource lifetimes are explicit (`using` / `Symbol.dispose`).
 *
 * ```ts
 * import { LlamaModel } from "@deno-llama/llama";
 *
 * using model = await LlamaModel.load("Qwen/Qwen2.5-0.5B-Instruct-GGUF", {
 *   quant: "q4_k_m",
 * });
 * using chat = model.session({ system: "You are concise." });
 * for await (const { text } of chat.respond("Name three primes.")) {
 *   await Deno.stdout.write(new TextEncoder().encode(text));
 * }
 * ```
 *
 * If the ergonomic layer's opinions chafe, the FFI layer (`@deno-llama/core`) is
 * right there.
 */

export { LlamaModel, type LoadOptions } from "./model.ts";
export { type ContextOptions, LlamaContext } from "./context.ts";
export { ChatSession, type RespondOptions } from "./session.ts";
export {
  type FinishReason,
  generate,
  type GeneratedToken,
  type GenerateOptions,
  type GenerateResult,
  generateResult,
} from "./generate.ts";
export {
  buildSampler,
  DEFAULT_SEED,
  type Sampler,
  type SamplerOptions,
} from "./sampler.ts";
export { applyChatTemplate, type ChatMessage, modelChatTemplate } from "./template.ts";
export { detokenize, Detokenizer, tokenize } from "./tokenize.ts";
export { type GgufProgress, resolveGguf, type ResolveGgufOptions } from "./hub.ts";
export { InferenceLock } from "./inference_lock.ts";
export { jsonSchemaToGrammar } from "./grammar/json_schema_to_gbnf.ts";
