/**
 * Minimal streaming chat.
 *
 *   deno task chat "Explain FFI in one sentence."
 */

import { loadDefaultModel } from "./shared/models.ts";

const prompt = Deno.args.join(" ") || "In one sentence, what is Deno?";

using model = await loadDefaultModel();
using chat = model.session({ system: "You are a concise, helpful assistant." });

const enc = new TextEncoder();
const t0 = performance.now();
let n = 0;
for await (
  const { id, text } of chat.respond(prompt, { maxTokens: 256, temperature: 0.7 })
) {
  await Deno.stdout.write(enc.encode(text));
  if (id >= 0) n++;
}
const dt = (performance.now() - t0) / 1000;
console.log(`\n\n[${n} tokens, ${(n / Math.max(dt, 1e-6)).toFixed(1)} tok/s]`);
