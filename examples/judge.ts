/**
 * Grammar-constrained JSON output: an LLM judge that emits a parseable verdict.
 *
 * The JSON schema is converted to GBNF internally and enforced by the sampler,
 * so the output always parses — the difference between a judge you can build on
 * and one that "usually" returns JSON.
 *
 *   deno task judge "What is the capital of France?" "Paris" "Berlin"
 *     args: <question> <reference answer> <candidate answer>
 */

import { loadDefaultModel } from "./shared/models.ts";

const [question, reference, candidate] = [
  Deno.args[0] ?? "What is 2 + 2?",
  Deno.args[1] ?? "4",
  Deno.args[2] ?? "5",
];

const schema = {
  type: "object",
  properties: {
    verdict: { type: "string", enum: ["correct", "partially_correct", "incorrect"] },
    confidence: { type: "number" },
    reason: { type: "string" },
  },
  required: ["verdict", "confidence", "reason"],
  additionalProperties: false,
} as const;

using model = await loadDefaultModel();
using chat = model.session({
  system: "You are a strict grading judge. Compare the candidate answer to the " +
    "reference and return a JSON verdict.",
});

const out = await chat.respondText(
  `Question: ${question}\nReference answer: ${reference}\nCandidate answer: ${candidate}\n` +
    `Grade the candidate answer.`,
  { jsonSchema: schema, temperature: 0, maxTokens: 256 },
);

const verdict = JSON.parse(out) as {
  verdict: string;
  confidence: number;
  reason: string;
};
console.log(JSON.stringify(verdict, null, 2));
