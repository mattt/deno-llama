/**
 * Build a llama.cpp sampler chain from ergonomic options.
 *
 * Order follows llama.cpp convention: grammar first (mask invalid tokens), then
 * penalties, then the truncation/temperature stack, then the final selector
 * (`dist` for temperature sampling, `greedy` for temperature <= 0).
 */

import type { Llama } from "@deno-llama/core";
import { cString, ptr } from "./util.ts";

type Raw = Llama["raw"];

/** llama.cpp's "pick a random seed" sentinel. */
export const DEFAULT_SEED = 0xffffffff;

export interface SamplerOptions {
  /** 0 (default) selects greedy decoding; > 0 enables temperature sampling. */
  temperature?: number;
  topK?: number;
  topP?: number;
  minP?: number;
  /** Repetition penalty (1.0 = off). */
  repeatPenalty?: number;
  /** Window of recent tokens the penalties consider (default 64). */
  penaltyLastN?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  /** Seed for reproducible sampling (temperature > 0). */
  seed?: number;
  /** GBNF grammar constraining output (see ./grammar). */
  grammar?: string;
}

/** An owned sampler chain; dispose to free it. */
export interface Sampler {
  readonly ptr: Deno.PointerValue;
  [Symbol.dispose](): void;
}

/** Construct a sampler chain for `opts`. Caller disposes it. */
export function buildSampler(
  raw: Raw,
  vocab: Deno.PointerValue,
  opts: SamplerOptions = {},
): Sampler {
  const params = raw.llama_sampler_chain_default_params() as Uint8Array;
  const chain = raw.llama_sampler_chain_init(params);

  // A stage that fails to construct comes back as NULL. llama.cpp does not
  // validate what goes into a chain, so a NULL member is only dereferenced later,
  // inside llama_sampler_sample — i.e. a segfault rather than an exception. Reject
  // it here, freeing the partially-built chain (freeing a chain frees its members).
  const add = (s: Deno.PointerValue, stage: string) => {
    if (!s) {
      raw.llama_sampler_free(chain);
      throw new Error(`Failed to initialize the "${stage}" sampler stage.`);
    }
    raw.llama_sampler_chain_add(chain, s);
  };

  // Logit-shaping stages (grammar, penalties) apply to greedy and sampling
  // alike, so they go before the greedy early-return.
  if (opts.grammar) {
    const g = cString(opts.grammar);
    const root = cString("root");
    const grammar = raw.llama_sampler_init_grammar(vocab, ptr(g), ptr(root));
    if (!grammar) {
      raw.llama_sampler_free(chain);
      throw new Error(
        "Invalid GBNF grammar: llama.cpp could not parse it. It must be valid GBNF " +
          "and define a `root` rule. (Grammars from jsonSchemaToGrammar always are.)",
      );
    }
    raw.llama_sampler_chain_add(chain, grammar);
  }

  const repeat = opts.repeatPenalty ?? 1;
  const freq = opts.frequencyPenalty ?? 0;
  const present = opts.presencePenalty ?? 0;
  if (repeat !== 1 || freq !== 0 || present !== 0) {
    // b10344's signature leads with n_vocab (varies across tags — hence pinning).
    const nVocab = raw.llama_vocab_n_tokens(vocab);
    add(
      raw.llama_sampler_init_penalties(
        nVocab,
        opts.penaltyLastN ?? 64,
        repeat,
        freq,
        present,
      ),
      "penalties",
    );
  }

  const temperature = opts.temperature ?? 0;
  if (temperature <= 0) {
    add(raw.llama_sampler_init_greedy(), "greedy");
    return chainHandle(raw, chain);
  }

  if (opts.topK !== undefined && opts.topK > 0) {
    add(raw.llama_sampler_init_top_k(opts.topK), "top_k");
  }
  if (opts.topP !== undefined && opts.topP < 1) {
    add(raw.llama_sampler_init_top_p(opts.topP, 1n), "top_p");
  }
  if (opts.minP !== undefined && opts.minP > 0) {
    add(raw.llama_sampler_init_min_p(opts.minP, 1n), "min_p");
  }
  add(raw.llama_sampler_init_temp(temperature), "temp");
  add(raw.llama_sampler_init_dist(opts.seed ?? DEFAULT_SEED), "dist");

  return chainHandle(raw, chain);
}

function chainHandle(raw: Raw, chain: Deno.PointerValue): Sampler {
  return {
    ptr: chain,
    [Symbol.dispose]() {
      raw.llama_sampler_free(chain);
    },
  };
}
