import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { InferenceLock } from "./inference_lock.ts";
import { resolveGguf } from "./hub.ts";
import { LlamaModel } from "./model.ts";
import { buildSampler } from "./sampler.ts";

// A repo id whose GGUF the integration tests load. Opt-in: set to run the
// model-backed tests (CI sets it; local runs skip to stay fast/offline).
const TEST_MODEL = Deno.env.get("DENO_LLAMA_TEST_MODEL");
const TEST_QUANT = Deno.env.get("DENO_LLAMA_TEST_QUANT") ?? "q4_k_m";

Deno.test("InferenceLock serializes in FIFO order", async () => {
  const lock = new InferenceLock();
  const order: number[] = [];
  await Promise.all([
    lock.run(async () => {
      await delay(15);
      order.push(1);
    }),
    lock.run(async () => {
      await delay(1);
      order.push(2);
    }),
    lock.run(() => {
      order.push(3);
      return Promise.resolve();
    }),
  ]);
  assertEquals(order, [1, 2, 3]);
});

Deno.test("InferenceLock respects an already-aborted signal", async () => {
  const lock = new InferenceLock();
  const ac = new AbortController();
  ac.abort();
  let threw = false;
  try {
    await lock.acquire(ac.signal);
  } catch {
    threw = true;
  }
  assert(threw, "expected acquire to reject on aborted signal");
});

Deno.test("InferenceLock: aborting a queued waiter does not deadlock the queue", async () => {
  const lock = new InferenceLock();
  const release1 = await lock.acquire(); // holder
  const ac = new AbortController();
  const queued = lock.acquire(ac.signal); // waits behind the holder
  const third = lock.acquire(); // waits behind the queued (soon-aborted) waiter

  ac.abort();
  release1();

  await assertRejects(() => queued); // the aborted waiter rejects
  // The third waiter must still get the lock — it would hang if the aborted
  // waiter had claimed and never released its slot.
  const release3 = await withTimeout(third, 1000, "third acquire deadlocked");
  release3();
});

Deno.test("InferenceLock: a queued waiter aborts without waiting for the holder", async () => {
  const lock = new InferenceLock();
  const release1 = await lock.acquire(); // holder, released only at the end
  const ac = new AbortController();
  const queued = lock.acquire(ac.signal);

  ac.abort();
  // Must reject now, not in however long the holder runs for.
  await withTimeout(
    assertRejects(() => queued),
    500,
    "aborting a queued waiter blocked on the holder",
  );

  // Mutual exclusion still holds: a later waiter must not overtake the holder.
  let gotLock = false;
  const third = lock.acquire().then((r) => {
    gotLock = true;
    return r;
  });
  await delay(20);
  assertEquals(gotLock, false, "third waiter overtook the running holder");
  release1();
  (await withTimeout(third, 1000, "third acquire deadlocked"))();
});

Deno.test("resolveGguf returns a local .gguf path unchanged", async () => {
  const tmp = await Deno.makeTempFile({ suffix: ".gguf" });
  try {
    assertEquals(await resolveGguf(tmp), tmp);
  } finally {
    await Deno.remove(tmp);
  }
});

Deno.test({
  name: "model: load, chat template, and streaming generation",
  ignore: !TEST_MODEL,
  fn: async () => {
    using model = await LlamaModel.load(TEST_MODEL!, {
      quant: TEST_QUANT,
      gpuLayers: 999,
    });
    assert(model.vocabSize > 0);
    assert(model.chatTemplate, "model should expose a chat template");

    using chat = model.session({ system: "You are concise." });
    let text = "";
    let finished = "";
    for await (
      const t of chat.respond("Reply with the single word: pong.", { maxTokens: 16 })
    ) {
      text += t.text;
      if (t.finishReason) finished = t.finishReason;
    }
    assert(text.length > 0, "expected non-empty generation");
    assert(finished === "stop" || finished === "length");
    assertEquals(chat.messages.at(-1)?.role, "assistant");
  },
});

Deno.test({
  name: "model: multi-turn respond() works (KV cache reset per turn)",
  ignore: !TEST_MODEL,
  fn: async () => {
    using model = await LlamaModel.load(TEST_MODEL!, {
      quant: TEST_QUANT,
      gpuLayers: 999,
    });
    using chat = model.session({ system: "You are concise." });
    // Before the KV-reset fix, the second turn threw "llama_decode failed: -1".
    const first = await chat.respondText("Say the word apple.", {
      maxTokens: 16,
      temperature: 0,
    });
    assert(first.length > 0, "first turn should produce output");
    const second = await chat.respondText("Now say the word banana.", {
      maxTokens: 16,
      temperature: 0,
    });
    assert(second.length > 0, "second turn should produce output");
    const assistantTurns = chat.messages.filter((m) => m.role === "assistant").length;
    assertEquals(assistantTurns, 2);
  },
});

Deno.test({
  name: "session: schema respondText throws on truncation instead of bad JSON",
  ignore: !TEST_MODEL,
  fn: async () => {
    using model = await LlamaModel.load(TEST_MODEL!, {
      quant: TEST_QUANT,
      gpuLayers: 999,
    });
    using chat = model.session();
    const schema = {
      type: "object",
      properties: { answer: { type: "string" } },
      required: ["answer"],
    };
    // maxTokens: 1 forces truncation mid-JSON; the contract is "throw, don't
    // return an unparseable prefix".
    await assertRejects(
      () => chat.respondText("Say hi.", { jsonSchema: schema, maxTokens: 1 }),
      Error,
      "Increase maxTokens",
    );
  },
});

Deno.test({
  name: "session: history stays consistent when the consumer stops early",
  ignore: !TEST_MODEL,
  fn: async () => {
    using model = await LlamaModel.load(TEST_MODEL!, {
      quant: TEST_QUANT,
      gpuLayers: 999,
    });
    using chat = model.session();
    const it = chat.respond("Count slowly from one to twenty.", { maxTokens: 200 });
    await it.next(); // consume a single token, then abandon the stream
    await it.return?.(undefined);
    const roles = chat.messages.map((m) => m.role);
    // The turn must remain paired: a user message followed by an assistant one.
    assertEquals(roles.at(-2), "user");
    assertEquals(roles.at(-1), "assistant");
  },
});

Deno.test({
  name: "model: grammar-constrained JSON output parses and conforms",
  ignore: !TEST_MODEL,
  fn: async () => {
    using model = await LlamaModel.load(TEST_MODEL!, {
      quant: TEST_QUANT,
      gpuLayers: 999,
    });
    using chat = model.session({ system: "You are a strict grading judge." });
    const schema = {
      type: "object",
      properties: {
        verdict: { type: "string", enum: ["correct", "incorrect"] },
        reason: { type: "string" },
      },
      required: ["verdict", "reason"],
      additionalProperties: false,
    };
    const out = await chat.respondText(
      'Question: What is 2+2? Answer given: "4". Grade it.',
      { jsonSchema: schema, maxTokens: 128, temperature: 0 },
    );
    const parsed = JSON.parse(out) as { verdict: string; reason: string };
    assert(["correct", "incorrect"].includes(parsed.verdict), `bad verdict: ${out}`);
    assertEquals(typeof parsed.reason, "string");
  },
});

Deno.test({
  name: "sampler: penalties apply under greedy decoding",
  ignore: !TEST_MODEL,
  fn: async () => {
    using model = await LlamaModel.load(TEST_MODEL!, {
      quant: TEST_QUANT,
      gpuLayers: 999,
    });
    // temperature 0 => greedy; a repeat penalty must still be added to the chain.
    using withPenalty = buildSampler(model.raw, model.vocab, {
      temperature: 0,
      repeatPenalty: 1.1,
    });
    using greedyOnly = buildSampler(model.raw, model.vocab, { temperature: 0 });
    const n = (s: Deno.PointerValue) => Number(model.raw.llama_sampler_chain_n(s));
    assertEquals(n(greedyOnly.ptr), 1); // greedy selector only
    assertEquals(n(withPenalty.ptr), 2); // penalties + greedy
  },
});

Deno.test({
  name: "sampler: a malformed grammar throws instead of crashing the process",
  ignore: !TEST_MODEL,
  fn: async () => {
    using model = await LlamaModel.load(TEST_MODEL!, {
      quant: TEST_QUANT,
      gpuLayers: 999,
    });
    // llama_sampler_init_grammar returns NULL here; adding that to the chain
    // used to segfault later, inside llama_sampler_sample.
    assertThrows(
      () => buildSampler(model.raw, model.vocab, { grammar: "root ::= (((" }),
      Error,
      "Invalid GBNF grammar",
    );
    // A grammar with no `root` rule is equally unusable.
    assertThrows(
      () => buildSampler(model.raw, model.vocab, { grammar: `nope ::= "x"` }),
      Error,
      "Invalid GBNF grammar",
    );
    // The sampler chain still works afterwards (the failed chain was freed).
    using ok = buildSampler(model.raw, model.vocab, { temperature: 0 });
    assertEquals(Number(model.raw.llama_sampler_chain_n(ok.ptr)), 1);
  },
});

Deno.test({
  name: "session: schema respondText accepts JSON completed on the last allowed token",
  ignore: !TEST_MODEL,
  fn: async () => {
    using model = await LlamaModel.load(TEST_MODEL!, {
      quant: TEST_QUANT,
      gpuLayers: 999,
    });
    using chat = model.session();
    const schema = {
      type: "object",
      properties: { ok: { type: "boolean" } },
      required: ["ok"],
      additionalProperties: false,
    };
    const opts = { jsonSchema: schema, temperature: 0 } as const;

    // Count the tokens a complete reply takes, then allow exactly that many:
    // generation now ends with finishReason "length" on text that is whole JSON.
    let exact = 0;
    for await (const t of chat.respond("Answer.", { ...opts, maxTokens: 64 })) {
      if (t.id >= 0) exact++;
    }
    assert(exact > 0, "expected a non-empty reply");
    chat.reset();

    const out = await chat.respondText("Answer.", { ...opts, maxTokens: exact });
    assertEquals(typeof (JSON.parse(out) as { ok: boolean }).ok, "boolean");
  },
});

Deno.test({
  name: "session: schema respondText rejects a reply cut short by a custom stop token",
  ignore: !TEST_MODEL,
  fn: async () => {
    using model = await LlamaModel.load(TEST_MODEL!, {
      quant: TEST_QUANT,
      gpuLayers: 999,
    });
    using chat = model.session();
    const schema = {
      type: "object",
      properties: { answer: { type: "string" } },
      required: ["answer"],
    };
    // Find the token the constrained reply actually opens with (greedy, so it is
    // reproducible), then make exactly that token a stop token. Generation ends
    // with finishReason "stop" and an empty body, which must not be reported as
    // complete JSON.
    let firstId = -1;
    for await (
      const t of chat.respond("Say hi.", {
        jsonSchema: schema,
        maxTokens: 4,
        temperature: 0,
      })
    ) {
      if (t.id >= 0) {
        firstId = t.id;
        break;
      }
    }
    assert(firstId >= 0, "expected at least one generated token");
    chat.reset(); // same prompt again, so greedy decoding repeats that token

    await assertRejects(
      () =>
        chat.respondText("Say hi.", {
          jsonSchema: schema,
          maxTokens: 64,
          temperature: 0,
          stopTokenIds: [firstId],
        }),
      Error,
      "before the JSON was complete",
    );
  },
});

function delay(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function withTimeout<T>(p: Promise<T>, ms: number, message: string): Promise<T> {
  return Promise.race([
    p,
    new Promise<T>((_, reject) => setTimeout(() => reject(new Error(message)), ms)),
  ]);
}
