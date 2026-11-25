import { assertEquals, assertRejects } from "@std/assert";
import type { LlamaContext } from "./context.ts";
import type { LlamaModel } from "./model.ts";
import { ChatSession } from "./session.ts";
import { cString, ptr, readCString } from "./util.ts";

// Render the message contents with one token per ASCII byte and always sample "x".
function testSession(): ChatSession {
  const template = cString("test");
  const raw = {
    llama_model_chat_template: () => ptr(template),
    llama_chat_apply_template: (_tmpl, chat, count, _add, out, cap) => {
      const messages = new Deno.UnsafePointerView(chat!);
      let prompt = "";
      for (let i = 0; i < Number(count); i++) {
        prompt += readCString(messages.getPointer(i * 16 + 8));
      }
      const bytes = new TextEncoder().encode(prompt);
      if (bytes.length <= cap) {
        new Uint8Array(Deno.UnsafePointerView.getArrayBuffer(out!, cap)).set(bytes);
      }
      return bytes.length;
    },
    llama_tokenize: (_vocab, _text, length) => length,
    llama_sampler_chain_default_params: () => new Uint8Array(1),
    llama_sampler_chain_init: () => ptr(template),
    llama_sampler_init_greedy: () => ptr(template),
    llama_sampler_chain_add: () => {},
    llama_sampler_free: () => {},
    llama_n_batch: () => 32,
    llama_decode_async: () => Promise.resolve(0),
    llama_sampler_sample: () => 1,
    llama_vocab_is_eog: () => false,
    llama_token_to_piece: (_vocab, _id, out) => {
      new Uint8Array(Deno.UnsafePointerView.getArrayBuffer(out!, 1))[0] = 0x78;
      return 1;
    },
  } satisfies Partial<LlamaModel["raw"]>;
  const model = { raw, ptr: null, vocab: null } as unknown as LlamaModel;
  const ctx = { contextLength: 32, clearMemory() {} } as LlamaContext;
  return new ChatSession(model, ctx, "system");
}

Deno.test("session: an oversized prompt leaves history usable for a short retry", async () => {
  const chat = testSession();
  await chat.respondText("one", { maxTokens: 1 });
  const prior = [...chat.messages];
  await assertRejects(() => chat.respondText("a".repeat(32)), Error, "exceeds context");
  const afterRejection = [...chat.messages];
  assertEquals(await chat.respondText("two", { maxTokens: 1 }), "x");
  assertEquals(afterRejection, prior);
  assertEquals(chat.messages, [
    ...prior,
    { role: "user", content: "two" },
    { role: "assistant", content: "x" },
  ]);
});

Deno.test("session: closing the stream keeps the reply yielded so far", async () => {
  const chat = testSession();
  const stream = chat.respond("one");
  assertEquals((await stream.next()).value?.text, "x");
  await stream.return(undefined);
  assertEquals(chat.messages, [
    { role: "system", content: "system" },
    { role: "user", content: "one" },
    { role: "assistant", content: "x" },
  ]);
});

Deno.test("session: a terminal marker records an empty completed reply", async () => {
  const chat = testSession();
  assertEquals(await chat.respondText("one", { maxTokens: 0 }), "");
  assertEquals(chat.messages, [
    { role: "system", content: "system" },
    { role: "user", content: "one" },
    { role: "assistant", content: "" },
  ]);
});
