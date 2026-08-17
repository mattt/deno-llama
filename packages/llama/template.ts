/**
 * Chat formatting via llama.cpp's built-in `llama_chat_apply_template`, so common
 * model families (Llama, Qwen, Gemma, ChatML, …) need no template engine of ours.
 * The template is read from the GGUF metadata unless one is supplied explicitly.
 */

import type { Llama } from "@deno-llama/core";
import { cString, ptr, ptrValue, readCString } from "./util.ts";

type Raw = Llama["raw"];

export interface ChatMessage {
  // Common roles with autocomplete, but any string is accepted.
  role: "system" | "user" | "assistant" | "tool" | (string & Record<never, never>);
  content: string;
}

/** The model's built-in chat template string, or undefined if it has none. */
export function modelChatTemplate(
  raw: Raw,
  model: Deno.PointerValue,
): string | undefined {
  const p = raw.llama_model_chat_template(model, null);
  const s = readCString(p);
  return s.length ? s : undefined;
}

/**
 * Apply a chat template to `messages`, returning the formatted prompt string.
 * Uses the model's built-in template unless `template` is given. When
 * `addAssistant` is true, the assistant turn is opened for generation.
 */
export function applyChatTemplate(
  raw: Raw,
  model: Deno.PointerValue,
  messages: ChatMessage[],
  addAssistant = true,
  template?: string,
): string {
  const tmpl = template ?? modelChatTemplate(raw, model);
  if (!tmpl) {
    throw new Error(
      "Model has no chat template in its GGUF metadata. Pass an explicit template " +
        "or use the raw prompt API.",
    );
  }
  const tmplBuf = cString(tmpl);

  // Build a C array of `llama_chat_message { const char* role; const char* content; }`.
  const keepAlive: Uint8Array[] = [];
  const arr = new Uint8Array(messages.length * 16);
  const view = new DataView(arr.buffer);
  messages.forEach((m, i) => {
    const roleBuf = cString(m.role);
    const contentBuf = cString(m.content);
    keepAlive.push(roleBuf, contentBuf);
    view.setBigUint64(i * 16, ptrValue(ptr(roleBuf)), true);
    view.setBigUint64(i * 16 + 8, ptrValue(ptr(contentBuf)), true);
  });

  const tmplPtr = ptr(tmplBuf);
  const chatPtr = ptr(arr);
  let cap = 4096;
  let out = new Uint8Array(cap);
  let n = raw.llama_chat_apply_template(
    tmplPtr,
    chatPtr,
    BigInt(messages.length),
    addAssistant,
    ptr(out),
    cap,
  );
  if (n < 0) throw new Error(`llama_chat_apply_template failed: ${n}`);
  if (n > cap) {
    cap = n;
    out = new Uint8Array(cap);
    n = raw.llama_chat_apply_template(
      tmplPtr,
      chatPtr,
      BigInt(messages.length),
      addAssistant,
      ptr(out),
      cap,
    );
    if (n < 0) throw new Error(`llama_chat_apply_template failed: ${n}`);
  }
  // keepAlive / tmplBuf stay referenced until here, covering the FFI calls.
  return new TextDecoder().decode(out.subarray(0, n));
}
