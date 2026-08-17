/**
 * deno-llama desktop preview — a loopback chat app for the experimental
 * `deno desktop` WebView backend.
 *
 * On first launch it downloads the pinned llama.cpp dylib and the default model,
 * then serves a tiny streaming chat UI on 127.0.0.1. This is the vertical slice
 * that proves the whole pipeline survives `deno desktop` packaging.
 *
 *   deno task desktop         # serve; open the URL in any browser
 *   deno task desktop:app     # run inside a deno desktop WebView
 */

import { loadDefaultModel, modelSource } from "../shared/models.ts";
import { UI } from "./ui.ts";

if (Deno.build.os !== "darwin" || Deno.build.arch !== "aarch64") {
  console.error("deno-llama desktop requires Apple Silicon macOS.");
  Deno.exit(1);
}

const HOST = "127.0.0.1";
const PORT = Number(Deno.env.get("PORT") ?? 8788);

console.log(`[deno-llama desktop] loading ${modelSource()} …`);
const model = await loadDefaultModel();
const session = model.session({ system: "You are a concise, helpful assistant." });
console.log("[deno-llama desktop] model ready");

const encoder = new TextEncoder();

Deno.serve({ hostname: HOST, port: PORT }, async (req) => {
  const url = new URL(req.url);

  if (req.method === "GET" && url.pathname === "/") {
    return new Response(UI, { headers: { "content-type": "text/html; charset=utf-8" } });
  }

  if (req.method === "POST" && url.pathname === "/chat") {
    const { message } = await req.json() as { message: string };
    const stream = new ReadableStream<Uint8Array>({
      async start(controller) {
        try {
          for await (
            const t of session.respond(message, { maxTokens: 512, temperature: 0.7 })
          ) {
            if (t.text) controller.enqueue(encoder.encode(t.text));
          }
        } finally {
          controller.close();
        }
      },
    });
    return new Response(stream, {
      headers: { "content-type": "text/plain; charset=utf-8" },
    });
  }

  return new Response("Not found", { status: 404 });
});

console.log(`[deno-llama desktop] http://${HOST}:${PORT}`);
console.log(
  "[deno-llama desktop] WebView: deno desktop --backend webview examples/desktop/main.ts",
);
