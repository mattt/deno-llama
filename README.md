# deno-llama

**llama.cpp for Deno** runs GGUF models in TypeScript on Apple Silicon,
without requiring Python, a Node compatibility layer, or a native build step.

deno-llama binds [llama.cpp](https://github.com/ggml-org/llama.cpp)
directly through Deno FFI (`Deno.dlopen`).
It downloads a prebuilt `libllama.dylib` on first use
and opens it at runtime.
This design supports `deno compile` and `deno desktop` packaging.
The TypeScript token loop calls `llama_decode`, samples the result,
and detokenizes it through an async iterator.
It does not use `UnsafeCallback`,
and the per-token FFI cost is small compared with a decode step.

## Status

This project supports macOS arm64
and is pinned to llama.cpp `b10344`.
The two packages pass `deno publish --dry-run`
but are not yet published.

## Requirements

- Apple Silicon Mac (arm64)
- macOS 14+
- [Deno](https://deno.land) 2.x

On first use, deno-llama downloads the official
`llama-b10344-bin-macos-arm64` release (about 11 MB),
verifies its checksum,
and caches it under `~/Library/Caches/deno-llama`.
It does not require Homebrew or a compiler.

## Quick start

```bash
deno task cli -- doctor                         # verify the toolchain and dylib
deno task chat "Explain FFI in one sentence."   # streaming chat
deno task judge "What is 2+2?" "4" "5"          # grammar-constrained JSON verdict
deno task desktop                               # loopback WebView chat app
```

```ts
import { LlamaModel } from "@deno-llama/llama";

using model = await LlamaModel.load("Qwen/Qwen2.5-0.5B-Instruct-GGUF", {
  quant: "q4_k_m",
});
using chat = model.session({ system: "You are concise." });

for await (const { text } of chat.respond("Name three primes.")) {
  await Deno.stdout.write(new TextEncoder().encode(text));
}
```

## Packages

| Package             | Role                                                          |
| ------------------- | ------------------------------------------------------------- |
| `@deno-llama/core`  | Generated llama.cpp FFI, dylib resolver/fetcher, `LlamaError` |
| `@deno-llama/llama` | `LlamaModel`, chat sessions, sampling, GBNF grammars, Hub     |

`@deno-llama/core` exports the raw symbol table,
`openLlama().raw`, pinned struct layouts, and enums.
Applications can use this FFI layer directly
instead of the higher-level API.

## Model API

```ts
import { LlamaModel } from "@deno-llama/llama";

// Load a Hub repo ID (owner/name[:file.gguf]) or a local .gguf path.
using model = await LlamaModel.load("owner/Model-GGUF", {
  quant: "Q4_K_M",
  gpuLayers: 999, // offload all layers to Metal (the default)
});

using chat = model.session({ system: "You are a strict judge." });

// Stream tokens.
for await (const { text } of chat.respond("Hello")) {
  // ...
}

// Convert a JSON schema to GBNF and enforce it during generation.
const verdict = await chat.respondText("Grade this answer.", {
  jsonSchema: {
    type: "object",
    properties: { ok: { type: "boolean" }, why: { type: "string" } },
    required: ["ok", "why"],
  },
  temperature: 0,
});
JSON.parse(verdict);
```

`respond` accepts sampler options such as `temperature`, `topK`, `topP`,
`minP`, `repeatPenalty`, and `seed`.
It also accepts a `maxTokens` budget and an `AbortSignal`.
Contexts and models use `using` / `Symbol.dispose`
for deterministic cleanup.
Concurrent calls on one session are serialized.

## Model loading

Pass a local `.gguf` path to use the file directly.
Pass a Hugging Face repository ID
to select and download a model through
[`@huggingface/hub`](https://www.npmjs.com/package/@huggingface/hub).
Downloads use the shared Hugging Face cache,
so deno-llama, `hf`, and `huggingface-cli` can use the same files.
Downloads resume after an interruption
and use Xet acceleration when the Hub file is Xet-backed.

Use `quant` to select a quantization
when a repository contains multiple GGUF files.
To select an exact file,
use `owner/name:file.gguf`.
For a sharded model,
deno-llama downloads every shard and returns the path to the first one.

The `revision` option accepts a branch, tag, or commit.
For gated repositories,
set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`,
or pass `accessToken`.

## FFI layer

```ts
import { ensureLibrary, openLlama, Struct } from "@deno-llama/core";

await ensureLibrary(); // download, verify, and cache the pinned dylib
const llama = openLlama(); // dlopen; llama_backend_init()
const params = Struct.from(
  "llama_context_params",
  llama.raw.llama_context_default_params() as Uint8Array,
);
params.set("n_ctx", 4096); // set a by-value struct field by name
```

A clang AST generator in `packages/core/codegen`
creates the symbol table from the pinned `llama.h`.
It flattens by-value structs such as `llama_model_params`,
`llama_context_params`, and `llama_batch`
into Deno FFI struct layouts with exact C ABI byte offsets.
Parameter getters then return the values that the caller set.

## Grammars

`jsonSchemaToGrammar`,
also available from `@deno-llama/llama/grammar`,
converts a JSON schema to GBNF.
It is ported from llama.cpp's converter
and includes the upstream test cases.
Grammar-constrained output lets applications require parseable verdicts.

## llama.cpp compatibility

Each deno-llama minor version binds one llama.cpp build tag.
The package version uses the form `2.<llama.cpp-build>.0`.
For example, `2.10344.0` binds llama.cpp `b10344`.
This follows the convention used by
[llama.swift](https://github.com/mattt/llama.swift).

Rebases to newer tags usually occur every four to eight weeks.
Because `llama.h` changes compatibility often,
the pinned version gives applications a known upgrade schedule.
For earlier support of a new model architecture,
build a custom dylib and set its path:

```bash
DENO_LLAMA_LIB_PATH=/path/to/your/libllama.dylib deno run -P=llama ...
```

To rebase, update `LLAMA_TAG`,
run `deno task codegen` with clang and network access,
update the package versions,
and review the generated symbol table, struct layouts, and checksum.

## Distribution (compiled apps)

Ship these files with a compiled release:

1. the compiled executable (or `.app`), and
2. vendored `libllama.dylib` and `libggml*` dependencies
   with relocatable `@loader_path` install names.

The resolver supports these layouts:

- `{execDir}/libllama.dylib` or `{execDir}/vendor/libllama.dylib`
- `{execDir}/../lib/libllama.dylib` (CLI archive)
- `{execDir}/../Frameworks/libllama.dylib` (macOS app bundle)

```bash
deno task compile:cli                # aarch64-apple-darwin CLI binary
deno task bundle:macos               # stage relocatable dylibs into dist/lib
deno run -A scripts/macos.ts release # CLI archive and desktop app
```

The `.app` uses
[`deno desktop`](https://docs.deno.com/runtime/reference/cli/desktop/)
and stores the native libraries in `Contents/Frameworks`.
With Developer ID secrets configured,
set `SIGN=1 NOTARIZE=1` to produce signed and notarized
CLI and app archives.

## Environment variables

| Variable                                | Effect                                 |
| --------------------------------------- | -------------------------------------- |
| `DENO_LLAMA_LIB_PATH`                   | Use a specific `libllama.dylib`        |
| `DENO_LLAMA_CACHE`                      | Override the dylib cache directory     |
| `DENO_LLAMA_VERBOSE`                    | Forward llama.cpp logs to stderr       |
| `DENO_LLAMA_MODEL` / `DENO_LLAMA_QUANT` | Default model + quant for the examples |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`   | Token for gated Hub repos              |

## Non-goals

This project does not provide a multi-backend abstraction
or support Node.js and Bun.
For those runtimes,
use [node-llama-cpp](https://github.com/withcatai/node-llama-cpp).

## License

deno-llama is available under the MIT license.
See the [LICENSE](/LICENSE) file for more info.

The downloaded llama.cpp and ggml dynamic libraries
remain under their upstream MIT license.
