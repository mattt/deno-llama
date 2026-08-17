/**
 * @deno-llama/core — raw FFI bindings over llama.cpp, pinned to one build tag.
 *
 * The symbol surface is generated from the pinned `llama.h` (see ./codegen) and
 * lives in ./generated. Open the library with {@link openLlama}; `.raw` gives
 * direct FFI symbols and `.checked` turns decode/encode status codes into thrown
 * {@link LlamaError}s. First-run acquisition of the prebuilt dylib is handled by
 * {@link ensureLibrary}, which downloads + verifies the pinned official release.
 *
 * The ergonomic layer (`@deno-llama/llama`) builds a `LlamaModel` on top of this.
 */

export { isOpen, LlamaError, openLlama } from "./ffi.ts";
export type { Llama } from "./ffi.ts";
export {
  cachedLibPath,
  cacheRoot,
  type DownloadProgress,
  ensureLibrary,
  LibraryAccessError,
  resolveLibPath,
} from "./resolver.ts";
export { ensureExtracted } from "./download.ts";

// By-value struct helpers (context/model/sampler params).
export { layoutOf, Struct } from "./structs.ts";

// Generated surface: symbol table, enums + struct layouts, and the pin metadata.
export { symbols } from "./generated/symbols.ts";
export * from "./generated/types.ts";
export {
  llamaTag,
  macosArm64Asset,
  macosArm64Sha256,
  skipped,
  statusReturning,
} from "./generated/meta.ts";
