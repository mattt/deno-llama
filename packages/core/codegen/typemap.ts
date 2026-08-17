/**
 * C leaf-type -> Deno FFI type mapping for llama.h / ggml.h.
 *
 * Unlike mlx-c, llama.cpp is a hand-written C API with real by-value structs
 * (`llama_model_params`, `llama_context_params`, `llama_batch`, …) and opaque
 * handles that are plain pointers (`struct llama_model *`), not single-field
 * structs. This module resolves *leaf* types (scalars, enums, pointers). The
 * by-value struct layouts are discovered from the clang AST in `generate.ts`,
 * which falls back to `mapLeaf` for each field.
 *
 * The leaf vocabulary is curated and pinned to the llama.cpp build tag. When a
 * rebase introduces a new leaf type, add it here; anything unknown returns
 * UNSUPPORTED so the dependent function is skipped and reported rather than
 * mis-bound.
 */

type NativeType = Deno.NativeType;

/** Sentinel for a type we deliberately don't (or can't) bind. */
export const UNSUPPORTED = Symbol("unsupported-type");

/** Opaque llama.cpp handles (`llama_model *`, `llama_context *`, …) are pointers. */
export const HANDLE: NativeType = "pointer";

/** C scalars and the fixed-width integer typedefs llama.h uses. */
const SCALARS: Record<string, NativeType> = {
  bool: "bool",
  char: "i8",
  "signed char": "i8",
  "unsigned char": "u8",
  short: "i16",
  "unsigned short": "u16",
  int: "i32",
  "unsigned int": "u32",
  unsigned: "u32",
  long: "i64",
  "unsigned long": "u64",
  "long long": "i64",
  "unsigned long long": "u64",
  float: "f32",
  double: "f64",
  size_t: "usize",
  ssize_t: "isize",
  ptrdiff_t: "isize",
  uintptr_t: "usize",
  intptr_t: "isize",
  int8_t: "i8",
  int16_t: "i16",
  int32_t: "i32",
  int64_t: "i64",
  uint8_t: "u8",
  uint16_t: "u16",
  uint32_t: "u32",
  uint64_t: "u64",
  // llama.h integer typedefs
  llama_token: "i32",
  llama_pos: "i32",
  llama_seq_id: "i32",
  llama_state_seq_flags: "u32",
  ggml_bf16_t: "u16",
  ggml_fp16_t: "u16",
};

/**
 * Typedefs that are pointers under the hood: opaque object handles and function
 * pointers. Inside a struct or as a parameter these are one machine pointer.
 */
const POINTER_TYPEDEFS = new Set([
  "llama_progress_callback",
  "llama_memory_t",
  "ggml_backend_dev_t",
  "ggml_backend_buffer_type_t",
  "ggml_backend_sched_eval_callback",
  "ggml_abort_callback",
  "ggml_log_callback",
  "ggml_threadpool_t",
]);

/** Strip `const` / `volatile` / `struct` / `enum` qualifiers and collapse whitespace. */
export function normalize(c: string): string {
  return c
    .replace(/\b(const|volatile|restrict)\b/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Map a C *leaf* type (not a by-value struct) to a Deno FFI type.
 * `enums` is the set of enum type names discovered from the AST.
 */
export function mapLeaf(
  cType: string,
  enums: ReadonlySet<string>,
): NativeType | typeof UNSUPPORTED {
  const t = normalize(cType);
  // Inline function pointer, e.g. `void (*)(void *)` — hand-bind if needed.
  if (t.includes("(*") || t.includes("(^")) return UNSUPPORTED;
  // Any pointer (including `char *`, `void *`, `struct X *`) is one pointer.
  if (t.endsWith("*")) return "pointer";
  if (t in SCALARS) return SCALARS[t];
  const bare = t.replace(/^enum /, "").replace(/^struct /, "");
  if (enums.has(bare)) return "i32";
  if (POINTER_TYPEDEFS.has(bare)) return "pointer";
  return UNSUPPORTED;
}

/**
 * Status-code returning functions (C `int32_t` where nonzero encodes an error).
 * Curated: llama.cpp does not use a uniform status convention, so only the
 * decode/encode family is treated as checked.
 */
export function isStatusReturn(name: string): boolean {
  return name === "llama_decode" || name === "llama_encode";
}
