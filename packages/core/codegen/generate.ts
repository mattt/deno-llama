/**
 * Generate Deno FFI bindings for llama.cpp, pinned to one build tag.
 *
 * Pipeline:
 *   1. download the pinned tag's headers (llama.h + the ggml headers it includes)
 *   2. download the pinned macOS arm64 release tarball; sha256 it; extract it
 *      (its libllama.dylib is the ground truth for `nm` validation)
 *   3. clang -ast-dump=json over llama.h  (a real parse, not regex)
 *   4. collect every `llama_*` FunctionDecl, plus the by-value structs and enums
 *      it references; map types via ./typemap.ts + AST-derived struct layouts
 *   5. compute C-ABI byte offsets for each by-value struct so the ergonomic layer
 *      can poke fields (n_gpu_layers, n_ctx, …) into the struct bytes
 *   6. validate bound names against the dylib's real exports (`nm -gU`)
 *   7. emit generated/{symbols,types,meta}.ts and `deno fmt` them
 *
 * Dev-time tool (needs clang/nm/tar and network). Generated output is committed
 * and pinned to the tag below. Rebase = bump LLAMA_TAG, re-run `deno task codegen`.
 *
 * Run: deno task codegen   (override tag with LLAMA_TAG=b#####)
 */

import { mapLeaf, normalize, UNSUPPORTED } from "./typemap.ts";

// ---- the pin -------------------------------------------------------------
const LLAMA_TAG = Deno.env.get("LLAMA_TAG") ?? "b10344";
const REPO = "ggml-org/llama.cpp";
const ASSET = `llama-${LLAMA_TAG}-bin-macos-arm64.tar.gz`;

type NativeType = Deno.NativeType;
type NativeResultType = Deno.NativeResultType; // NativeType | "void"

interface CFunction {
  name: string;
  ret: string;
  params: string[];
}

/** A resolved by-value struct: flat FFI field list + named byte offsets. */
interface StructLayout {
  flat: NativeType[];
  size: number;
  align: number;
  fields: {
    name: string;
    cType: string;
    ffi: NativeType;
    offset: number;
    count: number;
  }[];
}

async function run(cmd: string, args: string[]): Promise<string> {
  const out = await new Deno.Command(cmd, { args, stdout: "piped", stderr: "piped" })
    .output();
  if (!out.success) {
    throw new Error(
      `${cmd} ${args.join(" ")} failed:\n${new TextDecoder().decode(out.stderr)}`,
    );
  }
  return new TextDecoder().decode(out.stdout);
}

// ---- fetch headers + dylib ----------------------------------------------

async function fetchText(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} -> ${res.status}`);
  return await res.text();
}

/**
 * Download llama.h plus every header under ggml/include into a flat include dir
 * (clang only follows the ones llama.h transitively pulls). The ggml headers are
 * needed for more than `#include` resolution: llama.h's enum initializers
 * reference `GGML_*` macros defined there, and clang must see them to fold the
 * values the generator reads back out of the AST.
 */
async function fetchHeaders(dir: string): Promise<void> {
  const raw = (rel: string) =>
    `https://raw.githubusercontent.com/${REPO}/${LLAMA_TAG}/${rel}`;

  await Deno.writeTextFile(`${dir}/llama.h`, await fetchText(raw("include/llama.h")));

  const listRes = await fetch(
    `https://api.github.com/repos/${REPO}/contents/ggml/include?ref=${LLAMA_TAG}`,
    { headers: { accept: "application/vnd.github+json" } },
  );
  if (!listRes.ok) throw new Error(`list ggml/include -> ${listRes.status}`);
  const entries = await listRes.json() as { name: string; type: string }[];
  for (const e of entries) {
    if (e.type !== "file" || !e.name.endsWith(".h")) continue;
    const text = await fetchText(raw(`ggml/include/${e.name}`));
    await Deno.writeTextFile(`${dir}/${e.name}`, text);
  }
}

/** Download + sha256 + extract the release tarball; return {sha256, dylibPath}. */
async function fetchDylib(
  workDir: string,
): Promise<{ sha256: string; dylibPath: string }> {
  const url = `https://github.com/${REPO}/releases/download/${LLAMA_TAG}/${ASSET}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} -> ${res.status}`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const sha256 = [...new Uint8Array(digest)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  const tarPath = `${workDir}/${ASSET}`;
  await Deno.writeFile(tarPath, bytes);
  await run("tar", ["xzf", tarPath, "-C", workDir]);
  const dylibPath = `${workDir}/llama-${LLAMA_TAG}/libllama.dylib`;
  await Deno.stat(dylibPath); // fail loudly if the layout changed
  return { sha256, dylibPath };
}

// ---- clang AST -----------------------------------------------------------

async function dumpAst(includeDir: string): Promise<unknown> {
  const probe = `${includeDir}/__probe.c`;
  await Deno.writeTextFile(probe, '#include "llama.h"\n');
  const json = await run("clang", [
    "-Xclang",
    "-ast-dump=json",
    "-fsyntax-only",
    "-I",
    includeDir,
    probe,
  ]);
  return JSON.parse(json);
}

interface Collected {
  fns: CFunction[];
  enumNames: Set<string>;
  enums: Map<string, EnumDef>; // llama_* enums with clang-evaluated member values
  records: Map<string, string[]>; // struct name -> ordered field C types (with names dropped)
  recordFields: Map<string, { name: string; cType: string }[]>;
}

// deno-lint-ignore no-explicit-any
function collect(ast: any): Collected {
  const fns: CFunction[] = [];
  const enumNames = new Set<string>();
  const enums = new Map<string, EnumDef>();
  const records = new Map<string, string[]>();
  const recordFields = new Map<string, { name: string; cType: string }[]>();

  // deno-lint-ignore no-explicit-any
  function walk(n: any) {
    if (n.kind === "FunctionDecl" && n.name?.startsWith("llama_")) {
      const params = (n.inner ?? [])
        // deno-lint-ignore no-explicit-any
        .filter((c: any) => c.kind === "ParmVarDecl")
        // deno-lint-ignore no-explicit-any
        .map((c: any) => c.type.qualType as string);
      const full = (n.type?.qualType ?? "") as string;
      const ret = full.slice(0, full.lastIndexOf("(")).trim();
      fns.push({ name: n.name, ret, params });
    }
    // Collect every named enum (ggml_* included) so `enum X` fields/params map
    // to i32; only llama_* enums have their *values* emitted.
    if (n.kind === "EnumDecl" && n.name) {
      enumNames.add(n.name);
      if (n.name.startsWith("llama_") && !enums.has(n.name)) {
        const members = enumMembers(n.name, n);
        if (members.length) enums.set(n.name, { name: n.name, members });
      }
    }
    if (n.kind === "RecordDecl" && n.name?.startsWith("llama_") && n.inner) {
      const fields = (n.inner ?? [])
        // deno-lint-ignore no-explicit-any
        .filter((c: any) => c.kind === "FieldDecl")
        // deno-lint-ignore no-explicit-any
        .map((c: any) => ({ name: c.name as string, cType: c.type.qualType as string }));
      // Keep the definition that actually has fields (skip forward decls).
      if (fields.length && !records.has(n.name)) {
        records.set(n.name, fields.map((f: { cType: string }) => f.cType));
        recordFields.set(n.name, fields);
      }
    }
    for (const c of n.inner ?? []) walk(c);
  }
  walk(ast);

  const seen = new Set<string>();
  const uniqueFns = fns.filter((f) => !seen.has(f.name) && seen.add(f.name));
  return { fns: uniqueFns, enumNames, enums, records, recordFields };
}

/**
 * Read an enum's members from the AST. clang folds each initializer into a
 * `ConstantExpr` with the final value already computed, so initializers that
 * reference a macro from another header (`LLAMA_ROPE_TYPE_NEOX =
 * GGML_ROPE_TYPE_NEOX`), shift (`1 << 3`), or OR flags together all resolve
 * exactly — where reading the header text would have to re-implement the C
 * preprocessor and silently guess when it fell short. Members with no
 * initializer follow C's rule of "previous + 1".
 */
// deno-lint-ignore no-explicit-any
function enumMembers(enumName: string, decl: any): EnumDef["members"] {
  const members: EnumDef["members"] = [];
  let prev = -1;
  for (const c of decl.inner ?? []) {
    if (c.kind !== "EnumConstantDecl") continue;
    // deno-lint-ignore no-explicit-any
    const folded = (c.inner ?? []).find((i: any) =>
      i.kind === "ConstantExpr" && i.value !== undefined
    );
    const value = folded ? Number(folded.value) : prev + 1;
    if (!Number.isSafeInteger(value)) {
      throw new Error(
        `enum ${enumName}.${c.name}: value ${folded?.value} is not representable`,
      );
    }
    members.push({ name: c.name as string, value });
    prev = value;
  }
  return members;
}

// ---- type + struct-layout resolution ------------------------------------

function sizeAlign(t: NativeType, layouts: Map<string, StructLayout>): [number, number] {
  if (typeof t === "object" && "struct" in t) {
    // nested by-value struct — compute from its flat list
    return structSizeAlign(t.struct as NativeType[], layouts);
  }
  switch (t) {
    case "bool":
    case "u8":
    case "i8":
      return [1, 1];
    case "u16":
    case "i16":
      return [2, 2];
    case "u32":
    case "i32":
    case "f32":
      return [4, 4];
    case "u64":
    case "i64":
    case "f64":
    case "usize":
    case "isize":
    case "pointer":
    case "buffer":
    case "function":
      return [8, 8];
    default:
      throw new Error(`no size for FFI type ${JSON.stringify(t)}`);
  }
}

function structSizeAlign(
  flat: NativeType[],
  layouts: Map<string, StructLayout>,
): [number, number] {
  let offset = 0;
  let align = 1;
  for (const f of flat) {
    const [s, a] = sizeAlign(f, layouts);
    offset = alignUp(offset, a) + s;
    align = Math.max(align, a);
  }
  return [alignUp(offset, align), align];
}

const alignUp = (n: number, a: number) => Math.ceil(n / a) * a;

/** Split a field cType into base type and array count: `float[4]` -> ["float", 4]. */
function splitArray(cType: string): [string, number] {
  const m = cType.match(/^(.*)\[(\d+)\]$/);
  if (m) return [m[1].trim(), Number(m[2])];
  return [cType, 1];
}

/**
 * Resolve a by-value struct into a flat FFI list + named offsets, memoized.
 * Returns null if any field is unmappable (caller skips the dependent fn).
 */
function resolveStruct(
  name: string,
  ctx: Collected,
  layouts: Map<string, StructLayout>,
): StructLayout | null {
  const cached = layouts.get(name);
  if (cached) return cached;
  const fieldDecls = ctx.recordFields.get(name);
  if (!fieldDecls) return null;

  const flat: NativeType[] = [];
  const fields: StructLayout["fields"] = [];
  let offset = 0;
  let align = 1;

  for (const fd of fieldDecls) {
    const [base, count] = splitArray(fd.cType);
    const ffi = mapType(base, ctx, layouts);
    if (ffi === UNSUPPORTED) return null;
    const [s, a] = sizeAlign(ffi, layouts);
    offset = alignUp(offset, a);
    fields.push({ name: fd.name, cType: fd.cType, ffi, offset, count });
    for (let i = 0; i < count; i++) flat.push(ffi);
    offset += s * count;
    align = Math.max(align, a);
  }
  const size = alignUp(offset, align);
  const layout: StructLayout = { flat, size, align, fields };
  layouts.set(name, layout);
  return layout;
}

/** Map any C type (leaf or by-value struct) to a Deno FFI type, or UNSUPPORTED. */
function mapType(
  cType: string,
  ctx: Collected,
  layouts: Map<string, StructLayout>,
): NativeType | typeof UNSUPPORTED {
  const t = normalize(cType);
  if (!t.endsWith("*") && !t.includes("(*")) {
    const bare = t.replace(/^struct /, "").replace(/^enum /, "");
    if (ctx.records.has(bare)) {
      const layout = resolveStruct(bare, ctx, layouts);
      return layout ? { struct: layout.flat } : UNSUPPORTED;
    }
  }
  return mapLeaf(t, ctx.enumNames);
}

// ---- enum values (clang-evaluated; see enumMembers above) ----------------

interface EnumDef {
  name: string;
  members: { name: string; value: number }[];
}

// ---- emit ----------------------------------------------------------------

function typeLit(t: NativeResultType): string {
  if (typeof t === "string") return JSON.stringify(t);
  const s = t as { struct: NativeType[] };
  return `{ struct: [${s.struct.map(typeLit).join(", ")}] }`;
}

function pascal(enumName: string): string {
  return enumName.replace(/^llama_/, "").replace(
    /(^|_)(\w)/g,
    (_, __, c) => c.toUpperCase(),
  );
}

async function emit(x: {
  sha256: string;
  bound: { name: string; params: NativeType[]; result: NativeResultType }[];
  skipped: { name: string; reason: string }[];
  statusReturning: string[];
  enums: EnumDef[];
  layouts: Map<string, StructLayout>;
}) {
  const dir = new URL("../generated/", import.meta.url).pathname;
  await Deno.mkdir(dir, { recursive: true });
  const banner = `// GENERATED by packages/core/codegen/generate.ts — DO NOT EDIT.\n` +
    `// Source: llama.cpp ${LLAMA_TAG}. Regenerate with \`deno task codegen\`.\n\n`;

  // symbols.ts
  const syms = x.bound
    .map((b) =>
      `  ${b.name}: { parameters: [${b.params.map(typeLit).join(", ")}], result: ${
        typeLit(b.result)
      } },`
    )
    .join("\n");
  await Deno.writeTextFile(
    `${dir}symbols.ts`,
    `${banner}export const symbols = {\n${syms}\n} as const satisfies Deno.ForeignLibraryInterface;\n`,
  );

  // types.ts — enums + by-value struct layouts + handle alias
  const enumSrc = x.enums
    .map((e) => {
      const ts = pascal(e.name);
      const body = e.members.map((m) => `  ${m.name}: ${m.value},`).join("\n");
      return `/** llama.cpp \`enum ${e.name}\` */\nexport const ${ts} = {\n${body}\n} as const;\nexport type ${ts} = typeof ${ts}[keyof typeof ${ts}];`;
    })
    .join("\n\n");

  const layoutEntries = [...x.layouts.entries()]
    .map(([name, l]) => {
      const fields = l.fields
        .map((f) =>
          `    ${JSON.stringify(f.name)}: { offset: ${f.offset}, ffi: ${
            typeLit(f.ffi)
          }, count: ${f.count}, cType: ${JSON.stringify(f.cType)} },`
        )
        .join("\n");
      return `  ${
        JSON.stringify(name)
      }: {\n    size: ${l.size},\n    align: ${l.align},\n    fields: {\n${fields}\n    },\n  },`;
    })
    .join("\n");

  await Deno.writeTextFile(
    `${dir}types.ts`,
    `${banner}/** Opaque llama.cpp handles (\`llama_model *\`, …) are plain pointers. */\nexport const HANDLE = "pointer" as const;\n\n` +
      `/** Byte layouts for by-value structs, for poking fields into struct bytes. */\nexport interface StructField {\n  offset: number;\n  ffi: Deno.NativeType;\n  count: number;\n  cType: string;\n}\nexport interface StructLayout {\n  size: number;\n  align: number;\n  fields: Record<string, StructField>;\n}\nexport const structLayouts: Record<string, StructLayout> = {\n${layoutEntries}\n};\n\n${enumSrc}\n`,
  );

  // meta.ts
  await Deno.writeTextFile(
    `${dir}meta.ts`,
    `${banner}/** The pinned llama.cpp build tag. */\nexport const llamaTag = ${
      JSON.stringify(LLAMA_TAG)
    };\n\n` +
      `/** The macOS arm64 release asset the resolver downloads. */\nexport const macosArm64Asset = ${
        JSON.stringify(ASSET)
      };\n\n` +
      `/** SHA-256 of that asset, verified after download. */\nexport const macosArm64Sha256 = ${
        JSON.stringify(x.sha256)
      };\n\n` +
      `/** Functions whose C return is a status code (nonzero = error). */\nexport const statusReturning: ReadonlySet<string> = new Set(${
        JSON.stringify(x.statusReturning)
      });\n\n` +
      `/** Functions intentionally not bound (callbacks / unmappable types). */\nexport const skipped: ReadonlyArray<{ name: string; reason: string }> = ${
        JSON.stringify(x.skipped, null, 2)
      };\n`,
  );

  await run("deno", ["fmt", `${dir}symbols.ts`, `${dir}types.ts`, `${dir}meta.ts`]);
  console.log(`emitted ${dir}{symbols,types,meta}.ts`);
}

// ---- main ----------------------------------------------------------------

async function main() {
  const work = await Deno.makeTempDir({ prefix: "deno-llama-codegen-" });
  const includeDir = `${work}/include`;
  await Deno.mkdir(includeDir, { recursive: true });
  console.log(`llama.cpp tag: ${LLAMA_TAG}`);
  console.log(`work dir: ${work}`);

  await fetchHeaders(includeDir);
  const { sha256, dylibPath } = await fetchDylib(work);
  console.log(`dylib: ${dylibPath}`);
  console.log(`sha256: ${sha256}`);

  const ast = await dumpAst(includeDir);
  const ctx = collect(ast);
  const enums = [...ctx.enums.values()];
  console.log(
    `parsed ${ctx.fns.length} llama_ functions, ${enums.length} enums, ${ctx.records.size} structs`,
  );

  const layouts = new Map<string, StructLayout>();
  const bound: { name: string; params: NativeType[]; result: NativeResultType }[] = [];
  const skipped: { name: string; reason: string }[] = [];
  const statusReturning: string[] = [];

  for (const fn of ctx.fns) {
    // `void` is a valid return but not a NativeType (parameter/field) — map it here.
    const result: NativeResultType | typeof UNSUPPORTED = normalize(fn.ret) === "void"
      ? "void"
      : mapType(fn.ret, ctx, layouts);
    if (result === UNSUPPORTED) {
      skipped.push({ name: fn.name, reason: `return type ${fn.ret}` });
      continue;
    }
    let bad: string | null = null;
    const params: NativeType[] = [];
    for (const p of fn.params) {
      if (normalize(p) === "void") continue; // `(void)` empty param list
      const m = mapType(p, ctx, layouts);
      if (m === UNSUPPORTED) {
        bad = p;
        break;
      }
      params.push(m);
    }
    if (bad !== null) {
      skipped.push({ name: fn.name, reason: `param type ${bad}` });
      continue;
    }
    bound.push({ name: fn.name, params, result });
  }
  for (const b of bound) {
    if (b.name === "llama_decode" || b.name === "llama_encode") {
      statusReturning.push(b.name);
    }
  }

  // validate against the dylib's real exports
  const nm = await run("nm", ["-gU", dylibPath]);
  const exported = new Set(
    nm.split("\n")
      .map((l) => l.match(/ T (_llama_\w+)$/)?.[1]?.slice(1))
      .filter((x): x is string => !!x),
  );
  const missing = bound.filter((b) => !exported.has(b.name));
  if (missing.length) {
    // Some llama_* decls are static-inline helpers (no exported symbol) — drop them.
    const inlineDropped = missing.map((m) => m.name);
    for (const name of inlineDropped) {
      const i = bound.findIndex((b) => b.name === name);
      if (i >= 0) bound.splice(i, 1);
      skipped.push({ name, reason: "not exported by dylib (static inline?)" });
    }
    console.log(
      `dropped ${inlineDropped.length} non-exported decl(s): ${inlineDropped.join(", ")}`,
    );
  }

  await emit({ sha256, bound, skipped, statusReturning, enums, layouts });
  console.log(`\n✅ bound ${bound.length}, skipped ${skipped.length}`);
  console.log(
    `   emitted ${layouts.size} by-value struct layout(s): ${
      [...layouts.keys()].join(", ")
    }`,
  );
  await Deno.remove(work, { recursive: true });
}

if (import.meta.main) await main();
