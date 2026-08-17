import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import {
  cachedLibPath,
  ensureLibrary,
  isOpen,
  llamaTag,
  openLlama,
  resolveLibPath,
  Struct,
  structLayouts,
} from "./mod.ts";

Deno.test("cachedLibPath embeds the pinned tag", () => {
  assertStringIncludes(cachedLibPath(), `/${llamaTag}/libllama.dylib`);
});

Deno.test("resolveLibPath honors DENO_LLAMA_LIB_PATH", () => {
  const prev = Deno.env.get("DENO_LLAMA_LIB_PATH");
  const tmp = Deno.makeTempFileSync({ suffix: ".dylib" });
  try {
    Deno.env.set("DENO_LLAMA_LIB_PATH", tmp);
    assertEquals(resolveLibPath(), tmp);
  } finally {
    if (prev === undefined) Deno.env.delete("DENO_LLAMA_LIB_PATH");
    else Deno.env.set("DENO_LLAMA_LIB_PATH", prev);
    Deno.removeSync(tmp);
  }
});

Deno.test("resolveLibPath throws a clear error for a missing override", () => {
  const prev = Deno.env.get("DENO_LLAMA_LIB_PATH");
  try {
    Deno.env.set("DENO_LLAMA_LIB_PATH", "/no/such/libllama.dylib");
    let threw = false;
    try {
      resolveLibPath();
    } catch (e) {
      threw = true;
      assertStringIncludes((e as Error).message, "not found");
    }
    assert(threw, "expected resolveLibPath to throw");
  } finally {
    if (prev === undefined) Deno.env.delete("DENO_LLAMA_LIB_PATH");
    else Deno.env.set("DENO_LLAMA_LIB_PATH", prev);
  }
});

Deno.test("resolveLibPath blames a missing --allow-read, not a missing dylib", async () => {
  // Deno 2 raises NotCapable (not PermissionDenied) for a denied read, so this
  // needs a real subprocess running without --allow-read. It must be `deno run`:
  // `deno eval` has implicit access to every permission, which would make the
  // whole test vacuous.
  const entry = await Deno.makeTempFile({ suffix: ".ts" });
  try {
    await Deno.writeTextFile(
      entry,
      `import { resolveLibPath } from ${
        JSON.stringify(import.meta.resolve("./resolver.ts"))
      };
       try { resolveLibPath(); console.log("NO THROW"); }
       catch (e) { console.log(e.name + ": " + e.message); }`,
    );
    const out = await new Deno.Command(Deno.execPath(), {
      // The entry lives in a temp dir, so point it at the workspace config or
      // resolver.ts's transitive `@std/tar` import won't resolve.
      args: [
        "run",
        "--allow-env",
        "--config",
        new URL("../../deno.json", import.meta.url).pathname,
        entry,
      ],
      env: { DENO_LLAMA_LIB_PATH: "" },
      stdout: "piped",
      stderr: "piped",
    }).output();
    const text = new TextDecoder().decode(out.stdout) +
      new TextDecoder().decode(out.stderr);
    assertStringIncludes(text, "LibraryAccessError");
    assertStringIncludes(text, "--allow-read");
  } finally {
    await Deno.remove(entry);
  }
});

Deno.test("generated layouts include the by-value params structs", () => {
  for (const name of ["llama_model_params", "llama_context_params", "llama_batch"]) {
    assert(name in structLayouts, `missing layout for ${name}`);
    assert(structLayouts[name].size > 0);
  }
});

Deno.test("ffi opens and reads llama.cpp defaults at the right offsets", async () => {
  await ensureLibrary();
  const llama = openLlama();
  assertStringIncludes(llama.path, "libllama.dylib");

  const ctx = Struct.from(
    "llama_context_params",
    llama.raw.llama_context_default_params() as Uint8Array,
  );
  // These are stable llama.cpp defaults; if offsets were wrong they'd be garbage.
  assertEquals(ctx.get("n_ctx"), 512);
  assertEquals(ctx.get("n_batch"), 2048);
  assertEquals(ctx.get("n_ubatch"), 512);
  assertEquals(ctx.get("n_seq_max"), 1);
});

Deno.test("openLlama survives close() then reopen (fresh log callback)", async () => {
  await ensureLibrary();
  openLlama().close();
  assert(!isOpen());
  // Reopening must install a fresh log callback, not the freed one, and still work.
  const llama = openLlama();
  assert(isOpen());
  const ctx = Struct.from(
    "llama_context_params",
    llama.raw.llama_context_default_params() as Uint8Array,
  );
  assertEquals(ctx.get("n_ctx"), 512);
});
