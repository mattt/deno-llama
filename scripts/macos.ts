#!/usr/bin/env -S deno run -A
/**
 * macOS packaging for deno-llama.
 *
 *   deno run -A scripts/macos.ts bundle [--out dist]
 *       Stage relocatable llama.cpp / ggml dylibs (@loader_path install names).
 *   deno run -A scripts/macos.ts sign <path>...
 *       Developer ID sign binaries / dylibs / .app bundles (inside-out).
 *   deno run -A scripts/macos.ts notarize <target>
 *       Notarize + staple a zip / app / dmg / pkg via notarytool.
 *   deno run -A scripts/macos.ts release
 *       Compile the CLI, bundle dylibs, assemble CLI archive + .app; then
 *       sign/notarize when SIGN=1 / NOTARIZE=1.
 *
 * Env:
 *   sign:     MACOS_CODESIGN_IDENTITY (required), MACOS_ENTITLEMENTS (optional)
 *   notarize: APPLE_API_KEY_ID, APPLE_API_ISSUER, APPLE_API_KEY_PATH (required)
 */

import { llamaTag } from "../packages/core/generated/meta.ts";
import { cacheRoot, ensureLibrary } from "../packages/core/mod.ts";

// Version mirrors the package convention `2.<llama.cpp-build>.0` (llamaTag "b10344"
// -> "2.10344.0"), so it tracks the pin instead of drifting to a stale literal.
const RELEASE_VERSION = `2.${llamaTag.replace(/^b/, "")}.0`;

// Minimal entitlements for Deno FFI + local networking. Written to a temp file
// at sign time unless MACOS_ENTITLEMENTS points at a plist. Extend only if
// notarization requires it.
const ENTITLEMENTS_PLIST = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>com.apple.security.cs.allow-jit</key>
  <true/>
  <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
  <true/>
  <key>com.apple.security.network.client</key>
  <true/>
</dict>
</plist>
`;

// --------------------------------------------------------------------------
// shared helpers
// --------------------------------------------------------------------------

/** Run a command, capturing stdout; throws with stderr on failure. */
function run(cmd: string[], label = cmd[0]): string {
  const p = new Deno.Command(cmd[0], {
    args: cmd.slice(1),
    stdout: "piped",
    stderr: "piped",
  }).outputSync();
  const out = new TextDecoder().decode(p.stdout);
  const err = new TextDecoder().decode(p.stderr);
  if (p.code !== 0) throw new Error(`${label} failed: ${err || out}`);
  return out || err;
}

/** Run a command with inherited stdio (for streaming build output). */
function runInherit(cmd: string[], cwd?: string): void {
  console.log("$", cmd.join(" "));
  const p = new Deno.Command(cmd[0], {
    args: cmd.slice(1),
    cwd,
    stdout: "inherit",
    stderr: "inherit",
  }).outputSync();
  if (p.code !== 0) throw new Error(`failed: ${cmd.join(" ")}`);
}

function dirname(path: string): string {
  const i = path.lastIndexOf("/");
  return i <= 0 ? "." : path.slice(0, i);
}

function basename(path: string): string {
  const i = path.lastIndexOf("/");
  return i < 0 ? path : path.slice(i + 1);
}

function flag(name: string): string | undefined {
  const i = Deno.args.indexOf(name);
  return i >= 0 ? Deno.args[i + 1] : undefined;
}

// --------------------------------------------------------------------------
// bundle
// --------------------------------------------------------------------------

/** Dependencies of `path` (otool -L output minus the leading id line). */
function otoolDeps(path: string): string[] {
  return run(["otool", "-L", path], "otool")
    .split("\n")
    .slice(1)
    .map((l) => l.trim().split(" ")[0])
    .filter(Boolean);
}

/**
 * Reduce a dependency path to the base library name it should resolve to.
 * `@rpath/libggml-cpu.0.dylib` -> `libggml-cpu.dylib`; `libllama.0.0.10344.dylib`
 * -> `libllama.dylib`.
 */
function baseName(dep: string): string {
  const stem = basename(dep).replace(/\.dylib$/, "");
  const family = stem.replace(/(\.\d+)+$/, "");
  return `${family}.dylib`;
}

/** Directory holding the cached (or overridden) dylib set. */
function sourceDir(): string {
  const override = Deno.env.get("DENO_LLAMA_LIB_PATH");
  if (override) return dirname(override);
  return `${cacheRoot()}/${llamaTag}`;
}

/** Base-name dylibs to redistribute: libllama.dylib + every libggml*.dylib. */
function libsToStage(dir: string): string[] {
  const names: string[] = [];
  for (const entry of Deno.readDirSync(dir)) {
    const n = entry.name;
    if (!n.endsWith(".dylib")) continue;
    if (/\.\d/.test(n)) continue; // skip versioned files and .0.dylib symlinks
    if (n === "libllama.dylib" || /^libggml.*\.dylib$/.test(n)) names.push(n);
  }
  names.sort();
  if (!names.includes("libllama.dylib")) {
    throw new Error(`libllama.dylib not found in ${dir}`);
  }
  return names;
}

/** Copy + relocate the dylib set into `destDir`. */
function stage(srcDir: string, destDir: string, names: string[]): void {
  Deno.mkdirSync(destDir, { recursive: true });
  const staged = new Set(names);

  // Copy each library, resolving symlinks to the real versioned file.
  for (const name of names) {
    Deno.copyFileSync(Deno.realPathSync(`${srcDir}/${name}`), `${destDir}/${name}`);
  }

  // Rewrite ids and inter-library dependencies to @loader_path.
  for (const name of names) {
    const dst = `${destDir}/${name}`;
    run(["install_name_tool", "-id", `@loader_path/${name}`, dst], `id ${name}`);
    for (const dep of otoolDeps(dst)) {
      if (dep.startsWith("@loader_path/")) continue;
      const target = baseName(dep);
      const pointsHere = dep.startsWith("@rpath/") ||
        dep.startsWith(srcDir) || dep.includes("/deno-llama/");
      if (pointsHere && staged.has(target)) {
        run(
          ["install_name_tool", "-change", dep, `@loader_path/${target}`, dst],
          `rewrite dep ${dep}`,
        );
      }
    }
  }

  // Verify nothing escapes @loader_path or the system library paths.
  for (const name of names) {
    const dst = `${destDir}/${name}`;
    const bad = otoolDeps(dst).filter((d) =>
      !d.startsWith("@loader_path/") && !d.startsWith("/usr/lib/") &&
      !d.startsWith("/System/")
    );
    if (bad.length) {
      throw new Error(
        `${dst} still has non-relocatable dependencies:\n  ${bad.join("\n  ")}`,
      );
    }
  }
}

function isDir(path: string): boolean {
  try {
    return Deno.statSync(path).isDirectory;
  } catch {
    return false;
  }
}

/**
 * Stage dylibs into `<outRoot>/lib` (for the CLI archive) and, when a real
 * `DenoLlama.app` is present (built by `deno desktop`), into its
 * `Contents/Frameworks` — the resolver's `.app` layout. Returns the lib names.
 */
async function bundle(outRoot: string, quiet = false): Promise<string[]> {
  // Ensure the cache is populated (no-op if DENO_LLAMA_LIB_PATH / cache present).
  try {
    await ensureLibrary();
  } catch (err) {
    console.warn(`ensureLibrary() skipped: ${(err as Error).message}`);
  }

  const srcDir = sourceDir();
  const names = libsToStage(srcDir);
  const libDir = `${outRoot}/lib`;
  const dests = [libDir];

  const frameworksDir = `${outRoot}/DenoLlama.app/Contents/Frameworks`;
  if (isDir(`${outRoot}/DenoLlama.app/Contents/MacOS`)) dests.push(frameworksDir);

  if (!quiet) {
    console.log(`llama.cpp pin: ${llamaTag}`);
    console.log(`staging from: ${srcDir}`);
    console.log(`libraries:\n  ${names.join("\n  ")}`);
  }

  for (const dest of dests) stage(srcDir, dest, names);

  Deno.mkdirSync(`${outRoot}/licenses`, { recursive: true });
  Deno.writeTextFileSync(
    `${outRoot}/licenses/THIRD_PARTY_NATIVE.txt`,
    `This release bundles prebuilt llama.cpp / ggml dynamic libraries.
See the upstream project and license:
  https://github.com/ggml-org/llama.cpp
Pinned llama.cpp build tag: ${llamaTag}

Bundled libraries:
${names.map((n) => `  ${n}`).join("\n")}
`,
  );

  if (!quiet) {
    console.log("staged:");
    for (const dir of dests) {
      for (const name of names) console.log(`  ${dir}/${name}`);
    }
  }
  return names;
}

// --------------------------------------------------------------------------
// sign
// --------------------------------------------------------------------------

interface SignConfig {
  identity: string;
  entitlements: string;
}

function signConfig(): SignConfig {
  const identity = Deno.env.get("MACOS_CODESIGN_IDENTITY");
  if (!identity) throw new Error("MACOS_CODESIGN_IDENTITY is required for signing");
  return { identity, entitlements: resolveEntitlements() };
}

/** Path to an entitlements plist: MACOS_ENTITLEMENTS, else the inlined default. */
function resolveEntitlements(): string {
  const explicit = Deno.env.get("MACOS_ENTITLEMENTS");
  if (explicit) return explicit;
  const tmp = Deno.makeTempFileSync({
    prefix: "deno-llama-entitlements-",
    suffix: ".plist",
  });
  Deno.writeTextFileSync(tmp, ENTITLEMENTS_PLIST);
  return tmp;
}

function signOne(path: string, cfg: SignConfig, deep = false): void {
  const args = [
    "codesign",
    "--force",
    "--options",
    "runtime",
    "--timestamp",
    "--sign",
    cfg.identity,
  ];
  if (cfg.entitlements) args.push("--entitlements", cfg.entitlements);
  if (deep) args.push("--deep");
  args.push(path);
  console.log(`signing ${path}`);
  run(args, "codesign");
  run(["codesign", "--verify", "--verbose=2", path], "codesign --verify");
}

/** Sign each target; .app bundles are signed inside-out (Frameworks, binary, bundle). */
function sign(targets: string[], cfg: SignConfig): void {
  for (const t of targets) {
    const st = Deno.statSync(t);
    if (st.isDirectory && t.endsWith(".app")) {
      const frameworks = `${t}/Contents/Frameworks`;
      try {
        for (const e of Deno.readDirSync(frameworks)) {
          if (e.name.endsWith(".dylib")) signOne(`${frameworks}/${e.name}`, cfg);
        }
      } catch {
        // no Frameworks
      }
      try {
        for (const e of Deno.readDirSync(`${t}/Contents/MacOS`)) {
          signOne(`${t}/Contents/MacOS/${e.name}`, cfg);
        }
      } catch {
        // ignore
      }
      signOne(t, cfg, true);
    } else {
      signOne(t, cfg);
    }
  }
}

// --------------------------------------------------------------------------
// notarize
// --------------------------------------------------------------------------

interface NotaryConfig {
  keyId: string;
  issuer: string;
  keyPath: string;
}

function notaryConfig(): NotaryConfig {
  const keyId = Deno.env.get("APPLE_API_KEY_ID");
  const issuer = Deno.env.get("APPLE_API_ISSUER");
  const keyPath = Deno.env.get("APPLE_API_KEY_PATH");
  if (!keyId || !issuer || !keyPath) {
    throw new Error(
      "Need APPLE_API_KEY_ID, APPLE_API_ISSUER, and APPLE_API_KEY_PATH for notarization",
    );
  }
  return { keyId, issuer, keyPath };
}

/** Tickets can only be stapled to bundles, not to plain zip archives. */
function canStaple(target: string): boolean {
  return /\.(app|dmg|pkg)$/.test(target);
}

/** Submit a target for notarization and wait for the result. */
function notarizeSubmit(target: string, cfg: NotaryConfig): void {
  // notarytool accepts .zip/.dmg/.pkg but not a bare .app dir — wrap it in a
  // temp zip for submission (the ticket still staples to the .app afterward).
  let submitTarget = target;
  let temp: string | undefined;
  if (target.endsWith(".app")) {
    temp = `${target}.notary.zip`;
    run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", target, temp], "ditto");
    submitTarget = temp;
  }
  console.log(`submitting ${submitTarget}`);
  try {
    run([
      "xcrun",
      "notarytool",
      "submit",
      submitTarget,
      "--key",
      cfg.keyPath,
      "--key-id",
      cfg.keyId,
      "--issuer",
      cfg.issuer,
      "--wait",
    ], "notarytool submit");
  } finally {
    if (temp) {
      try {
        Deno.removeSync(temp);
      } catch {
        // best effort
      }
    }
  }
}

/** Staple a notarization ticket to a bundle (.app/.dmg/.pkg only). */
function staple(target: string): void {
  console.log(`stapling ${target}`);
  run(["xcrun", "stapler", "staple", target], "stapler staple");
  run(["xcrun", "stapler", "validate", target], "stapler validate");
  if (target.endsWith(".app") || target.endsWith(".dmg")) {
    run(["spctl", "-a", "-vv", "--type", "install", target], "spctl");
  }
}

function notarize(target: string, cfg: NotaryConfig): void {
  notarizeSubmit(target, cfg);
  if (canStaple(target)) {
    staple(target);
  } else {
    // A .zip carries no ticket; its enclosed signed binaries are notarized and
    // Gatekeeper verifies them online. Staple the .app/.dmg separately.
    console.log(`submitted ${target}; skipping staple (not a bundle)`);
  }
  console.log("notarization complete");
}

// --------------------------------------------------------------------------
// release
// --------------------------------------------------------------------------

async function release(dist: string): Promise<void> {
  Deno.mkdirSync(`${dist}/bin`, { recursive: true });

  // 1. Compile the CLI.
  runInherit([
    "deno",
    "compile",
    "-P=llama",
    "--allow-run",
    "--target",
    "aarch64-apple-darwin",
    "--output",
    `${dist}/bin/deno-llama`,
    "examples/cli/main.ts",
  ]);

  // 2. Build the desktop app with `deno desktop`, which bundles the WebView
  //    backend so DenoLlama.app opens a window. (Plain `deno compile` would only
  //    start a headless loopback server.)
  const app = `${dist}/DenoLlama.app`;
  runInherit([
    "deno",
    "desktop",
    "--output",
    app,
    "-P=llama",
    "--allow-run",
    "examples/desktop/main.ts",
  ]);

  // 3. Bundle dylibs into dist/lib and, now that the app exists, its Frameworks.
  const dylibs = await bundle(dist, true);

  // 4. Assemble the CLI archive tree.
  const cliRoot = `${dist}/deno-llama-cli`;
  Deno.mkdirSync(`${cliRoot}/bin`, { recursive: true });
  Deno.mkdirSync(`${cliRoot}/lib`, { recursive: true });
  Deno.copyFileSync(`${dist}/bin/deno-llama`, `${cliRoot}/bin/deno-llama`);
  for (const name of dylibs) {
    Deno.copyFileSync(`${dist}/lib/${name}`, `${cliRoot}/lib/${name}`);
  }
  Deno.copyFileSync(
    `${dist}/licenses/THIRD_PARTY_NATIVE.txt`,
    `${cliRoot}/THIRD_PARTY_NATIVE.txt`,
  );
  Deno.writeTextFileSync(
    `${cliRoot}/README.txt`,
    `deno-llama CLI (Apple Silicon)

Run:
  ./bin/deno-llama doctor
  ./bin/deno-llama chat "Hello"

Native libraries live in ./lib and are resolved relative to the executable.
llama.cpp pin: ${llamaTag}
`,
  );

  // 5. Checksums + metadata.
  const meta = {
    version: RELEASE_VERSION,
    llamaTag,
    deno: Deno.version.deno,
    target: "aarch64-apple-darwin",
    git: gitHead(),
  };
  Deno.writeTextFileSync(`${dist}/build-metadata.json`, JSON.stringify(meta, null, 2));

  const archive = "deno-llama-cli-aarch64.zip";
  const appArchive = "DenoLlama.app.zip";
  const zipCli = () =>
    runInherit([
      "bash",
      "-c",
      `cd ${dist} && ditto -c -k --sequesterRsrc --keepParent deno-llama-cli ${archive}`,
    ]);
  // Distribute the .app as a zip: a plain directory upload would lose the
  // executable's +x bit, and stapling needs the bundle preserved.
  const zipApp = () =>
    runInherit([
      "bash",
      "-c",
      `cd ${dist} && ditto -c -k --sequesterRsrc --keepParent DenoLlama.app ${appArchive}`,
    ]);

  // 6. Sign (rewrites binaries), then zip both artifacts. Staging dylibs into the
  //    app invalidated `deno desktop`'s ad-hoc signature, so re-seal: Developer ID
  //    when signing, else ad-hoc so the local bundle still launches.
  if (Deno.env.get("SIGN") === "1") {
    sign(
      [`${cliRoot}/bin/deno-llama`, ...dylibs.map((n) => `${cliRoot}/lib/${n}`), app],
      signConfig(),
    );
  } else {
    run(["codesign", "--force", "--sign", "-", "--deep", app], "adhoc codesign");
  }
  zipCli();

  // 7. Notarize both artifacts. The CLI zip carries no ticket (Gatekeeper checks
  //    its signed binaries online); the .app is submitted, stapled, then re-zipped.
  if (Deno.env.get("NOTARIZE") === "1") {
    const cfg = notaryConfig();
    notarize(`${dist}/${archive}`, cfg);
    notarize(app, cfg); // submits a temp zip of the bundle, then staples the .app
  }
  zipApp(); // (re)zip the app after any stapling

  // 8. Checksums LAST, over the final signed/notarized artifacts.
  runInherit([
    "bash",
    "-c",
    `cd ${dist} && shasum -a 256 ${archive} ${appArchive} > SHA256SUMS`,
  ]);

  console.log("release artifacts staged under dist/");
}

function gitHead(): string {
  try {
    return new TextDecoder()
      .decode(
        new Deno.Command("git", { args: ["rev-parse", "HEAD"] }).outputSync().stdout,
      )
      .trim();
  } catch {
    return "unknown";
  }
}

// --------------------------------------------------------------------------
// main
// --------------------------------------------------------------------------

function usage(): never {
  console.error(
    "usage: macos.ts <bundle [--out DIR] | sign PATH... | notarize TARGET | release>",
  );
  Deno.exit(2);
}

const [subcommand, ...rest] = Deno.args;
switch (subcommand) {
  case "bundle":
    await bundle(flag("--out") ?? "dist");
    break;
  case "sign": {
    const targets = rest.filter((a) => !a.startsWith("--"));
    if (targets.length === 0) usage();
    sign(targets, signConfig());
    console.log("signing complete");
    break;
  }
  case "notarize": {
    const target = rest.find((a) => !a.startsWith("--"));
    if (!target) usage();
    notarize(target, notaryConfig());
    break;
  }
  case "release":
    await release(flag("--out") ?? "dist");
    break;
  default:
    usage();
}
