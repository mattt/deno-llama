/**
 * libllama.dylib resolver.
 *
 * A compiled binary carries no native code, so the dylib is found at runtime.
 * Resolution order:
 *
 *   1. DENO_LLAMA_LIB_PATH            explicit override (also used by CI / tests)
 *   2. Cached pinned build            {cacheDir}/deno-llama/{tag}/libllama.dylib
 *   3. Vendored layouts (compiled apps), relative to the executable:
 *        {execDir}/libllama.dylib
 *        {execDir}/vendor/libllama.dylib
 *        {execDir}/../lib/libllama.dylib          (CLI archive layout)
 *        {execDir}/../Frameworks/libllama.dylib   (macOS .app layout)
 *
 * If none exist, {@link ensureLibrary} downloads + verifies + extracts the pinned
 * official release tarball into the cache (this is the only network step).
 */

import { llamaTag } from "./generated/meta.ts";
import { type DownloadProgress, ensureExtracted } from "./download.ts";

export type { DownloadProgress };

/** Root cache directory for downloaded llama.cpp builds. */
export function cacheRoot(): string {
  const override = Deno.env.get("DENO_LLAMA_CACHE");
  if (override) return override;
  const home = Deno.env.get("HOME") ?? ".";
  return `${home}/Library/Caches/deno-llama`;
}

/** The path a cached pinned dylib would occupy (may not exist yet). */
export function cachedLibPath(): string {
  return `${cacheRoot()}/${llamaTag}/libllama.dylib`;
}

/**
 * Find an already-present libllama.dylib, or throw.
 * Does not download; call {@link ensureLibrary} first for first-run acquisition.
 */
export function resolveLibPath(): string {
  const override = Deno.env.get("DENO_LLAMA_LIB_PATH");
  if (override) {
    if (!canStat(override)) {
      throw new Error(`DENO_LLAMA_LIB_PATH is set but not found: ${override}`);
    }
    return override;
  }

  const cached = cachedLibPath();
  if (canStat(cached)) return cached;

  for (const candidate of vendoredCandidates()) {
    if (canStat(candidate)) return candidate;
  }

  throw new Error(
    `libllama.dylib not found for llama.cpp ${llamaTag}. Call ensureLibrary() to ` +
      `download the pinned build, set DENO_LLAMA_LIB_PATH=/path/to/libllama.dylib, ` +
      `or vendor it beside the executable (./vendor/, ../lib/, ../Frameworks/).`,
  );
}

/**
 * Ensure a usable libllama.dylib exists and return its path, downloading the
 * pinned official release into the cache on first run. Honors
 * DENO_LLAMA_LIB_PATH and vendored/compiled layouts without any network access.
 */
export async function ensureLibrary(
  onProgress?: (p: DownloadProgress) => void,
): Promise<string> {
  // An explicit override must be honored exactly — if it's set but wrong, surface
  // the error instead of silently downloading a different (pinned) build.
  if (Deno.env.get("DENO_LLAMA_LIB_PATH")) return resolveLibPath();

  try {
    return resolveLibPath();
  } catch (err) {
    // "Can't look" is not "absent": downloading wouldn't fix a missing
    // --allow-read, and would fail again with a less useful message.
    if (err instanceof LibraryAccessError) throw err;
    // Not present anywhere — fetch the pinned build into the cache.
  }
  return await ensureExtracted(cacheRoot(), onProgress);
}

/** Locations checked for a vendored dylib, relative to the running executable. */
function vendoredCandidates(): string[] {
  let dir: string;
  try {
    dir = dirname(Deno.execPath());
  } catch {
    return [];
  }
  const parent = dirname(dir);
  return [
    `${dir}/libllama.dylib`,
    `${dir}/vendor/libllama.dylib`,
    `${parent}/lib/libllama.dylib`,
    `${parent}/Frameworks/libllama.dylib`,
  ];
}

function dirname(path: string): string {
  const i = path.lastIndexOf("/");
  return i <= 0 ? "." : path.slice(0, i);
}

/** Raised when the dylib can't even be looked for, as opposed to being absent. */
export class LibraryAccessError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "LibraryAccessError";
  }
}

function canStat(path: string): boolean {
  try {
    Deno.statSync(path);
    return true;
  } catch (err) {
    if (err instanceof Deno.errors.NotFound) return false;
    // Deno 2 raises NotCapable when a permission is missing; PermissionDenied is
    // the OS-level "you may not read this file". Both mean "can't tell", and both
    // deserve a better message than "dylib not found".
    if (
      err instanceof Deno.errors.NotCapable ||
      err instanceof Deno.errors.PermissionDenied
    ) {
      throw new LibraryAccessError(
        `Cannot read ${path}: missing --allow-read for the llama.cpp dylib. ` +
          `Grant read access or set DENO_LLAMA_LIB_PATH.`,
        { cause: err },
      );
    }
    return false;
  }
}
