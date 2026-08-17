/**
 * Download, verify, and extract the pinned llama.cpp release tarball.
 *
 * The macOS arm64 asset is a ~11 MB `.tar.gz` whose `libllama.dylib` links its
 * `libggml*.dylib` siblings via `@rpath`, so the whole directory is extracted
 * together and `libllama.dylib` is opened in place.
 */

import { UntarStream } from "@std/tar/untar-stream";
import { llamaTag, macosArm64Asset, macosArm64Sha256 } from "./generated/meta.ts";

export interface DownloadProgress {
  /** Bytes received so far. */
  received: number;
  /** Total bytes (from Content-Length), or undefined if unknown. */
  total?: number;
}

const RELEASE_BASE = "https://github.com/ggml-org/llama.cpp/releases/download";

/** SHA-256 of a byte buffer as lowercase hex. */
async function sha256Hex(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Ensure the pinned tarball is downloaded, verified, and extracted under
 * `<destRoot>/<tag>/`. Idempotent: if `libllama.dylib` is already present, returns
 * immediately. Returns the absolute path to `libllama.dylib`.
 */
export async function ensureExtracted(
  destRoot: string,
  onProgress?: (p: DownloadProgress) => void,
): Promise<string> {
  const tagDir = `${destRoot}/${llamaTag}`;
  const dylib = `${tagDir}/libllama.dylib`;
  if (await isFile(dylib)) return dylib;

  const url = `${RELEASE_BASE}/${llamaTag}/${macosArm64Asset}`;
  const res = await fetch(url);
  if (!res.ok || !res.body) {
    throw new Error(`Failed to download ${url}: ${res.status} ${res.statusText}`);
  }
  const total = Number(res.headers.get("content-length")) || undefined;

  // Stream into memory while reporting progress (asset is ~11 MB).
  const chunks: Uint8Array[] = [];
  let received = 0;
  for await (const chunk of res.body) {
    chunks.push(chunk);
    received += chunk.length;
    onProgress?.({ received, total });
  }
  const bytes = concat(chunks, received);

  const got = await sha256Hex(bytes);
  if (got !== macosArm64Sha256) {
    throw new Error(
      `Checksum mismatch for ${macosArm64Asset}:\n  expected ${macosArm64Sha256}\n  got      ${got}`,
    );
  }

  // Extract into a private staging dir, then atomically rename into place. The
  // staging path is unique per attempt and we never delete `tagDir` — so
  // concurrent first-run downloaders (e.g. parallel workspaces sharing the cache)
  // can't clobber each other's completed install.
  const staging = `${destRoot}/.staging-${llamaTag}-${crypto.randomUUID()}`;
  await Deno.mkdir(staging, { recursive: true });
  try {
    await extractTarGz(bytes, staging);

    // The archive top-level dir is `llama-<tag>/`; move it to `<tag>/`.
    const inner = `${staging}/llama-${llamaTag}`;
    const source = (await isDir(inner)) ? inner : staging;

    try {
      await Deno.rename(source, tagDir);
    } catch {
      // Another process won the race and populated tagDir; use theirs if valid.
      if (await isFile(dylib)) return dylib;
      throw new Error(`Failed to move extracted build into ${tagDir}`);
    }
  } finally {
    await rmrf(staging);
  }

  if (!(await isFile(dylib))) {
    throw new Error(`Extraction did not produce ${dylib}`);
  }
  return dylib;
}

/**
 * Gunzip + untar `bytes` into `dir`. The llama.cpp archive ships symlinks
 * (`libllama.dylib` -> `libllama.0.dylib`) and directories alongside regular
 * files, so entries are dispatched by tar typeflag: '5' dir, '2' symlink,
 * '1' hardlink, everything else a regular file.
 */
async function extractTarGz(bytes: Uint8Array<ArrayBuffer>, dir: string): Promise<void> {
  const stream = new Blob([bytes]).stream()
    .pipeThrough(new DecompressionStream("gzip"))
    .pipeThrough(new UntarStream());

  for await (const entry of stream) {
    const path = `${dir}/${entry.path}`;
    // deno-lint-ignore no-explicit-any
    const header = entry.header as any;
    const typeflag: string = header?.typeflag ?? "0";
    const linkname: string = header?.linkname ?? "";

    if (typeflag === "5" || entry.path.endsWith("/")) {
      await Deno.mkdir(path, { recursive: true });
      entry.readable?.cancel();
      continue;
    }
    await Deno.mkdir(dirname(path), { recursive: true });

    if (typeflag === "2") {
      // symlink — recreate it (relative link target within the extracted dir)
      await rmrf(path);
      await Deno.symlink(linkname, path);
      entry.readable?.cancel();
      continue;
    }
    if (typeflag === "1") {
      // hardlink — copy the already-extracted target
      await rmrf(path);
      await Deno.copyFile(`${dir}/${linkname}`, path);
      entry.readable?.cancel();
      continue;
    }
    if (!entry.readable) continue;

    using file = await Deno.open(path, { write: true, create: true, truncate: true });
    await entry.readable.pipeTo(file.writable);
    try {
      await Deno.chmod(path, 0o755); // dylibs must stay readable/executable
    } catch {
      // best effort (e.g. non-Unix fs)
    }
  }
}

function concat(chunks: Uint8Array[], total: number): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

function dirname(path: string): string {
  const i = path.lastIndexOf("/");
  return i <= 0 ? "." : path.slice(0, i);
}

async function isFile(path: string): Promise<boolean> {
  try {
    return (await Deno.stat(path)).isFile;
  } catch {
    return false;
  }
}

async function isDir(path: string): Promise<boolean> {
  try {
    return (await Deno.stat(path)).isDirectory;
  } catch {
    return false;
  }
}

async function rmrf(path: string): Promise<void> {
  try {
    await Deno.remove(path, { recursive: true });
  } catch {
    // ignore
  }
}
