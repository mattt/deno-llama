/**
 * Resolve a GGUF model to a local file, downloading from the Hugging Face Hub
 * into the shared HF cache when needed (so `hf`/`huggingface-cli` and this
 * library see the same files). A local `.gguf` path is used as-is.
 *
 * Downloads are delegated to `@huggingface/hub` (resumable, Xet-accelerated when
 * the file is Xet-backed). Progress is reported by wrapping the fetch that streams
 * the blob — a correctness feature for Glimmer-sized files, not a nicety.
 */

import { downloadFileToCacheDir, listFiles } from "@huggingface/hub";

export interface GgufProgress {
  file: string;
  received: number;
  total?: number;
}

export interface ResolveGgufOptions {
  /** Quantization to prefer, matched case-insensitively (e.g. "Q4_K_M"). */
  quant?: string;
  /** Git revision (branch, tag, or commit). Defaults to "main". */
  revision?: string;
  /** HF token for gated repos (defaults to HF_TOKEN / HUGGING_FACE_HUB_TOKEN). */
  accessToken?: string;
  onProgress?: (p: GgufProgress) => void;
}

/** True if `source` looks like a local filesystem path to a GGUF file. */
function isLocalGguf(source: string): boolean {
  if (!source.toLowerCase().endsWith(".gguf")) return false;
  // `owner/name:file.gguf` pins a file inside a repo — not a path (checked first
  // because the trailing `.gguf` would otherwise make it look like one).
  if (source.includes(":") && looksLikeRepoId(source.split(":")[0])) return false;
  if (source.includes("/") && !looksLikeRepoId(source)) return true;
  try {
    return Deno.statSync(source).isFile;
  } catch {
    return false;
  }
}

/** A repo id is `owner/name`; a path has more segments or a leading `.`/`/`. */
function looksLikeRepoId(s: string): boolean {
  if (s.startsWith(".") || s.startsWith("/") || s.startsWith("~")) return false;
  return s.split("/").length === 2 && !s.endsWith(".gguf");
}

/**
 * Resolve `source` (a Hub repo id, `repo:path.gguf`, or local `.gguf`) to a local
 * GGUF path, downloading if necessary. Returns the absolute file path.
 */
export async function resolveGguf(
  source: string,
  opts: ResolveGgufOptions = {},
): Promise<string> {
  if (isLocalGguf(source)) return source;

  // Allow "owner/name:file.gguf" to pin an exact file.
  let repoId = source;
  let explicitFile: string | undefined;
  const colon = source.indexOf(":");
  if (colon > 0 && source.slice(colon + 1).toLowerCase().endsWith(".gguf")) {
    repoId = source.slice(0, colon);
    explicitFile = source.slice(colon + 1);
  }

  const revision = opts.revision ?? "main";
  const accessToken = opts.accessToken ?? hfToken();
  const repo = { type: "model", name: repoId } as const;

  const file = explicitFile ??
    await pickGgufFile(repo, revision, accessToken, opts.quant);

  const onProgress = opts.onProgress;
  const download = (path: string) =>
    downloadFileToCacheDir({
      repo,
      path,
      revision,
      accessToken,
      fetch: onProgress
        ? progressFetch((received, total) => onProgress({ file: path, received, total }))
        : undefined,
    });

  // Sharded models (`…-00001-of-00003.gguf`) need every shard alongside the first
  // for llama.cpp to load them; download the whole set and return shard 1.
  const shard = file.match(/^(.*)-(\d+)-of-(\d+)\.gguf$/i);
  if (shard) {
    const [, prefix, idx, totalStr] = shard;
    const total = parseInt(totalStr, 10);
    const width = idx.length;
    let firstPath = "";
    for (let i = 1; i <= total; i++) {
      const name = `${prefix}-${String(i).padStart(width, "0")}-of-${totalStr}.gguf`;
      const p = await download(name);
      if (!p) {
        throw new Error(`Failed to download shard ${name} of ${repoId}@${revision}`);
      }
      if (i === 1) firstPath = p;
    }
    return firstPath;
  }

  const path = await download(file);
  if (!path) throw new Error(`Failed to download ${repoId}:${file}@${revision}`);
  return path;
}

/** List GGUF files in a repo and choose one by quant preference. */
async function pickGgufFile(
  repo: { type: "model"; name: string },
  revision: string,
  accessToken: string | undefined,
  quant?: string,
): Promise<string> {
  const ggufs: string[] = [];
  for await (const f of listFiles({ repo, revision, accessToken, recursive: true })) {
    if (f.type === "file" && f.path.toLowerCase().endsWith(".gguf")) ggufs.push(f.path);
  }
  if (ggufs.length === 0) {
    throw new Error(`No .gguf files found in ${repo.name}@${revision}`);
  }

  // Prefer the first shard of any sharded model so llama.cpp can load the set.
  const firstShards = ggufs.filter((p) => /-0*1-of-\d+\.gguf$/i.test(p));
  const shardBase = (p: string) => p.replace(/-0*\d+-of-\d+\.gguf$/i, "");
  const isShard = (p: string) => /-0*\d+-of-\d+\.gguf$/i.test(p);
  const candidates = ggufs.filter((p) => !isShard(p)).concat(firstShards);

  if (quant) {
    const q = quant.toLowerCase();
    const match = candidates.find((p) => p.toLowerCase().includes(q));
    if (match) return match;
    throw new Error(
      `No GGUF matching quant "${quant}" in ${repo.name}. Available:\n  ` +
        [...new Set(candidates.map((p) => isShard(p) ? shardBase(p) : p))].join("\n  "),
    );
  }

  if (candidates.length === 1) return candidates[0];
  throw new Error(
    `Multiple GGUF files in ${repo.name}; specify a quant (e.g. { quant: "Q4_K_M" }) ` +
      `or use "${repo.name}:<file>.gguf". Available:\n  ` +
      candidates.join("\n  "),
  );
}

function hfToken(): string | undefined {
  return Deno.env.get("HF_TOKEN") ?? Deno.env.get("HUGGING_FACE_HUB_TOKEN") ?? undefined;
}

/** Wrap fetch to report streamed byte progress on the response body. */
function progressFetch(
  onProgress: (received: number, total?: number) => void,
): typeof fetch {
  return async (input: string | URL | Request, init?: RequestInit) => {
    const res = await fetch(input, init);
    if (!res.body) return res;
    const total = Number(res.headers.get("content-length")) || undefined;
    // Only report on sizeable payloads (the model blob), not metadata calls.
    if (!total || total < 1_000_000) return res;

    let received = 0;
    const reader = res.body.getReader();
    const stream = new ReadableStream<Uint8Array>({
      async pull(controller) {
        const { done, value } = await reader.read();
        if (done) {
          controller.close();
          return;
        }
        received += value.length;
        onProgress(received, total);
        controller.enqueue(value);
      },
      cancel(reason) {
        reader.cancel(reason);
      },
    });
    return new Response(stream, {
      status: res.status,
      statusText: res.statusText,
      headers: res.headers,
    });
  };
}
