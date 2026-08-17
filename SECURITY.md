# Security

## Reporting

Please report suspected vulnerabilities privately via GitHub Security Advisories ("Report
a vulnerability") rather than a public issue. We aim to acknowledge within a few days.

## Trust boundaries

- **Native library.** deno-llama downloads the official llama.cpp macOS arm64 release for
  the pinned tag and verifies it against a SHA-256 committed in
  `packages/core/generated/meta.ts` before `dlopen`. A checksum mismatch is a hard error.
  To use your own build, set `DENO_LLAMA_LIB_PATH` (that path is trusted as-is — point it
  only at a library you trust).
- **Models.** GGUF files are downloaded from the Hugging Face Hub into the shared HF
  cache. Model weights are executed by llama.cpp; only load models from sources you trust.
- **FFI permissions.** Inference requires `--allow-ffi`, plus `--allow-read`,
  `--allow-env`, `--allow-sys=homedir`, and (for first-run downloads) `--allow-net` /
  `--allow-write`. The bundled `-P=llama` permission profile grants exactly this set. FFI
  is inherently capable of arbitrary native code; grant it only to code you trust.

## Supported versions

Security fixes target the latest pinned release. Older pins are not maintained; rebase to
the current tag.
