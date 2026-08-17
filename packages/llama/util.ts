/**
 * Small FFI helpers shared across the ergonomic layer: C-string marshalling and
 * pointer reading. Buffers returned by {@link cString} must be kept alive by the
 * caller for the duration of the FFI call that uses their pointer.
 */

const encoder = new TextEncoder();

/** Encode `s` as a NUL-terminated UTF-8 buffer for passing as `const char *`. */
export function cString(s: string): Uint8Array {
  const bytes = encoder.encode(s);
  const buf = new Uint8Array(bytes.length + 1);
  buf.set(bytes);
  buf[bytes.length] = 0;
  return buf;
}

/** A pointer to a JS-owned buffer (kept alive by the caller). */
export function ptr(buf: ArrayBufferView): Deno.PointerValue {
  return Deno.UnsafePointer.of(buf);
}

/** Read a NUL-terminated C string at `p`, or "" if null. */
export function readCString(p: Deno.PointerValue): string {
  return p ? new Deno.UnsafePointerView(p).getCString() : "";
}

/** Numeric value of a pointer (0 for null), for writing into struct fields. */
export function ptrValue(p: Deno.PointerValue): bigint {
  return p === null ? 0n : Deno.UnsafePointer.value(p);
}
