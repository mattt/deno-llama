/**
 * Read and write fields of by-value C structs held as raw bytes.
 *
 * llama.cpp's `*_default_params()` return a struct by value (Deno FFI hands it
 * back as a `Uint8Array`); we poke the handful of fields we override (n_ctx,
 * n_gpu_layers, …) into those bytes and pass them straight back to
 * `llama_init_from_model` / `llama_model_load_from_file`. Offsets come from the
 * generated {@link structLayouts} (computed with the platform C ABI).
 */

import { type StructLayout, structLayouts } from "./generated/types.ts";

const LE = true; // arm64 is little-endian

/** A byte-backed view over a by-value C struct with named field access. */
export class Struct {
  readonly bytes: Uint8Array;
  readonly layout: StructLayout;
  #view: DataView;

  constructor(layout: StructLayout, bytes?: Uint8Array) {
    this.layout = layout;
    this.bytes = bytes ?? new Uint8Array(layout.size);
    this.#view = new DataView(
      this.bytes.buffer,
      this.bytes.byteOffset,
      this.bytes.byteLength,
    );
  }

  /** Wrap the bytes returned by a `*_default_params()` FFI call. */
  static from(structName: string, bytes: Uint8Array): Struct {
    return new Struct(layoutOf(structName), bytes);
  }

  /** Read a scalar field as a JS number (bool -> 0/1, 64-bit -> Number). */
  get(name: string): number {
    const f = this.field(name);
    const o = f.offset;
    switch (f.ffi) {
      case "bool":
      case "u8":
        return this.#view.getUint8(o);
      case "i8":
        return this.#view.getInt8(o);
      case "u16":
        return this.#view.getUint16(o, LE);
      case "i16":
        return this.#view.getInt16(o, LE);
      case "u32":
        return this.#view.getUint32(o, LE);
      case "i32":
        return this.#view.getInt32(o, LE);
      case "f32":
        return this.#view.getFloat32(o, LE);
      case "f64":
        return this.#view.getFloat64(o, LE);
      case "u64":
      case "usize":
        return Number(this.#view.getBigUint64(o, LE));
      case "i64":
      case "isize":
        return Number(this.#view.getBigInt64(o, LE));
      default:
        throw new Error(`get(${name}): unsupported field type ${String(f.ffi)}`);
    }
  }

  /** Write a scalar field (number | boolean | bigint | pointer). */
  set(name: string, value: number | boolean | bigint | Deno.PointerValue): this {
    const f = this.field(name);
    const o = f.offset;
    switch (f.ffi) {
      case "bool":
        this.#view.setUint8(o, value ? 1 : 0);
        break;
      case "u8":
        this.#view.setUint8(o, Number(value));
        break;
      case "i8":
        this.#view.setInt8(o, Number(value));
        break;
      case "u16":
        this.#view.setUint16(o, Number(value), LE);
        break;
      case "i16":
        this.#view.setInt16(o, Number(value), LE);
        break;
      case "u32":
        this.#view.setUint32(o, Number(value), LE);
        break;
      case "i32":
        this.#view.setInt32(o, Number(value), LE);
        break;
      case "f32":
        this.#view.setFloat32(o, Number(value), LE);
        break;
      case "f64":
        this.#view.setFloat64(o, Number(value), LE);
        break;
      case "u64":
      case "usize":
      case "pointer":
        this.#view.setBigUint64(o, toBig(value), LE);
        break;
      case "i64":
      case "isize":
        this.#view.setBigInt64(o, toBig(value), LE);
        break;
      default:
        throw new Error(`set(${name}): unsupported field type ${String(f.ffi)}`);
    }
    return this;
  }

  private field(name: string) {
    const f = this.layout.fields[name];
    if (!f) throw new Error(`unknown field "${name}" in struct`);
    return f;
  }
}

/** Look up a generated struct layout by C name. */
export function layoutOf(structName: string): StructLayout {
  const layout = structLayouts[structName];
  if (!layout) throw new Error(`no generated layout for struct ${structName}`);
  return layout;
}

function toBig(value: number | boolean | bigint | Deno.PointerValue): bigint {
  if (typeof value === "bigint") return value;
  if (typeof value === "number") return BigInt(value);
  if (typeof value === "boolean") return value ? 1n : 0n;
  // Deno.PointerValue
  return value === null ? 0n : Deno.UnsafePointer.value(value);
}
