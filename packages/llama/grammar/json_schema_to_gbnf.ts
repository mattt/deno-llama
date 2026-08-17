/**
 * JSON Schema to GBNF grammar converter.
 *
 * Ported faithfully to typed TypeScript from llama.cpp's
 * `json_schema_to_grammar` reference implementation, pinned to tag `b10344`:
 * https://github.com/ggml-org/llama.cpp/blob/b10344/examples/json_schema_to_grammar.py
 *
 * The generated rule names and grammar output are kept identical to upstream so
 * behavior is faithful. Remote (`https://`) `$ref` fetching is intentionally not
 * implemented: it is disabled by default and throws a clear error when required
 * (see `allowFetch`).
 *
 * @module
 */

/** A built-in GBNF rule with its transitive rule dependencies. */
interface BuiltinRule {
  content: string;
  deps: string[];
}

const b = (content: string, deps: string[] = []): BuiltinRule => ({ content, deps });

// Constraining spaces to prevent model "running away".
const SPACE_RULE = String.raw`| " " | "\n"{1,2} [ \t]{0,20}`;

const PRIMITIVE_RULES: Record<string, BuiltinRule> = {
  "boolean": b('("true" | "false")'),
  "decimal-part": b("[0-9]{1,16}"),
  "integral-part": b("[0] | [1-9] [0-9]{0,15}"),
  "number": b(
    '("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?',
    ["integral-part", "decimal-part"],
  ),
  "integer": b('("-"? integral-part)', ["integral-part"]),
  "value": b("object | array | string | number | boolean | null", [
    "object",
    "array",
    "string",
    "number",
    "boolean",
    "null",
  ]),
  "object": b(
    '"{" space ( string ":" space value ("," space string ":" space value)* )? space "}"',
    ["string", "value"],
  ),
  "array": b('"[" space ( value ("," space value)* )? space "]"', ["value"]),
  "uuid": b(
    String
      .raw`"\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\""`,
  ),
  "char": b(
    String.raw`[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})`,
  ),
  "string": b(String.raw`"\"" char* "\""`, ["char"]),
  "null": b('"null"'),
};

// TODO: support "uri", "email" string formats
const STRING_FORMAT_RULES: Record<string, BuiltinRule> = {
  "date": b(
    '[0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )',
  ),
  "time": b(
    '([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )',
  ),
  "date-time": b('date "T" time', ["date", "time"]),
  "date-string": b(String.raw`"\"" date "\""`, ["date"]),
  "time-string": b(String.raw`"\"" time "\""`, ["time"]),
  "date-time-string": b(String.raw`"\"" date-time "\""`, ["date-time"]),
};

const DOTALL = String.raw`[\U00000000-\U0010FFFF]`;
const DOT = String.raw`[^\x0A\x0D]`;

const RESERVED_NAMES = new Set<string>([
  "root",
  "dot",
  ...Object.keys(PRIMITIVE_RULES),
  ...Object.keys(STRING_FORMAT_RULES),
]);

const INVALID_RULE_CHARS_RE = /[^a-zA-Z0-9-]+/g;
const GRAMMAR_LITERAL_ESCAPE_RE = /[\r\n"\\]/g;
const GRAMMAR_LITERAL_ESCAPES: Record<string, string> = {
  "\r": "\\r",
  "\n": "\\n",
  '"': '\\"',
  "\\": "\\\\",
};

const NON_LITERAL_SET = new Set("|.()[]{}*+?");
const ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = new Set("^$.[]()|{}*+?");

const MAX_SIZE = Number.MAX_SAFE_INTEGER;

/** Options controlling grammar generation. */
export interface JsonSchemaToGrammarOptions {
  /**
   * Property names defining the order of precedence for object properties.
   * Properties not listed keep their original schema order after listed ones.
   */
  propOrder?: string[];
  /**
   * Whether to allow fetching remote (`https://`) `$ref` schemas. Fetching is
   * not implemented in this port, so a remote `$ref` always throws; this flag
   * only changes the error message.
   */
  allowFetch?: boolean;
  /** Treat `.` in regex patterns as matching all chars including line breaks. */
  dotall?: boolean;
}

type SchemaObject = Record<string, unknown>;

function isObject(value: unknown): value is SchemaObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Mimics Python's `json.dumps` (default separators `", "` / `": "`,
 * `ensure_ascii=True`) so `const`/`enum` output matches upstream exactly.
 */
function pyJsonDumps(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return JSON.stringify(value);
  if (typeof value === "string") return pyJsonString(value);
  if (Array.isArray(value)) {
    return "[" + value.map((v) => pyJsonDumps(v)).join(", ") + "]";
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    return "{" +
      entries.map(([k, v]) => pyJsonString(k) + ": " + pyJsonDumps(v)).join(", ") +
      "}";
  }
  return JSON.stringify(value);
}

function pyJsonString(s: string): string {
  let out = '"';
  for (let idx = 0; idx < s.length; idx++) {
    const ch = s[idx];
    const code = s.charCodeAt(idx);
    switch (ch) {
      case '"':
        out += '\\"';
        break;
      case "\\":
        out += "\\\\";
        break;
      case "\n":
        out += "\\n";
        break;
      case "\r":
        out += "\\r";
        break;
      case "\t":
        out += "\\t";
        break;
      case "\b":
        out += "\\b";
        break;
      case "\f":
        out += "\\f";
        break;
      default:
        if (code < 0x20 || code > 0x7f) {
          out += "\\u" + code.toString(16).padStart(4, "0");
        } else {
          out += ch;
        }
    }
  }
  return out + '"';
}

function buildRepetition(
  itemRule: string,
  minItems: number,
  maxItems: number | undefined,
  separatorRule?: string,
): string {
  if (maxItems === 0) return "";

  if (minItems === 0 && maxItems === 1) return `${itemRule}?`;

  if (!separatorRule) {
    if (minItems === 1 && maxItems === undefined) return `${itemRule}+`;
    else if (minItems === 0 && maxItems === undefined) return `${itemRule}*`;
    else return `${itemRule}{${minItems},${maxItems ?? ""}}`;
  }

  const result = itemRule + " " +
    buildRepetition(
      `(${separatorRule} ${itemRule})`,
      minItems > 0 ? minItems - 1 : 0,
      maxItems === undefined ? undefined : maxItems - 1,
    );
  return minItems === 0 ? `(${result})?` : result;
}

function generateMinMaxInt(
  minValue: number | null,
  maxValue: number | null,
  out: string[],
  decimalsLeft = 16,
  topLevel = true,
): void {
  const digitRange = (fromChar: string, toChar: string): void => {
    out.push("[");
    if (fromChar === toChar) {
      out.push(fromChar);
    } else {
      out.push(fromChar);
      out.push("-");
      out.push(toChar);
    }
    out.push("]");
  };

  const moreDigits = (minDigits: number, maxDigits: number): void => {
    out.push("[0-9]");
    if (minDigits === maxDigits && minDigits === 1) return;
    out.push("{");
    out.push(String(minDigits));
    if (maxDigits !== minDigits) {
      out.push(",");
      if (maxDigits !== MAX_SIZE) out.push(String(maxDigits));
    }
    out.push("}");
  };

  const uniformRange = (fromStr: string, toStr: string): void => {
    let i = 0;
    while (i < fromStr.length && fromStr[i] === toStr[i]) i++;
    if (i > 0) {
      out.push('"');
      out.push(fromStr.slice(0, i));
      out.push('"');
    }
    if (i < fromStr.length) {
      if (i > 0) out.push(" ");
      const subLen = fromStr.length - i - 1;
      if (subLen > 0) {
        const fromSub = fromStr.slice(i + 1);
        const toSub = toStr.slice(i + 1);
        const subZeros = "0".repeat(subLen);
        const subNines = "9".repeat(subLen);

        let toReached = false;
        out.push("(");
        if (fromSub === subZeros) {
          digitRange(fromStr[i], String.fromCharCode(toStr.charCodeAt(i) - 1));
          out.push(" ");
          moreDigits(subLen, subLen);
        } else {
          out.push("[");
          out.push(fromStr[i]);
          out.push("] ");
          out.push("(");
          uniformRange(fromSub, subNines);
          out.push(")");
          if (fromStr.charCodeAt(i) < toStr.charCodeAt(i) - 1) {
            out.push(" | ");
            if (toSub === subNines) {
              digitRange(String.fromCharCode(fromStr.charCodeAt(i) + 1), toStr[i]);
              toReached = true;
            } else {
              digitRange(
                String.fromCharCode(fromStr.charCodeAt(i) + 1),
                String.fromCharCode(toStr.charCodeAt(i) - 1),
              );
            }
            out.push(" ");
            moreDigits(subLen, subLen);
          }
        }
        if (!toReached) {
          out.push(" | ");
          digitRange(toStr[i], toStr[i]);
          out.push(" ");
          uniformRange(subZeros, toSub);
        }
        out.push(")");
      } else {
        out.push("[");
        out.push(fromStr[i]);
        out.push("-");
        out.push(toStr[i]);
        out.push("]");
      }
    }
  };

  if (minValue !== null && maxValue !== null) {
    if (minValue < 0 && maxValue < 0) {
      out.push('"-" (');
      generateMinMaxInt(-maxValue, -minValue, out, decimalsLeft, true);
      out.push(")");
      return;
    }

    if (minValue < 0) {
      out.push('"-" (');
      generateMinMaxInt(0, -minValue, out, decimalsLeft, true);
      out.push(") | ");
      minValue = 0;
    }

    let minS = String(minValue);
    const maxS = String(maxValue);
    const minDigits = minS.length;
    const maxDigits = maxS.length;

    for (let digits = minDigits; digits < maxDigits; digits++) {
      uniformRange(minS, "9".repeat(digits));
      minS = "1" + "0".repeat(digits);
      out.push(" | ");
    }
    uniformRange(minS, maxS);
    return;
  }

  const lessDecimals = Math.max(decimalsLeft - 1, 1);

  if (minValue !== null) {
    if (minValue < 0) {
      out.push('"-" (');
      generateMinMaxInt(null, -minValue, out, decimalsLeft, false);
      out.push(") | [0] | [1-9] ");
      moreDigits(0, decimalsLeft - 1);
    } else if (minValue === 0) {
      if (topLevel) {
        out.push("[0] | [1-9] ");
        moreDigits(0, lessDecimals);
      } else {
        moreDigits(1, decimalsLeft);
      }
    } else if (minValue <= 9) {
      const c = String(minValue);
      const rangeStart = topLevel ? "1" : "0";
      if (c > rangeStart) {
        digitRange(rangeStart, String.fromCharCode(c.charCodeAt(0) - 1));
        out.push(" ");
        moreDigits(1, lessDecimals);
        out.push(" | ");
      }
      digitRange(c, "9");
      out.push(" ");
      moreDigits(0, lessDecimals);
    } else {
      const minS = String(minValue);
      const length = minS.length;
      const c = minS[0];

      if (c > "1") {
        digitRange(topLevel ? "1" : "0", String.fromCharCode(c.charCodeAt(0) - 1));
        out.push(" ");
        moreDigits(length, lessDecimals);
        out.push(" | ");
      }
      digitRange(c, c);
      out.push(" (");
      generateMinMaxInt(parseInt(minS.slice(1), 10), null, out, lessDecimals, false);
      out.push(")");
      if (c < "9") {
        out.push(" | ");
        digitRange(String.fromCharCode(c.charCodeAt(0) + 1), "9");
        out.push(" ");
        moreDigits(length - 1, lessDecimals);
      }
    }
    return;
  }

  if (maxValue !== null) {
    if (maxValue >= 0) {
      if (topLevel) {
        out.push('"-" [1-9] ');
        moreDigits(0, lessDecimals);
        out.push(" | ");
      }
      generateMinMaxInt(0, maxValue, out, decimalsLeft, true);
    } else {
      out.push('"-" (');
      generateMinMaxInt(-maxValue, null, out, decimalsLeft, false);
      out.push(")");
    }
    return;
  }

  throw new Error("At least one of minValue or maxValue must be set");
}

interface TrieNode {
  children: Map<string, TrieNode>;
  isEndOfString: boolean;
}

class SchemaConverter {
  private readonly propOrder: Map<string, number>;
  private readonly allowFetch: boolean;
  private readonly dotall: boolean;
  private readonly rawPattern: boolean;
  private readonly rules: Record<string, string>;
  private readonly refs: Map<string, unknown>;
  private readonly refsBeingResolved: Set<string>;

  constructor(opts: {
    propOrder: Map<string, number>;
    allowFetch: boolean;
    dotall: boolean;
    rawPattern: boolean;
  }) {
    this.propOrder = opts.propOrder;
    this.allowFetch = opts.allowFetch;
    this.dotall = opts.dotall;
    this.rawPattern = opts.rawPattern;
    this.rules = { "space": SPACE_RULE };
    this.refs = new Map();
    this.refsBeingResolved = new Set();
  }

  private formatLiteral(literal: string): string {
    const escaped = literal.replace(
      GRAMMAR_LITERAL_ESCAPE_RE,
      (m) => GRAMMAR_LITERAL_ESCAPES[m] ?? m,
    );
    return `"${escaped}"`;
  }

  private notStrings(strings: string[]): string {
    const makeNode = (): TrieNode => ({
      children: new Map(),
      isEndOfString: false,
    });
    const trie = makeNode();
    for (const s of strings) {
      let node = trie;
      for (const c of s) {
        let child = node.children.get(c);
        if (child === undefined) {
          child = makeNode();
          node.children.set(c, child);
        }
        node = child;
      }
      node.isEndOfString = true;
    }

    const charRule = this.addPrimitive("char", PRIMITIVE_RULES["char"]);
    const out: string[] = ['["] ( '];

    const visitNode = (node: TrieNode): void => {
      const rejects: string[] = [];
      let first = true;
      for (const c of [...node.children.keys()].sort()) {
        const child = node.children.get(c) as TrieNode;
        rejects.push(c);
        if (first) first = false;
        else out.push(" | ");
        out.push(`[${c}]`);
        if (child.children.size > 0) {
          out.push(" (");
          visitNode(child);
          out.push(")");
        } else if (child.isEndOfString) {
          out.push(` ${charRule}+`);
        }
      }
      if (node.children.size > 0) {
        if (!first) out.push(" | ");
        out.push(`[^"${rejects.join("")}] ${charRule}*`);
      }
    };
    visitNode(trie);

    out.push(` )${trie.isEndOfString ? "" : "?"} ["]`);
    return out.join("");
  }

  private addRule(name: string, rule: string): string {
    const escName = name.replace(INVALID_RULE_CHARS_RE, "-");
    let key: string;
    if (!(escName in this.rules) || this.rules[escName] === rule) {
      key = escName;
    } else {
      let i = 0;
      while (`${escName}${i}` in this.rules && this.rules[`${escName}${i}`] !== rule) {
        i++;
      }
      key = `${escName}${i}`;
    }
    this.rules[key] = rule;
    return key;
  }

  /**
   * Resolves all `$ref` fields in the schema, replacing local refs with
   * absolute references and populating `this.refs` with the referenced
   * subschemas. Remote (`https://`) refs are not fetched.
   */
  resolveRefs(schema: unknown, url: string): unknown {
    const visit = (n: unknown): unknown => {
      if (Array.isArray(n)) {
        for (const x of n) visit(x);
        return n;
      }
      if (!isObject(n)) return n;

      const rawRef = n["$ref"];
      if (rawRef !== undefined && rawRef !== null && !this.refs.has(String(rawRef))) {
        let ref = String(rawRef);
        let target: unknown;
        if (ref.startsWith("https://")) {
          if (!this.allowFetch) {
            throw new Error(
              "Fetching remote schemas is not allowed (enable allowFetch to force)",
            );
          }
          throw new Error(
            "Remote schema fetching over HTTPS is not implemented in this port",
          );
        } else if (ref.startsWith("#/")) {
          target = schema;
          ref = `${url}${ref}`;
          n["$ref"] = ref;
        } else {
          throw new Error(`Unsupported ref ${ref}`);
        }

        const selectors = (ref.split("#").pop() as string).split("/").slice(1);
        for (const sel of selectors) {
          if (target === undefined || target === null) {
            throw new Error(`Error resolving ref ${ref}: ${sel} not found`);
          }
          if (Array.isArray(target)) {
            const selIndex = Number(sel);
            if (
              !Number.isInteger(selIndex) || selIndex < 0 || selIndex >= target.length
            ) {
              throw new Error(`Error resolving ref ${ref}: ${sel} not in array`);
            }
            target = target[selIndex];
          } else if (isObject(target)) {
            if (!(sel in target)) {
              throw new Error(`Error resolving ref ${ref}: ${sel} not found`);
            }
            target = target[sel];
          } else {
            throw new Error(`Error resolving ref ${ref}: ${sel} not found`);
          }
        }

        this.refs.set(ref, target);
      } else {
        for (const v of Object.values(n)) visit(v);
      }
      return n;
    };
    return visit(schema);
  }

  private generateUnionRule(name: string, altSchemas: unknown[]): string {
    return altSchemas
      .map((alt, i) =>
        this.visit(alt as SchemaObject, `${name}${name ? "-" : "alternative-"}${i}`)
      )
      .join(" | ");
  }

  private visitPattern(pattern: string, name: string): string {
    if (!(pattern.startsWith("^") && pattern.endsWith("$"))) {
      throw new Error('Pattern must start with "^" and end with "$"');
    }
    pattern = pattern.slice(1, -1);
    const subRuleIds: Record<string, string> = {};

    let i = 0;
    const length = pattern.length;

    const toRule = (s: [string, boolean]): string => s[1] ? '"' + s[0] + '"' : s[0];

    const transform = (): [string, boolean] => {
      const start = i;
      const seq: Array<[string, boolean]> = [];

      const getDot = (): string => {
        const rule = this.dotall ? DOTALL : DOT;
        return this.addRule("dot", rule);
      };

      const joinSeq = (): [string, boolean] => {
        const ret: Array<[string, boolean]> = [];
        let gi = 0;
        while (gi < seq.length) {
          const isLiteral = seq[gi][1];
          let gj = gi;
          const group: Array<[string, boolean]> = [];
          while (gj < seq.length && seq[gj][1] === isLiteral) {
            group.push(seq[gj]);
            gj++;
          }
          if (isLiteral) ret.push([group.map((x) => x[0]).join(""), true]);
          else ret.push(...group);
          gi = gj;
        }
        if (ret.length === 1) return ret[0];
        return [seq.map((x) => toRule(x)).join(" "), false];
      };

      while (i < length) {
        const c = pattern[i];
        if (c === ".") {
          seq.push([getDot(), false]);
          i++;
        } else if (c === "(") {
          i++;
          if (i < length && pattern[i] === "?") {
            throw new Error(
              `Unsupported pattern syntax "${pattern[i]}" at index ${i} of /${pattern}/`,
            );
          }
          seq.push([`(${toRule(transform())})`, false]);
        } else if (c === ")") {
          i++;
          if (!(start > 0 && pattern[start - 1] === "(")) {
            throw new Error(
              `Unbalanced parentheses; start = ${start}, i = ${i}, pattern = ${pattern}`,
            );
          }
          return joinSeq();
        } else if (c === "[") {
          let squareBrackets = c;
          i++;
          while (i < length && pattern[i] !== "]") {
            if (pattern[i] === "\\") {
              squareBrackets += pattern.slice(i, i + 2);
              i += 2;
            } else {
              squareBrackets += pattern[i];
              i++;
            }
          }
          if (i >= length) {
            throw new Error(
              `Unbalanced square brackets; start = ${start}, i = ${i}, pattern = ${pattern}`,
            );
          }
          squareBrackets += "]";
          i++;
          seq.push([squareBrackets, false]);
        } else if (c === "|") {
          seq.push(["|", false]);
          i++;
        } else if (c === "*" || c === "+" || c === "?") {
          seq[seq.length - 1] = [toRule(seq[seq.length - 1]) + c, false];
          i++;
        } else if (c === "{") {
          let curlyBrackets = c;
          i++;
          while (i < length && pattern[i] !== "}") {
            curlyBrackets += pattern[i];
            i++;
          }
          if (i >= length) {
            throw new Error(
              `Unbalanced curly brackets; start = ${start}, i = ${i}, pattern = ${pattern}`,
            );
          }
          curlyBrackets += "}";
          i++;
          const nums = curlyBrackets.slice(1, -1).split(",").map((s) => s.trim());
          let minTimes = 0;
          let maxTimes: number | undefined = undefined;
          const parseNum = (s: string): number => {
            const v = parseInt(s, 10);
            if (Number.isNaN(v)) {
              throw new Error(`Invalid quantifier ${curlyBrackets} in /${pattern}/`);
            }
            return v;
          };
          if (nums.length === 1) {
            minTimes = parseNum(nums[0]);
            maxTimes = minTimes;
          } else {
            if (nums.length !== 2) {
              throw new Error(`Invalid quantifier ${curlyBrackets} in /${pattern}/`);
            }
            minTimes = nums[0] ? parseNum(nums[0]) : 0;
            maxTimes = nums[1] ? parseNum(nums[1]) : undefined;
          }

          let [sub, subIsLiteral] = seq[seq.length - 1];

          if (!subIsLiteral) {
            let id = subRuleIds[sub];
            if (id === undefined) {
              id = this.addRule(`${name}-${Object.keys(subRuleIds).length + 1}`, sub);
              subRuleIds[sub] = id;
            }
            sub = id;
          }

          seq[seq.length - 1] = [
            buildRepetition(subIsLiteral ? `"${sub}"` : sub, minTimes, maxTimes),
            false,
          ];
        } else {
          let literal = "";
          while (i < length) {
            if (pattern[i] === "\\" && i < length - 1) {
              const next = pattern[i + 1];
              if (ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.has(next)) {
                i++;
                literal += pattern[i];
                i++;
              } else {
                literal += pattern.slice(i, i + 2);
                i += 2;
              }
            } else if (pattern[i] === '"' && !this.rawPattern) {
              literal += '\\"';
              i++;
            } else if (
              !NON_LITERAL_SET.has(pattern[i]) &&
              (i === length - 1 || literal === "" || pattern[i + 1] === "." ||
                !NON_LITERAL_SET.has(pattern[i + 1]))
            ) {
              literal += pattern[i];
              i++;
            } else {
              break;
            }
          }
          if (literal) seq.push([literal, true]);
        }
      }

      return joinSeq();
    };

    return this.addRule(
      name,
      this.rawPattern ? toRule(transform()) : '"\\"" (' + toRule(transform()) + ') "\\""',
    );
  }

  private resolveRef(ref: string): string {
    const refFragment = ref.split("#").pop() as string;
    let refName = "ref" + refFragment.replace(/[^a-zA-Z0-9-]+/g, "-");
    if (!(refName in this.rules) && !this.refsBeingResolved.has(ref)) {
      this.refsBeingResolved.add(ref);
      const resolved = this.refs.get(ref);
      refName = this.visit(resolved as SchemaObject, refName);
      this.refsBeingResolved.delete(ref);
    }
    return refName;
  }

  private generateConstantRule(value: unknown): string {
    return this.formatLiteral(pyJsonDumps(value));
  }

  visit(schema: SchemaObject, name: string): string {
    const schemaType = schema["type"];
    const schemaFormat = schema["format"];
    const ruleName = RESERVED_NAMES.has(name) ? name + "-" : (name || "root");

    const ref = schema["$ref"];
    if (ref !== undefined && ref !== null) {
      return this.addRule(ruleName, this.resolveRef(String(ref)));
    }

    if ("oneOf" in schema || "anyOf" in schema) {
      let alts = schema["oneOf"];
      if (!alts || (Array.isArray(alts) && alts.length === 0)) alts = schema["anyOf"];
      return this.addRule(
        ruleName,
        this.generateUnionRule(name, alts as unknown[]),
      );
    }

    if (Array.isArray(schemaType)) {
      return this.addRule(
        ruleName,
        this.generateUnionRule(
          name,
          schemaType.map((t) => ({ ...schema, type: t })),
        ),
      );
    }

    if ("const" in schema) {
      return this.addRule(ruleName, this.generateConstantRule(schema["const"]));
    }

    if ("enum" in schema) {
      const rule = "(" +
        (schema["enum"] as unknown[]).map((v) => this.generateConstantRule(v)).join(
          " | ",
        ) +
        ")";
      return this.addRule(ruleName, rule);
    }

    if (
      (schemaType === undefined || schemaType === "object") &&
      ("properties" in schema ||
        ("additionalProperties" in schema && schema["additionalProperties"] !== true))
    ) {
      const required = new Set<string>(
        Array.isArray(schema["required"]) ? schema["required"] as string[] : [],
      );
      const properties: Array<[string, unknown]> = isObject(schema["properties"])
        ? Object.entries(schema["properties"])
        : [];
      return this.addRule(
        ruleName,
        this.buildObjectRule(properties, required, name, schema["additionalProperties"]),
      );
    }

    if (
      (schemaType === undefined || schemaType === "object" ||
        schemaType === "string") && "allOf" in schema
    ) {
      const required = new Set<string>();
      const properties: Array<[string, unknown]> = [];
      const enumSets: Array<Set<unknown>> = [];
      const hybridName = name;

      const addComponent = (compSchema: unknown, isRequired: boolean): void => {
        let cs = compSchema;
        if (isObject(cs) && cs["$ref"] !== undefined && cs["$ref"] !== null) {
          cs = this.refs.get(String(cs["$ref"]));
        }
        if (isObject(cs) && isObject(cs["properties"])) {
          for (const [propName, propSchema] of Object.entries(cs["properties"])) {
            properties.push([propName, propSchema]);
            if (isRequired) required.add(propName);
          }
        }
        if (isObject(cs) && "enum" in cs) {
          enumSets.push(new Set(cs["enum"] as unknown[]));
        }
      };

      for (const t of schema["allOf"] as unknown[]) {
        if (isObject(t) && "anyOf" in t) {
          for (const tt of t["anyOf"] as unknown[]) addComponent(tt, false);
        } else {
          addComponent(t, true);
        }
      }

      if (enumSets.length > 0) {
        let intersection = enumSets[0];
        for (let k = 1; k < enumSets.length; k++) {
          const s = enumSets[k];
          intersection = new Set([...intersection].filter((x) => s.has(x)));
        }
        if (intersection.size > 0) {
          const rule = "(" +
            [...intersection].sort().map((v) => this.generateConstantRule(v)).join(
              " | ",
            ) +
            ")";
          return this.addRule(ruleName, rule);
        }
      }

      return this.addRule(
        ruleName,
        this.buildObjectRule(properties, required, hybridName, undefined),
      );
    }

    if (
      (schemaType === undefined || schemaType === "array") &&
      ("items" in schema || "prefixItems" in schema)
    ) {
      const items = "items" in schema ? schema["items"] : schema["prefixItems"];
      if (Array.isArray(items)) {
        return this.addRule(
          ruleName,
          '"[" space ' +
            items
              .map((item, i) =>
                this.visit(item as SchemaObject, `${name}${name ? "-" : ""}tuple-${i}`)
              )
              .join(' "," space ') +
            ' space "]"',
        );
      }
      const itemRuleName = this.visit(
        items as SchemaObject,
        `${name}${name ? "-" : ""}item`,
      );
      const minItems = (schema["minItems"] as number) ?? 0;
      const maxItems = schema["maxItems"] as number | undefined;
      return this.addRule(
        ruleName,
        '"[" space ' +
          buildRepetition(itemRuleName, minItems, maxItems, '"," space') +
          ' space "]"',
      );
    }

    if ((schemaType === undefined || schemaType === "string") && "pattern" in schema) {
      return this.visitPattern(schema["pattern"] as string, ruleName);
    }

    if (
      (schemaType === undefined || schemaType === "string") &&
      typeof schemaFormat === "string" && /^uuid[1-5]?$/.test(schemaFormat)
    ) {
      return this.addPrimitive(
        ruleName === "root" ? "root" : schemaFormat,
        PRIMITIVE_RULES["uuid"],
      );
    }

    if (
      (schemaType === undefined || schemaType === "string") &&
      `${schemaFormat}-string` in STRING_FORMAT_RULES
    ) {
      const primName = `${schemaFormat}-string`;
      return this.addRule(
        ruleName,
        this.addPrimitive(primName, STRING_FORMAT_RULES[primName]),
      );
    }

    if (
      schemaType === "string" && ("minLength" in schema || "maxLength" in schema)
    ) {
      const charRule = this.addPrimitive("char", PRIMITIVE_RULES["char"]);
      const minLen = (schema["minLength"] as number) ?? 0;
      const maxLen = schema["maxLength"] as number | undefined;
      return this.addRule(
        ruleName,
        `"\\"" ${buildRepetition(charRule, minLen, maxLen)} "\\""`,
      );
    }

    if (
      (schemaType === undefined || schemaType === "integer") &&
      ("minimum" in schema || "exclusiveMinimum" in schema ||
        "maximum" in schema || "exclusiveMaximum" in schema)
    ) {
      let minValue: number | null = null;
      let maxValue: number | null = null;
      if ("minimum" in schema) minValue = schema["minimum"] as number;
      else if ("exclusiveMinimum" in schema) {
        minValue = (schema["exclusiveMinimum"] as number) + 1;
      }
      if ("maximum" in schema) maxValue = schema["maximum"] as number;
      else if ("exclusiveMaximum" in schema) {
        maxValue = (schema["exclusiveMaximum"] as number) - 1;
      }

      const out: string[] = ["("];
      generateMinMaxInt(minValue, maxValue, out);
      out.push(")");
      return this.addRule(ruleName, out.join(""));
    }

    if (schemaType === "object" || Object.keys(schema).length === 0) {
      return this.addRule(
        ruleName,
        this.addPrimitive("object", PRIMITIVE_RULES["object"]),
      );
    }

    if (schemaType === undefined) {
      // No type constraint and no recognized structural keywords (e.g.
      // {"description": "..."}). Per JSON Schema semantics this accepts any value.
      return this.addRule(ruleName, this.addPrimitive("value", PRIMITIVE_RULES["value"]));
    }

    if (typeof schemaType !== "string" || !(schemaType in PRIMITIVE_RULES)) {
      throw new Error(`Unrecognized schema: ${JSON.stringify(schema)}`);
    }
    return this.addPrimitive(
      ruleName === "root" ? "root" : schemaType,
      PRIMITIVE_RULES[schemaType],
    );
  }

  private addPrimitive(name: string, rule: BuiltinRule): string {
    const n = this.addRule(name, rule.content);
    for (const dep of rule.deps) {
      const depRule = PRIMITIVE_RULES[dep] ?? STRING_FORMAT_RULES[dep];
      if (depRule === undefined) throw new Error(`Rule ${dep} not known`);
      if (!(dep in this.rules)) this.addPrimitive(dep, depRule);
    }
    return n;
  }

  private buildObjectRule(
    properties: Array<[string, unknown]>,
    required: Set<string>,
    name: string,
    additionalProperties: unknown,
  ): string {
    const propOrder = this.propOrder;
    const sortedProps = properties
      .map((kv, idx) => ({ key: kv[0], idx }))
      .sort((a, bItem) => {
        const oa = propOrder.has(a.key) ? propOrder.get(a.key) as number : propOrder.size;
        const ob = propOrder.has(bItem.key)
          ? propOrder.get(bItem.key) as number
          : propOrder.size;
        if (oa !== ob) return oa - ob;
        return a.idx - bItem.idx;
      })
      .map((x) => x.key);

    const propKvRuleNames: Record<string, string> = {};
    for (const [propName, propSchema] of properties) {
      const propRuleName = this.visit(
        propSchema as SchemaObject,
        `${name}${name ? "-" : ""}${propName}`,
      );
      propKvRuleNames[propName] = this.addRule(
        `${name}${name ? "-" : ""}${propName}-kv`,
        `${this.formatLiteral(pyJsonDumps(propName))} space ":" space ${propRuleName}`,
      );
    }
    const requiredProps = sortedProps.filter((k) => required.has(k));
    const optionalProps = sortedProps.filter((k) => !required.has(k));

    if (
      additionalProperties !== undefined && additionalProperties !== null &&
      additionalProperties !== false
    ) {
      const subName = `${name}${name ? "-" : ""}additional`;
      const valueRule = isObject(additionalProperties)
        ? this.visit(additionalProperties, `${subName}-value`)
        : this.addPrimitive("value", PRIMITIVE_RULES["value"]);
      const keyRule = sortedProps.length === 0
        ? this.addPrimitive("string", PRIMITIVE_RULES["string"])
        : this.addRule(`${subName}-k`, this.notStrings(sortedProps));

      propKvRuleNames["*"] = this.addRule(
        `${subName}-kv`,
        `${keyRule} ":" space ${valueRule}`,
      );
      optionalProps.push("*");
    }

    let rule = '"{" space ';
    rule += requiredProps.map((k) => propKvRuleNames[k]).join(' "," space ');

    if (optionalProps.length > 0) {
      rule += " (";
      if (requiredProps.length > 0) rule += ' "," space ( ';

      const getRecursiveRefs = (ks: string[], firstIsOptional: boolean): string => {
        const [k, ...rest] = ks;
        const kvRuleName = propKvRuleNames[k];
        const commaRef = `( "," space ${kvRuleName} )`;
        let res: string;
        if (firstIsOptional) {
          res = commaRef + (k === "*" ? "*" : "?");
        } else {
          res = kvRuleName + (k === "*" ? " " + commaRef + "*" : "");
        }
        if (rest.length > 0) {
          res += " " +
            this.addRule(
              `${name}${name ? "-" : ""}${k}-rest`,
              getRecursiveRefs(rest, true),
            );
        }
        return res;
      };

      rule += optionalProps
        .map((_, i) => getRecursiveRefs(optionalProps.slice(i), false))
        .join(" | ");
      if (requiredProps.length > 0) rule += " )";
      rule += " )?";
    }

    rule += ' space "}"';

    return rule;
  }

  formatGrammar(): string {
    return Object.entries(this.rules)
      .sort((a, bItem) => (a[0] < bItem[0] ? -1 : a[0] > bItem[0] ? 1 : 0))
      .map(([n, r]) => `${n} ::= ${r}`)
      .join("\n");
  }
}

/**
 * Converts a JSON Schema into a GBNF grammar string suitable for constraining
 * llama.cpp generation.
 *
 * @param schema The JSON Schema (as a parsed object).
 * @param options Optional generation settings.
 * @returns A deterministic GBNF grammar string.
 */
export function jsonSchemaToGrammar(
  schema: unknown,
  options: JsonSchemaToGrammarOptions = {},
): string {
  if (!isObject(schema)) {
    throw new Error("Schema must be a JSON object");
  }

  const propOrder = new Map<string, number>(
    (options.propOrder ?? []).map((n, i) => [n, i] as [string, number]),
  );
  const converter = new SchemaConverter({
    propOrder,
    allowFetch: options.allowFetch ?? false,
    dotall: options.dotall ?? false,
    rawPattern: false,
  });
  const resolved = converter.resolveRefs(schema, "");
  converter.visit(resolved as SchemaObject, "");
  return converter.formatGrammar();
}
