import { assertEquals, assertStringIncludes } from "@std/assert";
import { jsonSchemaToGrammar } from "./json_schema_to_gbnf.ts";

const SPACE = String.raw`space ::= | " " | "\n"{1,2} [ \t]{0,20}`;

Deno.test("primitive: string", () => {
  const g = jsonSchemaToGrammar({ type: "string" });
  assertEquals(
    g,
    [
      String.raw`char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})`,
      String.raw`root ::= "\"" char* "\""`,
      SPACE,
    ].join("\n"),
  );
});

Deno.test("primitive: integer references integral-part rule", () => {
  const g = jsonSchemaToGrammar({ type: "integer" });
  assertEquals(
    g,
    [
      "integral-part ::= [0] | [1-9] [0-9]{0,15}",
      `root ::= ("-"? integral-part)`,
      SPACE,
    ].join("\n"),
  );
});

Deno.test("primitive: number", () => {
  const g = jsonSchemaToGrammar({ type: "number" });
  assertStringIncludes(
    g,
    `root ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?`,
  );
  assertStringIncludes(g, "decimal-part ::= [0-9]{1,16}");
});

Deno.test("primitive: boolean", () => {
  const g = jsonSchemaToGrammar({ type: "boolean" });
  assertEquals(g, [`root ::= ("true" | "false")`, SPACE].join("\n"));
});

Deno.test("primitive: null", () => {
  const g = jsonSchemaToGrammar({ type: "null" });
  assertEquals(g, [`root ::= "null"`, SPACE].join("\n"));
});

Deno.test("object with required + optional props", () => {
  const g = jsonSchemaToGrammar({
    type: "object",
    properties: {
      b: { type: "string" },
      a: { type: "string" },
      d: { type: "string" },
      c: { type: "string" },
    },
    required: ["a", "b"],
    additionalProperties: false,
  });
  // required props appear (in original schema order), optional ones are nested
  assertStringIncludes(g, `"{" space b-kv "," space a-kv`);
  assertStringIncludes(g, `a-kv ::= "\\"a\\"" space ":" space string`);
  assertStringIncludes(g, `d-rest ::= ( "," space c-kv )?`);
  assertStringIncludes(g, `space "}"`);
});

Deno.test("nested objects", () => {
  const g = jsonSchemaToGrammar({
    type: "object",
    properties: {
      inner: {
        type: "object",
        properties: { x: { type: "integer" } },
        required: ["x"],
        additionalProperties: false,
      },
    },
    required: ["inner"],
    additionalProperties: false,
  });
  assertStringIncludes(g, `root ::= "{" space inner-kv space "}"`);
  assertStringIncludes(g, `inner-kv ::= "\\"inner\\"" space ":" space inner`);
  assertStringIncludes(g, `inner ::= "{" space inner-x-kv space "}"`);
  assertStringIncludes(g, `inner-x-kv ::= "\\"x\\"" space ":" space integer`);
});

Deno.test("array with items produces a repetition", () => {
  const g = jsonSchemaToGrammar({ type: "array", items: { type: "string" } });
  assertStringIncludes(
    g,
    `root ::= "[" space (string ("," space string)*)? space "]"`,
  );
  assertStringIncludes(g, `string ::= "\\"" char* "\\""`);
});

Deno.test("array with minItems + maxItems", () => {
  const g = jsonSchemaToGrammar({
    items: { type: "boolean" },
    minItems: 2,
    maxItems: 4,
  });
  assertStringIncludes(
    g,
    `root ::= "[" space boolean ("," space boolean){1,3} space "]"`,
  );
});

Deno.test("enum produces alternation with |", () => {
  const g = jsonSchemaToGrammar({
    enum: ["red", "amber", "green", null, 42, ["foo"]],
  });
  assertEquals(
    g,
    [
      `root ::= ("\\"red\\"" | "\\"amber\\"" | "\\"green\\"" | "null" | "42" | "[\\"foo\\"]")`,
      SPACE,
    ].join("\n"),
  );
});

Deno.test("const", () => {
  assertEquals(
    jsonSchemaToGrammar({ const: "foo" }),
    [`root ::= "\\"foo\\""`, SPACE].join("\n"),
  );
  assertEquals(
    jsonSchemaToGrammar({ const: 123 }),
    [`root ::= "123"`, SPACE].join("\n"),
  );
});

Deno.test("oneOf / anyOf union", () => {
  const oneOf = jsonSchemaToGrammar({
    oneOf: [{ type: "string" }, { type: "boolean" }],
  });
  assertStringIncludes(oneOf, "root ::= string | boolean");

  const anyOf = jsonSchemaToGrammar({
    anyOf: [{ type: "string" }, { type: "number" }],
  });
  assertStringIncludes(anyOf, "root ::= string | number");
});

Deno.test("type as array (nullable) union", () => {
  const g = jsonSchemaToGrammar({
    type: ["array", "null"],
    prefixItems: { type: "string" },
  });
  assertStringIncludes(g, "root ::= alternative-0 | null");
  assertStringIncludes(
    g,
    `alternative-0 ::= "[" space (string ("," space string)*)? space "]"`,
  );
});

Deno.test("$ref to definitions", () => {
  const g = jsonSchemaToGrammar({
    $ref: "#/definitions/foo",
    definitions: {
      foo: {
        type: "object",
        properties: { a: { type: "string" } },
        required: ["a"],
        additionalProperties: false,
      },
    },
  });
  assertEquals(
    g,
    [
      String.raw`char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})`,
      `ref-definitions-foo ::= "{" space ref-definitions-foo-a-kv space "}"`,
      `ref-definitions-foo-a-kv ::= "\\"a\\"" space ":" space string`,
      "root ::= ref-definitions-foo",
      SPACE,
      String.raw`string ::= "\"" char* "\""`,
    ].join("\n"),
  );
});

Deno.test("$ref to $defs", () => {
  const g = jsonSchemaToGrammar({
    type: "array",
    items: { $ref: "#/$defs/item" },
    $defs: {
      item: {
        type: "object",
        properties: { n: { type: "integer" } },
        required: ["n"],
        additionalProperties: false,
      },
    },
  });
  assertStringIncludes(g, `root ::= "[" space (item ("," space item)*)? space "]"`);
  assertStringIncludes(g, "item ::= ref-defs-item");
  assertStringIncludes(g, `ref-defs-item ::= "{" space ref-defs-item-n-kv space "}"`);
});

Deno.test("string with pattern", () => {
  const g = jsonSchemaToGrammar({ type: "string", pattern: "^abc?d*efg+(hij)?kl$" });
  assertEquals(
    g,
    [
      String.raw`root ::= "\"" ("ab" "c"? "d"* "ef" "g"+ ("hij")? "kl") "\""`,
      SPACE,
    ].join("\n"),
  );
});

Deno.test("format: date-time", () => {
  const g = jsonSchemaToGrammar({ type: "string", format: "date-time" });
  assertStringIncludes(g, "root ::= date-time-string");
  assertStringIncludes(g, `date-time-string ::= "\\"" date-time "\\""`);
  assertStringIncludes(g, `date-time ::= date "T" time`);
});

Deno.test("format: uuid is inlined by primitive rule name", () => {
  const g = jsonSchemaToGrammar({ type: "string", format: "uuid" });
  assertStringIncludes(
    g,
    String
      .raw`root ::= "\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\""`,
  );
});

Deno.test("integer with minimum/maximum range", () => {
  assertEquals(
    jsonSchemaToGrammar({ type: "integer", minimum: 3 }),
    ["root ::= ([1-2] [0-9]{1,15} | [3-9] [0-9]{0,15})", SPACE].join("\n"),
  );
  assertEquals(
    jsonSchemaToGrammar({ type: "integer", minimum: 0, maximum: 23 }),
    ["root ::= ([0-9] | ([1] [0-9] | [2] [0-3]))", SPACE].join("\n"),
  );
});

Deno.test("empty schema accepts any JSON object primitive", () => {
  const g = jsonSchemaToGrammar({});
  assertStringIncludes(g, "root ::= object");
  assertStringIncludes(g, "value ::= object | array | string | number | boolean | null");
});

Deno.test("propOrder controls property precedence", () => {
  const g = jsonSchemaToGrammar(
    {
      type: "object",
      properties: {
        a: { type: "string" },
        b: { type: "string" },
        c: { type: "string" },
      },
      required: ["a", "b", "c"],
      additionalProperties: false,
    },
    { propOrder: ["c", "b", "a"] },
  );
  assertStringIncludes(
    g,
    `root ::= "{" space c-kv "," space b-kv "," space a-kv space "}"`,
  );
});

Deno.test("remote $ref throws when fetching disabled", () => {
  let threw = false;
  try {
    jsonSchemaToGrammar({ $ref: "https://example.com/schema.json" });
  } catch (e) {
    threw = true;
    assertStringIncludes((e as Error).message, "not allowed");
  }
  assertEquals(threw, true);
});
