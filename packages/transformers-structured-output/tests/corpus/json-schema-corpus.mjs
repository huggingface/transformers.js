import assert from "node:assert/strict";
import test from "node:test";

import { assertEveryBytePrefixAccepted, assertJsonSchema, assertJsonSchemaCompileError, jsonSchemaAccepts } from "./helpers.mjs";

// Stable in-repo corpus covering the supported JSON Schema profile. Cases were
// consolidated from the package's former external corpus runner.
const valueCases = [
  { name: "null", schema: { type: "null" }, accepted: [null], rejected: [false, 0, "null"] },
  { name: "boolean", schema: { type: "boolean" }, accepted: [true, false], rejected: [null, 0, "true"] },
  { name: "integer", schema: { type: "integer" }, accepted: [-12, 0, 42], rejected: [1.5, "1", true] },
  { name: "number", schema: { type: "number" }, accepted: [-1.25, 0, 2], rejected: ["2", null] },
  { name: "string", schema: { type: "string" }, accepted: ["", "hello", "😀"], rejected: [1, null, false] },
  { name: "type union", schema: { type: ["string", "null"] }, accepted: ["x", null], rejected: [1, true] },
  { name: "const", schema: { const: { answer: [1, true] } }, accepted: [{ answer: [1, true] }], rejected: [{ answer: [1, false] }] },
  { name: "enum", schema: { enum: ["en", "de", null] }, accepted: ["en", "de", null], rejected: ["fr", 0] },
  { name: "integer bounds", schema: { type: "integer", minimum: -2, exclusiveMaximum: 3 }, accepted: [-2, 0, 2], rejected: [-3, 3] },
  { name: "number multipleOf", schema: { type: "number", multipleOf: 0.25 }, accepted: [-1.5, 0, 2.25], rejected: [0.1, 1.3] },
  { name: "string length", schema: { type: "string", minLength: 2, maxLength: 3 }, accepted: ["ab", "😀x", "abc"], rejected: ["", "a", "abcd"] },
  { name: "array length", schema: { type: "array", minItems: 1, maxItems: 2 }, accepted: [[1], [1, 2]], rejected: [[], [1, 2, 3]] },
  { name: "homogeneous array", schema: { type: "array", items: { type: "integer" } }, accepted: [[], [1, 2]], rejected: [[1, "2"]] },
  {
    name: "tuple",
    schema: { type: "array", prefixItems: [{ type: "string" }, { type: "integer" }], items: false },
    accepted: [["x", 1]],
    rejected: [
      ["x", "1"],
      ["x", 1, true],
    ],
  },
  {
    name: "unique array",
    schema: { type: "array", uniqueItems: true },
    accepted: [
      [1, 2],
      [{ a: 1 }, { a: 2 }],
    ],
    rejected: [
      [1, 1],
      [{ a: [1] }, { a: [1] }],
    ],
  },
  {
    name: "contains",
    schema: { type: "array", contains: { type: "integer", minimum: 2 }, minContains: 2, maxContains: 2 },
    accepted: [["x", 2, 3]],
    rejected: [
      [2, "x"],
      [2, 3, 4],
    ],
  },
  { name: "required object", schema: { type: "object", properties: { answer: { type: "integer" } }, required: ["answer"], additionalProperties: false }, accepted: [{ answer: 1 }], rejected: [{}, { answer: "1" }, { answer: 1, extra: true }] },
];

for (const { name, schema, accepted, rejected } of valueCases) {
  test(name, () => {
    for (const value of accepted) assertJsonSchema(schema, value, true);
    for (const value of rejected) assertJsonSchema(schema, value, false);
  });
}

test("composition", () => {
  const schema = {
    allOf: [{ type: "integer" }, { minimum: 1 }],
    anyOf: [{ maximum: 2 }, { const: 5 }],
    not: { const: 0 },
  };
  for (const value of [1, 2, 5]) assertJsonSchema(schema, value, true);
  for (const value of [0, 3, "1"]) assertJsonSchema(schema, value, false);
  assertJsonSchema({ oneOf: [{ type: "number" }, { type: "integer" }] }, 1, false);
  assertJsonSchema({ oneOf: [{ type: "number" }, { type: "string" }] }, 1, true);
});

test("conditionals", () => {
  const schema = {
    type: "object",
    properties: { kind: { enum: ["text", "count"] }, value: true },
    required: ["kind", "value"],
    if: { properties: { kind: { const: "text" } }, required: ["kind"] },
    then: { properties: { value: { type: "string" } } },
    else: { properties: { value: { type: "integer" } } },
  };
  assertEveryBytePrefixAccepted(schema, '{"value":"ok","kind":"text"}');
  assertEveryBytePrefixAccepted(schema, '{"kind":"count","value":2}');
  assert.equal(jsonSchemaAccepts(schema, undefined, '{"kind":"text","value":2}'), false);
  assert.equal(jsonSchemaAccepts(schema, undefined, '{"kind":"count","value":"no"}'), false);
});

test("property patterns and dependencies", () => {
  const schema = {
    type: "object",
    properties: { code: { type: "integer" }, card: { type: "string" }, billing: { type: "string" }, country: { type: "string" } },
    patternProperties: { "^code$": { minimum: 1 }, "^x-": { type: "string" } },
    dependentRequired: { card: ["billing"] },
    dependentSchemas: { billing: { properties: { country: { const: "US" } }, required: ["country"] } },
    additionalProperties: false,
  };
  assertEveryBytePrefixAccepted(schema, '{"x-note":"ok","code":2,"country":"US","billing":"x","card":"1"}');
  assert.equal(jsonSchemaAccepts(schema, undefined, '{"code":0}'), false);
  assert.equal(jsonSchemaAccepts(schema, undefined, '{"card":"1"}'), false);
  assert.equal(jsonSchemaAccepts(schema, undefined, '{"billing":"x","country":"CA"}'), false);
});

test("local and recursive references", () => {
  const linkedList = {
    $defs: {
      node: {
        type: "object",
        properties: { value: { type: "string" }, next: { anyOf: [{ $ref: "#/$defs/node" }, { type: "null" }] } },
        required: ["value", "next"],
        additionalProperties: false,
      },
    },
    $ref: "#/$defs/node",
  };
  assertEveryBytePrefixAccepted(linkedList, '{"value":"a","next":{"value":"b","next":null}}');
  assert.equal(jsonSchemaAccepts(linkedList, undefined, '{"value":"a","next":2}'), false);

  const draft07 = { definitions: { answer: { type: "integer", minimum: 1 } }, $ref: "#/definitions/answer" };
  assertJsonSchema(draft07, 2, true);
  assertJsonSchema(draft07, 0, false);
});

test("draft-07 dependencies and tuple items", () => {
  const dependencies = {
    type: "object",
    properties: { a: true, b: true, c: { type: "integer" } },
    dependencies: { a: ["b"], b: { required: ["c"] } },
  };
  assertEveryBytePrefixAccepted(dependencies, '{"a":1,"b":2,"c":3}');
  assert.equal(jsonSchemaAccepts(dependencies, undefined, '{"a":1}'), false);
  assert.equal(jsonSchemaAccepts(dependencies, undefined, '{"b":2}'), false);

  const tuple = { type: "array", items: [{ type: "string" }, { type: "integer" }], additionalItems: false };
  assertEveryBytePrefixAccepted(tuple, '["x",1]');
  assert.equal(jsonSchemaAccepts(tuple, undefined, '["x",1,true]'), false);
});

test("exact numeric equality", () => {
  assertEveryBytePrefixAccepted({ type: "array", uniqueItems: true }, "[9007199254740992,9007199254740993]");
  assert.equal(jsonSchemaAccepts({ type: "array", uniqueItems: true }, undefined, "[1,1.0]"), false);
});

test("unknown formats are annotations", () => {
  assertEveryBytePrefixAccepted({ type: "string", format: "vendor-format" }, '"anything"');
});

test("x-guidance separators", () => {
  const schema = {
    type: "object",
    properties: { a: { type: "integer" }, b: { type: "integer" } },
    required: ["a", "b"],
    additionalProperties: false,
    "x-guidance": { item_separator: "-", key_separator: "_ ", whitespace_flexible: false },
  };
  assertEveryBytePrefixAccepted(schema, '{"a"_ 1-"b"_ 2}');
  assert.equal(jsonSchemaAccepts(schema, undefined, '{"a":1,"b":2}'), false);
});

test("invalid and non-incremental schemas fail at compilation", () => {
  const invalid = [{ not: [] }, { patternProperties: { "[": true } }, { dependentRequired: { a: ["b", "b"] } }, { minContains: -1 }, { uniqueItems: "yes" }, { unevaluatedProperties: false }];
  for (const schema of invalid) assertJsonSchemaCompileError(schema);
  assertJsonSchemaCompileError({ type: "string", pattern: "^yes$" }, /cannot be enforced incrementally/);
  assertJsonSchemaCompileError({ type: "string", format: "date" }, /cannot be enforced incrementally/);
  assertJsonSchemaCompileError({ $ref: "other.json#/value" }, /external JSON Schema reference/i);
});
