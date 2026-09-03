import assert from "node:assert/strict";

import { Tensor } from "@huggingface/transformers";

import { StructuredOutputProcessor } from "../../dist/index.js";

const EOS_TOKEN_ID = 256;
const encoder = new TextEncoder();
const tokenizer = {
  tokens: [...Array.from({ length: 256 }, (_, byte) => [byte]), []],
  eos_token_id: EOS_TOKEN_ID,
  special_token_ids: [EOS_TOKEN_ID],
};

function scores() {
  return new Tensor("float32", new Float32Array(EOS_TOKEN_ID + 1), [1, EOS_TOKEN_ID + 1]);
}

function allowed(logits, tokenId) {
  return Number.isFinite(logits.data[tokenId]);
}

export function compileJsonSchema(schema) {
  return new StructuredOutputProcessor(tokenizer, { type: "json_schema", json_schema: schema });
}

export function jsonSchemaAccepts(schema, value, jsonText) {
  const text = jsonText ?? JSON.stringify(value);
  let constraint;
  try {
    constraint = compileJsonSchema(schema);
  } catch {
    return false;
  }
  const inputIds = [0n];
  for (const tokenId of encoder.encode(text)) {
    const logits = scores();
    try {
      constraint([inputIds], logits);
      if (!allowed(logits, tokenId)) return false;
      inputIds.push(BigInt(tokenId));
    } catch {
      return false;
    }
  }
  const logits = scores();
  try {
    constraint([inputIds], logits);
  } catch {
    return false;
  }
  return allowed(logits, EOS_TOKEN_ID);
}

export function assertJsonSchema(schema, value, expected, jsonText) {
  const text = jsonText ?? JSON.stringify(value);
  assert.equal(jsonSchemaAccepts(schema, value, text), expected, `${text} should ${expected ? "match" : "not match"} ${JSON.stringify(schema)}`);
}

export function assertEveryBytePrefixAccepted(schema, text) {
  assert.equal(jsonSchemaAccepts(schema, undefined, text), true, `${text} contains a pruned byte prefix`);
}

export function assertJsonSchemaCompileError(schema, pattern) {
  assert.throws(() => compileJsonSchema(schema), pattern);
}
