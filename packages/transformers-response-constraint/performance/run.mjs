import { Tensor } from "@huggingface/transformers";

import { ResponseConstraint } from "../dist/index.js";
import complexJson from "./complex-json.mjs";
import simpleJson from "./simple-json.mjs";
import simpleRegex from "./simple-regex.mjs";

const WARMUP_RUNS = 2;
const MEASURED_RUNS = 10;
const EOS_TOKEN_ID = 256;
const VOCAB_SIZE = Number(process.env.VOCAB_SIZE ?? 8192);
const encoder = new TextEncoder();
const tokenizer = {
  tokens: [
    ...Array.from({ length: EOS_TOKEN_ID }, (_, tokenId) => [tokenId]),
    [],
    ...Array.from({ length: VOCAB_SIZE - EOS_TOKEN_ID - 1 }, (_, index) => [
      ...encoder.encode(` token-${index.toString(36)}`),
    ]),
  ],
  eos_token_id: EOS_TOKEN_ID,
  special_token_ids: [EOS_TOKEN_ID],
};
const encodedCases = [simpleJson, simpleRegex, complexJson].map((testCase) => ({
  ...testCase,
  tokenIds: encoder.encode(testCase.output),
}));
const coldResults = new Map();

for (const testCase of encodedCases) {
  coldResults.set(testCase, await measure(testCase));
}

for (const testCase of encodedCases) {
  for (let run = 0; run < WARMUP_RUNS; ++run) await measure(testCase);
}

const results = [];
for (const testCase of encodedCases) {
  let elapsedMs = 0;
  let processorMs = 0;
  let updateMs = 0;
  for (let run = 0; run < MEASURED_RUNS; ++run) {
    const result = await measure(testCase);
    elapsedMs += result.elapsedMs;
    processorMs += result.processorMs;
    updateMs += result.updateMs;
  }
  results.push({
    constraint: testCase.name,
    vocabulary: VOCAB_SIZE,
    tokens: testCase.tokenIds.length + 1,
    "cold ms": format(coldResults.get(testCase).elapsedMs),
    "total ms": format(elapsedMs / MEASURED_RUNS),
    "ms/token": format(
      elapsedMs / MEASURED_RUNS / (testCase.tokenIds.length + 1),
    ),
    "processor ms": format(processorMs / MEASURED_RUNS),
    "update ms": format(updateMs / MEASURED_RUNS),
  });
}

console.table(results);

async function measure(testCase) {
  const constraint = await ResponseConstraint.fromResponseFormat(
    tokenizer,
    testCase.responseFormat,
  );
  const scores = new Tensor("float32", new Float32Array(VOCAB_SIZE), [
    1,
    VOCAB_SIZE,
  ]);
  const inputIds = [0n];
  const startedAt = performance.now();
  let processorMs = 0;
  let updateMs = 0;
  for (const tokenId of [...testCase.tokenIds, EOS_TOKEN_ID]) {
    scores.data.fill(0);
    let stepStartedAt = performance.now();
    constraint.logits_processor([inputIds], scores);
    processorMs += performance.now() - stepStartedAt;
    if (!Number.isFinite(scores.data[tokenId]))
      throw new Error(`${testCase.name} rejected token ${tokenId}`);
    inputIds.push(BigInt(tokenId));
    stepStartedAt = performance.now();
    constraint.logits_processor.onTokensSampled([tokenId], [inputIds]);
    constraint.stopping_criteria([inputIds]);
    updateMs += performance.now() - stepStartedAt;
  }
  if (!constraint.stopping_criteria([inputIds])[0])
    throw new Error(`${testCase.name} did not stop after EOS`);
  return { elapsedMs: performance.now() - startedAt, processorMs, updateMs };
}

function format(value) {
  return value.toFixed(3);
}
