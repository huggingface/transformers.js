import { spawnSync } from "child_process";

const MODULE_NAME = "@huggingface/transformers";

const CODE_BODY = `
const model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM";
const generator = await pipeline("text-generation", model_id, { dtype: "fp32" });
const result = await generator("hello", { max_new_tokens: 3, return_full_text: false });
process.stdout.write(result[0].generated_text);
`;

const TARGET_OUTPUT = "erdingsdelete mely";

const wrap_async_iife = (code) => `(async function() { ${code} })();`;

const check = (code, module = false) => {
  const args = ["-e", code];
  if (module) args.push("--input-type=module");
  const { status, stdout, stderr } = spawnSync("node", args);
  expect(stderr.toString()).toEqual(""); // No warnings or errors are printed
  expect(stdout.toString()).toEqual(TARGET_OUTPUT); // The output should match
  expect(status).toEqual(0); // The process should exit cleanly
};

describe("Testing the bundle", () => {
  it("ECMAScript Module (ESM)", () => {
    check(`import { pipeline } from "${MODULE_NAME}";${CODE_BODY}`, true);
  });

  it("CommonJS (CJS) with require", () => {
    check(`const { pipeline } = require("${MODULE_NAME}");${wrap_async_iife(CODE_BODY)}`);
  });

  it("CommonJS (CJS) with dynamic import", () => {
    check(`${wrap_async_iife(`const { pipeline } = await import("${MODULE_NAME}");${CODE_BODY}`)}`);
  });
});
