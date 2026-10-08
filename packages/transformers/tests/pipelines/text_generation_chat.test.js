import { jest } from "@jest/globals";
import { Tensor, TextGenerationPipeline } from "../../src/transformers.js";

describe("Text generation chat validation", () => {
  function createPipeline() {
    const tokenizer = Object.assign((inputs) => ({ input_ids: new Tensor("int64", new BigInt64Array(inputs.length), [inputs.length, 1]) }), {
      apply_chat_template: jest.fn(() => "prompt"),
      batch_decode: (tokens) => Array(tokens.dims[0]).fill("prompt"),
    });
    const model = { generate: jest.fn(async ({ input_ids }) => input_ids) };
    return { pipe: new TextGenerationPipeline({ task: "text-generation", tokenizer, model }), tokenizer, model };
  }

  it.each([false, true])("accepts contentless assistant tool calls (batched=%s)", async (batched) => {
    const { pipe, tokenizer } = createPipeline();
    const messages = [
      { role: "user", content: "What is 4+5?" },
      { role: "assistant", thinking: "Use add.", tool_calls: [{ type: "function", function: { name: "add", arguments: { a: 4, b: 5 } } }] },
      { role: "tool", name: "add", content: "9" },
    ];
    const output = await pipe(batched ? [messages, messages] : messages);
    expect(tokenizer.apply_chat_template).toHaveBeenCalledWith(messages, expect.anything());
    expect(tokenizer.apply_chat_template).toHaveBeenCalledTimes(batched ? 2 : 1);
    const result = batched ? output[0] : output;
    expect(result[0].generated_text.slice(0, -1)).toEqual(messages);
  });

  it.each([null, "invalid", { role: "user", tool_calls: [] }, { role: "assistant", tool_calls: [] }, { role: "assistant", tool_calls: "invalid" }])("rejects invalid chat messages: %j", async (message) => {
    const { pipe, model } = createPipeline();
    await expect(pipe([{ role: "user", content: "hi" }, message])).rejects.toThrow("Input must be");
    expect(model.generate).not.toHaveBeenCalled();
  });
});
