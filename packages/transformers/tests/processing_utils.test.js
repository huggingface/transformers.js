import { jest } from "@jest/globals";

import { Processor } from "../src/processing_utils.js";

describe("Processor response parsing", () => {
  it("forwards to its tokenizer", () => {
    const tokenizer = { parse_response: jest.fn(() => ({ role: "assistant", content: "answer" })) };
    const processor = new Processor({}, { tokenizer }, null);
    expect(processor.parse_response("answer", { prefix: "" })).toEqual({ role: "assistant", content: "answer" });
    expect(tokenizer.parse_response).toHaveBeenCalledWith("answer", { prefix: "" });
    expect(() => new Processor({}, {}, null).parse_response("answer", { prefix: "" })).toThrow("without a tokenizer");
  });
});
