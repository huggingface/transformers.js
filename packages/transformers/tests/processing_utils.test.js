import { jest } from "@jest/globals";

import { Processor } from "../src/processing_utils.js";

describe("Processor response parsing", () => {
  it("forwards explicit response parsing to its tokenizer", () => {
    const tokenizer = {
      parse_response: jest.fn(() => ({ role: "assistant", content: "answer" })),
      get_response_parser: jest.fn(() => ({ feed: jest.fn(), finalize: jest.fn() })),
    };
    const processor = new Processor({}, { tokenizer }, null);

    expect(processor.parse_response("answer", { prefix: "" })).toEqual({
      role: "assistant",
      content: "answer",
    });
    expect(tokenizer.parse_response).toHaveBeenCalledWith("answer", { prefix: "" });

    const parser = processor.get_response_parser({ prefix: "" });
    expect(parser).toBe(tokenizer.get_response_parser.mock.results[0].value);
    expect(tokenizer.get_response_parser).toHaveBeenCalledWith({ prefix: "" });
  });

  it("requires a tokenizer", () => {
    const processor = new Processor({}, {}, null);
    expect(() => processor.parse_response("answer", { prefix: "" })).toThrow("Unable to parse response without a tokenizer");
    expect(() => processor.get_response_parser({ prefix: "" })).toThrow("Unable to create a response parser without a tokenizer");
  });
});
