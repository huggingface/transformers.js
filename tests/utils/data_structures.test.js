import { PriorityQueue, DictionarySplitter } from "../../src/utils/data-structures.js";

describe("Priority queue", () => {
  const EXAMPLE_ARRAY = [2, 5, 3, 1, 4];
  it("default (max heap)", () => {
    const queue = new PriorityQueue();
    queue.extend(EXAMPLE_ARRAY);
    expect(queue.pop()).toBe(5);
  });

  it("min heap", () => {
    const queue = new PriorityQueue((a, b) => a < b);
    queue.extend(EXAMPLE_ARRAY);
    expect(queue.pop()).toBe(1);
  });

  it("heap w/ max size", () => {
    const queue = new PriorityQueue((a, b) => a > b, 3);
    queue.extend([1, 2, 3, 4, 5, 4, 3, 2, 1]);
    expect(queue.pop()).toBe(5);

    // Test with random sizes
    const sizes = [1, 3, 4, 5, 8, 9, 15, 16, 31, 32, 127, 128];
    const arr = Array.from({ length: 100 }, (_) => Math.random());
    const max = Math.max(...arr);
    for (const size of sizes) {
      const queue = new PriorityQueue((a, b) => a > b, size);
      queue.extend(arr);
      expect(queue.pop()).toBe(max);
      expect(queue.size).toBeLessThanOrEqual(size);
    }
  });
});

describe("Dictionary splitter", () => {
  it("should split on a defined dictionary", () => {
    const splitter = new DictionarySplitter(
      ["a", "b", "c", "abc"],
      null, // no split regex
    );
    const text = ".a.b.cc.abcdef.";
    const expected = [".", "a", ".", "b", ".", "c", "c", ".", "abc", "def."];
    const result = splitter.split(text);
    expect(result).toEqual(expected);
  });
  it("should split on a defined dictionary w/ split regex", () => {
    const splitter = new DictionarySplitter(
      ["a", "b", "c", "abc"],
      /\s+/, // split on whitespace
    );
    const text = "a  b  c";
    const expected = ["a", "b", "c"];
    const result = splitter.split(text);
    expect(result).toEqual(expected);
  });

  it("should handle multi-byte characters", () => {
    const text = "before🤗after\ud83etest";
    const splitter = new DictionarySplitter(
      ["🤗" /* '\ud83e\udd17' */, "\ud83e"],
      null, // no split regex
    );
    const expected = ["before", "🤗", "after", "\ud83e", "test"];
    const result = splitter.split(text);
    expect(result).toEqual(expected);
  });
});
