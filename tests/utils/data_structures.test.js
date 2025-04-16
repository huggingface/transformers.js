import { PriorityQueue, DictionarySplitter, LRUCache } from "../../src/utils/data-structures.js";

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
    const splitter = new DictionarySplitter(["a", "b", "c", "abc"]);
    const text = ".a.b.cc.abcdef.";
    const expected = [".", "a", ".", "b", ".", "c", "c", ".", "abc", "def."];
    const result = splitter.split(text);
    expect(result).toEqual(expected);
  });

  it("should handle multi-byte characters", () => {
    const text = "before🤗after\ud83etest";
    const splitter = new DictionarySplitter(["🤗" /* '\ud83e\udd17' */, "\ud83e"]);
    const expected = ["before", "🤗", "after", "\ud83e", "test"];
    const result = splitter.split(text);
    expect(result).toEqual(expected);
  });
});

describe("LRUCache", () => {
  it("should return undefined for non-existent keys", () => {
    const cache = new LRUCache(2);
    expect(cache.get("nonexistent")).toEqual(undefined);
  });

  it("should store and retrieve values correctly", () => {
    const cache = new LRUCache(2);
    cache.put("a", 1);
    cache.put("b", 2);
    expect(cache.get("a")).toEqual(1);
    expect(cache.get("b")).toEqual(2);
  });

  it("should update the value and refresh the usage", () => {
    const cache = new LRUCache(2);
    cache.put("a", 1);
    cache.put("b", 2);
    // Update key "a"
    cache.put("a", 10);
    expect(cache.get("a")).toEqual(10);
    // Access "a" so "b" becomes the LRU
    cache.get("a");
    cache.put("c", 3);
    // "b" should be evicted since it is the least recently used.
    expect(cache.get("b")).toEqual(undefined);
    expect(cache.get("c")).toEqual(3);
  });

  it("should evict the least recently used item when capacity is exceeded", () => {
    const cache = new LRUCache(3);
    cache.put("a", 1);
    cache.put("b", 2);
    cache.put("c", 3);
    // Access "a" to refresh its recentness.
    cache.get("a");
    // Insert a new key, this should evict "b" as it is the least recently used.
    cache.put("d", 4);
    expect(cache.get("b")).toEqual(undefined);
    expect(cache.get("a")).toEqual(1);
    expect(cache.get("c")).toEqual(3);
    expect(cache.get("d")).toEqual(4);
  });

  it("should update the usage order on get", () => {
    const cache = new LRUCache(3);
    cache.put("a", "apple");
    cache.put("b", "banana");
    cache.put("c", "cherry");
    // Access "a" making it most recently used.
    expect(cache.get("a")).toEqual("apple");
    // Insert new element to evict the least recently used ("b").
    cache.put("d", "date");
    expect(cache.get("b")).toEqual(undefined);
    // "a", "c", and "d" should be present.
    expect(cache.get("a")).toEqual("apple");
    expect(cache.get("c")).toEqual("cherry");
    expect(cache.get("d")).toEqual("date");
  });

  it("should clear the cache", () => {
    const cache = new LRUCache(2);
    cache.put("a", 1);
    cache.put("b", 2);
    cache.clear();
    expect(cache.get("a")).toEqual(undefined);
    expect(cache.get("b")).toEqual(undefined);
  });
});
