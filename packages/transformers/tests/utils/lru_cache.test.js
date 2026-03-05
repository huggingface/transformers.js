import { LRUCache } from "../../src/utils/lru_cache.js";

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

  it("should return true when deleting an existing key", () => {
    const cache = new LRUCache(2);
    cache.put("a", 1);
    expect(cache.delete("a")).toEqual(true);
  });

  it("should return false when deleting a non-existent key", () => {
    const cache = new LRUCache(2);
    expect(cache.delete("nonexistent")).toEqual(false);
  });

  it("should make a deleted key unretrievable", () => {
    const cache = new LRUCache(2);
    cache.put("a", 1);
    cache.delete("a");
    expect(cache.get("a")).toEqual(undefined);
  });

  it("should allow re-inserting a deleted key", () => {
    const cache = new LRUCache(2);
    cache.put("a", 1);
    cache.delete("a");
    cache.put("a", 2);
    expect(cache.get("a")).toEqual(2);
  });

  it("should not affect other entries when deleting a key", () => {
    const cache = new LRUCache(3);
    cache.put("a", 1);
    cache.put("b", 2);
    cache.put("c", 3);
    cache.delete("b");
    expect(cache.get("a")).toEqual(1);
    expect(cache.get("b")).toEqual(undefined);
    expect(cache.get("c")).toEqual(3);
  });

  it("should free up space after deletion, preventing unwanted eviction", () => {
    const cache = new LRUCache(2);
    cache.put("a", 1);
    cache.put("b", 2);
    cache.delete("a");
    // With "a" deleted, inserting "c" should not evict "b"
    cache.put("c", 3);
    expect(cache.get("b")).toEqual(2);
    expect(cache.get("c")).toEqual(3);
  });
});
