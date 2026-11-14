import { CONTENT_PARSERS } from "../../src/utils/chat_parsing/content_parsers.js";

const json = (text, args = {}) => CONTENT_PARSERS.json(text, args);

describe("reassessment content regressions", () => {
  it.each(["{NaN:1}", "{Infinity:1}", "{-Infinity:1}", "[NaNx]", "[Infinity0]", "[1NaN]", '{"x":NaNfoo}', '{"x":1,NaN:2}'])("rejects malformed bare tokens: %s", (text) => {
    expect(() => json(text)).toThrow();
    expect(json(text, { allow_non_json: true })).toBe(text);
  });

  it("preserves every original string while reviving non-finite values", () => {
    const strings = ["\u0001NaN\u0002", "n:NaN", "s:n:Infinity", "NaN", "\u0001Infinity\u0002"];
    const result = json(`{"NaN":NaN,"values":${JSON.stringify(strings)},"nested":[Infinity,{"-Infinity":-Infinity}]}`);
    expect(result.NaN).toBeNaN();
    expect(result.values).toEqual(strings);
    expect(result.nested).toEqual([Infinity, { "-Infinity": -Infinity }]);
  });

  it("preserves sentinel-looking strings with custom delimiters", () => {
    const result = json('{"text":<<\\u0001NaN\\u0002>>, "n":NaN}', { string_delims: [["<<", ">>"]] });
    expect(result.text).toBe("\\u0001NaN\\u0002");
    expect(result.n).toBeNaN();
    const escaped = json('{"value":"\\u0001NaN\\u0002","text":<<normal>>,"n":NaN}', { string_delims: [["<<", ">>"]] });
    expect(escaped).toEqual({ value: "\u0001NaN\u0002", text: "normal", n: NaN });
  });

  it.each(['{city: "Paris", days: 2}', '{ city : "Paris", days : 2 }', '{\ncity\t:\t"Paris",\n days:2}'])("supports whitespace around opted-in bare keys: %s", (text) => {
    expect(json(text, { unquoted_keys: true })).toEqual({ city: "Paris", days: 2 });
    expect(() => json(text)).toThrow();
  });

  it("does not rewrite key-like text inside strings", () => {
    const value = '{ key : "quoted" }';
    expect(json(`{ city: ${JSON.stringify(value)} }`, { unquoted_keys: true })).toEqual({ city: value });
  });

  it.each([
    [[[1, 2]], [1, 2]],
    [
      [[1, 2], [3]],
      [[1, 2], [3]],
    ],
    [
      [1, 2],
      [1, 2],
    ],
    [
      [1, [2], 3],
      [1, [2], 3],
    ],
    [
      [[1], 2, [3]],
      [[1], 2, [3]],
    ],
  ])("collects duplicate occurrences without flattening: %j", (values, expected) => {
    const text = values.map((value) => `<x>${JSON.stringify(value)}</x>`).join("");
    const args = { tag_pattern: "<(?P<key>[^>]+)>(?P<value>.*?)</[^>]+>", value_parser: { name: "json" } };
    expect(CONTENT_PARSERS["xml-inline"](text, { ...args, merge_duplicates: true })).toEqual({ x: expected });
    expect(CONTENT_PARSERS["xml-inline"](text, args)).toEqual({ x: values.at(-1) });
  });
});
