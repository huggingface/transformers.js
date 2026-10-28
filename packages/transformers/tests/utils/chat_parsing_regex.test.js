import { compile_pattern, search } from "../../src/utils/chat_parsing/regex.js";

describe("Python regex translation", () => {
  const matches = (source, text) => search(compile_pattern(source), text, 0);

  it.each([String.raw`\s`, String.raw`[\s]`])("uses Unicode whitespace for %s", (source) => {
    expect(matches(source, "\u0085")).toMatchObject({ start: 0, end: 1 });
    expect(matches(source, "\uFEFF")).toBeNull();
  });

  it.each([String.raw`\S`, String.raw`[\S]`])("uses Unicode non-whitespace for %s", (source) => {
    expect(matches(source, "\u0085")).toBeNull();
    expect(matches(source, "\uFEFF")).toMatchObject({ start: 0, end: 1 });
  });

  it("lifts complement shorthands into single-code-point character-set atoms", () => {
    expect(matches(String.raw`[\W_]+`, "a_!😀é")).toMatchObject({ start: 1, end: 5 });
    expect(matches(String.raw`[^\W_]+`, "_!😀é\u200D")).toMatchObject({ start: 4, end: 6 });
    expect(matches(String.raw`[\W^]`, "^")).toMatchObject({ start: 0, end: 1 });
    expect(matches(String.raw`[\W]`, "\u200D")).toBeNull();
    expect(matches(String.raw`[\W]`, "²")).toMatchObject({ start: 0, end: 1 });
  });

  it("preserves ranges and literal hyphens when composing character classes", () => {
    expect(matches(String.raw`[\Wa-z]+`, "abc!😀")).toMatchObject({ start: 0, end: 6 });
    expect(matches(String.raw`[\W-]+`, "-!")).toMatchObject({ start: 0, end: 2 });
    expect(matches(String.raw`[-\W]+`, "-!")).toMatchObject({ start: 0, end: 2 });
    expect(matches(String.raw`[a-c-e]+`, "abc-e")).toMatchObject({ start: 0, end: 5 });
    for (const source of [String.raw`[a-\W]`, String.raw`[\W-z]`, String.raw`[a-\d]`, String.raw`[\d-z]`]) {
      expect(() => compile_pattern(source)).toThrow("bad character range");
    }
  });

  it("keeps complement classes as whole atoms in streaming partial matches", () => {
    const pattern = compile_pattern(String.raw`<[^\W_]+>[\W_]END`);
    expect(search(pattern, "<é>😀EN", 0, true)).toEqual({ start: 0, end: 7, groups: {}, partial: true });
    expect(search(pattern, "<é>😀END", 0, true)).toEqual({ start: 0, end: 8, groups: {}, partial: false });
    expect(search(pattern, "<é>aEN", 0, true)).toMatchObject({ start: 6, partial: true });
  });

  it("translates the bell escape both inside and outside a character class", () => {
    expect(matches(String.raw`\a[\a]`, "\x07\x07")).toMatchObject({ start: 0, end: 2 });
  });

  it("preserves Python-specific anchors and DOTALL behavior", () => {
    expect(matches(String.raw`a\Z`, "a\n")).toBeNull();
    expect(matches("a$", "a\n")).toMatchObject({ start: 0, end: 1 });
    expect(matches("^a", "\na")).toBeNull();
    expect(matches("a.b", "a\nb")).toMatchObject({ start: 0, end: 3 });
  });
});
