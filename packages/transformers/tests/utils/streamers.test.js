import { TextStreamer } from "../../src/generation/streamers.js";

/**
 * Minimal tokenizer stub that models the two behaviours the streamer relies on:
 *
 * 1. The leading space of the first token in a decoded sequence is stripped, so text
 *    computed from a slice of the stream is not simply a slice of the whole text.
 * 2. A byte-fallback token decodes to U+FFFD until the token completing its UTF-8
 *    sequence arrives.
 *
 * Every decode call is recorded, so tests can assert how much decoding work the
 * streamer performs.
 */
class StubTokenizer {
  static SPECIAL = 0;
  // Regular pieces, each prefixed by a space (like SentencePiece's "▁word").
  static WORDS = [" hello", " world", " again", " today", " and", " some", " more", " text", " 中"];
  // A two-token UTF-8 sequence for "é", via byte fallback.
  static BYTE1 = 100;
  static BYTE2 = 101;

  constructor() {
    this.decode_calls = [];
  }

  get all_special_ids() {
    return [StubTokenizer.SPECIAL];
  }

  /**
   * @param {bigint[]} ids
   * @returns {string}
   */
  decode(ids) {
    this.decode_calls.push(ids.length);
    let out = "";
    for (let i = 0; i < ids.length; ++i) {
      const id = Number(ids[i]);
      if (id === StubTokenizer.BYTE1) {
        if (i + 1 < ids.length && Number(ids[i + 1]) === StubTokenizer.BYTE2) {
          out += "é";
          ++i;
        } else {
          out += "\uFFFD";
        }
        continue;
      }
      if (id === StubTokenizer.BYTE2) {
        out += "\uFFFD";
        continue;
      }
      if (id === StubTokenizer.SPECIAL) {
        out += "<s>";
        continue;
      }
      let piece = StubTokenizer.WORDS[id % StubTokenizer.WORDS.length];
      if (i === 0) {
        piece = piece.replace(/^ /, "");
      }
      out += piece;
    }
    return out;
  }

  /** Total number of tokens handed to `decode` across all calls. */
  get decoded_token_count() {
    return this.decode_calls.reduce((a, b) => a + b, 0);
  }

  /**
   * The text the whole token sequence stands for, with the leading space of every piece
   * kept (unlike `decode`, nothing is stripped).
   * @param {number[]} ids
   * @returns {string}
   */
  render(ids) {
    let out = "";
    for (let i = 0; i < ids.length; ++i) {
      if (ids[i] === StubTokenizer.BYTE1 && ids[i + 1] === StubTokenizer.BYTE2) {
        out += " é";
        ++i;
      } else if (ids[i] === StubTokenizer.BYTE1 || ids[i] === StubTokenizer.BYTE2) {
        out += " \uFFFD";
      } else if (ids[i] === StubTokenizer.SPECIAL) {
        out += "<s>";
      } else {
        out += StubTokenizer.WORDS[ids[i] % StubTokenizer.WORDS.length];
      }
    }
    return out;
  }
}

const isChineseChar = (cp) => (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf) || (cp >= 0x20000 && cp <= 0x2a6df) || (cp >= 0x2a700 && cp <= 0x2b73f) || (cp >= 0x2b740 && cp <= 0x2b81f) || (cp >= 0x2b820 && cp <= 0x2ceaf) || (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x2f800 && cp <= 0x2fa1f);

/**
 * The pre-existing streaming algorithm: decode the whole line on every token. Kept here as
 * the reference behaviour that the incremental implementation must reproduce exactly.
 */
class ReferenceStreamer {
  constructor(tokenizer, { skip_prompt = false, skip_special_tokens = true, callback_function } = {}) {
    this.tokenizer = tokenizer;
    this.skip_prompt = skip_prompt;
    this.callback_function = callback_function;
    this.decode_kwargs = { skip_special_tokens };
    this.token_cache = [];
    this.print_len = 0;
    this.next_tokens_are_prompt = true;
    this.special_ids = new Set(tokenizer.all_special_ids.map(BigInt));
  }

  put(value) {
    const is_prompt = this.next_tokens_are_prompt;
    if (is_prompt) {
      this.next_tokens_are_prompt = false;
      if (this.skip_prompt) return;
    }
    const tokens = value[0];

    if (tokens.length === 1 && this.special_ids.has(tokens[0])) {
      if (this.decode_kwargs.skip_special_tokens) return;
      if (this.token_cache.length > 0) {
        const text = this.tokenizer.decode(this.token_cache, this.decode_kwargs);
        this.emit(text.slice(this.print_len));
        this.token_cache = [];
        this.print_len = 0;
      }
      this.emit(this.tokenizer.decode(tokens, this.decode_kwargs));
      return;
    }

    this.token_cache = this.token_cache.concat(tokens);
    const text = this.tokenizer.decode(this.token_cache, this.decode_kwargs);

    let printable_text;
    if (is_prompt || text.endsWith("\n")) {
      printable_text = text.slice(this.print_len);
      this.token_cache = [];
      this.print_len = 0;
    } else if (text.length > 0 && isChineseChar(text.charCodeAt(text.length - 1))) {
      printable_text = text.slice(this.print_len);
      this.print_len += printable_text.length;
    } else {
      printable_text = text.slice(this.print_len, text.lastIndexOf(" ") + 1);
      this.print_len += printable_text.length;
    }
    this.emit(printable_text);
  }

  end() {
    let printable_text = "";
    if (this.token_cache.length > 0) {
      const text = this.tokenizer.decode(this.token_cache, this.decode_kwargs);
      printable_text = text.slice(this.print_len);
      this.token_cache = [];
      this.print_len = 0;
    }
    this.next_tokens_are_prompt = true;
    this.emit(printable_text);
  }

  emit(text) {
    if (text.length > 0) {
      this.callback_function?.(text);
    }
  }
}

/**
 * Stream `ids` through the given streamer class and return the emitted chunks.
 */
const stream = (StreamerClass, ids, options = {}) => {
  const tokenizer = new StubTokenizer();
  const chunks = [];
  const streamer = new StreamerClass(tokenizer, { callback_function: (text) => chunks.push(text), ...options });
  for (const id of ids) {
    streamer.put([[BigInt(id)]]);
  }
  streamer.end();
  return { chunks, tokenizer };
};

const word_ids = (count, start = 1) => Array.from({ length: count }, (_, i) => start + i);

describe("TextStreamer", () => {
  it("emits the same chunks as decoding the whole line on every token", () => {
    const ids = word_ids(300);
    const reference = stream(ReferenceStreamer, ids);
    const actual = stream(TextStreamer, ids);

    expect(actual.chunks).toEqual(reference.chunks);
    expect(actual.chunks.join("").replace(/ /g, "")).toBe(new StubTokenizer().render(ids).replace(/ /g, ""));
  });

  it("matches the reference across newlines, special tokens and CJK", () => {
    const ids = [
      ...word_ids(3),
      9, // "中" — flushes on the CJK branch
      ...word_ids(4, 2),
      0, // special token
      ...word_ids(3, 5),
    ];

    for (const options of [{ skip_special_tokens: false }, { skip_special_tokens: true }, { skip_prompt: true }]) {
      expect(stream(TextStreamer, ids, options).chunks).toEqual(stream(ReferenceStreamer, ids, options).chunks);
    }
  });

  it("does not drop characters when a byte-fallback token waits for its continuation", () => {
    const ids = [StubTokenizer.BYTE1, ...word_ids(4), StubTokenizer.BYTE2, StubTokenizer.BYTE1, StubTokenizer.BYTE2];
    const reference = stream(ReferenceStreamer, ids);
    const actual = stream(TextStreamer, ids);

    expect(actual.chunks).toEqual(reference.chunks);
    expect(actual.chunks.join("")).toContain("é");
  });

  it("keeps decoding work linear in the number of streamed tokens", () => {
    const short = stream(TextStreamer, word_ids(40));
    const long = stream(TextStreamer, word_ids(400));
    const reference_long = stream(ReferenceStreamer, word_ids(400));

    // The reference decodes n*(n+1)/2 tokens for a line of n tokens, so growing the line
    // 10x costs ~100x. Extending only the undecoded tail keeps the cost proportional to n.
    expect(long.tokenizer.decoded_token_count).toBeLessThan(reference_long.tokenizer.decoded_token_count / 20);
    expect(long.tokenizer.decoded_token_count).toBeLessThan(short.tokenizer.decoded_token_count * 12);
  });
});
