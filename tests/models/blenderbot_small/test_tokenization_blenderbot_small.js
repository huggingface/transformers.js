import { BlenderbotSmallTokenizer } from "../../../src/tokenizers.js";
import { BASE_TEST_STRINGS, BLENDERBOT_SMALL_TEST_STRINGS } from "../test_strings.js";

export const TOKENIZER_CLASS = BlenderbotSmallTokenizer;

// NOTE: `.tokenize()` is disabled for BlenderbotSmallTokenizer
export const TEST_CONFIG = {
  "Xenova/blenderbot_small-90M": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      // "tokens": ["how", "are", "you", "doing", "?"],
      ids: [102, 46, 15, 267, 20],
      decoded: "how are you doing?",
    },
    SIMPLE_WITH_PUNCTUATION: {
      text: BASE_TEST_STRINGS.SIMPLE_WITH_PUNCTUATION,
      // "tokens": ["you", "should", "'", "ve", "done", "this"],
      ids: [15, 197, 8, 117, 369, 36],
      decoded: "you should've done this",
    },
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      // "tokens": ["0@@", "1@@", "2@@", "3@@", "4@@", "5@@", "6@@", "7@@", "89", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "100", "1000"],
      ids: [1988, 2388, 735, 801, 827, 948, 981, 1110, 4814, 520, 143, 176, 216, 260, 253, 345, 374, 420, 475, 316, 773, 6217],
      decoded: "0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      // "tokens": ["the", "company", "was", "founded", "in", "2016", "."],
      ids: [7, 293, 18, 912, 13, 845, 5],
      decoded: "the company was founded in 2016.",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      // "tokens": ["a", "__newln__", "'", "ll", "!", "!@@", "to", "?", "'", "d", "'", "'", "d", "of", ",", "can", "'", "t", "."],
      ids: [12, 4, 8, 97, 37, 3, 11, 20, 8, 85, 8, 8, 85, 10, 6, 62, 8, 30, 5],
      decoded: "a __newln__'ll! __unk__ to?'d'' d of, can't.",
    },
    PYTHON_CODE: {
      text: BASE_TEST_STRINGS.PYTHON_CODE,
      // "tokens": ["def", "main", "(", ")@@", ":", "__newln__", "pass"],
      ids: [21996, 550, 40, 3, 106, 4, 1314],
      decoded: "def main ( __unk__ : __newln__ pass",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      // "tokens": ["let", "a", "=", "ob@@", "j", ".@@", "to@@", "string", "(", ")@@", ";", "__newln__", "to@@", "string", "(", ")@@", ";"],
      ids: [131, 12, 1381, 2808, 755, 3, 752, 4529, 40, 3, 118, 4, 752, 4529, 40, 3, 118],
      decoded: "let a = obj __unk__ tostring ( __unk__ ; __newln__ tostring ( __unk__ ;",
    },
    NEWLINES: {
      text: BASE_TEST_STRINGS.NEWLINES,
      // "tokens": ["this", "__newln__", "is", "__newln__", "a", "__newln__", "test", "."],
      ids: [36, 4, 24, 4, 12, 4, 1248, 5],
      decoded: "this __newln__ is __newln__ a __newln__ test.",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      // "tokens": ["un@@", "wan@@", "t@@", "\u00e9@@", "d", ",@@", "running"],
      ids: [204, 4151, 291, 1677, 85, 3, 785],
      decoded: "unwant\u00e9d __unk__ running",
    },
    CONTROL_TOKENS: {
      text: BASE_TEST_STRINGS.CONTROL_TOKENS,
      // "tokens": ["1@@", "\u0000@@", "2@@", "\ufffd@@", "3"],
      ids: [2388, 3, 735, 3, 216],
      decoded: "1__unk__ 2__unk__ 3",
    },
    HELLO_WORLD_TITLECASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_TITLECASE,
      // "tokens": ["hello", "world"],
      ids: [880, 159],
      decoded: "hello world",
    },
    HELLO_WORLD_LOWERCASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_LOWERCASE,
      // "tokens": ["hello", "world"],
      ids: [880, 159],
      decoded: "hello world",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      // "tokens": ["\u751f@@", "\u6d3b@@", "\u7684@@", "\u771f@@", "\u8c1b@@", "\u662f"],
      ids: [30488, 32756, 29891, 30813, 3, 34037],
      decoded: "\u751f\u6d3b\u7684\u771f__unk__ \u662f",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      // "tokens": ["leading", "space"],
      ids: [1164, 833],
      decoded: "leading space",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      // "tokens": ["trailing", "space"],
      ids: [12499, 833],
      decoded: "trailing space",
    },
    DOUBLE_SPACE: {
      text: BASE_TEST_STRINGS.DOUBLE_SPACE,
      // "tokens": ["hi", "hello"],
      ids: [792, 880],
      decoded: "hi hello",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      // "tokens": ["test", "$@@", "1", "r@@", "2", "#@@", "3", "\u20ac@@", "4", "\u00a3@@", "5", "\u00a5@@", "6", "\u20a3@@", "7", "\u20b9@@", "8", "\u20b1@@", "9", "test"],
      ids: [1248, 3, 143, 510, 176, 3, 216, 3, 260, 3, 253, 3, 345, 3, 374, 3, 420, 3, 475, 1248],
      decoded: "test __unk__ 1 r2 __unk__ 3 __unk__ 4 __unk__ 5 __unk__ 6 __unk__ 7 __unk__ 8 __unk__ 9 test",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      // "tokens": ["i", "bought", "an", "apple", "for", "$@@", "1", ".@@", "00", "at", "the", "store", "."],
      ids: [14, 1890, 50, 4758, 26, 3, 143, 3, 1966, 32, 7, 1640, 5],
      decoded: "i bought an apple for __unk__ 1 __unk__ 00 at the store.",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      // "tokens": ["you@@", "\u2026"],
      ids: [7984, 1244],
      decoded: "you\u2026",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      // "tokens": ["you@@", "\u2026"],
      ids: [7984, 1244],
      decoded: "you\u2026",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      // "tokens": ["you@@", "\u2026", "you@@", "\u2026"],
      ids: [7984, 1244, 7984, 1244],
      decoded: "you\u2026 you\u2026",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      // "tokens": ["weird", "\uff5e", "edge", "\uff5e", "case"],
      ids: [2614, 30831, 1649, 30831, 543],
      decoded: "weird \uff5e edge \uff5e case",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      // "tokens": ["\u2581@@", "this", "\u2581@@", "is", "\u2581@@", "a", "\u2581@@", "test", "\u2581", "."],
      ids: [3, 36, 3, 24, 3, 12, 3, 1248, 50106, 5],
      decoded: "__unk__ this __unk__ is __unk__ a __unk__ test \u2581.",
    },
    SPECIAL_TOKENS: {
      text: BLENDERBOT_SMALL_TEST_STRINGS.SPECIAL_TOKENS,
      // "tokens": ["__start__", "hello", "world", "__end__"],
      ids: [1, 880, 159, 2],
      decoded: "__start__ hello world __end__",
    },
    WHITESPACE_1: {
      text: BLENDERBOT_SMALL_TEST_STRINGS.WHITESPACE_1,
      // "tokens": ["__start__", "hey", "__end__"],
      ids: [1, 226, 2],
      decoded: "__start__ hey __end__",
    },
    WHITESPACE_2: {
      text: BLENDERBOT_SMALL_TEST_STRINGS.WHITESPACE_2,
      // "tokens": ["__start__", "hey", "__end__"],
      ids: [1, 226, 2],
      decoded: "__start__ hey __end__",
    },
  },
};
