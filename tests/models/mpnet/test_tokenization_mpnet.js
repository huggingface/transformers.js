import { MPNetTokenizer } from "../../../src/tokenizers.js";
import { BASE_TEST_STRINGS } from "../test_strings.js";

export const TOKENIZER_CLASS = MPNetTokenizer;
export const TEST_CONFIG = {
  "Xenova/all-mpnet-base-v2": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      tokens: ["how", "are", "you", "doing", "?"],
      ids: [0, 2133, 2028, 2021, 2729, 1033, 2],
      decoded: "<s> how are you doing? </s>",
    },
    SIMPLE_WITH_PUNCTUATION: {
      text: BASE_TEST_STRINGS.SIMPLE_WITH_PUNCTUATION,
      tokens: ["you", "should", "'", "ve", "done", "this"],
      ids: [0, 2021, 2327, 1009, 2314, 2593, 2027, 2],
      decoded: "<s> you should've done this </s>",
    },
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["01", "##23", "##45", "##6", "##7", "##8", "##9", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "100", "1000"],
      ids: [0, 5894, 21930, 19965, 2579, 2585, 2624, 2687, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 2188, 2535, 6698, 2],
      decoded: "<s> 0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000 </s>",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["the", "company", "was", "founded", "in", "2016", "."],
      ids: [0, 2000, 2198, 2005, 2635, 2003, 2359, 1016, 2],
      decoded: "<s> the company was founded in 2016. </s>",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["a", "'", "ll", "!", "!", "to", "?", "'", "d", "'", "'", "d", "of", ",", "can", "'", "t", "."],
      ids: [0, 1041, 1009, 2226, 1003, 1003, 2004, 1033, 1009, 1044, 1009, 1009, 1044, 2001, 1014, 2068, 1009, 1060, 1016, 2],
      decoded: "<s> a'll!! to?'d'' d of, can't. </s>",
    },
    PYTHON_CODE: {
      text: BASE_TEST_STRINGS.PYTHON_CODE,
      tokens: ["def", "main", "(", ")", ":", "pass"],
      ids: [0, 13370, 2368, 1010, 1011, 1028, 3417, 2],
      decoded: "<s> def main ( ) : pass </s>",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["let", "a", "=", "ob", "##j", ".", "to", "##st", "##ring", "(", ")", ";", "to", "##st", "##ring", "(", ")", ";"],
      ids: [0, 2296, 1041, 1031, 27889, 3505, 1016, 2004, 3371, 4896, 1010, 1011, 1029, 2004, 3371, 4896, 1010, 1011, 1029, 2],
      decoded: "<s> let a = obj. tostring ( ) ; tostring ( ) ; </s>",
    },
    NEWLINES: {
      text: BASE_TEST_STRINGS.NEWLINES,
      tokens: ["this", "is", "a", "test", "."],
      ids: [0, 2027, 2007, 1041, 3235, 1016, 2],
      decoded: "<s> this is a test. </s>",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["unwanted", ",", "running"],
      ids: [0, 18166, 1014, 2774, 2],
      decoded: "<s> unwanted, running </s>",
    },
    CONTROL_TOKENS: {
      text: BASE_TEST_STRINGS.CONTROL_TOKENS,
      tokens: ["123"],
      ids: [0, 13142, 2],
      decoded: "<s> 123 </s>",
    },
    HELLO_WORLD_TITLECASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_TITLECASE,
      tokens: ["hello", "world"],
      ids: [0, 7596, 2092, 2],
      decoded: "<s> hello world </s>",
    },
    HELLO_WORLD_LOWERCASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_LOWERCASE,
      tokens: ["hello", "world"],
      ids: [0, 7596, 2092, 2],
      decoded: "<s> hello world </s>",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u751f", "[UNK]", "\u7684", "\u771f", "[UNK]", "[UNK]"],
      ids: [0, 1914, 104, 1920, 1925, 104, 104, 2],
      decoded: "<s> \u751f [UNK] \u7684 \u771f [UNK] [UNK] </s>",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      tokens: ["leading", "space"],
      ids: [0, 2881, 2690, 2],
      decoded: "<s> leading space </s>",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["trailing", "space"],
      ids: [0, 12546, 2690, 2],
      decoded: "<s> trailing space </s>",
    },
    DOUBLE_SPACE: {
      text: BASE_TEST_STRINGS.DOUBLE_SPACE,
      tokens: ["hi", "hello"],
      ids: [0, 7636, 7596, 2],
      decoded: "<s> hi hello </s>",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["test", "$", "1", "r", "##2", "#", "3", "\u20ac", "##4", "\u00a35", "\u00a5", "##6", "[UNK]", "\u20b9", "##8", "\u20b1", "##9", "test"],
      ids: [0, 3235, 1006, 1019, 1058, 2479, 1005, 1021, 1578, 2553, 27817, 1075, 2579, 104, 1580, 2624, 1579, 2687, 3235, 2],
      decoded: "<s> test $ 1 r2 # 3 \u20ac4 \u00a35 \u00a56 [UNK] \u20b98 \u20b19 test </s>",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["i", "bought", "an", "apple", "for", "$", "1", ".", "00", "at", "the", "store", "."],
      ids: [0, 1049, 4153, 2023, 6211, 2009, 1006, 1019, 1016, 4006, 2016, 2000, 3577, 1016, 2],
      decoded: "<s> i bought an apple for $ 1. 00 at the store. </s>",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["you", "\u2026"],
      ids: [0, 2021, 1533, 2],
      decoded: "<s> you \u2026 </s>",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["you", "\u2026"],
      ids: [0, 2021, 1533, 2],
      decoded: "<s> you \u2026 </s>",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["you", "\u2026", "you", "\u2026"],
      ids: [0, 2021, 1533, 2021, 1533, 2],
      decoded: "<s> you \u2026 you \u2026 </s>",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["weird", "\uff5e", "edge", "\uff5e", "case"],
      ids: [0, 6885, 1999, 3345, 1999, 2557, 2],
      decoded: "<s> weird \uff5e edge \uff5e case </s>",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      tokens: ["[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "."],
      ids: [0, 104, 104, 104, 104, 104, 1016, 2],
      decoded: "<s> [UNK] [UNK] [UNK] [UNK] [UNK]. </s>",
    },
    POPULAR_EMOJIS: {
      text: BASE_TEST_STRINGS.POPULAR_EMOJIS,
      tokens: ["[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]"],
      ids: [0, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 2],
      decoded: "<s> [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] </s>",
    },
    MULTIBYTE_EMOJIS: {
      text: BASE_TEST_STRINGS.MULTIBYTE_EMOJIS,
      tokens: ["[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]"],
      ids: [0, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 2],
      decoded: "<s> [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] </s>",
    },
  },
};
