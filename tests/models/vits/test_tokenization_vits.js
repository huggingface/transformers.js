import { VitsTokenizer } from "../../../src/tokenizers.js";
import { BASE_TEST_STRINGS, VITS_TEST_STRINGS } from "../test_strings.js";

export const TOKENIZER_CLASS = VitsTokenizer;
export const TEST_CONFIG = {
  "Xenova/mms-tts-eng": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      tokens: ["k", "h", "k", "o", "k", "w", "k", " ", "k", "a", "k", "r", "k", "e", "k", " ", "k", "y", "k", "o", "k", "u", "k", " ", "k", "d", "k", "o", "k", "i", "k", "n", "k", "g", "k"],
      ids: [0, 6, 0, 22, 0, 9, 0, 19, 0, 26, 0, 25, 0, 7, 0, 19, 0, 3, 0, 22, 0, 4, 0, 19, 0, 5, 0, 22, 0, 18, 0, 29, 0, 37, 0],
      decoded: "how are you doing",
    },
    SIMPLE_WITH_PUNCTUATION: {
      text: BASE_TEST_STRINGS.SIMPLE_WITH_PUNCTUATION,
      tokens: ["k", "y", "k", "o", "k", "u", "k", " ", "k", "s", "k", "h", "k", "o", "k", "u", "k", "l", "k", "d", "k", "'", "k", "v", "k", "e", "k", " ", "k", "d", "k", "o", "k", "n", "k", "e", "k", " ", "k", "t", "k", "h", "k", "i", "k", "s", "k"],
      ids: [0, 3, 0, 22, 0, 4, 0, 19, 0, 8, 0, 6, 0, 22, 0, 4, 0, 21, 0, 5, 0, 1, 0, 32, 0, 7, 0, 19, 0, 5, 0, 22, 0, 29, 0, 7, 0, 19, 0, 33, 0, 6, 0, 18, 0, 8, 0],
      decoded: "you should've done this",
    },
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["k", "0", "k", "1", "k", "2", "k", "3", "k", "4", "k", "5", "k", "6", "k", " ", "k", "0", "k", " ", "k", "1", "k", " ", "k", "2", "k", " ", "k", "3", "k", " ", "k", "4", "k", " ", "k", "5", "k", " ", "k", "6", "k", " ", "k", " ", "k", " ", "k", " ", "k", "1", "k", "0", "k", " ", "k", "1", "k", "0", "k", "0", "k", " ", "k", "1", "k", "0", "k", "0", "k", "0", "k"],
      ids: [0, 23, 0, 15, 0, 28, 0, 11, 0, 27, 0, 35, 0, 36, 0, 19, 0, 23, 0, 19, 0, 15, 0, 19, 0, 28, 0, 19, 0, 11, 0, 19, 0, 27, 0, 19, 0, 35, 0, 19, 0, 36, 0, 19, 0, 19, 0, 19, 0, 19, 0, 15, 0, 23, 0, 19, 0, 15, 0, 23, 0, 23, 0, 19, 0, 15, 0, 23, 0, 23, 0, 23, 0],
      decoded: "0123456 0 1 2 3 4 5 6    10 100 1000",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["k", "t", "k", "h", "k", "e", "k", " ", "k", "c", "k", "o", "k", "m", "k", "p", "k", "a", "k", "n", "k", "y", "k", " ", "k", "w", "k", "a", "k", "s", "k", " ", "k", "f", "k", "o", "k", "u", "k", "n", "k", "d", "k", "e", "k", "d", "k", " ", "k", "i", "k", "n", "k", " ", "k", "2", "k", "0", "k", "1", "k", "6", "k"],
      ids: [0, 33, 0, 6, 0, 7, 0, 19, 0, 12, 0, 22, 0, 17, 0, 13, 0, 26, 0, 29, 0, 3, 0, 19, 0, 9, 0, 26, 0, 8, 0, 19, 0, 20, 0, 22, 0, 4, 0, 29, 0, 5, 0, 7, 0, 5, 0, 19, 0, 18, 0, 29, 0, 19, 0, 28, 0, 23, 0, 15, 0, 36, 0],
      decoded: "the company was founded in 2016",
    },
    NEWLINES: {
      text: BASE_TEST_STRINGS.NEWLINES,
      tokens: ["k", "t", "k", "h", "k", "i", "k", "s", "k", "i", "k", "s", "k", "a", "k", "t", "k", "e", "k", "s", "k", "t", "k"],
      ids: [0, 33, 0, 6, 0, 18, 0, 8, 0, 18, 0, 8, 0, 26, 0, 33, 0, 7, 0, 8, 0, 33, 0],
      decoded: "thisisatest",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: [],
      ids: [],
      decoded: "",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      tokens: ["k", "l", "k", "e", "k", "a", "k", "d", "k", "i", "k", "n", "k", "g", "k", " ", "k", "s", "k", "p", "k", "a", "k", "c", "k", "e", "k"],
      ids: [0, 21, 0, 7, 0, 26, 0, 5, 0, 18, 0, 29, 0, 37, 0, 19, 0, 8, 0, 13, 0, 26, 0, 12, 0, 7, 0],
      decoded: "leading space",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["k", "t", "k", "r", "k", "a", "k", "i", "k", "l", "k", "i", "k", "n", "k", "g", "k", " ", "k", "s", "k", "p", "k", "a", "k", "c", "k", "e", "k"],
      ids: [0, 33, 0, 25, 0, 26, 0, 18, 0, 21, 0, 18, 0, 29, 0, 37, 0, 19, 0, 8, 0, 13, 0, 26, 0, 12, 0, 7, 0],
      decoded: "trailing space",
    },
    SURROUNDING_SPACE: {
      text: BASE_TEST_STRINGS.SURROUNDING_SPACE,
      tokens: ["k", "s", "k", "u", "k", "r", "k", "r", "k", "o", "k", "u", "k", "n", "k", "d", "k", "i", "k", "n", "k", "g", "k", " ", "k", "s", "k", "p", "k", "a", "k", "c", "k", "e", "k"],
      ids: [0, 8, 0, 4, 0, 25, 0, 25, 0, 22, 0, 4, 0, 29, 0, 5, 0, 18, 0, 29, 0, 37, 0, 19, 0, 8, 0, 13, 0, 26, 0, 12, 0, 7, 0],
      decoded: "surrounding space",
    },
    SPECIAL_CHARACTERS: {
      text: VITS_TEST_STRINGS.SPECIAL_CHARACTERS,
      tokens: [],
      ids: [],
      decoded: "",
    },
  },
  "Xenova/mms-tts-ron": {
    SPECIAL_CHARACTERS: {
      text: VITS_TEST_STRINGS.SPECIAL_CHARACTERS,
      tokens: ["c", "\u0163", "c", " ", "c", "\u0163", "c"],
      ids: [0, 32, 0, 28, 0, 32, 0],
      decoded: "\u0163 \u0163",
    },
  },
};
