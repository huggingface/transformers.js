import { BloomTokenizer } from "../../../src/tokenizers.js";
import { BASE_TEST_STRINGS, BLOOM_TEST_STRINGS, SENTENCEPIECE_TEST_STRINGS } from "../test_strings.js";

export const TOKENIZER_CLASS = BloomTokenizer;
export const TEST_CONFIG = {
  "Xenova/bloom-560m": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      tokens: ["How", "\u0120are", "\u0120you", "\u0120doing", "?"],
      ids: [7572, 1306, 1152, 12491, 34],
      decoded: "How are you doing?",
    },
    SIMPLE_WITH_PUNCTUATION: {
      text: BASE_TEST_STRINGS.SIMPLE_WITH_PUNCTUATION,
      tokens: ["You", "\u0120should", "'ve", "\u0120done", "\u0120this"],
      ids: [5448, 3403, 7300, 11541, 1119],
      decoded: "You should've done this",
    },
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["0123", "456789", "\u01200", "\u01201", "\u01202", "\u01203", "\u01204", "\u01205", "\u01206", "\u01207", "\u01208", "\u01209", "\u012010", "\u0120100", "\u01201000"],
      ids: [166660, 145647, 931, 404, 415, 735, 934, 973, 1231, 1392, 1445, 1575, 1581, 4334, 19526],
      decoded: "0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["The", "\u0120company", "\u0120was", "\u0120founded", "\u0120in", "\u01202016", "."],
      ids: [2175, 16333, 1620, 88289, 361, 5854, 17],
      decoded: "The company was founded in 2016.",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["A", "\u010a", "'ll", "\u0120!!", "to", "?", "'d", "''", "d", "\u0120of", ",", "\u0120can't", "."],
      ids: [36, 189, 8722, 49825, 1025, 34, 10628, 2328, 71, 461, 15, 11229, 17],
      decoded: "A\n'll !!to?'d''d of, can't.",
    },
    PYTHON_CODE: {
      text: BASE_TEST_STRINGS.PYTHON_CODE,
      tokens: ["def", "\u0120main", "()", ":", "\u010a\u0109", "pass"],
      ids: [7564, 4291, 883, 29, 1582, 12608],
      decoded: "def main():\n\tpass",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["let", "\u0120a", "\u0120=", "\u0120obj", ".", "toString", "()", ";", "\u010a", "toString", "()", ";"],
      ids: [2963, 267, 564, 17949, 17, 27392, 883, 30, 189, 27392, 883, 30],
      decoded: "let a = obj.toString();\ntoString();",
    },
    NEWLINES: {
      text: BASE_TEST_STRINGS.NEWLINES,
      tokens: ["This", "\u010a\u010a", "is", "\u010a", "a", "\u010a", "test", "."],
      ids: [6168, 603, 290, 189, 68, 189, 9234, 17],
      decoded: "This\n\nis\na\ntest.",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["UN", "want", "\u00c3\u00a9d", ",", "running"],
      ids: [5777, 75642, 2454, 15, 101897],
      decoded: "UNwant\u00e9d,running",
    },
    CONTROL_TOKENS: {
      text: BASE_TEST_STRINGS.CONTROL_TOKENS,
      tokens: ["1", "\u0100", "2", "\u00ef\u00bf\u00bd", "3"],
      ids: [20, 179, 21, 23181, 22],
      decoded: "1\u00002\ufffd3",
    },
    HELLO_WORLD_TITLECASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_TITLECASE,
      tokens: ["Hello", "\u0120World"],
      ids: [59414, 12155],
      decoded: "Hello World",
    },
    HELLO_WORLD_LOWERCASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_LOWERCASE,
      tokens: ["hello", "\u0120world"],
      ids: [101579, 8876],
      decoded: "hello world",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u00e7\u0136\u0141\u00e6\u00b4\u00bb\u00e7\u013c\u0126", "\u00e7\u013e\u0141", "\u00e8\u00b0", "\u013d", "\u00e6\u013a\u00af"],
      ids: [71167, 4137, 1927, 239, 644],
      decoded: "\u751f\u6d3b\u7684\u771f\u8c1b\u662f",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      tokens: ["\u0120\u0120", "\u0120leading", "\u0120space"],
      ids: [250, 36128, 12978],
      decoded: "   leading space",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["tra", "iling", "\u0120space", "\u0120\u0120\u0120"],
      ids: [1900, 17022, 12978, 416],
      decoded: "trailing space   ",
    },
    SURROUNDING_SPACE: {
      text: BASE_TEST_STRINGS.SURROUNDING_SPACE,
      tokens: ["\u0120\u0120", "\u0120surrounding", "\u0120space", "\u0120\u0120\u0120"],
      ids: [250, 66599, 12978, 416],
      decoded: "   surrounding space   ",
    },
    DOUBLE_SPACE: {
      text: BASE_TEST_STRINGS.DOUBLE_SPACE,
      tokens: ["Hi", "\u0120", "\u0120Hello"],
      ids: [30050, 210, 86153],
      decoded: "Hi  Hello",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["test", "\u0120$1", "\u0120R2", "\u0120#3", "\u0120\u00e2\u0124\u00ac", "4", "\u0120\u00c2\u00a3", "5", "\u0120\u00c2\u00a5", "6", "\u0120\u00e2\u0124", "\u00a3", "7", "\u0120\u00e2\u0124\u00b9", "8", "\u0120\u00e2\u0124", "\u00b1", "9", "\u0120test"],
      ids: [9234, 41448, 80774, 201642, 20117, 23, 40300, 24, 62153, 25, 72279, 100, 26, 120434, 27, 72279, 113, 28, 4006],
      decoded: "test $1 R2 #3 \u20ac4 \u00a35 \u00a56 \u20a37 \u20b98 \u20b19 test",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["I", "\u0120bought", "\u0120an", "\u0120apple", "\u0120for", "\u0120$1", ".", "00", "\u0120at", "\u0120the", "\u0120store", "."],
      ids: [44, 87926, 660, 101091, 613, 41448, 17, 462, 919, 368, 18706, 17],
      decoded: "I bought an apple for $1.00 at the store.",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u0120\u0120"],
      ids: [23438, 4346, 250],
      decoded: "you\u2026  ",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u00c2\u0142\u00c2\u0142"],
      ids: [23438, 4346, 12361],
      decoded: "you\u2026\u00a0\u00a0",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u00c2\u0142\u00c2\u0142", "you", "\u00e2\u0122\u00a6", "\u00c2\u0142\u00c2\u0142"],
      ids: [23438, 4346, 12361, 23438, 4346, 12361],
      decoded: "you\u2026\u00a0\u00a0you\u2026\u00a0\u00a0",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["we", "ird", "\u0120\u00ef\u00bd", "\u0140", "\u0120edge", "\u0120\u00ef\u00bd", "\u0140", "\u0120case"],
      ids: [2136, 7589, 122354, 242, 29655, 122354, 242, 4462],
      decoded: "weird \uff5e edge \uff5e case",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      tokens: ["\u00e2\u0138", "\u0123", "This", "\u0120\u00e2\u0138", "\u0123", "is", "\u0120\u00e2\u0138", "\u0123", "a", "\u0120\u00e2\u0138", "\u0123", "test", "\u0120\u00e2\u0138", "\u0123", "."],
      ids: [26127, 213, 6168, 15299, 213, 290, 15299, 213, 68, 15299, 213, 9234, 15299, 213, 17],
      decoded: "\u2581This \u2581is \u2581a \u2581test \u2581.",
    },
    POPULAR_EMOJIS: {
      text: BASE_TEST_STRINGS.POPULAR_EMOJIS,
      tokens: ["\u00f0\u0141\u013a", "\u0124", "\u0120\u00f0\u0141", "\u0133", "\u012f", "\u0120\u00f0\u0141", "\u00a4", "\u00a3", "\u0120\u00f0\u0141\u013a", "\u012f", "\u0120\u00f0\u0141\u013a", "\u0143", "\u0120\u00f0\u0141", "\u0130", "\u012b", "\u0120\u00f0\u0141", "\u013b", "\u0131", "\u0120\u00f0\u0141\u013a", "\u012c", "\u0120\u00f0\u0141", "\u0136", "\u00a5", "\u0120\u00f0\u0141\u013a", "\u0123", "\u0120\u00f0\u0141\u013a", "\u0127", "\u0120\u00f0\u0141", "\u00a4", "\u0139", "\u0120\u00f0\u0141\u013a", "\u0128", "\u0120\u00f0\u0141", "\u0133", "\u0131", "\u0120\u00e2\u013f", "\u00a4", "\u00ef\u00b8\u0131", "\u0120\u00f0\u0141", "\u0134", "\u013e", "\u0120\u00f0\u0141", "\u0134", "\u013c", "\u0120\u00f0\u0141", "\u0134", "\u0139", "\u0120\u00f0\u0141", "\u0134", "\u013b", "\u0120\u00f0\u0141", "\u0138", "\u00a4", "\u0120\u00f0\u0141\u013a", "\u0130", "\u0120\u00f0\u0141", "\u0133", "\u012e", "\u0120\u00f0\u0141", "\u00a5", "\u00b3", "\u0120\u00f0\u0141", "\u0134", "\u00aa", "\u0120\u00e2\u013e", "\u00a8", "\u0120\u00f0\u0141", "\u0133", "\u012b", "\u0120\u00f0\u0141", "\u0133", "\u0122", "\u0120\u00f0\u0141", "\u0134", "\u00af", "\u0120\u00f0\u0141", "\u0130", "\u012a", "\u0120\u00f0\u0141", "\u013b", "\u012a", "\u0120\u00f0\u0141", "\u013b", "\u012e", "\u0120\u00f0\u0141", "\u0134", "\u0122", "\u0120\u00f0\u0141", "\u0133", "\u0129", "\u0120\u00f0\u0141", "\u0133", "\u012d", "\u0120\u00e2\u013e", "\u0127", "\u0120\u00f0\u0141", "\u0130", "\u0123", "\u0120\u00f0\u0141", "\u012e", "\u0140", "\u0120\u00f0\u0141", "\u012e", "\u00b8", "\u0120\u00f0\u0141", "\u0134", "\u00b0"],
      ids: [127322, 214, 41234, 229, 225, 41234, 101, 100, 126342, 225, 126342, 245, 41234, 226, 221, 41234, 237, 227, 126342, 222, 41234, 232, 102, 126342, 213, 126342, 217, 41234, 101, 235, 126342, 218, 41234, 229, 227, 189367, 101, 116057, 41234, 230, 240, 41234, 230, 238, 41234, 230, 235, 41234, 230, 237, 41234, 234, 101, 126342, 226, 41234, 229, 224, 41234, 102, 115, 41234, 230, 107, 76758, 105, 41234, 229, 221, 41234, 229, 212, 41234, 230, 111, 41234, 226, 220, 41234, 237, 220, 41234, 237, 224, 41234, 230, 212, 41234, 229, 219, 41234, 229, 223, 76758, 217, 41234, 226, 213, 41234, 224, 242, 41234, 224, 120, 41234, 230, 112],
      decoded: "\ud83d\ude02 \ud83d\udc4d \ud83e\udd23 \ud83d\ude0d \ud83d\ude2d \ud83c\udf89 \ud83d\ude4f \ud83d\ude0a \ud83d\udd25 \ud83d\ude01 \ud83d\ude05 \ud83e\udd17 \ud83d\ude06 \ud83d\udc4f \u2764\ufe0f \ud83d\udc9c \ud83d\udc9a \ud83d\udc97 \ud83d\udc99 \ud83d\udda4 \ud83d\ude0e \ud83d\udc4c \ud83e\udd73 \ud83d\udcaa \u2728 \ud83d\udc49 \ud83d\udc40 \ud83d\udcaf \ud83c\udf88 \ud83d\ude48 \ud83d\ude4c \ud83d\udc80 \ud83d\udc47 \ud83d\udc4b \u2705 \ud83c\udf81 \ud83c\udf1e \ud83c\udf38 \ud83d\udcb0",
    },
    MULTIBYTE_EMOJIS: {
      text: BASE_TEST_STRINGS.MULTIBYTE_EMOJIS,
      tokens: ["\u00e2\u013e", "\u00a8", "\u0120\u00f0\u0141", "\u00a4", "\u0139", "\u0120\u00f0\u0141", "\u0133", "\u0123", "\u00ef\u00b8\u0131", "\u0120\u00f0\u0141", "\u0133", "\u00b1", "\u00f0\u0141\u0131", "\u00bb", "\u0120\u00f0\u0141", "\u0137", "\u00b5", "\u00e2\u0122\u012f", "\u00e2\u013b", "\u0124", "\u00ef\u00b8\u0131", "\u0120\u00f0\u0141", "\u00a7", "\u013b", "\u00f0\u0141\u0131", "\u00bb", "\u00e2\u0122\u012f", "\u00e2\u013b", "\u0124", "\u0120\u00f0\u0141", "\u0133", "\u00a8", "\u00f0\u0141\u0131", "\u00bb", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u012e", "\u00be", "\u0120\u00f0\u0141", "\u00a7", "\u0133", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u00a4", "\u013f", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u00a7", "\u0133", "\u0120\u00f0\u0141", "\u0133", "\u00a9", "\u00e2\u0122\u012f", "\u00e2\u013f", "\u00a4", "\u00e2\u0122\u012f", "\u00f0\u0141\u0134", "\u012d", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u0133", "\u00a8", "\u0120\u00f0\u0141", "\u0133", "\u00a9", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u0133", "\u00a9", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u0133", "\u00a7", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u0133", "\u00a6", "\u0120\u00f0\u0141", "\u00a7", "\u0133", "\u00f0\u0141\u0131", "\u00bb", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u00a4", "\u013f", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u00a7", "\u0133", "\u00f0\u0141\u0131", "\u00bb", "\u0120\u00f0\u0141", "\u0131", "\u00b4", "\u00f3", "\u0142", "\u0123", "\u00a7", "\u00f3", "\u0142", "\u0123", "\u00a2", "\u00f3", "\u0142", "\u0123", "\u00a5", "\u00f3", "\u0142", "\u0123", "\u00ae", "\u00f3", "\u0142", "\u0123", "\u00a7", "\u00f3", "\u0142", "\u0123", "\u00bf", "\u0120\u00f0\u0141", "\u0133", "\u00a8", "\u00f0\u0141\u0131", "\u00bb", "\u00e2\u0122\u012f", "\u00e2\u013f", "\u00a4", "\u00ef\u00b8\u0131", "\u00e2\u0122\u012f", "\u00f0\u0141\u0134", "\u012d", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u0133", "\u00a8", "\u00f0\u0141\u0131", "\u00bc"],
      ids: [120709, 105, 41234, 101, 235, 41234, 229, 213, 116057, 41234, 229, 113, 244635, 123, 41234, 233, 117, 1553, 15596, 214, 116057, 41234, 104, 237, 244635, 123, 1553, 15596, 214, 41234, 229, 105, 244635, 123, 1553, 22618, 224, 126, 41234, 104, 229, 1553, 22618, 101, 241, 1553, 22618, 104, 229, 41234, 229, 106, 1553, 157147, 101, 1553, 139500, 223, 1553, 22618, 229, 105, 41234, 229, 106, 1553, 22618, 229, 106, 1553, 22618, 229, 104, 1553, 22618, 229, 103, 41234, 104, 229, 244635, 123, 1553, 22618, 101, 241, 1553, 22618, 104, 229, 244635, 123, 41234, 227, 116, 177, 244, 213, 104, 177, 244, 213, 99, 177, 244, 213, 102, 177, 244, 213, 110, 177, 244, 213, 104, 177, 244, 213, 127, 41234, 229, 105, 244635, 123, 1553, 157147, 101, 116057, 1553, 139500, 223, 1553, 22618, 229, 105, 244635, 124],
      decoded: "\u2728 \ud83e\udd17 \ud83d\udc41\ufe0f \ud83d\udc71\ud83c\udffb \ud83d\udd75\u200d\u2642\ufe0f \ud83e\uddd9\ud83c\udffb\u200d\u2642 \ud83d\udc68\ud83c\udffb\u200d\ud83c\udf3e \ud83e\uddd1\u200d\ud83e\udd1d\u200d\ud83e\uddd1 \ud83d\udc69\u200d\u2764\u200d\ud83d\udc8b\u200d\ud83d\udc68 \ud83d\udc69\u200d\ud83d\udc69\u200d\ud83d\udc67\u200d\ud83d\udc66 \ud83e\uddd1\ud83c\udffb\u200d\ud83e\udd1d\u200d\ud83e\uddd1\ud83c\udffb \ud83c\udff4\udb40\udc67\udb40\udc62\udb40\udc65\udb40\udc6e\udb40\udc67\udb40\udc7f \ud83d\udc68\ud83c\udffb\u200d\u2764\ufe0f\u200d\ud83d\udc8b\u200d\ud83d\udc68\ud83c\udffc",
    },
    ONLY_WHITESPACE: {
      text: BASE_TEST_STRINGS.ONLY_WHITESPACE,
      tokens: ["\u0120\u0109", "\u010a"],
      ids: [33651, 189],
      decoded: " \t\n",
    },
    END_OF_SENTENCE_PUNCTUATION: {
      text: BLOOM_TEST_STRINGS.END_OF_SENTENCE_PUNCTUATION,
      tokens: ["test", ".", "\u0120test", ",", "\u0120test", "!", "\u0120test", "?", "\u0120test", "\u00e2\u0122\u00a6", "\u0120test", "\u00e3\u0122\u0124", "\u0120test", "\u00ef\u00bc\u012e", "\u0120test", "\u00e3\u0122\u0123", "\u0120test", "\u00e0\u00a5\u00a4", "\u0120test", "\u00db\u0136", "\u0120test", "\u00d8\u012e", "\u0120test"],
      ids: [9234, 17, 4006, 15, 4006, 4, 4006, 34, 4006, 4346, 4006, 420, 4006, 355, 4006, 594, 4006, 527, 4006, 1174, 4006, 687, 4006],
      decoded: "test. test, test! test? test\u2026 test\u3002 test\uff0c test\u3001 test\u0964 test\u06d4 test\u060c test",
    },
    SPECIAL_WITH_TRAILING_WHITESPACE: {
      text: SENTENCEPIECE_TEST_STRINGS.SPECIAL_WITH_TRAILING_WHITESPACE,
      tokens: ["<s>", "\u010a"],
      ids: [1, 189],
      decoded: "<s>\n",
    },
    SPECIAL_SURROUNDED_BY_WHITESPACE: {
      text: SENTENCEPIECE_TEST_STRINGS.SPECIAL_SURROUNDED_BY_WHITESPACE,
      tokens: ["\u0120", "</s>", "\u0120test", "\u0120", "</s>", "\u0120"],
      ids: [210, 2, 4006, 210, 2, 210],
      decoded: " </s> test </s> ",
    },
    SPECIAL_NO_WHITESPACE: {
      text: SENTENCEPIECE_TEST_STRINGS.SPECIAL_NO_WHITESPACE,
      tokens: ["</s>", "test", "</s>"],
      ids: [2, 9234, 2],
      decoded: "</s>test</s>",
    },
  },
};
