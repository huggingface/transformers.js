import { Qwen2Tokenizer } from "../../../src/tokenizers.js";
import { BASE_TEST_STRINGS } from "../test_strings.js";

export const TOKENIZER_CLASS = Qwen2Tokenizer;
export const TEST_CONFIG = {
  "Xenova/Qwen1.5-0.5B-Chat": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      tokens: ["How", "\u0120are", "\u0120you", "\u0120doing", "?"],
      ids: [4340, 525, 498, 3730, 30],
      decoded: "How are you doing?",
    },
    SIMPLE_WITH_PUNCTUATION: {
      text: BASE_TEST_STRINGS.SIMPLE_WITH_PUNCTUATION,
      tokens: ["You", "\u0120should", "'ve", "\u0120done", "\u0120this"],
      ids: [2610, 1265, 3003, 2814, 419],
      decoded: "You should've done this",
    },
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "\u0120", "0", "\u0120", "1", "\u0120", "2", "\u0120", "3", "\u0120", "4", "\u0120", "5", "\u0120", "6", "\u0120", "7", "\u0120", "8", "\u0120", "9", "\u0120", "1", "0", "\u0120", "1", "0", "0", "\u0120", "1", "0", "0", "0"],
      ids: [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 220, 15, 220, 16, 220, 17, 220, 18, 220, 19, 220, 20, 220, 21, 220, 22, 220, 23, 220, 24, 220, 16, 15, 220, 16, 15, 15, 220, 16, 15, 15, 15],
      decoded: "0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["The", "\u0120company", "\u0120was", "\u0120founded", "\u0120in", "\u0120", "2", "0", "1", "6", "."],
      ids: [785, 2813, 572, 18047, 304, 220, 17, 15, 16, 21, 13],
      decoded: "The company was founded in 2016.",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["A", "\u010a", "'ll", "\u0120!!", "to", "?'", "d", "''", "d", "\u0120of", ",", "\u0120can", "'t", "."],
      ids: [32, 198, 3278, 11015, 983, 20224, 67, 4605, 67, 315, 11, 646, 944, 13],
      decoded: "A\n'll !!to?'d''d of, can't.",
    },
    PYTHON_CODE: {
      text: BASE_TEST_STRINGS.PYTHON_CODE,
      tokens: ["def", "\u0120main", "():\u010a", "\u0109pass"],
      ids: [750, 1887, 3932, 41431],
      decoded: "def main():\n\tpass",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["let", "\u0120a", "\u0120=", "\u0120obj", ".toString", "();\u010a", "toString", "();"],
      ids: [1149, 264, 284, 2839, 5070, 543, 6575, 2129],
      decoded: "let a = obj.toString();\ntoString();",
    },
    NEWLINES: {
      text: BASE_TEST_STRINGS.NEWLINES,
      tokens: ["This", "\u010a\u010a", "is", "\u010a", "a", "\u010a", "test", "."],
      ids: [1986, 271, 285, 198, 64, 198, 1944, 13],
      decoded: "This\n\nis\na\ntest.",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["UN", "want", "\u00c3\u00a9d", ",", "running"],
      ids: [1861, 52657, 15083, 11, 27173],
      decoded: "UNwant\u00e9d,running",
    },
    CONTROL_TOKENS: {
      text: BASE_TEST_STRINGS.CONTROL_TOKENS,
      tokens: ["1", "\u0100", "2", "\u00ef\u00bf\u00bd", "3"],
      ids: [16, 188, 17, 5691, 18],
      decoded: "1\u00002\ufffd3",
    },
    HELLO_WORLD_TITLECASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_TITLECASE,
      tokens: ["Hello", "\u0120World"],
      ids: [9707, 4337],
      decoded: "Hello World",
    },
    HELLO_WORLD_LOWERCASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_LOWERCASE,
      tokens: ["hello", "\u0120world"],
      ids: [14990, 1879],
      decoded: "hello world",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u00e7\u0136\u0141\u00e6\u00b4\u00bb\u00e7\u013c\u0126", "\u00e7\u013e\u0141", "\u00e8\u00b0\u013d", "\u00e6\u013a\u00af"],
      ids: [105301, 88051, 116109, 20412],
      decoded: "\u751f\u6d3b\u7684\u771f\u8c1b\u662f",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      tokens: ["\u0120\u0120", "\u0120leading", "\u0120space"],
      ids: [256, 6388, 3550],
      decoded: "   leading space",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["tr", "ailing", "\u0120space", "\u0120\u0120\u0120"],
      ids: [376, 14277, 3550, 262],
      decoded: "trailing space   ",
    },
    DOUBLE_SPACE: {
      text: BASE_TEST_STRINGS.DOUBLE_SPACE,
      tokens: ["Hi", "\u0120", "\u0120Hello"],
      ids: [13048, 220, 21927],
      decoded: "Hi  Hello",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["test", "\u0120$", "1", "\u0120R", "2", "\u0120#", "3", "\u0120\u00e2\u0124\u00ac", "4", "\u0120\u00c2\u00a3", "5", "\u0120\u00c2\u00a5", "6", "\u0120\u00e2", "\u0124", "\u00a3", "7", "\u0120\u00e2\u0124\u00b9", "8", "\u0120\u00e2", "\u0124", "\u00b1", "9", "\u0120test"],
      ids: [1944, 400, 16, 431, 17, 671, 18, 12984, 19, 6938, 20, 71488, 21, 2858, 224, 96, 22, 89791, 23, 2858, 224, 109, 24, 1273],
      decoded: "test $1 R2 #3 \u20ac4 \u00a35 \u00a56 \u20a37 \u20b98 \u20b19 test",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["I", "\u0120bought", "\u0120an", "\u0120apple", "\u0120for", "\u0120$", "1", ".", "0", "0", "\u0120at", "\u0120the", "\u0120store", "."],
      ids: [40, 10788, 458, 23268, 369, 400, 16, 13, 15, 15, 518, 279, 3553, 13],
      decoded: "I bought an apple for $1.00 at the store.",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u0120\u0120"],
      ids: [9330, 1940, 256],
      decoded: "you\u2026  ",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u00c2\u0142\u00c2\u0142"],
      ids: [9330, 1940, 9238],
      decoded: "you\u2026\u00a0\u00a0",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u00c2\u0142", "\u00c2\u0142", "you", "\u00e2\u0122\u00a6", "\u00c2\u0142\u00c2\u0142"],
      ids: [9330, 1940, 4102, 4102, 9330, 1940, 9238],
      decoded: "you\u2026\u00a0\u00a0you\u2026\u00a0\u00a0",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["we", "ird", "\u0120", "\u00ef\u00bd\u0140", "\u0120edge", "\u0120", "\u00ef\u00bd\u0140", "\u0120case"],
      ids: [896, 2603, 220, 21216, 6821, 220, 21216, 1142],
      decoded: "weird \uff5e edge \uff5e case",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      tokens: ["\u00e2\u0138", "\u0123", "This", "\u0120\u00e2\u0138", "\u0123", "is", "\u0120\u00e2\u0138", "\u0123", "a", "\u0120\u00e2\u0138", "\u0123", "test", "\u0120\u00e2\u0138", "\u0123", "."],
      ids: [10417, 223, 1986, 14520, 223, 285, 14520, 223, 64, 14520, 223, 1944, 14520, 223, 13],
      decoded: "\u2581This \u2581is \u2581a \u2581test \u2581.",
    },
    POPULAR_EMOJIS: {
      text: BASE_TEST_STRINGS.POPULAR_EMOJIS,
      tokens: ["\u00f0\u0141\u013a\u0124", "\u0120\u00f0\u0141\u0133", "\u012f", "\u0120\u00f0\u0141", "\u00a4", "\u00a3", "\u0120\u00f0\u0141\u013a", "\u012f", "\u0120\u00f0\u0141\u013a", "\u0143", "\u0120\u00f0\u0141", "\u0130", "\u012b", "\u0120\u00f0\u0141", "\u013b", "\u0131", "\u0120\u00f0\u0141\u013a", "\u012c", "\u0120\u00f0\u0141\u0136", "\u00a5", "\u0120\u00f0\u0141\u013a", "\u0123", "\u0120\u00f0\u0141\u013a", "\u0127", "\u0120\u00f0\u0141", "\u00a4", "\u0139", "\u0120\u00f0\u0141\u013a", "\u0128", "\u0120\u00f0\u0141\u0133", "\u0131", "\u0120\u00e2\u013f\u00a4", "\u00ef\u00b8\u0131", "\u0120\u00f0\u0141\u0134", "\u013e", "\u0120\u00f0\u0141\u0134", "\u013c", "\u0120\u00f0\u0141\u0134", "\u0139", "\u0120\u00f0\u0141\u0134", "\u013b", "\u0120\u00f0\u0141", "\u0138", "\u00a4", "\u0120\u00f0\u0141\u013a", "\u0130", "\u0120\u00f0\u0141\u0133", "\u012e", "\u0120\u00f0\u0141", "\u00a5", "\u00b3", "\u0120\u00f0\u0141\u0134", "\u00aa", "\u0120\u00e2\u013e", "\u00a8", "\u0120\u00f0\u0141\u0133", "\u012b", "\u0120\u00f0\u0141\u0133", "\u0122", "\u0120\u00f0\u0141\u0134", "\u00af", "\u0120\u00f0\u0141", "\u0130", "\u012a", "\u0120\u00f0\u0141", "\u013b", "\u012a", "\u0120\u00f0\u0141", "\u013b", "\u012e", "\u0120\u00f0\u0141\u0134", "\u0122", "\u0120\u00f0\u0141\u0133", "\u0129", "\u0120\u00f0\u0141\u0133", "\u012d", "\u0120\u00e2\u013e", "\u0127", "\u0120\u00f0\u0141", "\u0130", "\u0123", "\u0120\u00f0\u0141", "\u012e", "\u0140", "\u0120\u00f0\u0141", "\u012e", "\u00b8", "\u0120\u00f0\u0141\u0134", "\u00b0"],
      ids: [144185, 61804, 235, 11162, 97, 96, 26525, 235, 26525, 255, 11162, 236, 231, 11162, 247, 237, 26525, 232, 95069, 98, 26525, 223, 26525, 227, 11162, 97, 245, 26525, 228, 61804, 237, 70470, 30543, 63039, 250, 63039, 248, 63039, 245, 63039, 247, 11162, 244, 97, 26525, 236, 61804, 234, 11162, 98, 111, 63039, 103, 25521, 101, 61804, 231, 61804, 222, 63039, 107, 11162, 236, 230, 11162, 247, 230, 11162, 247, 234, 63039, 222, 61804, 229, 61804, 233, 25521, 227, 11162, 236, 223, 11162, 234, 252, 11162, 234, 116, 63039, 108],
      decoded: "\ud83d\ude02 \ud83d\udc4d \ud83e\udd23 \ud83d\ude0d \ud83d\ude2d \ud83c\udf89 \ud83d\ude4f \ud83d\ude0a \ud83d\udd25 \ud83d\ude01 \ud83d\ude05 \ud83e\udd17 \ud83d\ude06 \ud83d\udc4f \u2764\ufe0f \ud83d\udc9c \ud83d\udc9a \ud83d\udc97 \ud83d\udc99 \ud83d\udda4 \ud83d\ude0e \ud83d\udc4c \ud83e\udd73 \ud83d\udcaa \u2728 \ud83d\udc49 \ud83d\udc40 \ud83d\udcaf \ud83c\udf88 \ud83d\ude48 \ud83d\ude4c \ud83d\udc80 \ud83d\udc47 \ud83d\udc4b \u2705 \ud83c\udf81 \ud83c\udf1e \ud83c\udf38 \ud83d\udcb0",
    },
    MULTIBYTE_EMOJIS: {
      text: BASE_TEST_STRINGS.MULTIBYTE_EMOJIS,
      tokens: ["\u00e2\u013e\u00a8", "\u0120\u00f0\u0141", "\u00a4", "\u0139", "\u0120\u00f0\u0141\u0133", "\u0123", "\u00ef\u00b8\u0131", "\u0120\u00f0\u0141\u0133", "\u00b1", "\u00f0\u0141\u0131\u00bb", "\u0120\u00f0\u0141", "\u0137", "\u00b5", "\u00e2\u0122", "\u012f", "\u00e2\u013b\u0124", "\u00ef\u00b8\u0131", "\u0120\u00f0\u0141", "\u00a7", "\u013b", "\u00f0\u0141\u0131\u00bb", "\u00e2\u0122", "\u012f", "\u00e2\u013b\u0124", "\u0120\u00f0\u0141\u0133", "\u00a8", "\u00f0\u0141\u0131\u00bb", "\u00e2\u0122", "\u012f", "\u00f0\u0141\u012e\u00be", "\u0120\u00f0\u0141", "\u00a7", "\u0133", "\u00e2\u0122", "\u012f", "\u00f0\u0141\u00a4\u013f", "\u00e2\u0122", "\u012f", "\u00f0\u0141\u00a7\u0133", "\u0120\u00f0\u0141\u0133", "\u00a9", "\u00e2\u0122", "\u012f", "\u00e2\u013f\u00a4", "\u00e2\u0122", "\u012f", "\u00f0\u0141\u0134\u012d", "\u00e2\u0122", "\u012f", "\u00f0\u0141\u0133\u00a8", "\u0120\u00f0\u0141\u0133", "\u00a9", "\u00e2\u0122", "\u012f", "\u00f0\u0141\u0133\u00a9", "\u00e2\u0122", "\u012f", "\u00f0\u0141\u0133\u00a7", "\u00e2\u0122", "\u012f", "\u00f0\u0141\u0133\u00a6", "\u0120\u00f0\u0141", "\u00a7", "\u0133", "\u00f0\u0141\u0131\u00bb", "\u00e2\u0122", "\u012f", "\u00f0\u0141\u00a4\u013f", "\u00e2\u0122", "\u012f", "\u00f0\u0141\u00a7\u0133", "\u00f0\u0141\u0131\u00bb", "\u0120\u00f0\u0141", "\u0131", "\u00b4", "\u00f3", "\u0142\u0123", "\u00a7", "\u00f3", "\u0142\u0123", "\u00a2", "\u00f3", "\u0142\u0123", "\u00a5", "\u00f3", "\u0142\u0123", "\u00ae", "\u00f3", "\u0142\u0123", "\u00a7", "\u00f3", "\u0142\u0123", "\u00bf", "\u0120\u00f0\u0141\u0133", "\u00a8", "\u00f0\u0141\u0131\u00bb", "\u00e2\u0122", "\u012f", "\u00e2\u013f\u00a4", "\u00ef\u00b8\u0131", "\u00e2\u0122", "\u012f", "\u00f0\u0141\u0134\u012d", "\u00e2\u0122", "\u012f", "\u00f0\u0141\u0133\u00a8", "\u00f0\u0141\u0131\u00bc"],
      ids: [144232, 11162, 97, 245, 61804, 223, 30543, 61804, 109, 144321, 11162, 243, 113, 378, 235, 144693, 30543, 11162, 100, 247, 144321, 378, 235, 144693, 61804, 101, 144321, 378, 235, 146467, 11162, 100, 239, 378, 235, 146392, 378, 235, 148738, 61804, 102, 378, 235, 141390, 378, 235, 145002, 378, 235, 145367, 61804, 102, 378, 235, 145233, 378, 235, 145665, 378, 235, 145988, 11162, 100, 239, 144321, 378, 235, 146392, 378, 235, 148738, 144321, 11162, 237, 112, 175, 15675, 100, 175, 15675, 95, 175, 15675, 98, 175, 15675, 106, 175, 15675, 100, 175, 15675, 123, 61804, 101, 144321, 378, 235, 141390, 30543, 378, 235, 145002, 378, 235, 145367, 144784],
      decoded: "\u2728 \ud83e\udd17 \ud83d\udc41\ufe0f \ud83d\udc71\ud83c\udffb \ud83d\udd75\u200d\u2642\ufe0f \ud83e\uddd9\ud83c\udffb\u200d\u2642 \ud83d\udc68\ud83c\udffb\u200d\ud83c\udf3e \ud83e\uddd1\u200d\ud83e\udd1d\u200d\ud83e\uddd1 \ud83d\udc69\u200d\u2764\u200d\ud83d\udc8b\u200d\ud83d\udc68 \ud83d\udc69\u200d\ud83d\udc69\u200d\ud83d\udc67\u200d\ud83d\udc66 \ud83e\uddd1\ud83c\udffb\u200d\ud83e\udd1d\u200d\ud83e\uddd1\ud83c\udffb \ud83c\udff4\udb40\udc67\udb40\udc62\udb40\udc65\udb40\udc6e\udb40\udc67\udb40\udc7f \ud83d\udc68\ud83c\udffb\u200d\u2764\ufe0f\u200d\ud83d\udc8b\u200d\ud83d\udc68\ud83c\udffc",
    },
  },
};
