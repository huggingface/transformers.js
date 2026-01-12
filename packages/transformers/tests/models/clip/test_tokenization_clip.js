import { CLIPTokenizer } from "../../../src/tokenizers.js";
import { BASE_TEST_STRINGS } from "../test_strings.js";

export const TOKENIZER_CLASS = CLIPTokenizer;
export const TEST_CONFIG = {
  "Xenova/clip-vit-base-patch16": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      tokens: ["how</w>", "are</w>", "you</w>", "doing</w>", "?</w>"],
      ids: [49406, 829, 631, 592, 1960, 286, 49407],
      decoded: "<|startoftext|>how are you doing? <|endoftext|>",
    },
    SIMPLE_WITH_PUNCTUATION: {
      text: BASE_TEST_STRINGS.SIMPLE_WITH_PUNCTUATION,
      tokens: ["you</w>", "should</w>", "'ve</w>", "done</w>", "this</w>"],
      ids: [49406, 592, 1535, 1200, 1700, 589, 49407],
      decoded: "<|startoftext|>you should've done this <|endoftext|>",
    },
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["0</w>", "1</w>", "2</w>", "3</w>", "4</w>", "5</w>", "6</w>", "7</w>", "8</w>", "9</w>", "0</w>", "1</w>", "2</w>", "3</w>", "4</w>", "5</w>", "6</w>", "7</w>", "8</w>", "9</w>", "1</w>", "0</w>", "1</w>", "0</w>", "0</w>", "1</w>", "0</w>", "0</w>", "0</w>"],
      ids: [49406, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 272, 271, 272, 271, 271, 272, 271, 271, 271, 49407],
      decoded: "<|startoftext|>0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 1 0 1 0 0 1 0 0 0 <|endoftext|>",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["the</w>", "company</w>", "was</w>", "founded</w>", "in</w>", "2</w>", "0</w>", "1</w>", "6</w>", ".</w>"],
      ids: [49406, 518, 2634, 739, 12240, 530, 273, 271, 272, 277, 269, 49407],
      decoded: "<|startoftext|>the company was founded in 2 0 1 6. <|endoftext|>",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["a</w>", "'ll</w>", "!!</w>", "to</w>", "?'</w>", "d</w>", "''</w>", "d</w>", "of</w>", ",</w>", "can</w>", "'t</w>", ".</w>"],
      ids: [49406, 320, 1342, 748, 531, 13610, 323, 8445, 323, 539, 267, 753, 713, 269, 49407],
      decoded: "<|startoftext|>a 'll!! to?' d '' d of, can 't. <|endoftext|>",
    },
    PYTHON_CODE: {
      text: BASE_TEST_STRINGS.PYTHON_CODE,
      tokens: ["def</w>", "main</w>", "(", "):</w>", "pass</w>"],
      ids: [49406, 11649, 2623, 7, 4143, 3511, 49407],
      decoded: "<|startoftext|>def main (): pass <|endoftext|>",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["let</w>", "a</w>", "=</w>", "ob", "j</w>", ".</w>", "to", "string</w>", "(", ");</w>", "to", "string</w>", "(", ");</w>"],
      ids: [49406, 1094, 320, 284, 1411, 329, 269, 580, 9696, 7, 19686, 580, 9696, 7, 19686, 49407],
      decoded: "<|startoftext|>let a = obj. tostring (); tostring (); <|endoftext|>",
    },
    NEWLINES: {
      text: BASE_TEST_STRINGS.NEWLINES,
      tokens: ["this</w>", "is</w>", "a</w>", "test</w>", ".</w>"],
      ids: [49406, 589, 533, 320, 1628, 269, 49407],
      decoded: "<|startoftext|>this is a test. <|endoftext|>",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["un", "want", "\u00c3\u00a9", "d</w>", ",</w>", "running</w>"],
      ids: [49406, 569, 18356, 3459, 323, 267, 2761, 49407],
      decoded: "<|startoftext|>unwant\u00e9d, running <|endoftext|>",
    },
    CONTROL_TOKENS: {
      text: BASE_TEST_STRINGS.CONTROL_TOKENS,
      tokens: ["1</w>", "\u0100</w>", "2</w>", "\u00ef\u00bf\u00bd</w>", "3</w>"],
      ids: [49406, 272, 444, 273, 39802, 274, 49407],
      decoded: "<|startoftext|>1 \u0000 2 \ufffd 3 <|endoftext|>",
    },
    HELLO_WORLD_TITLECASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_TITLECASE,
      tokens: ["hello</w>", "world</w>"],
      ids: [49406, 3306, 1002, 49407],
      decoded: "<|startoftext|>hello world <|endoftext|>",
    },
    HELLO_WORLD_LOWERCASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_LOWERCASE,
      tokens: ["hello</w>", "world</w>"],
      ids: [49406, 3306, 1002, 49407],
      decoded: "<|startoftext|>hello world <|endoftext|>",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u00e7\u0136\u0141", "\u00e6", "\u00b4", "\u00bb", "\u00e7", "\u013c", "\u0126", "\u00e7\u013e\u0141", "\u00e8", "\u00b0", "\u013d", "\u00e6\u013a", "\u00af</w>"],
      ids: [49406, 33375, 162, 112, 119, 163, 248, 226, 41570, 164, 108, 249, 42891, 363, 49407],
      decoded: "<|startoftext|>\u751f\u6d3b\u7684\u771f\u8c1b\u662f <|endoftext|>",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      tokens: ["leading</w>", "space</w>"],
      ids: [49406, 3833, 2138, 49407],
      decoded: "<|startoftext|>leading space <|endoftext|>",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["trailing</w>", "space</w>"],
      ids: [49406, 37427, 2138, 49407],
      decoded: "<|startoftext|>trailing space <|endoftext|>",
    },
    DOUBLE_SPACE: {
      text: BASE_TEST_STRINGS.DOUBLE_SPACE,
      tokens: ["hi</w>", "hello</w>"],
      ids: [49406, 1883, 3306, 49407],
      decoded: "<|startoftext|>hi hello <|endoftext|>",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["test</w>", "$</w>", "1</w>", "r</w>", "2</w>", "#</w>", "3</w>", "\u00e2\u0124\u00ac</w>", "4</w>", "\u00c2\u00a3</w>", "5</w>", "\u00c2\u00a5</w>", "6</w>", "\u00e2\u0124", "\u00a3</w>", "7</w>", "\u00e2\u0124\u00b9</w>", "8</w>", "\u00e2\u0124", "\u00b1</w>", "9</w>", "test</w>"],
      ids: [49406, 1628, 259, 272, 337, 273, 258, 274, 6309, 275, 1950, 276, 20199, 277, 5227, 352, 278, 21777, 279, 5227, 365, 280, 1628, 49407],
      decoded: "<|startoftext|>test $ 1 r 2 # 3 \u20ac 4 \u00a3 5 \u00a5 6 \u20a3 7 \u20b9 8 \u20b1 9 test <|endoftext|>",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["i</w>", "bought</w>", "an</w>", "apple</w>", "for</w>", "$</w>", "1</w>", ".</w>", "0</w>", "0</w>", "at</w>", "the</w>", "store</w>", ".</w>"],
      ids: [49406, 328, 4142, 550, 3055, 556, 259, 272, 269, 271, 271, 536, 518, 2183, 269, 49407],
      decoded: "<|startoftext|>i bought an apple for $ 1. 0 0 at the store. <|endoftext|>",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["you</w>", "\u00e2\u0122\u00a6</w>"],
      ids: [49406, 592, 959, 49407],
      decoded: "<|startoftext|>you \u2026 <|endoftext|>",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["you</w>", "\u00e2\u0122\u00a6</w>"],
      ids: [49406, 592, 959, 49407],
      decoded: "<|startoftext|>you \u2026 <|endoftext|>",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["you</w>", "\u00e2\u0122\u00a6</w>", "you</w>", "\u00e2\u0122\u00a6</w>"],
      ids: [49406, 592, 959, 592, 959, 49407],
      decoded: "<|startoftext|>you \u2026 you \u2026 <|endoftext|>",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["weird</w>", "\u00ef", "\u00bd", "\u0140</w>", "edge</w>", "\u00ef", "\u00bd", "\u0140</w>", "case</w>"],
      ids: [49406, 5613, 171, 121, 508, 5461, 171, 121, 508, 2068, 49407],
      decoded: "<|startoftext|>weird \uff5e edge \uff5e case <|endoftext|>",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      tokens: ["\u00e2\u0138", "\u0123</w>", "this</w>", "\u00e2\u0138", "\u0123</w>", "is</w>", "\u00e2\u0138", "\u0123</w>", "a</w>", "\u00e2\u0138", "\u0123</w>", "test</w>", "\u00e2\u0138", "\u0123", ".</w>"],
      ids: [49406, 4168, 479, 589, 4168, 479, 533, 4168, 479, 320, 4168, 479, 1628, 4168, 223, 269, 49407],
      decoded: "<|startoftext|>\u2581 this \u2581 is \u2581 a \u2581 test \u2581. <|endoftext|>",
    },
    POPULAR_EMOJIS: {
      text: BASE_TEST_STRINGS.POPULAR_EMOJIS,
      tokens: ["\u00f0\u0141\u013a\u0124</w>", "\u00f0\u0141\u0133\u012f</w>", "\u00f0\u0141\u00a4\u00a3</w>", "\u00f0\u0141\u013a\u012f</w>", "\u00f0\u0141\u013a\u0143</w>", "\u00f0\u0141\u0130\u012b</w>", "\u00f0\u0141\u013b\u0131</w>", "\u00f0\u0141\u013a\u012c</w>", "\u00f0\u0141\u0136\u00a5</w>", "\u00f0\u0141\u013a\u0123</w>", "\u00f0\u0141\u013a\u0127</w>", "\u00f0\u0141\u00a4\u0139</w>", "\u00f0\u0141\u013a\u0128</w>", "\u00f0\u0141\u0133\u0131</w>", "\u00e2\u013f\u00a4\u00ef\u00b8\u0131</w>", "\u00f0\u0141\u0134\u013e</w>", "\u00f0\u0141\u0134\u013c</w>", "\u00f0\u0141\u0134\u0139</w>", "\u00f0\u0141\u0134\u013b</w>", "\u00f0\u0141\u0138\u00a4</w>", "\u00f0\u0141\u013a\u0130</w>", "\u00f0\u0141\u0133\u012e</w>", "\u00f0\u0141\u00a5\u00b3</w>", "\u00f0\u0141\u0134\u00aa</w>", "\u00e2\u013e\u00a8</w>", "\u00f0\u0141\u0133\u012b</w>", "\u00f0\u0141\u0133\u0122</w>", "\u00f0\u0141\u0134\u00af</w>", "\u00f0\u0141\u0130\u012a</w>", "\u00f0\u0141\u013b\u012a</w>", "\u00f0\u0141\u013b\u012e</w>", "\u00f0\u0141\u0134\u0122</w>", "\u00f0\u0141\u0133\u0129</w>", "\u00f0\u0141\u0133\u012d</w>", "\u00e2\u013e\u0127</w>", "\u00f0\u0141\u0130\u0123</w>", "\u00f0\u0141\u012e\u0140</w>", "\u00f0\u0141\u012e\u00b8</w>", "\u00f0\u0141\u0134\u00b0</w>"],
      ids: [49406, 1558, 4201, 9909, 1754, 3915, 3986, 5503, 3020, 3016, 4821, 9188, 10465, 10943, 4829, 1752, 4882, 6521, 6690, 4074, 10860, 4345, 4494, 28055, 6440, 3531, 3988, 5908, 7018, 14448, 9516, 4855, 12158, 7475, 17686, 5564, 13462, 12980, 10980, 14078, 49407],
      decoded: "<|startoftext|>\ud83d\ude02 \ud83d\udc4d \ud83e\udd23 \ud83d\ude0d \ud83d\ude2d \ud83c\udf89 \ud83d\ude4f \ud83d\ude0a \ud83d\udd25 \ud83d\ude01 \ud83d\ude05 \ud83e\udd17 \ud83d\ude06 \ud83d\udc4f \u2764\ufe0f \ud83d\udc9c \ud83d\udc9a \ud83d\udc97 \ud83d\udc99 \ud83d\udda4 \ud83d\ude0e \ud83d\udc4c \ud83e\udd73 \ud83d\udcaa \u2728 \ud83d\udc49 \ud83d\udc40 \ud83d\udcaf \ud83c\udf88 \ud83d\ude48 \ud83d\ude4c \ud83d\udc80 \ud83d\udc47 \ud83d\udc4b \u2705 \ud83c\udf81 \ud83c\udf1e \ud83c\udf38 \ud83d\udcb0 <|endoftext|>",
    },
    MULTIBYTE_EMOJIS: {
      text: BASE_TEST_STRINGS.MULTIBYTE_EMOJIS,
      tokens: ["\u00e2\u013e\u00a8</w>", "\u00f0\u0141\u00a4\u0139</w>", "\u00f0\u0141\u0133\u0123", "\u00ef\u00b8\u0131</w>", "\u00f0\u0141\u0133", "\u00b1", "\u00f0\u0141\u0131\u00bb</w>", "\u00f0\u0141\u0137", "\u00b5", "\u00e2\u0122\u012f\u00e2\u013b\u0124\u00ef\u00b8\u0131</w>", "\u00f0\u0141\u00a7", "\u013b", "\u00f0\u0141\u0131\u00bb", "\u00e2\u0122\u012f\u00e2\u013b", "\u0124</w>", "\u00f0\u0141\u0133\u00a8", "\u00f0\u0141\u0131\u00bb\u00e2\u0122\u012f", "\u00f0\u0141\u012e\u00be</w>", "\u00f0\u0141\u00a7", "\u0133", "\u00e2\u0122\u012f", "\u00f0\u0141\u00a4", "\u013f", "\u00e2\u0122\u012f", "\u00f0\u0141\u00a7", "\u0133</w>", "\u00f0\u0141\u0133\u00a9\u00e2\u0122\u012f", "\u00e2\u013f\u00a4", "\u00e2\u0122\u012f", "\u00f0\u0141\u0134\u012d", "\u00e2\u0122\u012f", "\u00f0\u0141\u0133", "\u00a8</w>", "\u00f0\u0141\u0133\u00a9\u00e2\u0122\u012f", "\u00f0\u0141\u0133\u00a9\u00e2\u0122\u012f", "\u00f0\u0141\u0133\u00a7", "\u00e2\u0122\u012f", "\u00f0\u0141\u0133", "\u00a6</w>", "\u00f0\u0141\u00a7", "\u0133", "\u00f0\u0141\u0131\u00bb\u00e2\u0122\u012f", "\u00f0\u0141\u00a4", "\u013f", "\u00e2\u0122\u012f", "\u00f0\u0141\u00a7", "\u0133", "\u00f0\u0141\u0131\u00bb</w>", "\u00f0\u0141\u0131\u00b4", "\u00f3", "\u0142", "\u0123", "\u00a7", "\u00f3", "\u0142", "\u0123", "\u00a2", "\u00f3", "\u0142", "\u0123", "\u00a5", "\u00f3", "\u0142", "\u0123", "\u00ae", "\u00f3", "\u0142", "\u0123", "\u00a7", "\u00f3", "\u0142", "\u0123", "\u00bf</w>", "\u00f0\u0141\u0133\u00a8", "\u00f0\u0141\u0131\u00bb\u00e2\u0122\u012f", "\u00e2\u013f\u00a4\u00ef\u00b8\u0131", "\u00e2\u0122\u012f", "\u00f0\u0141\u0134\u012d", "\u00e2\u0122\u012f", "\u00f0\u0141\u0133\u00a8", "\u00f0\u0141\u0131\u00bc</w>"],
      ids: [49406, 3531, 10465, 47796, 1001, 964, 109, 3702, 7692, 113, 10613, 8792, 247, 5042, 5177, 480, 18966, 46250, 39796, 8792, 239, 4244, 1793, 251, 4244, 8792, 495, 26304, 1266, 4244, 12217, 4244, 964, 357, 26304, 26304, 48938, 4244, 964, 355, 8792, 239, 46250, 1793, 251, 4244, 8792, 239, 3702, 39690, 175, 254, 223, 100, 175, 254, 223, 95, 175, 254, 223, 98, 175, 254, 223, 106, 175, 254, 223, 100, 175, 254, 223, 379, 18966, 46250, 2626, 4244, 12217, 4244, 18966, 4027, 49407],
      decoded: "<|startoftext|>\u2728 \ud83e\udd17 \ud83d\udc41\ufe0f \ud83d\udc71\ud83c\udffb \ud83d\udd75\u200d\u2642\ufe0f \ud83e\uddd9\ud83c\udffb\u200d\u2642 \ud83d\udc68\ud83c\udffb\u200d\ud83c\udf3e \ud83e\uddd1\u200d\ud83e\udd1d\u200d\ud83e\uddd1 \ud83d\udc69\u200d\u2764\u200d\ud83d\udc8b\u200d\ud83d\udc68 \ud83d\udc69\u200d\ud83d\udc69\u200d\ud83d\udc67\u200d\ud83d\udc66 \ud83e\uddd1\ud83c\udffb\u200d\ud83e\udd1d\u200d\ud83e\uddd1\ud83c\udffb \ud83c\udff4\udb40\udc67\udb40\udc62\udb40\udc65\udb40\udc6e\udb40\udc67\udb40\udc7f \ud83d\udc68\ud83c\udffb\u200d\u2764\ufe0f\u200d\ud83d\udc8b\u200d\ud83d\udc68\ud83c\udffc <|endoftext|>",
    },
  },
  "Xenova/owlvit-base-patch32": {
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["a</w>", "'ll</w>", "!", "!", "to</w>", "?'</w>", "d</w>", "''</w>", "d</w>", "of</w>", ",</w>", "can</w>", "'t</w>", ".</w>"],
      ids: [49406, 320, 1342, 0, 0, 531, 13610, 323, 8445, 323, 539, 267, 753, 713, 269, 49407],
      decoded: "<|startoftext|>a 'll!!to?' d '' d of, can 't. <|endoftext|>",
    },
  },
};
