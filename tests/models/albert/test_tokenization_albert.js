import { AlbertTokenizer } from "../../../src/tokenizers.js";
import { BASE_TEST_STRINGS, BERT_TEST_STRINGS } from "../test_strings.js";

export const TOKENIZER_CLASS = AlbertTokenizer;
export const TEST_CONFIG = {
  // - uses `StripAccents` normalizer
  "Xenova/albert-base-v2": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      tokens: ["\u2581how", "\u2581are", "\u2581you", "\u2581doing", "?"],
      ids: [2, 184, 50, 42, 845, 60, 3],
      decoded: "[CLS] how are you doing?[SEP]",
    },
    SIMPLE_WITH_PUNCTUATION: {
      text: BASE_TEST_STRINGS.SIMPLE_WITH_PUNCTUATION,
      tokens: ["\u2581you", "\u2581should", "'", "ve", "\u2581done", "\u2581this"],
      ids: [2, 42, 378, 22, 195, 677, 48, 3],
      decoded: "[CLS] you should've done this[SEP]",
    },
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["\u25810", "12", "345", "67", "89", "\u25810", "\u25811", "\u25812", "\u25813", "\u25814", "\u25815", "\u25816", "\u25817", "\u25818", "\u25819", "\u258110", "\u2581100", "\u25811000"],
      ids: [2, 713, 918, 21997, 4167, 3877, 713, 137, 172, 203, 268, 331, 400, 453, 469, 561, 332, 808, 6150, 3],
      decoded: "[CLS] 0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000[SEP]",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["\u2581the", "\u2581company", "\u2581was", "\u2581founded", "\u2581in", "\u25812016", "."],
      ids: [2, 14, 237, 23, 785, 19, 690, 9, 3],
      decoded: "[CLS] the company was founded in 2016.[SEP]",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["\u2581a", "\u2581", "'", "ll", "\u2581", "!!", "to", "?'", "d", '"', "d", "\u2581of", ",", "\u2581can", "'", "t", "."],
      ids: [2, 21, 13, 22, 211, 13, 19015, 262, 5663, 43, 7, 43, 16, 15, 92, 22, 38, 9, 3],
      decoded: "[CLS] a 'll!!to?'d\"d of, can't.[SEP]",
    },
    PYTHON_CODE: {
      text: BASE_TEST_STRINGS.PYTHON_CODE,
      tokens: ["\u2581def", "\u2581main", "(", ")", ":", "\u2581pass"],
      ids: [2, 6312, 407, 5, 6, 45, 1477, 3],
      decoded: "[CLS] def main(): pass[SEP]",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["\u2581let", "\u2581a", "\u2581=", "\u2581ob", "j", ".", "to", "string", "(", ")", ";", "\u2581to", "string", "(", ")", ";"],
      ids: [2, 408, 21, 800, 5122, 728, 9, 262, 11130, 5, 6, 73, 20, 11130, 5, 6, 73, 3],
      decoded: "[CLS] let a = obj.tostring(); tostring();[SEP]",
    },
    NEWLINES: {
      text: BASE_TEST_STRINGS.NEWLINES,
      tokens: ["\u2581this", "\u2581is", "\u2581a", "\u2581test", "."],
      ids: [2, 48, 25, 21, 1289, 9, 3],
      decoded: "[CLS] this is a test.[SEP]",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["\u2581unwanted", ",", "running"],
      ids: [2, 21095, 15, 11325, 3],
      decoded: "[CLS] unwanted,running[SEP]",
    },
    CONTROL_TOKENS: {
      text: BASE_TEST_STRINGS.CONTROL_TOKENS,
      tokens: ["\u25811", "\u0000", "2", "\u25813"],
      ids: [2, 137, 1, 135, 203, 3],
      decoded: "[CLS] 1<unk>2 3[SEP]",
    },
    HELLO_WORLD_TITLECASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_TITLECASE,
      tokens: ["\u2581hello", "\u2581world"],
      ids: [2, 10975, 126, 3],
      decoded: "[CLS] hello world[SEP]",
    },
    HELLO_WORLD_LOWERCASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_LOWERCASE,
      tokens: ["\u2581hello", "\u2581world"],
      ids: [2, 10975, 126, 3],
      decoded: "[CLS] hello world[SEP]",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u2581", "\u751f\u6d3b\u7684\u771f\u8c1b\u662f"],
      ids: [2, 13, 1, 3],
      decoded: "[CLS] <unk>[SEP]",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      tokens: ["\u2581leading", "\u2581space"],
      ids: [2, 1005, 726, 3],
      decoded: "[CLS] leading space[SEP]",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["\u2581trailing", "\u2581space"],
      ids: [2, 14323, 726, 3],
      decoded: "[CLS] trailing space[SEP]",
    },
    DOUBLE_SPACE: {
      text: BASE_TEST_STRINGS.DOUBLE_SPACE,
      tokens: ["\u2581hi", "\u2581hello"],
      ids: [2, 4148, 10975, 3],
      decoded: "[CLS] hi hello[SEP]",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["\u2581test", "\u2581$1", "\u2581r", "2", "\u2581#3", "\u2581", "\u20ac", "4", "\u2581", "\u00a3", "5", "\u2581", "\u00a5", "6", "\u2581", "\u20a3", "7", "\u2581", "\u20b9", "8", "\u2581", "\u20b1", "9", "\u2581test"],
      ids: [2, 1289, 3742, 761, 135, 11489, 13, 12, 300, 13, 11, 264, 13, 1, 379, 13, 1, 465, 13, 1, 457, 13, 1, 518, 1289, 3],
      decoded: "[CLS] test $1 r2 #3 \u20ac4 \u00a35 <unk>6 <unk>7 <unk>8 <unk>9 test[SEP]",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["\u2581i", "\u2581bought", "\u2581an", "\u2581apple", "\u2581for", "\u2581$1", ".", "00", "\u2581at", "\u2581the", "\u2581store", "."],
      ids: [2, 31, 2448, 40, 4037, 26, 3742, 9, 2032, 35, 14, 1718, 9, 3],
      decoded: "[CLS] i bought an apple for $1.00 at the store.[SEP]",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["\u2581you", ".", ".", "."],
      ids: [2, 42, 9, 9, 9, 3],
      decoded: "[CLS] you...[SEP]",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["\u2581you", ".", ".", "."],
      ids: [2, 42, 9, 9, 9, 3],
      decoded: "[CLS] you...[SEP]",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["\u2581you", ".", ".", ".", "\u2581you", ".", ".", "."],
      ids: [2, 42, 9, 9, 9, 42, 9, 9, 9, 3],
      decoded: "[CLS] you... you...[SEP]",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["\u2581weird", "\u2581", "~", "\u2581edge", "\u2581", "~", "\u2581case"],
      ids: [2, 5455, 13, 1, 1407, 13, 1, 610, 3],
      decoded: "[CLS] weird <unk> edge <unk> case[SEP]",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      tokens: ["\u2581this", "\u2581is", "\u2581a", "\u2581test", "\u2581", "."],
      ids: [2, 48, 25, 21, 1289, 13, 9, 3],
      decoded: "[CLS] this is a test.[SEP]",
    },
    POPULAR_EMOJIS: {
      text: BASE_TEST_STRINGS.POPULAR_EMOJIS,
      tokens: ["\u2581", "\ud83d\ude02", "\u2581", "\ud83d\udc4d", "\u2581", "\ud83e\udd23", "\u2581", "\ud83d\ude0d", "\u2581", "\ud83d\ude2d", "\u2581", "\ud83c\udf89", "\u2581", "\ud83d\ude4f", "\u2581", "\ud83d\ude0a", "\u2581", "\ud83d\udd25", "\u2581", "\ud83d\ude01", "\u2581", "\ud83d\ude05", "\u2581", "\ud83e\udd17", "\u2581", "\ud83d\ude06", "\u2581", "\ud83d\udc4f", "\u2581", "\u2764", "\u2581", "\ud83d\udc9c", "\u2581", "\ud83d\udc9a", "\u2581", "\ud83d\udc97", "\u2581", "\ud83d\udc99", "\u2581", "\ud83d\udda4", "\u2581", "\ud83d\ude0e", "\u2581", "\ud83d\udc4c", "\u2581", "\ud83e\udd73", "\u2581", "\ud83d\udcaa", "\u2581", "\u2728", "\u2581", "\ud83d\udc49", "\u2581", "\ud83d\udc40", "\u2581", "\ud83d\udcaf", "\u2581", "\ud83c\udf88", "\u2581", "\ud83d\ude48", "\u2581", "\ud83d\ude4c", "\u2581", "\ud83d\udc80", "\u2581", "\ud83d\udc47", "\u2581", "\ud83d\udc4b", "\u2581", "\u2705", "\u2581", "\ud83c\udf81", "\u2581", "\ud83c\udf1e", "\u2581", "\ud83c\udf38", "\u2581", "\ud83d\udcb0"],
      ids: [2, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 3],
      decoded: "[CLS] <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>[SEP]",
    },
    MULTIBYTE_EMOJIS: {
      text: BASE_TEST_STRINGS.MULTIBYTE_EMOJIS,
      tokens: ["\u2581", "\u2728", "\u2581", "\ud83e\udd17", "\u2581", "\ud83d\udc41", "\u2581", "\ud83d\udc71\ud83c\udffb", "\u2581", "\ud83d\udd75", "\u2581", "\u2642", "\u2581", "\ud83e\uddd9\ud83c\udffb", "\u2581", "\u2642", "\u2581", "\ud83d\udc68\ud83c\udffb", "\u2581", "\ud83c\udf3e", "\u2581", "\ud83e\uddd1", "\u2581", "\ud83e\udd1d", "\u2581", "\ud83e\uddd1", "\u2581", "\ud83d\udc69", "\u2581", "\u2764", "\u2581", "\ud83d\udc8b", "\u2581", "\ud83d\udc68", "\u2581", "\ud83d\udc69", "\u2581", "\ud83d\udc69", "\u2581", "\ud83d\udc67", "\u2581", "\ud83d\udc66", "\u2581", "\ud83e\uddd1\ud83c\udffb", "\u2581", "\ud83e\udd1d", "\u2581", "\ud83e\uddd1\ud83c\udffb", "\u2581", "\ud83c\udff4\udb40\udc67\udb40\udc62\udb40\udc65\udb40\udc6e\udb40\udc67\udb40\udc7f", "\u2581", "\ud83d\udc68\ud83c\udffb", "\u2581", "\u2764", "\u2581", "\ud83d\udc8b", "\u2581", "\ud83d\udc68\ud83c\udffc"],
      ids: [2, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 3],
      decoded: "[CLS] <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>[SEP]",
    },
    CHINESE_LATIN_MIXED: {
      text: BERT_TEST_STRINGS.CHINESE_LATIN_MIXED,
      tokens: ["\u2581", "ah", "\u535a\u63a8", "zz"],
      ids: [2, 13, 1307, 1, 5092, 3],
      decoded: "[CLS] ah<unk>zz[SEP]",
    },
    SIMPLE_WITH_ACCENTS: {
      text: BERT_TEST_STRINGS.SIMPLE_WITH_ACCENTS,
      tokens: ["\u2581hello"],
      ids: [2, 10975, 3],
      decoded: "[CLS] hello[SEP]",
    },
    MIXED_CASE_WITHOUT_ACCENTS: {
      text: BERT_TEST_STRINGS.MIXED_CASE_WITHOUT_ACCENTS,
      tokens: ["\u2581hello", "!", "how", "\u2581are", "\u2581you", "?"],
      ids: [2, 10975, 187, 1544, 50, 42, 60, 3],
      decoded: "[CLS] hello!how are you?[SEP]",
    },
    MIXED_CASE_WITH_ACCENTS: {
      text: BERT_TEST_STRINGS.MIXED_CASE_WITH_ACCENTS,
      tokens: ["\u2581hall", "o", "!", "how", "\u2581are", "\u2581you", "?"],
      ids: [2, 554, 111, 187, 1544, 50, 42, 60, 3],
      decoded: "[CLS] hallo!how are you?[SEP]",
    },
  },
};
