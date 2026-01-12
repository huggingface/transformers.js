import { M2M100Tokenizer } from "../../../src/tokenizers.js";
import { BASE_TEST_STRINGS, M2M_100_TEST_STRINGS } from "../test_strings.js";

export const TOKENIZER_CLASS = M2M100Tokenizer;

// NOTE: The slow tokenizer (used by transformers) has minor inconsistencies against the fast tokenizer.
// For this reason, we may override the expected results for certain tests.
export const TEST_CONFIG = {
  "Xenova/m2m100_418M": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      tokens: ["\u2581How", "\u2581are", "\u2581you", "\u2581doing", "?"],
      ids: [128022, 34226, 4234, 8251, 123047, 24, 2],
      decoded: "__en__ How are you doing?</s>",
    },
    SIMPLE_WITH_PUNCTUATION: {
      text: BASE_TEST_STRINGS.SIMPLE_WITH_PUNCTUATION,
      tokens: ["\u2581You", "\u2581should", "'", "ve", "\u2581done", "\u2581this"],
      ids: [128022, 14921, 119092, 12, 470, 111108, 15911, 2],
      decoded: "__en__ You should've done this</s>",
    },
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["\u25810", "123", "45", "6", "78", "9", "\u25810", "\u25811", "\u25812", "\u25813", "\u25814", "\u25815", "\u25816", "\u25817", "\u25818", "\u25819", "\u258110", "\u2581100", "\u25811000"],
      ids: [128022, 847, 78596, 3834, 435, 7049, 718, 847, 161, 168, 205, 273, 265, 376, 442, 455, 572, 301, 1245, 7336, 2],
      decoded: "__en__ 0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000</s>",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["\u2581The", "\u2581company", "\u2581was", "\u2581found", "ed", "\u2581in", "\u25812016."],
      ids: [128022, 1658, 66486, 1513, 118728, 241, 28, 8860, 2],
      decoded: "__en__ The company was founded in 2016.</s>",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["\u2581A", "\u2581'", "ll", "\u2581!!", "to", "?'", "d", "''", "d", "\u2581of", ",", "\u2581can", "'", "t", "."],
      ids: [128022, 58, 244, 2279, 9403, 428, 72956, 173, 8471, 173, 432, 4, 3154, 12, 88, 5, 2],
      decoded: "__en__ A 'll!!to?'d''d of, can't.</s>",
    },
    PYTHON_CODE: {
      text: BASE_TEST_STRINGS.PYTHON_CODE,
      tokens: ["\u2581def", "\u2581main", "(", "):", "\u2581pass"],
      ids: [128022, 8268, 9359, 249, 2825, 4799, 2],
      decoded: "__en__ def main(): pass</s>",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["\u2581let", "\u2581a", "\u2581=", "\u2581ob", "j", ".", "to", "Str", "ing", "(", ");", "\u2581to", "Str", "ing", "(", ");"],
      ids: [128022, 2507, 8, 3255, 607, 189, 5, 428, 41549, 150, 249, 5294, 128, 41549, 150, 249, 5294, 2],
      decoded: "__en__ let a = obj.toString(); toString();</s>",
    },
    NEWLINES: {
      text: BASE_TEST_STRINGS.NEWLINES,
      tokens: ["\u2581This", "\u2581is", "\u2581a", "\u2581test", "."],
      ids: [128022, 36606, 117, 8, 4183, 5, 2],
      decoded: "__en__ This is a test.</s>",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["\u2581UN", "want", "\u00e9d", ",", "run", "ning"],
      ids: [128022, 6984, 108054, 7151, 4, 18634, 656, 2],
      decoded: "__en__ UNwant\u00e9d,running</s>",
    },
    CONTROL_TOKENS: {
      text: BASE_TEST_STRINGS.CONTROL_TOKENS,
      tokens: ["\u25811", "\u0000", "2", "\u25813"],
      ids: [128022, 161, 4163, 339, 205, 2],
      decoded: "__en__ 1\u00002 3</s>",
    },
    HELLO_WORLD_TITLECASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_TITLECASE,
      tokens: ["\u2581Hello", "\u2581World"],
      ids: [128022, 65761, 10581, 2],
      decoded: "__en__ Hello World</s>",
    },
    HELLO_WORLD_LOWERCASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_LOWERCASE,
      tokens: ["\u2581hello", "\u2581world"],
      ids: [128022, 110013, 55185, 2],
      decoded: "__en__ hello world</s>",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u2581", "\u751f\u6d3b", "\u7684", "\u771f", /* "\u8c1b" */ "<unk>", "\u662f"],
      ids: [128022, 22, 8523, 80, 10418, 3, 775, 2],
      decoded: "__en__ \u751f\u6d3b\u7684\u771f<unk>\u662f</s>",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      tokens: ["\u2581leading", "\u2581space"],
      ids: [128022, 124476, 118561, 2],
      decoded: "__en__ leading space</s>",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["\u2581tra", "iling", "\u2581space"],
      ids: [128022, 1368, 19217, 118561, 2],
      decoded: "__en__ trailing space</s>",
    },
    SURROUNDING_SPACE: {
      text: BASE_TEST_STRINGS.SURROUNDING_SPACE,
      tokens: ["\u2581surround", "ing", "\u2581space"],
      ids: [128022, 124728, 150, 118561, 2],
      decoded: "__en__ surrounding space</s>",
    },
    DOUBLE_SPACE: {
      text: BASE_TEST_STRINGS.DOUBLE_SPACE,
      tokens: ["\u2581Hi", "\u2581Hello"],
      ids: [128022, 7676, 65761, 2],
      decoded: "__en__ Hi Hello</s>",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["\u2581test", "\u2581$", "1", "\u2581R", "2", "\u2581#", "3", "\u2581\u20ac", "4", "\u2581\u00a3", "5", "\u2581", "\u00a5", "6", "\u2581", /* "\u20a3" */ "<unk>", "7", "\u2581", "\u20b9", "8", "\u2581", /* "\u20b1" */ "<unk>", "9", "\u2581test"],
      ids: [128022, 4183, 4352, 451, 180, 339, 584, 425, 4257, 465, 13506, 679, 22, 43832, 435, 22, 3, 622, 22, 115056, 677, 22, 3, 718, 4183, 2],
      decoded: "__en__ test $1 R2 #3 \u20ac4 \u00a35 \u00a56 <unk>7 \u20b98 <unk>9 test</s>",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["\u2581I", "\u2581bought", "\u2581an", "\u2581ap", "ple", "\u2581for", "\u2581$", "1.", "00", "\u2581at", "\u2581the", "\u2581store", "."],
      ids: [128022, 203, 127797, 48, 722, 6857, 193, 4352, 2023, 1365, 120, 1197, 9160, 5, 2],
      decoded: "__en__ I bought an apple for $1.00 at the store.</s>",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["\u2581you", "..."],
      ids: [128022, 8251, 26, 2],
      decoded: "__en__ you...</s>",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["\u2581you", "..."],
      ids: [128022, 8251, 26, 2],
      decoded: "__en__ you...</s>",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["\u2581you", "...", "\u2581you", "..."],
      ids: [128022, 8251, 26, 8251, 26, 2],
      decoded: "__en__ you... you...</s>",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["\u2581we", "ird", "\u2581", "\uff5e", "\u2581ed", "ge", "\u2581", "\uff5e", "\u2581case"],
      ids: [128022, 1710, 13067, 22, 14691, 1500, 568, 22, 14691, 24306, 2],
      decoded: "__en__ weird \uff5e edge \uff5e case</s>",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      tokens: ["\u2581This", "\u2581is", "\u2581a", "\u2581test", "\u2581."],
      ids: [128022, 36606, 117, 8, 4183, 237, 2],
      decoded: "__en__ This is a test.</s>",
    },
    POPULAR_EMOJIS: {
      text: BASE_TEST_STRINGS.POPULAR_EMOJIS,
      tokens: ["\u2581", "\ud83d\ude02", "\u2581", "\ud83d\udc4d", "\u2581", "\ud83e\udd23", "\u2581", "\ud83d\ude0d", "\u2581", "\ud83d\ude2d", "\u2581", "\ud83c\udf89", "\u2581", "\ud83d\ude4f", "\u2581", "\ud83d\ude0a", "\u2581", "\ud83d\udd25", "\u2581", "\ud83d\ude01", "\u2581", "\ud83d\ude05", "\u2581", "\ud83e\udd17", "\u2581", "\ud83d\ude06", "\u2581", "\ud83d\udc4f", "\u2581\u2764", "\ufe0f", "\u2581", "\ud83d\udc9c", "\u2581", "\ud83d\udc9a", "\u2581", "\ud83d\udc97", "\u2581", "\ud83d\udc99", "\u2581", "\ud83d\udda4", "\u2581", "\ud83d\ude0e", "\u2581", "\ud83d\udc4c", "\u2581", /* "\ud83e\udd73" */ "<unk>", "\u2581", "\ud83d\udcaa", "\u2581", "\u2728", "\u2581", "\ud83d\udc49", "\u2581", "\ud83d\udc40", "\u2581", "\ud83d\udcaf", "\u2581", "\ud83c\udf88", "\u2581", "\ud83d\ude48", "\u2581", "\ud83d\ude4c", "\u2581", /* "\ud83d\udc80" */ "<unk>", "\u2581", "\ud83d\udc47", "\u2581", "\ud83d\udc4b", "\u2581", "\u2705", "\u2581", "\ud83c\udf81", "\u2581", /* "\ud83c\udf1e" */ "<unk>", "\u2581", "\ud83c\udf38", "\u2581", "\ud83d\udcb0"],
      ids: [128022, 22, 74222, 22, 118514, 22, 124385, 22, 99683, 22, 123842, 22, 124821, 22, 117689, 22, 103111, 22, 121924, 22, 121088, 22, 124207, 22, 123955, 22, 120137, 22, 123534, 66038, 18905, 22, 125385, 22, 125317, 22, 126071, 22, 124787, 22, 127396, 22, 120119, 22, 122813, 22, 3, 22, 123482, 22, 120563, 22, 117995, 22, 127978, 22, 126507, 22, 127269, 22, 126179, 22, 125300, 22, 3, 22, 120807, 22, 127143, 22, 118682, 22, 125350, 22, 3, 22, 123790, 22, 126948, 2],
      decoded: /* "__en__ \ud83d\ude02 \ud83d\udc4d \ud83e\udd23 \ud83d\ude0d \ud83d\ude2d \ud83c\udf89 \ud83d\ude4f \ud83d\ude0a \ud83d\udd25 \ud83d\ude01 \ud83d\ude05 \ud83e\udd17 \ud83d\ude06 \ud83d\udc4f \u2764\ufe0f \ud83d\udc9c \ud83d\udc9a \ud83d\udc97 \ud83d\udc99 \ud83d\udda4 \ud83d\ude0e \ud83d\udc4c <unk>\ud83d\udcaa \u2728 \ud83d\udc49 \ud83d\udc40 \ud83d\udcaf \ud83c\udf88 \ud83d\ude48 \ud83d\ude4c <unk>\ud83d\udc47 \ud83d\udc4b \u2705 \ud83c\udf81 <unk>\ud83c\udf38 \ud83d\udcb0</s>" */ "__en__ \ud83d\ude02 \ud83d\udc4d \ud83e\udd23 \ud83d\ude0d \ud83d\ude2d \ud83c\udf89 \ud83d\ude4f \ud83d\ude0a \ud83d\udd25 \ud83d\ude01 \ud83d\ude05 \ud83e\udd17 \ud83d\ude06 \ud83d\udc4f \u2764\ufe0f \ud83d\udc9c \ud83d\udc9a \ud83d\udc97 \ud83d\udc99 \ud83d\udda4 \ud83d\ude0e \ud83d\udc4c <unk> \ud83d\udcaa \u2728 \ud83d\udc49 \ud83d\udc40 \ud83d\udcaf \ud83c\udf88 \ud83d\ude48 \ud83d\ude4c <unk> \ud83d\udc47 \ud83d\udc4b \u2705 \ud83c\udf81 <unk> \ud83c\udf38 \ud83d\udcb0</s>",
    },
    MULTIBYTE_EMOJIS: {
      text: BASE_TEST_STRINGS.MULTIBYTE_EMOJIS,
      tokens: ["\u2581", "\u2728", "\u2581", "\ud83e\udd17", "\u2581", "\ud83d\udc41", "\ufe0f", "\u2581", /* "\ud83d\udc71" */ "<unk>", "\ud83c\udffb", "\u2581", /* "\ud83d\udd75" */ "<unk>", "\u2581", "\u2642", "\ufe0f", "\u2581", /* "\ud83e\uddd9" */ "<unk>", "\ud83c\udffb", "\u2581", "\u2642", "\u2581", "\ud83d\udc68", "\ud83c\udffb", "\u2581", /* "\ud83c\udf3e" */ "<unk>", "\u2581", /* "\ud83e\uddd1" */ "<unk>", "\u2581", /* "\ud83e\udd1d" */ "<unk>", "\u2581", /* "\ud83e\uddd1" */ "<unk>", "\u2581", "\ud83d\udc69", "\u2581\u2764", "\u2581", "\ud83d\udc8b", "\u2581", "\ud83d\udc68", "\u2581", "\ud83d\udc69", "\u2581", "\ud83d\udc69", "\u2581", /* "\ud83d\udc67" */ "<unk>", "\u2581", /* "\ud83d\udc66" */ "<unk>", "\u2581", /* "\ud83e\uddd1" */ "<unk>", "\ud83c\udffb", "\u2581", /* "\ud83e\udd1d" */ "<unk>", "\u2581", /* "\ud83e\uddd1" */ "<unk>", "\ud83c\udffb", "\u2581", /* "\ud83c\udff4\udb40\udc67\udb40\udc62\udb40\udc65\udb40\udc6e\udb40\udc67\udb40\udc7f"*/ "<unk>", "\u2581", "\ud83d\udc68", "\ud83c\udffb", "\u2581\u2764", "\ufe0f", "\u2581", "\ud83d\udc8b", "\u2581", "\ud83d\udc68", "\ud83c\udffc"],
      ids: [128022, 22, 120563, 22, 123955, 22, 121442, 18905, 22, 3, 116617, 22, 3, 22, 122517, 18905, 22, 3, 116617, 22, 122517, 22, 127603, 116617, 22, 3, 22, 3, 22, 3, 22, 3, 22, 126739, 66038, 22, 126237, 22, 127603, 22, 126739, 22, 126739, 22, 3, 22, 3, 22, 3, 116617, 22, 3, 22, 3, 116617, 22, 3, 22, 127603, 116617, 66038, 18905, 22, 126237, 22, 127603, 123285, 2],
      decoded: /* "__en__ \u2728 \ud83e\udd17 \ud83d\udc41\ufe0f <unk>\ud83c\udffb <unk>\u2642\ufe0f <unk>\ud83c\udffb \u2642 \ud83d\udc68\ud83c\udffb <unk><unk><unk><unk>\ud83d\udc69 \u2764 \ud83d\udc8b \ud83d\udc68 \ud83d\udc69 \ud83d\udc69 <unk><unk><unk>\ud83c\udffb <unk><unk>\ud83c\udffb <unk>\ud83d\udc68\ud83c\udffb \u2764\ufe0f \ud83d\udc8b \ud83d\udc68\ud83c\udffc</s>" */ "__en__ \u2728 \ud83e\udd17 \ud83d\udc41\ufe0f <unk>\ud83c\udffb <unk> \u2642\ufe0f <unk>\ud83c\udffb \u2642 \ud83d\udc68\ud83c\udffb <unk> <unk> <unk> <unk> \ud83d\udc69 \u2764 \ud83d\udc8b \ud83d\udc68 \ud83d\udc69 \ud83d\udc69 <unk> <unk> <unk>\ud83c\udffb <unk> <unk>\ud83c\udffb <unk> \ud83d\udc68\ud83c\udffb \u2764\ufe0f \ud83d\udc8b \ud83d\udc68\ud83c\udffc</s>",
    },
    ONLY_WHITESPACE: {
      text: BASE_TEST_STRINGS.ONLY_WHITESPACE,
      tokens: [],
      ids: [128022, 2],
      decoded: /* "__en__ </s>" */ "__en__</s>",
    },
    TRANSLATION_INPUTS: {
      text: M2M_100_TEST_STRINGS.TRANSLATION_INPUTS,
      tokens: ["__en__", "\u2581hello", "\u2581world", "</s>"],
      ids: [128022, 128022, 110013, 55185, 2, 2],
      decoded: /* "__en__ __en__ hello world</s></s>" */ "__en____en__ hello world</s></s>",
    },
  },
};
