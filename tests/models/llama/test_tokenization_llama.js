import { LlamaTokenizer } from "../../../src/tokenizers.js";
import { BASE_TEST_STRINGS, LLAMA_TEST_STRINGS } from "../test_strings.js";

export const TOKENIZER_CLASS = LlamaTokenizer;
export const TEST_CONFIG = {
  "Xenova/llama-tokenizer": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      tokens: ["\u2581How", "\u2581are", "\u2581you", "\u2581doing", "?"],
      ids: [1, 1128, 526, 366, 2599, 29973],
      decoded: "<s> How are you doing?",
    },
    SIMPLE_WITH_PUNCTUATION: {
      text: BASE_TEST_STRINGS.SIMPLE_WITH_PUNCTUATION,
      tokens: ["\u2581You", "\u2581should", "'", "ve", "\u2581done", "\u2581this"],
      ids: [1, 887, 881, 29915, 345, 2309, 445],
      decoded: "<s> You should've done this",
    },
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["\u2581", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "\u2581", "0", "\u2581", "1", "\u2581", "2", "\u2581", "3", "\u2581", "4", "\u2581", "5", "\u2581", "6", "\u2581", "7", "\u2581", "8", "\u2581", "9", "\u2581", "1", "0", "\u2581", "1", "0", "0", "\u2581", "1", "0", "0", "0"],
      ids: [1, 29871, 29900, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929, 29871, 29900, 29871, 29896, 29871, 29906, 29871, 29941, 29871, 29946, 29871, 29945, 29871, 29953, 29871, 29955, 29871, 29947, 29871, 29929, 29871, 29896, 29900, 29871, 29896, 29900, 29900, 29871, 29896, 29900, 29900, 29900],
      decoded: "<s> 0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["\u2581The", "\u2581company", "\u2581was", "\u2581founded", "\u2581in", "\u2581", "2", "0", "1", "6", "."],
      ids: [1, 450, 5001, 471, 11091, 297, 29871, 29906, 29900, 29896, 29953, 29889],
      decoded: "<s> The company was founded in 2016.",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["\u2581A", "<0x0A>", "'", "ll", "\u2581!!", "to", "?'", "d", "''", "d", "\u2581of", ",", "\u2581can", "'", "t", "."],
      ids: [1, 319, 13, 29915, 645, 21443, 517, 17901, 29881, 4907, 29881, 310, 29892, 508, 29915, 29873, 29889],
      decoded: "<s> A\n'll !!to?'d''d of, can't.",
    },
    PYTHON_CODE: {
      text: BASE_TEST_STRINGS.PYTHON_CODE,
      tokens: ["\u2581def", "\u2581main", "():", "<0x0A>", "<0x09>", "pass"],
      ids: [1, 822, 1667, 7295, 13, 12, 3364],
      decoded: "<s> def main():\n\tpass",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["\u2581let", "\u2581a", "\u2581=", "\u2581obj", ".", "toString", "();", "<0x0A>", "toString", "();"],
      ids: [1, 1235, 263, 353, 5446, 29889, 7711, 890, 13, 7711, 890],
      decoded: "<s> let a = obj.toString();\ntoString();",
    },
    NEWLINES: {
      text: LLAMA_TEST_STRINGS.NEWLINES,
      tokens: ["\u2581ax", "<0x0A>", "####", "<0x0A>", "bo", "o"],
      ids: [1, 4853, 13, 4136, 13, 833, 29877],
      decoded: "<s> ax\n####\nboo",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["\u2581UN", "w", "ant", "\u00e9d", ",", "running"],
      ids: [1, 8291, 29893, 424, 2487, 29892, 21094],
      decoded: "<s> UNwant\u00e9d,running",
    },
    CONTROL_TOKENS: {
      text: BASE_TEST_STRINGS.CONTROL_TOKENS,
      tokens: ["\u2581", "1", "<0x00>", "2", "\ufffd", "3"],
      ids: [1, 29871, 29896, 3, 29906, 30140, 29941],
      decoded: "<s> 1\u00002\ufffd3",
    },
    HELLO_WORLD_TITLECASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_TITLECASE,
      tokens: ["\u2581Hello", "\u2581World"],
      ids: [1, 15043, 2787],
      decoded: "<s> Hello World",
    },
    HELLO_WORLD_LOWERCASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_LOWERCASE,
      tokens: ["\u2581hello", "\u2581world"],
      ids: [1, 22172, 3186],
      decoded: "<s> hello world",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u2581", "\u751f", "\u6d3b", "\u7684", "\u771f", "<0xE8>", "<0xB0>", "<0x9B>", "\u662f"],
      ids: [1, 29871, 30486, 31704, 30210, 30848, 235, 179, 158, 30392],
      decoded: "<s> \u751f\u6d3b\u7684\u771f\u8c1b\u662f",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      tokens: ["\u2581\u2581\u2581", "\u2581leading", "\u2581space"],
      ids: [1, 1678, 8236, 2913],
      decoded: "<s>    leading space",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["\u2581trailing", "\u2581space", "\u2581\u2581\u2581"],
      ids: [1, 25053, 2913, 1678],
      decoded: "<s> trailing space   ",
    },
    DOUBLE_SPACE: {
      text: BASE_TEST_STRINGS.DOUBLE_SPACE,
      tokens: ["\u2581Hi", "\u2581", "\u2581Hello"],
      ids: [1, 6324, 29871, 15043],
      decoded: "<s> Hi  Hello",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["\u2581test", "\u2581$", "1", "\u2581R", "2", "\u2581#", "3", "\u2581\u20ac", "4", "\u2581\u00a3", "5", "\u2581", "\u00a5", "6", "\u2581", "<0xE2>", "<0x82>", "<0xA3>", "7", "\u2581", "\u20b9", "8", "\u2581", "<0xE2>", "<0x82>", "<0xB1>", "9", "\u2581test"],
      ids: [1, 1243, 395, 29896, 390, 29906, 396, 29941, 25540, 29946, 15151, 29945, 29871, 30563, 29953, 29871, 229, 133, 166, 29955, 29871, 30620, 29947, 29871, 229, 133, 180, 29929, 1243],
      decoded: "<s> test $1 R2 #3 \u20ac4 \u00a35 \u00a56 \u20a37 \u20b98 \u20b19 test",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["\u2581I", "\u2581bought", "\u2581an", "\u2581apple", "\u2581for", "\u2581$", "1", ".", "0", "0", "\u2581at", "\u2581the", "\u2581store", "."],
      ids: [1, 306, 18093, 385, 26163, 363, 395, 29896, 29889, 29900, 29900, 472, 278, 3787, 29889],
      decoded: "<s> I bought an apple for $1.00 at the store.",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["\u2581you", "\u2026", "\u2581\u2581"],
      ids: [1, 366, 30098, 259],
      decoded: "<s> you\u2026  ",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["\u2581you", "\u2026", "\u00a0\u00a0"],
      ids: [1, 366, 30098, 8655],
      decoded: "<s> you\u2026\u00a0\u00a0",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["\u2581you", "\u2026", "\u00a0\u00a0", "you", "\u2026", "\u00a0\u00a0"],
      ids: [1, 366, 30098, 8655, 6293, 30098, 8655],
      decoded: "<s> you\u2026\u00a0\u00a0you\u2026\u00a0\u00a0",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["\u2581weird", "\u2581", "\uff5e", "\u2581edge", "\u2581", "\uff5e", "\u2581case"],
      ids: [1, 13543, 29871, 30739, 7636, 29871, 30739, 1206],
      decoded: "<s> weird \uff5e edge \uff5e case",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      tokens: ["\u2581", "\u2581This", "\u2581", "\u2581is", "\u2581", "\u2581a", "\u2581", "\u2581test", "\u2581", "\u2581."],
      ids: [1, 29871, 910, 29871, 338, 29871, 263, 29871, 1243, 29871, 869],
      decoded: "<s>  This  is  a  test  .",
    },
    POPULAR_EMOJIS: {
      text: BASE_TEST_STRINGS.POPULAR_EMOJIS,
      tokens: ["\u2581", "<0xF0>", "<0x9F>", "<0x98>", "<0x82>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0x8D>", "\u2581", "<0xF0>", "<0x9F>", "<0xA4>", "<0xA3>", "\u2581", "<0xF0>", "<0x9F>", "<0x98>", "<0x8D>", "\u2581", "<0xF0>", "<0x9F>", "<0x98>", "<0xAD>", "\u2581", "<0xF0>", "<0x9F>", "<0x8E>", "<0x89>", "\u2581", "<0xF0>", "<0x9F>", "<0x99>", "<0x8F>", "\u2581", "<0xF0>", "<0x9F>", "<0x98>", "<0x8A>", "\u2581", "<0xF0>", "<0x9F>", "<0x94>", "<0xA5>", "\u2581", "<0xF0>", "<0x9F>", "<0x98>", "<0x81>", "\u2581", "<0xF0>", "<0x9F>", "<0x98>", "<0x85>", "\u2581", "<0xF0>", "<0x9F>", "<0xA4>", "<0x97>", "\u2581", "<0xF0>", "<0x9F>", "<0x98>", "<0x86>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0x8F>", "\u2581", "<0xE2>", "<0x9D>", "<0xA4>", "\ufe0f", "\u2581", "<0xF0>", "<0x9F>", "<0x92>", "<0x9C>", "\u2581", "<0xF0>", "<0x9F>", "<0x92>", "<0x9A>", "\u2581", "<0xF0>", "<0x9F>", "<0x92>", "<0x97>", "\u2581", "<0xF0>", "<0x9F>", "<0x92>", "<0x99>", "\u2581", "<0xF0>", "<0x9F>", "<0x96>", "<0xA4>", "\u2581", "<0xF0>", "<0x9F>", "<0x98>", "<0x8E>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0x8C>", "\u2581", "<0xF0>", "<0x9F>", "<0xA5>", "<0xB3>", "\u2581", "<0xF0>", "<0x9F>", "<0x92>", "<0xAA>", "\u2581", "<0xE2>", "<0x9C>", "<0xA8>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0x89>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0x80>", "\u2581", "<0xF0>", "<0x9F>", "<0x92>", "<0xAF>", "\u2581", "<0xF0>", "<0x9F>", "<0x8E>", "<0x88>", "\u2581", "<0xF0>", "<0x9F>", "<0x99>", "<0x88>", "\u2581", "<0xF0>", "<0x9F>", "<0x99>", "<0x8C>", "\u2581", "<0xF0>", "<0x9F>", "<0x92>", "<0x80>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0x87>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0x8B>", "\u2581", "\u2705", "\u2581", "<0xF0>", "<0x9F>", "<0x8E>", "<0x81>", "\u2581", "<0xF0>", "<0x9F>", "<0x8C>", "<0x9E>", "\u2581", "<0xF0>", "<0x9F>", "<0x8C>", "<0xB8>", "\u2581", "<0xF0>", "<0x9F>", "<0x92>", "<0xB0>"],
      ids: [1, 29871, 243, 162, 155, 133, 29871, 243, 162, 148, 144, 29871, 243, 162, 167, 166, 29871, 243, 162, 155, 144, 29871, 243, 162, 155, 176, 29871, 243, 162, 145, 140, 29871, 243, 162, 156, 146, 29871, 243, 162, 155, 141, 29871, 243, 162, 151, 168, 29871, 243, 162, 155, 132, 29871, 243, 162, 155, 136, 29871, 243, 162, 167, 154, 29871, 243, 162, 155, 137, 29871, 243, 162, 148, 146, 29871, 229, 160, 167, 30598, 29871, 243, 162, 149, 159, 29871, 243, 162, 149, 157, 29871, 243, 162, 149, 154, 29871, 243, 162, 149, 156, 29871, 243, 162, 153, 167, 29871, 243, 162, 155, 145, 29871, 243, 162, 148, 143, 29871, 243, 162, 168, 182, 29871, 243, 162, 149, 173, 29871, 229, 159, 171, 29871, 243, 162, 148, 140, 29871, 243, 162, 148, 131, 29871, 243, 162, 149, 178, 29871, 243, 162, 145, 139, 29871, 243, 162, 156, 139, 29871, 243, 162, 156, 143, 29871, 243, 162, 149, 131, 29871, 243, 162, 148, 138, 29871, 243, 162, 148, 142, 29871, 31681, 29871, 243, 162, 145, 132, 29871, 243, 162, 143, 161, 29871, 243, 162, 143, 187, 29871, 243, 162, 149, 179],
      decoded: "<s> \ud83d\ude02 \ud83d\udc4d \ud83e\udd23 \ud83d\ude0d \ud83d\ude2d \ud83c\udf89 \ud83d\ude4f \ud83d\ude0a \ud83d\udd25 \ud83d\ude01 \ud83d\ude05 \ud83e\udd17 \ud83d\ude06 \ud83d\udc4f \u2764\ufe0f \ud83d\udc9c \ud83d\udc9a \ud83d\udc97 \ud83d\udc99 \ud83d\udda4 \ud83d\ude0e \ud83d\udc4c \ud83e\udd73 \ud83d\udcaa \u2728 \ud83d\udc49 \ud83d\udc40 \ud83d\udcaf \ud83c\udf88 \ud83d\ude48 \ud83d\ude4c \ud83d\udc80 \ud83d\udc47 \ud83d\udc4b \u2705 \ud83c\udf81 \ud83c\udf1e \ud83c\udf38 \ud83d\udcb0",
    },
    MULTIBYTE_EMOJIS: {
      text: BASE_TEST_STRINGS.MULTIBYTE_EMOJIS,
      tokens: ["\u2581", "<0xE2>", "<0x9C>", "<0xA8>", "\u2581", "<0xF0>", "<0x9F>", "<0xA4>", "<0x97>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0x81>", "\ufe0f", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0xB1>", "<0xF0>", "<0x9F>", "<0x8F>", "<0xBB>", "\u2581", "<0xF0>", "<0x9F>", "<0x95>", "<0xB5>", "\u200d", "\u2642", "\ufe0f", "\u2581", "<0xF0>", "<0x9F>", "<0xA7>", "<0x99>", "<0xF0>", "<0x9F>", "<0x8F>", "<0xBB>", "\u200d", "\u2642", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0xA8>", "<0xF0>", "<0x9F>", "<0x8F>", "<0xBB>", "\u200d", "<0xF0>", "<0x9F>", "<0x8C>", "<0xBE>", "\u2581", "<0xF0>", "<0x9F>", "<0xA7>", "<0x91>", "\u200d", "<0xF0>", "<0x9F>", "<0xA4>", "<0x9D>", "\u200d", "<0xF0>", "<0x9F>", "<0xA7>", "<0x91>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0xA9>", "\u200d", "<0xE2>", "<0x9D>", "<0xA4>", "\u200d", "<0xF0>", "<0x9F>", "<0x92>", "<0x8B>", "\u200d", "<0xF0>", "<0x9F>", "<0x91>", "<0xA8>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0xA9>", "\u200d", "<0xF0>", "<0x9F>", "<0x91>", "<0xA9>", "\u200d", "<0xF0>", "<0x9F>", "<0x91>", "<0xA7>", "\u200d", "<0xF0>", "<0x9F>", "<0x91>", "<0xA6>", "\u2581", "<0xF0>", "<0x9F>", "<0xA7>", "<0x91>", "<0xF0>", "<0x9F>", "<0x8F>", "<0xBB>", "\u200d", "<0xF0>", "<0x9F>", "<0xA4>", "<0x9D>", "\u200d", "<0xF0>", "<0x9F>", "<0xA7>", "<0x91>", "<0xF0>", "<0x9F>", "<0x8F>", "<0xBB>", "\u2581", "<0xF0>", "<0x9F>", "<0x8F>", "<0xB4>", "<0xF3>", "<0xA0>", "<0x81>", "<0xA7>", "<0xF3>", "<0xA0>", "<0x81>", "<0xA2>", "<0xF3>", "<0xA0>", "<0x81>", "<0xA5>", "<0xF3>", "<0xA0>", "<0x81>", "<0xAE>", "<0xF3>", "<0xA0>", "<0x81>", "<0xA7>", "<0xF3>", "<0xA0>", "<0x81>", "<0xBF>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0xA8>", "<0xF0>", "<0x9F>", "<0x8F>", "<0xBB>", "\u200d", "<0xE2>", "<0x9D>", "<0xA4>", "\ufe0f", "\u200d", "<0xF0>", "<0x9F>", "<0x92>", "<0x8B>", "\u200d", "<0xF0>", "<0x9F>", "<0x91>", "<0xA8>", "<0xF0>", "<0x9F>", "<0x8F>", "<0xBC>"],
      ids: [1, 29871, 229, 159, 171, 29871, 243, 162, 167, 154, 29871, 243, 162, 148, 132, 30598, 29871, 243, 162, 148, 180, 243, 162, 146, 190, 29871, 243, 162, 152, 184, 30722, 31135, 30598, 29871, 243, 162, 170, 156, 243, 162, 146, 190, 30722, 31135, 29871, 243, 162, 148, 171, 243, 162, 146, 190, 30722, 243, 162, 143, 193, 29871, 243, 162, 170, 148, 30722, 243, 162, 167, 160, 30722, 243, 162, 170, 148, 29871, 243, 162, 148, 172, 30722, 229, 160, 167, 30722, 243, 162, 149, 142, 30722, 243, 162, 148, 171, 29871, 243, 162, 148, 172, 30722, 243, 162, 148, 172, 30722, 243, 162, 148, 170, 30722, 243, 162, 148, 169, 29871, 243, 162, 170, 148, 243, 162, 146, 190, 30722, 243, 162, 167, 160, 30722, 243, 162, 170, 148, 243, 162, 146, 190, 29871, 243, 162, 146, 183, 246, 163, 132, 170, 246, 163, 132, 165, 246, 163, 132, 168, 246, 163, 132, 177, 246, 163, 132, 170, 246, 163, 132, 194, 29871, 243, 162, 148, 171, 243, 162, 146, 190, 30722, 229, 160, 167, 30598, 30722, 243, 162, 149, 142, 30722, 243, 162, 148, 171, 243, 162, 146, 191],
      decoded: "<s> \u2728 \ud83e\udd17 \ud83d\udc41\ufe0f \ud83d\udc71\ud83c\udffb \ud83d\udd75\u200d\u2642\ufe0f \ud83e\uddd9\ud83c\udffb\u200d\u2642 \ud83d\udc68\ud83c\udffb\u200d\ud83c\udf3e \ud83e\uddd1\u200d\ud83e\udd1d\u200d\ud83e\uddd1 \ud83d\udc69\u200d\u2764\u200d\ud83d\udc8b\u200d\ud83d\udc68 \ud83d\udc69\u200d\ud83d\udc69\u200d\ud83d\udc67\u200d\ud83d\udc66 \ud83e\uddd1\ud83c\udffb\u200d\ud83e\udd1d\u200d\ud83e\uddd1\ud83c\udffb \ud83c\udff4\udb40\udc67\udb40\udc62\udb40\udc65\udb40\udc6e\udb40\udc67\udb40\udc7f \ud83d\udc68\ud83c\udffb\u200d\u2764\ufe0f\u200d\ud83d\udc8b\u200d\ud83d\udc68\ud83c\udffc",
    },
    BPE_SCORES_PRIORITY_1: {
      text: LLAMA_TEST_STRINGS.BPE_SCORES_PRIORITY_1,
      tokens: ["\u2581gra", "bb", "ed"],
      ids: [1, 2646, 1327, 287],
      decoded: "<s> grabbed",
    },
    BPE_SCORES_PRIORITY_2: {
      text: LLAMA_TEST_STRINGS.BPE_SCORES_PRIORITY_2,
      tokens: ["\u2581", "\u2581gra", "bb", "ed"],
      ids: [1, 29871, 2646, 1327, 287],
      decoded: "<s>  grabbed",
    },
    BPE_SCORES_PRIORITY_3: {
      text: LLAMA_TEST_STRINGS.BPE_SCORES_PRIORITY_3,
      tokens: ["\u2581\u2581\u2581\u2581\u2581\u2581\u2581\u2581\u2581\u2581\u2581", "\u2581gra", "bb", "ed"],
      ids: [1, 9651, 2646, 1327, 287],
      decoded: "<s>            grabbed",
    },
    NEWLINE: {
      text: LLAMA_TEST_STRINGS.NEWLINE,
      tokens: ["\u2581", "<0x0A>"],
      ids: [1, 29871, 13],
      decoded: "<s> \n",
    },
    NEWLINE_WITH_LEADING_SPACE: {
      text: LLAMA_TEST_STRINGS.NEWLINE_WITH_LEADING_SPACE,
      tokens: ["\u2581\u2581", "<0x0A>"],
      ids: [1, 259, 13],
      decoded: "<s>  \n",
    },
    TABS: {
      text: LLAMA_TEST_STRINGS.TABS,
      tokens: ["\u2581", "<0x09>", "tabs", "<0x09>", "<0x09>", "<0x09>", "<0x09>", "out", "\u2581here"],
      ids: [1, 29871, 12, 21175, 12, 12, 12, 12, 449, 1244],
      decoded: "<s> \ttabs\t\t\t\tout here",
    },
    NEWLINE_AND_TAB: {
      text: LLAMA_TEST_STRINGS.NEWLINE_AND_TAB,
      tokens: ["\u2581", "<0x0A>", "<0x09>", "<0x0A>"],
      ids: [1, 29871, 13, 12, 13],
      decoded: "<s> \n\t\n",
    },
    CHINESE_LETTER: {
      text: LLAMA_TEST_STRINGS.CHINESE_LETTER,
      tokens: ["\u2581", "\u9547"],
      ids: [1, 29871, 30411],
      decoded: "<s> \u9547",
    },
    EMOJIS_1: {
      text: LLAMA_TEST_STRINGS.EMOJIS_1,
      tokens: ["\u2581", "<0xF0>", "<0x9F>", "<0xA6>", "<0x99>"],
      ids: [1, 29871, 243, 162, 169, 156],
      decoded: "<s> \ud83e\udd99",
    },
    EMOJIS_2: {
      text: LLAMA_TEST_STRINGS.EMOJIS_2,
      tokens: ["\u2581", "<0xF0>", "<0x9F>", "<0xA6>", "<0x99>", "<0xEA>", "<0x99>", "<0x8A>"],
      ids: [1, 29871, 243, 162, 169, 156, 237, 156, 141],
      decoded: "<s> \ud83e\udd99\ua64a",
    },
    EMOJIS_3: {
      text: LLAMA_TEST_STRINGS.EMOJIS_3,
      tokens: ["\u2581", "<0xEA>", "<0x99>", "<0x8A>", "<0xF0>", "<0x9F>", "<0xA6>", "<0x99>"],
      ids: [1, 29871, 237, 156, 141, 243, 162, 169, 156],
      decoded: "<s> \ua64a\ud83e\udd99",
    },
    PARAGRAPH: {
      text: LLAMA_TEST_STRINGS.PARAGRAPH,
      tokens: ["\u2581The", "\u2581ll", "ama", "\u2581(/", "\u02c8", "l", "\u0251", "\u02d0", "m", "\u0259", "/", ";", "\u2581", "<0xF0>", "<0x9F>", "<0xA6>", "<0x99>", "Span", "ish", "\u2581pron", "unci", "ation", ":", "\u2581[", "\u02c8", "\u028e", "ama", "])", "\u2581(", "L", "ama", "\u2581gl", "ama", ")", "\u2581is", "\u2581a", "\u2581domestic", "ated", "\u2581South", "\u2581American", "\u2581cam", "el", "id", ",", "\u2581widely", "\u2581used", "\u2581as", "\u2581a", "\u2581meat", "\u2581and", "\u2581pack", "\u2581animal", "\u2581by", "\u2581And", "e", "an", "\u2581cult", "ures", "\u2581since", "\u2581the", "\u2581Pre", "-", "Col", "umb", "ian", "\u2581era", ".", "\u2581L", "lam", "as", "\u2581are", "\u2581social", "\u2581animals", "\u2581and", "\u2581live", "\u2581with", "\u2581others", "\u2581as", "\u2581a", "\u2581her", "d", ".", "\u2581Their", "\u2581w", "ool", "\u2581is", "\u2581soft", "\u2581and", "\u2581contains", "\u2581only", "\u2581a", "\u2581small", "\u2581amount", "\u2581of", "\u2581lan", "olin", ".[", "2", "]", "\u2581L", "lam", "as", "\u2581can", "\u2581learn", "\u2581simple", "\u2581tasks", "\u2581after", "\u2581a", "\u2581few", "\u2581repet", "itions", ".", "\u2581When", "\u2581using", "\u2581a", "\u2581pack", ",", "\u2581they", "\u2581can", "\u2581carry", "\u2581about", "\u2581", "2", "5", "\u2581to", "\u2581", "3", "0", "%", "\u2581of", "\u2581their", "\u2581body", "\u2581weight", "\u2581for", "\u2581", "8", "\u2581to", "\u2581", "1", "3", "\u2581km", "\u2581(", "5", "\u2013", "8", "\u2581miles", ").", "[", "3", "]", "\u2581The", "\u2581name", "\u2581ll", "ama", "\u2581(", "in", "\u2581the", "\u2581past", "\u2581also", "\u2581sp", "elled", '\u2581"', "l", "ama", '"', "\u2581or", '\u2581"', "gl", "ama", '")', "\u2581was", "\u2581adopted", "\u2581by", "\u2581European", "\u2581sett", "lers", "\u2581from", "\u2581native", "\u2581Peru", "vi", "ans", ".[", "4", "]", "\u2581The", "\u2581ancest", "ors", "\u2581of", "\u2581llam", "as", "\u2581are", "\u2581thought", "\u2581to", "\u2581have", "\u2581origin", "ated", "\u2581from", "\u2581the", "\u2581Great", "\u2581Pla", "ins", "\u2581of", "\u2581North", "\u2581America", "\u2581about", "\u2581", "4", "0", "\u2581million", "\u2581years", "\u2581ago", ",", "\u2581and", "\u2581subsequently", "\u2581migr", "ated", "\u2581to", "\u2581South", "\u2581America", "\u2581about", "\u2581three", "\u2581million", "\u2581years", "\u2581ago", "\u2581during", "\u2581the", "\u2581Great", "\u2581American", "\u2581Inter", "change", ".", "\u2581By", "\u2581the", "\u2581end", "\u2581of", "\u2581the", "\u2581last", "\u2581ice", "\u2581age", "\u2581(", "1", "0", ",", "0", "0", "0", "\u2013", "1", "2", ",", "0", "0", "0", "\u2581years", "\u2581ago", "),", "\u2581cam", "el", "ids", "\u2581were", "\u2581ext", "inct", "\u2581in", "\u2581North", "\u2581America", ".[", "3", "]", "\u2581As", "\u2581of", "\u2581", "2", "0", "0", "7", ",", "\u2581there", "\u2581were", "\u2581over", "\u2581seven", "\u2581million", "\u2581llam", "as", "\u2581and", "\u2581al", "p", "ac", "as", "\u2581in", "\u2581South", "\u2581America", "\u2581and", "\u2581over", "\u2581", "1", "5", "8", ",", "0", "0", "0", "\u2581llam", "as", "\u2581and", "\u2581", "1", "0", "0", ",", "0", "0", "0", "<0xEA>", "<0x99>", "<0x8A>", "<0xF0>", "<0x9F>", "<0xA6>", "<0x99>", "\u2581al", "p", "ac", "as", ",", "\u2581desc", "ended", "\u2581from", "\u2581pro", "gen", "itors", "\u2581imported", "\u2581late", "\u2581in", "\u2581the", "\u2581", "2", "0", "th", "\u2581century", ",", "\u2581in", "\u2581the", "\u2581United", "\u2581States", "\u2581and", "\u2581Canada", ".[", "5", "]", "\u2581In", "\u2581A", "ym", "ara", "\u2581myth", "ology", ",", "\u2581llam", "as", "\u2581are", "\u2581important", "\u2581be", "ings", ".", "\u2581The", "\u2581Heaven", "ly", "\u2581L", "l", "ama", "\u2581is", "\u2581said", "\u2581to", "\u2581drink", "\u2581water", "\u2581from", "\u2581the", "\u2581ocean", "\u2581and", "\u2581ur", "in", "ates", "\u2581as", "\u2581it", "\u2581ra", "ins", ".[", "6", "]", "\u2581According", "\u2581to", "\u2581A", "ym", "ara", "\u2581es", "chat", "ology", ",", "\u2581llam", "as", "\u2581will", "\u2581return", "\u2581to", "\u2581the", "\u2581water", "\u2581spr", "ings", "\u2581and", "\u2581l", "ago", "ons", "\u2581where", "\u2581they", "\u2581come", "\u2581from", "\u2581at", "\u2581the", "\u2581end", "\u2581of", "\u2581time", ".[", "6", "]"],
      ids: [1, 450, 11148, 3304, 20374, 30176, 29880, 30426, 30215, 29885, 30184, 29914, 29936, 29871, 243, 162, 169, 156, 15495, 728, 11504, 11173, 362, 29901, 518, 30176, 31743, 3304, 2314, 313, 29931, 3304, 3144, 3304, 29897, 338, 263, 21849, 630, 4275, 3082, 3949, 295, 333, 29892, 17644, 1304, 408, 263, 27654, 322, 4870, 13019, 491, 1126, 29872, 273, 4185, 1973, 1951, 278, 4721, 29899, 1625, 3774, 713, 3152, 29889, 365, 5288, 294, 526, 5264, 15006, 322, 5735, 411, 4045, 408, 263, 902, 29881, 29889, 11275, 281, 1507, 338, 4964, 322, 3743, 871, 263, 2319, 5253, 310, 10906, 22878, 7226, 29906, 29962, 365, 5288, 294, 508, 5110, 2560, 9595, 1156, 263, 2846, 21159, 2187, 29889, 1932, 773, 263, 4870, 29892, 896, 508, 8677, 1048, 29871, 29906, 29945, 304, 29871, 29941, 29900, 29995, 310, 1009, 3573, 7688, 363, 29871, 29947, 304, 29871, 29896, 29941, 2383, 313, 29945, 29994, 29947, 7800, 467, 29961, 29941, 29962, 450, 1024, 11148, 3304, 313, 262, 278, 4940, 884, 805, 14356, 376, 29880, 3304, 29908, 470, 376, 3820, 3304, 1159, 471, 16356, 491, 7824, 3604, 9306, 515, 7531, 25493, 1403, 550, 7226, 29946, 29962, 450, 19525, 943, 310, 11829, 294, 526, 2714, 304, 505, 3978, 630, 515, 278, 7027, 13494, 1144, 310, 4644, 6813, 1048, 29871, 29946, 29900, 7284, 2440, 8020, 29892, 322, 17602, 9725, 630, 304, 4275, 6813, 1048, 2211, 7284, 2440, 8020, 2645, 278, 7027, 3082, 4124, 3167, 29889, 2648, 278, 1095, 310, 278, 1833, 14890, 5046, 313, 29896, 29900, 29892, 29900, 29900, 29900, 29994, 29896, 29906, 29892, 29900, 29900, 29900, 2440, 8020, 511, 3949, 295, 4841, 892, 1294, 5562, 297, 4644, 6813, 7226, 29941, 29962, 1094, 310, 29871, 29906, 29900, 29900, 29955, 29892, 727, 892, 975, 9881, 7284, 11829, 294, 322, 394, 29886, 562, 294, 297, 4275, 6813, 322, 975, 29871, 29896, 29945, 29947, 29892, 29900, 29900, 29900, 11829, 294, 322, 29871, 29896, 29900, 29900, 29892, 29900, 29900, 29900, 237, 156, 141, 243, 162, 169, 156, 394, 29886, 562, 294, 29892, 5153, 2760, 515, 410, 1885, 17259, 19673, 5683, 297, 278, 29871, 29906, 29900, 386, 6462, 29892, 297, 278, 3303, 3900, 322, 7400, 7226, 29945, 29962, 512, 319, 962, 2518, 22082, 3002, 29892, 11829, 294, 526, 4100, 367, 886, 29889, 450, 22977, 368, 365, 29880, 3304, 338, 1497, 304, 13748, 4094, 515, 278, 23474, 322, 5065, 262, 1078, 408, 372, 1153, 1144, 7226, 29953, 29962, 7579, 304, 319, 962, 2518, 831, 13496, 3002, 29892, 11829, 294, 674, 736, 304, 278, 4094, 7689, 886, 322, 301, 4425, 787, 988, 896, 2041, 515, 472, 278, 1095, 310, 931, 7226, 29953, 29962],
      decoded: '<s> The llama (/\u02c8l\u0251\u02d0m\u0259/; \ud83e\udd99Spanish pronunciation: [\u02c8\u028eama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the Pre-Columbian era. Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5\u20138 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4] The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000\u201312,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000\ua64a\ud83e\udd99 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5] In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology, llamas will return to the water springs and lagoons where they come from at the end of time.[6]',
    },
  },
  "Xenova/llama3-tokenizer": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      tokens: ["How", "\u0120are", "\u0120you", "\u0120doing", "?"],
      ids: [4438, 527, 499, 3815, 30],
      decoded: "How are you doing?",
    },
    SIMPLE_WITH_PUNCTUATION: {
      text: BASE_TEST_STRINGS.SIMPLE_WITH_PUNCTUATION,
      tokens: ["You", "\u0120should", "'ve", "\u0120done", "\u0120this"],
      ids: [2675, 1288, 3077, 2884, 420],
      decoded: "You should've done this",
    },
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["012", "345", "678", "9", "\u0120", "0", "\u0120", "1", "\u0120", "2", "\u0120", "3", "\u0120", "4", "\u0120", "5", "\u0120", "6", "\u0120", "7", "\u0120", "8", "\u0120", "9", "\u0120", "10", "\u0120", "100", "\u0120", "100", "0"],
      ids: [11531, 12901, 17458, 24, 220, 15, 220, 16, 220, 17, 220, 18, 220, 19, 220, 20, 220, 21, 220, 22, 220, 23, 220, 24, 220, 605, 220, 1041, 220, 1041, 15],
      decoded: "0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["The", "\u0120company", "\u0120was", "\u0120founded", "\u0120in", "\u0120", "201", "6", "."],
      ids: [791, 2883, 574, 18538, 304, 220, 679, 21, 13],
      decoded: "The company was founded in 2016.",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["A", "\u010a", "'ll", "\u0120!!", "to", "?'", "d", "''", "d", "\u0120of", ",", "\u0120can", "'t", "."],
      ids: [32, 198, 3358, 11261, 998, 20837, 67, 4708, 67, 315, 11, 649, 956, 13],
      decoded: "A\n'll!!to?'d''d of, can't.",
    },
    PYTHON_CODE: {
      text: BASE_TEST_STRINGS.PYTHON_CODE,
      tokens: ["def", "\u0120main", "():\u010a", "\u0109pass"],
      ids: [755, 1925, 4019, 42531],
      decoded: "def main():\n\tpass",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["let", "\u0120a", "\u0120=", "\u0120obj", ".toString", "();\u010a", "toString", "();"],
      ids: [1169, 264, 284, 2909, 5180, 545, 6712, 2178],
      decoded: "let a = obj.toString();\ntoString();",
    },
    NEWLINES: {
      text: LLAMA_TEST_STRINGS.NEWLINES,
      tokens: ["ax", "\u010a", "####\u010a", "boo"],
      ids: [710, 198, 71050, 34093],
      decoded: "ax\n####\nboo",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["UN", "want", "\u00c3\u00a9d", ",", "running"],
      ids: [1899, 53757, 15433, 11, 28272],
      decoded: "UNwant\u00e9d,running",
    },
    CONTROL_TOKENS: {
      text: BASE_TEST_STRINGS.CONTROL_TOKENS,
      tokens: ["1", "\u0100", "2", "\u00ef\u00bf\u00bd", "3"],
      ids: [16, 188, 17, 5809, 18],
      decoded: "1\u00002\ufffd3",
    },
    HELLO_WORLD_TITLECASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_TITLECASE,
      tokens: ["Hello", "\u0120World"],
      ids: [9906, 4435],
      decoded: "Hello World",
    },
    HELLO_WORLD_LOWERCASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_LOWERCASE,
      tokens: ["hello", "\u0120world"],
      ids: [15339, 1917],
      decoded: "hello world",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u00e7\u0136\u0141\u00e6\u00b4\u00bb", "\u00e7\u013c\u0126", "\u00e7\u013e\u0141", "\u00e8\u00b0", "\u013d", "\u00e6\u013a\u00af"],
      ids: [104654, 9554, 89151, 39013, 249, 21043],
      decoded: "\u751f\u6d3b\u7684\u771f\u8c1b\u662f",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      tokens: ["\u0120\u0120", "\u0120leading", "\u0120space"],
      ids: [256, 6522, 3634],
      decoded: "   leading space",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["tr", "ailing", "\u0120space", "\u0120\u0120\u0120"],
      ids: [376, 14612, 3634, 262],
      decoded: "trailing space   ",
    },
    DOUBLE_SPACE: {
      text: BASE_TEST_STRINGS.DOUBLE_SPACE,
      tokens: ["Hi", "\u0120", "\u0120Hello"],
      ids: [13347, 220, 22691],
      decoded: "Hi  Hello",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["test", "\u0120$", "1", "\u0120R", "2", "\u0120#", "3", "\u0120\u00e2\u0124\u00ac", "4", "\u0120\u00c2\u00a3", "5", "\u0120\u00c2\u00a5", "6", "\u0120\u00e2\u0124", "\u00a3", "7", "\u0120\u00e2\u0124\u00b9", "8", "\u0120\u00e2\u0124", "\u00b1", "9", "\u0120test"],
      ids: [1985, 400, 16, 432, 17, 674, 18, 13281, 19, 7083, 20, 72588, 21, 113384, 96, 22, 90891, 23, 113384, 109, 24, 1296],
      decoded: "test $1 R2 #3 \u20ac4 \u00a35 \u00a56 \u20a37 \u20b98 \u20b19 test",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["I", "\u0120bought", "\u0120an", "\u0120apple", "\u0120for", "\u0120$", "1", ".", "00", "\u0120at", "\u0120the", "\u0120store", "."],
      ids: [40, 11021, 459, 24149, 369, 400, 16, 13, 410, 520, 279, 3637, 13],
      decoded: "I bought an apple for $1.00 at the store.",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u0120\u0120"],
      ids: [9514, 1981, 256],
      decoded: "you\u2026  ",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u00c2\u0142\u00c2\u0142"],
      ids: [9514, 1981, 9421],
      decoded: "you\u2026\u00a0\u00a0",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u00c2\u0142", "\u00c2\u0142", "you", "\u00e2\u0122\u00a6", "\u00c2\u0142\u00c2\u0142"],
      ids: [9514, 1981, 4194, 4194, 9514, 1981, 9421],
      decoded: "you\u2026\u00a0\u00a0you\u2026\u00a0\u00a0",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["we", "ird", "\u0120\u00ef\u00bd\u0140", "\u0120edge", "\u0120\u00ef\u00bd\u0140", "\u0120case"],
      ids: [906, 2668, 111942, 6964, 111942, 1162],
      decoded: "weird \uff5e edge \uff5e case",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      tokens: ["\u00e2\u0138", "\u0123", "This", "\u0120\u00e2\u0138", "\u0123", "is", "\u0120\u00e2\u0138", "\u0123", "a", "\u0120\u00e2\u0138", "\u0123", "test", "\u0120\u00e2\u0138", "\u0123", "."],
      ids: [10634, 223, 2028, 14860, 223, 285, 14860, 223, 64, 14860, 223, 1985, 14860, 223, 13],
      decoded: "\u2581This \u2581is \u2581a \u2581test \u2581.",
    },
    POPULAR_EMOJIS: {
      text: BASE_TEST_STRINGS.POPULAR_EMOJIS,
      tokens: ["\u00f0\u0141\u013a", "\u0124", "\u0120\u00f0\u0141\u0133", "\u012f", "\u0120\u00f0\u0141", "\u00a4", "\u00a3", "\u0120\u00f0\u0141\u013a", "\u012f", "\u0120\u00f0\u0141\u013a", "\u0143", "\u0120\u00f0\u0141", "\u0130", "\u012b", "\u0120\u00f0\u0141", "\u013b", "\u0131", "\u0120\u00f0\u0141\u013a", "\u012c", "\u0120\u00f0\u0141\u0136", "\u00a5", "\u0120\u00f0\u0141\u013a", "\u0123", "\u0120\u00f0\u0141\u013a", "\u0127", "\u0120\u00f0\u0141", "\u00a4", "\u0139", "\u0120\u00f0\u0141\u013a", "\u0128", "\u0120\u00f0\u0141\u0133", "\u0131", "\u0120\u00e2\u013f\u00a4", "\u00ef\u00b8\u0131", "\u0120\u00f0\u0141\u0134", "\u013e", "\u0120\u00f0\u0141\u0134", "\u013c", "\u0120\u00f0\u0141\u0134", "\u0139", "\u0120\u00f0\u0141\u0134", "\u013b", "\u0120\u00f0\u0141", "\u0138", "\u00a4", "\u0120\u00f0\u0141\u013a", "\u0130", "\u0120\u00f0\u0141\u0133", "\u012e", "\u0120\u00f0\u0141", "\u00a5", "\u00b3", "\u0120\u00f0\u0141\u0134", "\u00aa", "\u0120\u00e2\u013e", "\u00a8", "\u0120\u00f0\u0141\u0133", "\u012b", "\u0120\u00f0\u0141\u0133", "\u0122", "\u0120\u00f0\u0141\u0134", "\u00af", "\u0120\u00f0\u0141", "\u0130", "\u012a", "\u0120\u00f0\u0141", "\u013b", "\u012a", "\u0120\u00f0\u0141", "\u013b", "\u012e", "\u0120\u00f0\u0141\u0134", "\u0122", "\u0120\u00f0\u0141\u0133", "\u0129", "\u0120\u00f0\u0141\u0133", "\u012d", "\u0120\u00e2\u013e", "\u0127", "\u0120\u00f0\u0141", "\u0130", "\u0123", "\u0120\u00f0\u0141", "\u012e", "\u0140", "\u0120\u00f0\u0141", "\u012e", "\u00b8", "\u0120\u00f0\u0141\u0134", "\u00b0"],
      ids: [76460, 224, 62904, 235, 11410, 97, 96, 27623, 235, 27623, 255, 11410, 236, 231, 11410, 247, 237, 27623, 232, 96169, 98, 27623, 223, 27623, 227, 11410, 97, 245, 27623, 228, 62904, 237, 71570, 31643, 64139, 250, 64139, 248, 64139, 245, 64139, 247, 11410, 244, 97, 27623, 236, 62904, 234, 11410, 98, 111, 64139, 103, 26602, 101, 62904, 231, 62904, 222, 64139, 107, 11410, 236, 230, 11410, 247, 230, 11410, 247, 234, 64139, 222, 62904, 229, 62904, 233, 26602, 227, 11410, 236, 223, 11410, 234, 252, 11410, 234, 116, 64139, 108],
      decoded: "\ud83d\ude02 \ud83d\udc4d \ud83e\udd23 \ud83d\ude0d \ud83d\ude2d \ud83c\udf89 \ud83d\ude4f \ud83d\ude0a \ud83d\udd25 \ud83d\ude01 \ud83d\ude05 \ud83e\udd17 \ud83d\ude06 \ud83d\udc4f \u2764\ufe0f \ud83d\udc9c \ud83d\udc9a \ud83d\udc97 \ud83d\udc99 \ud83d\udda4 \ud83d\ude0e \ud83d\udc4c \ud83e\udd73 \ud83d\udcaa \u2728 \ud83d\udc49 \ud83d\udc40 \ud83d\udcaf \ud83c\udf88 \ud83d\ude48 \ud83d\ude4c \ud83d\udc80 \ud83d\udc47 \ud83d\udc4b \u2705 \ud83c\udf81 \ud83c\udf1e \ud83c\udf38 \ud83d\udcb0",
    },
    MULTIBYTE_EMOJIS: {
      text: BASE_TEST_STRINGS.MULTIBYTE_EMOJIS,
      tokens: ["\u00e2\u013e", "\u00a8", "\u0120\u00f0\u0141", "\u00a4", "\u0139", "\u0120\u00f0\u0141\u0133", "\u0123", "\u00ef\u00b8\u0131", "\u0120\u00f0\u0141\u0133", "\u00b1", "\u00f0\u0141", "\u0131", "\u00bb", "\u0120\u00f0\u0141", "\u0137", "\u00b5", "\u00e2\u0122\u012f", "\u00e2\u013b", "\u0124", "\u00ef\u00b8\u0131", "\u0120\u00f0\u0141", "\u00a7", "\u013b", "\u00f0\u0141", "\u0131", "\u00bb", "\u00e2\u0122\u012f", "\u00e2\u013b", "\u0124", "\u0120\u00f0\u0141\u0133", "\u00a8", "\u00f0\u0141", "\u0131", "\u00bb", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u012e", "\u00be", "\u0120\u00f0\u0141", "\u00a7", "\u0133", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u00a4", "\u013f", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u00a7", "\u0133", "\u0120\u00f0\u0141\u0133", "\u00a9", "\u00e2\u0122\u012f", "\u00e2\u013f\u00a4", "\u00e2\u0122\u012f", "\u00f0\u0141\u0134", "\u012d", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u0133", "\u00a8", "\u0120\u00f0\u0141\u0133", "\u00a9", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u0133", "\u00a9", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u0133", "\u00a7", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u0133", "\u00a6", "\u0120\u00f0\u0141", "\u00a7", "\u0133", "\u00f0\u0141", "\u0131", "\u00bb", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u00a4", "\u013f", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u00a7", "\u0133", "\u00f0\u0141", "\u0131", "\u00bb", "\u0120\u00f0\u0141", "\u0131", "\u00b4", "\u00f3", "\u0142\u0123", "\u00a7", "\u00f3", "\u0142\u0123", "\u00a2", "\u00f3", "\u0142\u0123", "\u00a5", "\u00f3", "\u0142\u0123", "\u00ae", "\u00f3", "\u0142\u0123", "\u00a7", "\u00f3", "\u0142\u0123", "\u00bf", "\u0120\u00f0\u0141\u0133", "\u00a8", "\u00f0\u0141", "\u0131", "\u00bb", "\u00e2\u0122\u012f", "\u00e2\u013f\u00a4", "\u00ef\u00b8\u0131", "\u00e2\u0122\u012f", "\u00f0\u0141\u0134", "\u012d", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u0133", "\u00a8", "\u00f0\u0141", "\u0131", "\u00bc"],
      ids: [38798, 101, 11410, 97, 245, 62904, 223, 31643, 62904, 109, 9468, 237, 119, 11410, 243, 113, 102470, 17245, 224, 31643, 11410, 100, 247, 9468, 237, 119, 102470, 17245, 224, 62904, 101, 9468, 237, 119, 102470, 9468, 234, 122, 11410, 100, 239, 102470, 9468, 97, 251, 102470, 9468, 100, 239, 62904, 102, 102470, 121643, 102470, 93273, 233, 102470, 9468, 239, 101, 62904, 102, 102470, 9468, 239, 102, 102470, 9468, 239, 100, 102470, 9468, 239, 99, 11410, 100, 239, 9468, 237, 119, 102470, 9468, 97, 251, 102470, 9468, 100, 239, 9468, 237, 119, 11410, 237, 112, 175, 16050, 100, 175, 16050, 95, 175, 16050, 98, 175, 16050, 106, 175, 16050, 100, 175, 16050, 123, 62904, 101, 9468, 237, 119, 102470, 121643, 31643, 102470, 93273, 233, 102470, 9468, 239, 101, 9468, 237, 120],
      decoded: "\u2728 \ud83e\udd17 \ud83d\udc41\ufe0f \ud83d\udc71\ud83c\udffb \ud83d\udd75\u200d\u2642\ufe0f \ud83e\uddd9\ud83c\udffb\u200d\u2642 \ud83d\udc68\ud83c\udffb\u200d\ud83c\udf3e \ud83e\uddd1\u200d\ud83e\udd1d\u200d\ud83e\uddd1 \ud83d\udc69\u200d\u2764\u200d\ud83d\udc8b\u200d\ud83d\udc68 \ud83d\udc69\u200d\ud83d\udc69\u200d\ud83d\udc67\u200d\ud83d\udc66 \ud83e\uddd1\ud83c\udffb\u200d\ud83e\udd1d\u200d\ud83e\uddd1\ud83c\udffb \ud83c\udff4\udb40\udc67\udb40\udc62\udb40\udc65\udb40\udc6e\udb40\udc67\udb40\udc7f \ud83d\udc68\ud83c\udffb\u200d\u2764\ufe0f\u200d\ud83d\udc8b\u200d\ud83d\udc68\ud83c\udffc",
    },
    BPE_SCORES_PRIORITY_1: {
      text: LLAMA_TEST_STRINGS.BPE_SCORES_PRIORITY_1,
      tokens: ["grab", "bed"],
      ids: [59312, 2788],
      decoded: "grabbed",
    },
    BPE_SCORES_PRIORITY_2: {
      text: LLAMA_TEST_STRINGS.BPE_SCORES_PRIORITY_2,
      tokens: ["\u0120grabbed"],
      ids: [30418],
      decoded: " grabbed",
    },
    BPE_SCORES_PRIORITY_3: {
      text: LLAMA_TEST_STRINGS.BPE_SCORES_PRIORITY_3,
      tokens: ["\u0120\u0120\u0120\u0120\u0120\u0120\u0120\u0120\u0120\u0120", "\u0120grabbed"],
      ids: [1881, 30418],
      decoded: "           grabbed",
    },
    NEWLINE: {
      text: LLAMA_TEST_STRINGS.NEWLINE,
      tokens: ["\u010a"],
      ids: [198],
      decoded: "\n",
    },
    NEWLINE_WITH_LEADING_SPACE: {
      text: LLAMA_TEST_STRINGS.NEWLINE_WITH_LEADING_SPACE,
      tokens: ["\u0120\u010a"],
      ids: [720],
      decoded: " \n",
    },
    TABS: {
      text: LLAMA_TEST_STRINGS.TABS,
      tokens: ["\u0109t", "abs", "\u0109\u0109\u0109", "\u0109out", "\u0120here"],
      ids: [3324, 3518, 573, 14294, 1618],
      decoded: "\ttabs\t\t\t\tout here",
    },
    NEWLINE_AND_TAB: {
      text: LLAMA_TEST_STRINGS.NEWLINE_AND_TAB,
      tokens: ["\u010a\u0109\u010a"],
      ids: [18108],
      decoded: "\n\t\n",
    },
    CHINESE_LETTER: {
      text: LLAMA_TEST_STRINGS.CHINESE_LETTER,
      tokens: ["\u00e9\u0137\u0129"],
      ids: [104643],
      decoded: "\u9547",
    },
    EMOJIS_1: {
      text: LLAMA_TEST_STRINGS.EMOJIS_1,
      tokens: ["\u00f0\u0141", "\u00a6", "\u013b"],
      ids: [9468, 99, 247],
      decoded: "\ud83e\udd99",
    },
    EMOJIS_2: {
      text: LLAMA_TEST_STRINGS.EMOJIS_2,
      tokens: ["\u00f0\u0141", "\u00a6", "\u013b", "\u00ea", "\u013b", "\u012c"],
      ids: [9468, 99, 247, 166, 247, 232],
      decoded: "\ud83e\udd99\ua64a",
    },
    EMOJIS_3: {
      text: LLAMA_TEST_STRINGS.EMOJIS_3,
      tokens: ["\u00ea", "\u013b", "\u012c", "\u00f0\u0141", "\u00a6", "\u013b"],
      ids: [166, 247, 232, 9468, 99, 247],
      decoded: "\ua64a\ud83e\udd99",
    },
    PARAGRAPH: {
      text: LLAMA_TEST_STRINGS.PARAGRAPH,
      tokens: ["The", "\u0120llama", "\u0120(/", "\u00cb", "\u012a", "l", "\u00c9", "\u0133", "\u00cb", "\u0132", "m", "\u00c9\u013b", "/", ";", "\u0120\u00f0\u0141", "\u00a6", "\u013b", "Spanish", "\u0120pronunciation", ":", "\u0120[", "\u00cb", "\u012a", "\u00ca", "\u0130", "ama", "])", "\u0120(", "L", "ama", "\u0120gl", "ama", ")", "\u0120is", "\u0120a", "\u0120domestic", "ated", "\u0120South", "\u0120American", "\u0120camel", "id", ",", "\u0120widely", "\u0120used", "\u0120as", "\u0120a", "\u0120meat", "\u0120and", "\u0120pack", "\u0120animal", "\u0120by", "\u0120And", "ean", "\u0120cultures", "\u0120since", "\u0120the", "\u0120Pre", "-C", "olum", "bian", "\u0120era", ".", "\u0120L", "lam", "as", "\u0120are", "\u0120social", "\u0120animals", "\u0120and", "\u0120live", "\u0120with", "\u0120others", "\u0120as", "\u0120a", "\u0120herd", ".", "\u0120Their", "\u0120wool", "\u0120is", "\u0120soft", "\u0120and", "\u0120contains", "\u0120only", "\u0120a", "\u0120small", "\u0120amount", "\u0120of", "\u0120lan", "olin", ".[", "2", "]", "\u0120L", "lam", "as", "\u0120can", "\u0120learn", "\u0120simple", "\u0120tasks", "\u0120after", "\u0120a", "\u0120few", "\u0120repetitions", ".", "\u0120When", "\u0120using", "\u0120a", "\u0120pack", ",", "\u0120they", "\u0120can", "\u0120carry", "\u0120about", "\u0120", "25", "\u0120to", "\u0120", "30", "%", "\u0120of", "\u0120their", "\u0120body", "\u0120weight", "\u0120for", "\u0120", "8", "\u0120to", "\u0120", "13", "\u0120km", "\u0120(", "5", "\u00e2\u0122\u0135", "8", "\u0120miles", ").[", "3", "]", "\u0120The", "\u0120name", "\u0120llama", "\u0120(", "in", "\u0120the", "\u0120past", "\u0120also", "\u0120spelled", '\u0120"', "lama", '"', "\u0120or", '\u0120"', "gl", "ama", '")', "\u0120was", "\u0120adopted", "\u0120by", "\u0120European", "\u0120settlers", "\u0120from", "\u0120native", "\u0120Per", "uv", "ians", ".[", "4", "]", "\u0120The", "\u0120ancestors", "\u0120of", "\u0120ll", "amas", "\u0120are", "\u0120thought", "\u0120to", "\u0120have", "\u0120originated", "\u0120from", "\u0120the", "\u0120Great", "\u0120Plains", "\u0120of", "\u0120North", "\u0120America", "\u0120about", "\u0120", "40", "\u0120million", "\u0120years", "\u0120ago", ",", "\u0120and", "\u0120subsequently", "\u0120migrated", "\u0120to", "\u0120South", "\u0120America", "\u0120about", "\u0120three", "\u0120million", "\u0120years", "\u0120ago", "\u0120during", "\u0120the", "\u0120Great", "\u0120American", "\u0120Inter", "change", ".", "\u0120By", "\u0120the", "\u0120end", "\u0120of", "\u0120the", "\u0120last", "\u0120ice", "\u0120age", "\u0120(", "10", ",", "000", "\u00e2\u0122\u0135", "12", ",", "000", "\u0120years", "\u0120ago", "),", "\u0120camel", "ids", "\u0120were", "\u0120extinct", "\u0120in", "\u0120North", "\u0120America", ".[", "3", "]", "\u0120As", "\u0120of", "\u0120", "200", "7", ",", "\u0120there", "\u0120were", "\u0120over", "\u0120seven", "\u0120million", "\u0120ll", "amas", "\u0120and", "\u0120al", "pac", "as", "\u0120in", "\u0120South", "\u0120America", "\u0120and", "\u0120over", "\u0120", "158", ",", "000", "\u0120ll", "amas", "\u0120and", "\u0120", "100", ",", "000", "\u00ea", "\u013b", "\u012c", "\u00f0\u0141", "\u00a6", "\u013b", "\u0120al", "pac", "as", ",", "\u0120descended", "\u0120from", "\u0120progen", "itors", "\u0120imported", "\u0120late", "\u0120in", "\u0120the", "\u0120", "20", "th", "\u0120century", ",", "\u0120in", "\u0120the", "\u0120United", "\u0120States", "\u0120and", "\u0120Canada", ".[", "5", "]", "\u0120In", "\u0120A", "ym", "ara", "\u0120mythology", ",", "\u0120ll", "amas", "\u0120are", "\u0120important", "\u0120beings", ".", "\u0120The", "\u0120Heavenly", "\u0120L", "lama", "\u0120is", "\u0120said", "\u0120to", "\u0120drink", "\u0120water", "\u0120from", "\u0120the", "\u0120ocean", "\u0120and", "\u0120ur", "in", "ates", "\u0120as", "\u0120it", "\u0120rains", ".[", "6", "]", "\u0120According", "\u0120to", "\u0120A", "ym", "ara", "\u0120es", "chat", "ology", ",", "\u0120ll", "amas", "\u0120will", "\u0120return", "\u0120to", "\u0120the", "\u0120water", "\u0120springs", "\u0120and", "\u0120l", "ago", "ons", "\u0120where", "\u0120they", "\u0120come", "\u0120from", "\u0120at", "\u0120the", "\u0120end", "\u0120of", "\u0120time", ".[", "6", "]"],
      ids: [791, 94776, 47325, 135, 230, 75, 133, 239, 135, 238, 76, 99638, 14, 26, 11410, 99, 247, 62897, 71722, 25, 510, 135, 230, 134, 236, 3105, 2526, 320, 43, 3105, 2840, 3105, 8, 374, 264, 13018, 660, 4987, 3778, 50252, 307, 11, 13882, 1511, 439, 264, 13339, 323, 3854, 10065, 555, 1628, 5420, 27833, 2533, 279, 5075, 7813, 1152, 13464, 11639, 13, 445, 24705, 300, 527, 3674, 10099, 323, 3974, 449, 3885, 439, 264, 59213, 13, 11205, 39640, 374, 8579, 323, 5727, 1193, 264, 2678, 3392, 315, 31791, 37737, 8032, 17, 60, 445, 24705, 300, 649, 4048, 4382, 9256, 1306, 264, 2478, 86066, 13, 3277, 1701, 264, 3854, 11, 814, 649, 6920, 922, 220, 914, 311, 220, 966, 4, 315, 872, 2547, 4785, 369, 220, 23, 311, 220, 1032, 13437, 320, 20, 4235, 23, 8931, 94638, 18, 60, 578, 836, 94776, 320, 258, 279, 3347, 1101, 68918, 330, 81101, 1, 477, 330, 6200, 3105, 909, 574, 18306, 555, 7665, 61107, 505, 10068, 3700, 12328, 5493, 8032, 19, 60, 578, 38618, 315, 9507, 29189, 527, 3463, 311, 617, 44853, 505, 279, 8681, 63911, 315, 4892, 5270, 922, 220, 1272, 3610, 1667, 4227, 11, 323, 28520, 73691, 311, 4987, 5270, 922, 2380, 3610, 1667, 4227, 2391, 279, 8681, 3778, 5783, 3455, 13, 3296, 279, 842, 315, 279, 1566, 10054, 4325, 320, 605, 11, 931, 4235, 717, 11, 931, 1667, 4227, 705, 50252, 3447, 1051, 69918, 304, 4892, 5270, 8032, 18, 60, 1666, 315, 220, 1049, 22, 11, 1070, 1051, 927, 8254, 3610, 9507, 29189, 323, 453, 46051, 300, 304, 4987, 5270, 323, 927, 220, 11286, 11, 931, 9507, 29189, 323, 220, 1041, 11, 931, 166, 247, 232, 9468, 99, 247, 453, 46051, 300, 11, 58842, 505, 84360, 12170, 25973, 3389, 304, 279, 220, 508, 339, 9478, 11, 304, 279, 3723, 4273, 323, 7008, 8032, 20, 60, 763, 362, 1631, 5169, 59492, 11, 9507, 29189, 527, 3062, 23837, 13, 578, 88150, 445, 81101, 374, 1071, 311, 7172, 3090, 505, 279, 18435, 323, 4433, 258, 988, 439, 433, 62555, 8032, 21, 60, 10771, 311, 362, 1631, 5169, 1560, 9884, 2508, 11, 9507, 29189, 690, 471, 311, 279, 3090, 42242, 323, 326, 6438, 2439, 1405, 814, 2586, 505, 520, 279, 842, 315, 892, 8032, 21, 60],
      decoded: 'The llama (/\u02c8l\u0251\u02d0m\u0259/; \ud83e\udd99Spanish pronunciation: [\u02c8\u028eama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the Pre-Columbian era. Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5\u20138 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4] The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000\u201312,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000\ua64a\ud83e\udd99 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5] In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology, llamas will return to the water springs and lagoons where they come from at the end of time.[6]',
    },
  },

  // - Sequence PostProcessor
  // - "ignore_merges": true
  "Xenova/llama3-tokenizer-new": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      tokens: ["How", "\u0120are", "\u0120you", "\u0120doing", "?"],
      ids: [128000, 4438, 527, 499, 3815, 30],
      decoded: "<|begin_of_text|>How are you doing?",
    },
    SIMPLE_WITH_PUNCTUATION: {
      text: BASE_TEST_STRINGS.SIMPLE_WITH_PUNCTUATION,
      tokens: ["You", "\u0120should", "'ve", "\u0120done", "\u0120this"],
      ids: [128000, 2675, 1288, 3077, 2884, 420],
      decoded: "<|begin_of_text|>You should've done this",
    },
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["012", "345", "678", "9", "\u0120", "0", "\u0120", "1", "\u0120", "2", "\u0120", "3", "\u0120", "4", "\u0120", "5", "\u0120", "6", "\u0120", "7", "\u0120", "8", "\u0120", "9", "\u0120", "10", "\u0120", "100", "\u0120", "100", "0"],
      ids: [128000, 11531, 12901, 17458, 24, 220, 15, 220, 16, 220, 17, 220, 18, 220, 19, 220, 20, 220, 21, 220, 22, 220, 23, 220, 24, 220, 605, 220, 1041, 220, 1041, 15],
      decoded: "<|begin_of_text|>0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["The", "\u0120company", "\u0120was", "\u0120founded", "\u0120in", "\u0120", "201", "6", "."],
      ids: [128000, 791, 2883, 574, 18538, 304, 220, 679, 21, 13],
      decoded: "<|begin_of_text|>The company was founded in 2016.",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["A", "\u010a", "'ll", "\u0120!!", "to", "?'", "d", "''", "d", "\u0120of", ",", "\u0120can", "'t", "."],
      ids: [128000, 32, 198, 3358, 11261, 998, 20837, 67, 4708, 67, 315, 11, 649, 956, 13],
      decoded: "<|begin_of_text|>A\n'll!!to?'d''d of, can't.",
    },
    PYTHON_CODE: {
      text: BASE_TEST_STRINGS.PYTHON_CODE,
      tokens: ["def", "\u0120main", "():\u010a", "\u0109pass"],
      ids: [128000, 755, 1925, 4019, 42531],
      decoded: "<|begin_of_text|>def main():\n\tpass",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["let", "\u0120a", "\u0120=", "\u0120obj", ".toString", "();\u010a", "toString", "();"],
      ids: [128000, 1169, 264, 284, 2909, 5180, 545, 6712, 2178],
      decoded: "<|begin_of_text|>let a = obj.toString();\ntoString();",
    },
    NEWLINES: {
      text: LLAMA_TEST_STRINGS.NEWLINES,
      tokens: ["ax", "\u010a", "####\u010a", "boo"],
      ids: [128000, 710, 198, 71050, 34093],
      decoded: "<|begin_of_text|>ax\n####\nboo",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["UN", "want", "\u00c3\u00a9d", ",", "running"],
      ids: [128000, 1899, 53757, 15433, 11, 28272],
      decoded: "<|begin_of_text|>UNwant\u00e9d,running",
    },
    CONTROL_TOKENS: {
      text: BASE_TEST_STRINGS.CONTROL_TOKENS,
      tokens: ["1", "\u0100", "2", "\u00ef\u00bf\u00bd", "3"],
      ids: [128000, 16, 188, 17, 5809, 18],
      decoded: "<|begin_of_text|>1\u00002\ufffd3",
    },
    HELLO_WORLD_TITLECASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_TITLECASE,
      tokens: ["Hello", "\u0120World"],
      ids: [128000, 9906, 4435],
      decoded: "<|begin_of_text|>Hello World",
    },
    HELLO_WORLD_LOWERCASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_LOWERCASE,
      tokens: ["hello", "\u0120world"],
      ids: [128000, 15339, 1917],
      decoded: "<|begin_of_text|>hello world",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u00e7\u0136\u0141\u00e6\u00b4\u00bb", "\u00e7\u013c\u0126", "\u00e7\u013e\u0141", "\u00e8\u00b0", "\u013d", "\u00e6\u013a\u00af"],
      ids: [128000, 104654, 9554, 89151, 39013, 249, 21043],
      decoded: "<|begin_of_text|>\u751f\u6d3b\u7684\u771f\u8c1b\u662f",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      tokens: ["\u0120\u0120", "\u0120leading", "\u0120space"],
      ids: [128000, 256, 6522, 3634],
      decoded: "<|begin_of_text|>   leading space",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["tr", "ailing", "\u0120space", "\u0120\u0120\u0120"],
      ids: [128000, 376, 14612, 3634, 262],
      decoded: "<|begin_of_text|>trailing space   ",
    },
    DOUBLE_SPACE: {
      text: BASE_TEST_STRINGS.DOUBLE_SPACE,
      tokens: ["Hi", "\u0120", "\u0120Hello"],
      ids: [128000, 13347, 220, 22691],
      decoded: "<|begin_of_text|>Hi  Hello",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["test", "\u0120$", "1", "\u0120R", "2", "\u0120#", "3", "\u0120\u00e2\u0124\u00ac", "4", "\u0120\u00c2\u00a3", "5", "\u0120\u00c2\u00a5", "6", "\u0120\u00e2\u0124", "\u00a3", "7", "\u0120\u00e2\u0124\u00b9", "8", "\u0120\u00e2\u0124", "\u00b1", "9", "\u0120test"],
      ids: [128000, 1985, 400, 16, 432, 17, 674, 18, 13281, 19, 7083, 20, 72588, 21, 113384, 96, 22, 90891, 23, 113384, 109, 24, 1296],
      decoded: "<|begin_of_text|>test $1 R2 #3 \u20ac4 \u00a35 \u00a56 \u20a37 \u20b98 \u20b19 test",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["I", "\u0120bought", "\u0120an", "\u0120apple", "\u0120for", "\u0120$", "1", ".", "00", "\u0120at", "\u0120the", "\u0120store", "."],
      ids: [128000, 40, 11021, 459, 24149, 369, 400, 16, 13, 410, 520, 279, 3637, 13],
      decoded: "<|begin_of_text|>I bought an apple for $1.00 at the store.",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u0120\u0120"],
      ids: [128000, 9514, 1981, 256],
      decoded: "<|begin_of_text|>you\u2026  ",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u00c2\u0142\u00c2\u0142"],
      ids: [128000, 9514, 1981, 9421],
      decoded: "<|begin_of_text|>you\u2026\u00a0\u00a0",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u00c2\u0142", "\u00c2\u0142", "you", "\u00e2\u0122\u00a6", "\u00c2\u0142\u00c2\u0142"],
      ids: [128000, 9514, 1981, 4194, 4194, 9514, 1981, 9421],
      decoded: "<|begin_of_text|>you\u2026\u00a0\u00a0you\u2026\u00a0\u00a0",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["we", "ird", "\u0120\u00ef\u00bd\u0140", "\u0120edge", "\u0120\u00ef\u00bd\u0140", "\u0120case"],
      ids: [128000, 906, 2668, 111942, 6964, 111942, 1162],
      decoded: "<|begin_of_text|>weird \uff5e edge \uff5e case",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      tokens: ["\u00e2\u0138", "\u0123", "This", "\u0120\u00e2\u0138", "\u0123", "is", "\u0120\u00e2\u0138", "\u0123", "a", "\u0120\u00e2\u0138", "\u0123", "test", "\u0120\u00e2\u0138", "\u0123", "."],
      ids: [128000, 10634, 223, 2028, 14860, 223, 285, 14860, 223, 64, 14860, 223, 1985, 14860, 223, 13],
      decoded: "<|begin_of_text|>\u2581This \u2581is \u2581a \u2581test \u2581.",
    },
    POPULAR_EMOJIS: {
      text: BASE_TEST_STRINGS.POPULAR_EMOJIS,
      tokens: ["\u00f0\u0141\u013a", "\u0124", "\u0120\u00f0\u0141\u0133", "\u012f", "\u0120\u00f0\u0141", "\u00a4", "\u00a3", "\u0120\u00f0\u0141\u013a", "\u012f", "\u0120\u00f0\u0141\u013a", "\u0143", "\u0120\u00f0\u0141", "\u0130", "\u012b", "\u0120\u00f0\u0141", "\u013b", "\u0131", "\u0120\u00f0\u0141\u013a", "\u012c", "\u0120\u00f0\u0141\u0136", "\u00a5", "\u0120\u00f0\u0141\u013a", "\u0123", "\u0120\u00f0\u0141\u013a", "\u0127", "\u0120\u00f0\u0141", "\u00a4", "\u0139", "\u0120\u00f0\u0141\u013a", "\u0128", "\u0120\u00f0\u0141\u0133", "\u0131", "\u0120\u00e2\u013f\u00a4", "\u00ef\u00b8\u0131", "\u0120\u00f0\u0141\u0134", "\u013e", "\u0120\u00f0\u0141\u0134", "\u013c", "\u0120\u00f0\u0141\u0134", "\u0139", "\u0120\u00f0\u0141\u0134", "\u013b", "\u0120\u00f0\u0141", "\u0138", "\u00a4", "\u0120\u00f0\u0141\u013a", "\u0130", "\u0120\u00f0\u0141\u0133", "\u012e", "\u0120\u00f0\u0141", "\u00a5", "\u00b3", "\u0120\u00f0\u0141\u0134", "\u00aa", "\u0120\u00e2\u013e", "\u00a8", "\u0120\u00f0\u0141\u0133", "\u012b", "\u0120\u00f0\u0141\u0133", "\u0122", "\u0120\u00f0\u0141\u0134", "\u00af", "\u0120\u00f0\u0141", "\u0130", "\u012a", "\u0120\u00f0\u0141", "\u013b", "\u012a", "\u0120\u00f0\u0141", "\u013b", "\u012e", "\u0120\u00f0\u0141\u0134", "\u0122", "\u0120\u00f0\u0141\u0133", "\u0129", "\u0120\u00f0\u0141\u0133", "\u012d", "\u0120\u00e2\u013e", "\u0127", "\u0120\u00f0\u0141", "\u0130", "\u0123", "\u0120\u00f0\u0141", "\u012e", "\u0140", "\u0120\u00f0\u0141", "\u012e", "\u00b8", "\u0120\u00f0\u0141\u0134", "\u00b0"],
      ids: [128000, 76460, 224, 62904, 235, 11410, 97, 96, 27623, 235, 27623, 255, 11410, 236, 231, 11410, 247, 237, 27623, 232, 96169, 98, 27623, 223, 27623, 227, 11410, 97, 245, 27623, 228, 62904, 237, 71570, 31643, 64139, 250, 64139, 248, 64139, 245, 64139, 247, 11410, 244, 97, 27623, 236, 62904, 234, 11410, 98, 111, 64139, 103, 26602, 101, 62904, 231, 62904, 222, 64139, 107, 11410, 236, 230, 11410, 247, 230, 11410, 247, 234, 64139, 222, 62904, 229, 62904, 233, 26602, 227, 11410, 236, 223, 11410, 234, 252, 11410, 234, 116, 64139, 108],
      decoded: "<|begin_of_text|>\ud83d\ude02 \ud83d\udc4d \ud83e\udd23 \ud83d\ude0d \ud83d\ude2d \ud83c\udf89 \ud83d\ude4f \ud83d\ude0a \ud83d\udd25 \ud83d\ude01 \ud83d\ude05 \ud83e\udd17 \ud83d\ude06 \ud83d\udc4f \u2764\ufe0f \ud83d\udc9c \ud83d\udc9a \ud83d\udc97 \ud83d\udc99 \ud83d\udda4 \ud83d\ude0e \ud83d\udc4c \ud83e\udd73 \ud83d\udcaa \u2728 \ud83d\udc49 \ud83d\udc40 \ud83d\udcaf \ud83c\udf88 \ud83d\ude48 \ud83d\ude4c \ud83d\udc80 \ud83d\udc47 \ud83d\udc4b \u2705 \ud83c\udf81 \ud83c\udf1e \ud83c\udf38 \ud83d\udcb0",
    },
    MULTIBYTE_EMOJIS: {
      text: BASE_TEST_STRINGS.MULTIBYTE_EMOJIS,
      tokens: ["\u00e2\u013e", "\u00a8", "\u0120\u00f0\u0141", "\u00a4", "\u0139", "\u0120\u00f0\u0141\u0133", "\u0123", "\u00ef\u00b8\u0131", "\u0120\u00f0\u0141\u0133", "\u00b1", "\u00f0\u0141", "\u0131", "\u00bb", "\u0120\u00f0\u0141", "\u0137", "\u00b5", "\u00e2\u0122\u012f", "\u00e2\u013b", "\u0124", "\u00ef\u00b8\u0131", "\u0120\u00f0\u0141", "\u00a7", "\u013b", "\u00f0\u0141", "\u0131", "\u00bb", "\u00e2\u0122\u012f", "\u00e2\u013b", "\u0124", "\u0120\u00f0\u0141\u0133", "\u00a8", "\u00f0\u0141", "\u0131", "\u00bb", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u012e", "\u00be", "\u0120\u00f0\u0141", "\u00a7", "\u0133", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u00a4", "\u013f", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u00a7", "\u0133", "\u0120\u00f0\u0141\u0133", "\u00a9", "\u00e2\u0122\u012f", "\u00e2\u013f\u00a4", "\u00e2\u0122\u012f", "\u00f0\u0141\u0134", "\u012d", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u0133", "\u00a8", "\u0120\u00f0\u0141\u0133", "\u00a9", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u0133", "\u00a9", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u0133", "\u00a7", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u0133", "\u00a6", "\u0120\u00f0\u0141", "\u00a7", "\u0133", "\u00f0\u0141", "\u0131", "\u00bb", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u00a4", "\u013f", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u00a7", "\u0133", "\u00f0\u0141", "\u0131", "\u00bb", "\u0120\u00f0\u0141", "\u0131", "\u00b4", "\u00f3", "\u0142\u0123", "\u00a7", "\u00f3", "\u0142\u0123", "\u00a2", "\u00f3", "\u0142\u0123", "\u00a5", "\u00f3", "\u0142\u0123", "\u00ae", "\u00f3", "\u0142\u0123", "\u00a7", "\u00f3", "\u0142\u0123", "\u00bf", "\u0120\u00f0\u0141\u0133", "\u00a8", "\u00f0\u0141", "\u0131", "\u00bb", "\u00e2\u0122\u012f", "\u00e2\u013f\u00a4", "\u00ef\u00b8\u0131", "\u00e2\u0122\u012f", "\u00f0\u0141\u0134", "\u012d", "\u00e2\u0122\u012f", "\u00f0\u0141", "\u0133", "\u00a8", "\u00f0\u0141", "\u0131", "\u00bc"],
      ids: [128000, 38798, 101, 11410, 97, 245, 62904, 223, 31643, 62904, 109, 9468, 237, 119, 11410, 243, 113, 102470, 17245, 224, 31643, 11410, 100, 247, 9468, 237, 119, 102470, 17245, 224, 62904, 101, 9468, 237, 119, 102470, 9468, 234, 122, 11410, 100, 239, 102470, 9468, 97, 251, 102470, 9468, 100, 239, 62904, 102, 102470, 121643, 102470, 93273, 233, 102470, 9468, 239, 101, 62904, 102, 102470, 9468, 239, 102, 102470, 9468, 239, 100, 102470, 9468, 239, 99, 11410, 100, 239, 9468, 237, 119, 102470, 9468, 97, 251, 102470, 9468, 100, 239, 9468, 237, 119, 11410, 237, 112, 175, 16050, 100, 175, 16050, 95, 175, 16050, 98, 175, 16050, 106, 175, 16050, 100, 175, 16050, 123, 62904, 101, 9468, 237, 119, 102470, 121643, 31643, 102470, 93273, 233, 102470, 9468, 239, 101, 9468, 237, 120],
      decoded: "<|begin_of_text|>\u2728 \ud83e\udd17 \ud83d\udc41\ufe0f \ud83d\udc71\ud83c\udffb \ud83d\udd75\u200d\u2642\ufe0f \ud83e\uddd9\ud83c\udffb\u200d\u2642 \ud83d\udc68\ud83c\udffb\u200d\ud83c\udf3e \ud83e\uddd1\u200d\ud83e\udd1d\u200d\ud83e\uddd1 \ud83d\udc69\u200d\u2764\u200d\ud83d\udc8b\u200d\ud83d\udc68 \ud83d\udc69\u200d\ud83d\udc69\u200d\ud83d\udc67\u200d\ud83d\udc66 \ud83e\uddd1\ud83c\udffb\u200d\ud83e\udd1d\u200d\ud83e\uddd1\ud83c\udffb \ud83c\udff4\udb40\udc67\udb40\udc62\udb40\udc65\udb40\udc6e\udb40\udc67\udb40\udc7f \ud83d\udc68\ud83c\udffb\u200d\u2764\ufe0f\u200d\ud83d\udc8b\u200d\ud83d\udc68\ud83c\udffc",
    },
    BPE_SCORES_PRIORITY_1: {
      text: LLAMA_TEST_STRINGS.BPE_SCORES_PRIORITY_1,
      tokens: ["grab", "bed"],
      ids: [128000, 59312, 2788],
      decoded: "<|begin_of_text|>grabbed",
    },
    BPE_SCORES_PRIORITY_2: {
      text: LLAMA_TEST_STRINGS.BPE_SCORES_PRIORITY_2,
      tokens: ["\u0120grabbed"],
      ids: [128000, 30418],
      decoded: "<|begin_of_text|> grabbed",
    },
    BPE_SCORES_PRIORITY_3: {
      text: LLAMA_TEST_STRINGS.BPE_SCORES_PRIORITY_3,
      tokens: ["\u0120\u0120\u0120\u0120\u0120\u0120\u0120\u0120\u0120\u0120", "\u0120grabbed"],
      ids: [128000, 1881, 30418],
      decoded: "<|begin_of_text|>           grabbed",
    },
    NEWLINE: {
      text: LLAMA_TEST_STRINGS.NEWLINE,
      tokens: ["\u010a"],
      ids: [128000, 198],
      decoded: "<|begin_of_text|>\n",
    },
    NEWLINE_WITH_LEADING_SPACE: {
      text: LLAMA_TEST_STRINGS.NEWLINE_WITH_LEADING_SPACE,
      tokens: ["\u0120\u010a"],
      ids: [128000, 720],
      decoded: "<|begin_of_text|> \n",
    },
    TABS: {
      text: LLAMA_TEST_STRINGS.TABS,
      tokens: ["\u0109t", "abs", "\u0109\u0109\u0109", "\u0109out", "\u0120here"],
      ids: [128000, 3324, 3518, 573, 14294, 1618],
      decoded: "<|begin_of_text|>\ttabs\t\t\t\tout here",
    },
    NEWLINE_AND_TAB: {
      text: LLAMA_TEST_STRINGS.NEWLINE_AND_TAB,
      tokens: ["\u010a\u0109\u010a"],
      ids: [128000, 18108],
      decoded: "<|begin_of_text|>\n\t\n",
    },
    CHINESE_LETTER: {
      text: LLAMA_TEST_STRINGS.CHINESE_LETTER,
      tokens: ["\u00e9\u0137\u0129"],
      ids: [128000, 104643],
      decoded: "<|begin_of_text|>\u9547",
    },
    EMOJIS_1: {
      text: LLAMA_TEST_STRINGS.EMOJIS_1,
      tokens: ["\u00f0\u0141", "\u00a6", "\u013b"],
      ids: [128000, 9468, 99, 247],
      decoded: "<|begin_of_text|>\ud83e\udd99",
    },
    EMOJIS_2: {
      text: LLAMA_TEST_STRINGS.EMOJIS_2,
      tokens: ["\u00f0\u0141", "\u00a6", "\u013b", "\u00ea", "\u013b", "\u012c"],
      ids: [128000, 9468, 99, 247, 166, 247, 232],
      decoded: "<|begin_of_text|>\ud83e\udd99\ua64a",
    },
    EMOJIS_3: {
      text: LLAMA_TEST_STRINGS.EMOJIS_3,
      tokens: ["\u00ea", "\u013b", "\u012c", "\u00f0\u0141", "\u00a6", "\u013b"],
      ids: [128000, 166, 247, 232, 9468, 99, 247],
      decoded: "<|begin_of_text|>\ua64a\ud83e\udd99",
    },
    PARAGRAPH: {
      text: LLAMA_TEST_STRINGS.PARAGRAPH,
      tokens: ["The", "\u0120llama", "\u0120(/", "\u00cb", "\u012a", "l", "\u00c9", "\u0133", "\u00cb", "\u0132", "m", "\u00c9\u013b", "/", ";", "\u0120\u00f0\u0141", "\u00a6", "\u013b", "Spanish", "\u0120pronunciation", ":", "\u0120[", "\u00cb", "\u012a", "\u00ca", "\u0130", "ama", "])", "\u0120(", "L", "ama", "\u0120gl", "ama", ")", "\u0120is", "\u0120a", "\u0120domestic", "ated", "\u0120South", "\u0120American", "\u0120camel", "id", ",", "\u0120widely", "\u0120used", "\u0120as", "\u0120a", "\u0120meat", "\u0120and", "\u0120pack", "\u0120animal", "\u0120by", "\u0120And", "ean", "\u0120cultures", "\u0120since", "\u0120the", "\u0120Pre", "-C", "olum", "bian", "\u0120era", ".", "\u0120L", "lam", "as", "\u0120are", "\u0120social", "\u0120animals", "\u0120and", "\u0120live", "\u0120with", "\u0120others", "\u0120as", "\u0120a", "\u0120herd", ".", "\u0120Their", "\u0120wool", "\u0120is", "\u0120soft", "\u0120and", "\u0120contains", "\u0120only", "\u0120a", "\u0120small", "\u0120amount", "\u0120of", "\u0120lan", "olin", ".[", "2", "]", "\u0120L", "lam", "as", "\u0120can", "\u0120learn", "\u0120simple", "\u0120tasks", "\u0120after", "\u0120a", "\u0120few", "\u0120repetitions", ".", "\u0120When", "\u0120using", "\u0120a", "\u0120pack", ",", "\u0120they", "\u0120can", "\u0120carry", "\u0120about", "\u0120", "25", "\u0120to", "\u0120", "30", "%", "\u0120of", "\u0120their", "\u0120body", "\u0120weight", "\u0120for", "\u0120", "8", "\u0120to", "\u0120", "13", "\u0120km", "\u0120(", "5", "\u00e2\u0122\u0135", "8", "\u0120miles", ").[", "3", "]", "\u0120The", "\u0120name", "\u0120llama", "\u0120(", "in", "\u0120the", "\u0120past", "\u0120also", "\u0120spelled", '\u0120"', "lama", '"', "\u0120or", '\u0120"', "gl", "ama", '")', "\u0120was", "\u0120adopted", "\u0120by", "\u0120European", "\u0120settlers", "\u0120from", "\u0120native", "\u0120Per", "uv", "ians", ".[", "4", "]", "\u0120The", "\u0120ancestors", "\u0120of", "\u0120ll", "amas", "\u0120are", "\u0120thought", "\u0120to", "\u0120have", "\u0120originated", "\u0120from", "\u0120the", "\u0120Great", "\u0120Plains", "\u0120of", "\u0120North", "\u0120America", "\u0120about", "\u0120", "40", "\u0120million", "\u0120years", "\u0120ago", ",", "\u0120and", "\u0120subsequently", "\u0120migrated", "\u0120to", "\u0120South", "\u0120America", "\u0120about", "\u0120three", "\u0120million", "\u0120years", "\u0120ago", "\u0120during", "\u0120the", "\u0120Great", "\u0120American", "\u0120Inter", "change", ".", "\u0120By", "\u0120the", "\u0120end", "\u0120of", "\u0120the", "\u0120last", "\u0120ice", "\u0120age", "\u0120(", "10", ",", "000", "\u00e2\u0122\u0135", "12", ",", "000", "\u0120years", "\u0120ago", "),", "\u0120camel", "ids", "\u0120were", "\u0120extinct", "\u0120in", "\u0120North", "\u0120America", ".[", "3", "]", "\u0120As", "\u0120of", "\u0120", "200", "7", ",", "\u0120there", "\u0120were", "\u0120over", "\u0120seven", "\u0120million", "\u0120ll", "amas", "\u0120and", "\u0120al", "pac", "as", "\u0120in", "\u0120South", "\u0120America", "\u0120and", "\u0120over", "\u0120", "158", ",", "000", "\u0120ll", "amas", "\u0120and", "\u0120", "100", ",", "000", "\u00ea", "\u013b", "\u012c", "\u00f0\u0141", "\u00a6", "\u013b", "\u0120al", "pac", "as", ",", "\u0120descended", "\u0120from", "\u0120progen", "itors", "\u0120imported", "\u0120late", "\u0120in", "\u0120the", "\u0120", "20", "th", "\u0120century", ",", "\u0120in", "\u0120the", "\u0120United", "\u0120States", "\u0120and", "\u0120Canada", ".[", "5", "]", "\u0120In", "\u0120A", "ym", "ara", "\u0120mythology", ",", "\u0120ll", "amas", "\u0120are", "\u0120important", "\u0120beings", ".", "\u0120The", "\u0120Heavenly", "\u0120L", "lama", "\u0120is", "\u0120said", "\u0120to", "\u0120drink", "\u0120water", "\u0120from", "\u0120the", "\u0120ocean", "\u0120and", "\u0120ur", "in", "ates", "\u0120as", "\u0120it", "\u0120rains", ".[", "6", "]", "\u0120According", "\u0120to", "\u0120A", "ym", "ara", "\u0120es", "chat", "ology", ",", "\u0120ll", "amas", "\u0120will", "\u0120return", "\u0120to", "\u0120the", "\u0120water", "\u0120springs", "\u0120and", "\u0120l", "ago", "ons", "\u0120where", "\u0120they", "\u0120come", "\u0120from", "\u0120at", "\u0120the", "\u0120end", "\u0120of", "\u0120time", ".[", "6", "]"],
      ids: [128000, 791, 94776, 47325, 135, 230, 75, 133, 239, 135, 238, 76, 99638, 14, 26, 11410, 99, 247, 62897, 71722, 25, 510, 135, 230, 134, 236, 3105, 2526, 320, 43, 3105, 2840, 3105, 8, 374, 264, 13018, 660, 4987, 3778, 50252, 307, 11, 13882, 1511, 439, 264, 13339, 323, 3854, 10065, 555, 1628, 5420, 27833, 2533, 279, 5075, 7813, 1152, 13464, 11639, 13, 445, 24705, 300, 527, 3674, 10099, 323, 3974, 449, 3885, 439, 264, 59213, 13, 11205, 39640, 374, 8579, 323, 5727, 1193, 264, 2678, 3392, 315, 31791, 37737, 8032, 17, 60, 445, 24705, 300, 649, 4048, 4382, 9256, 1306, 264, 2478, 86066, 13, 3277, 1701, 264, 3854, 11, 814, 649, 6920, 922, 220, 914, 311, 220, 966, 4, 315, 872, 2547, 4785, 369, 220, 23, 311, 220, 1032, 13437, 320, 20, 4235, 23, 8931, 94638, 18, 60, 578, 836, 94776, 320, 258, 279, 3347, 1101, 68918, 330, 81101, 1, 477, 330, 6200, 3105, 909, 574, 18306, 555, 7665, 61107, 505, 10068, 3700, 12328, 5493, 8032, 19, 60, 578, 38618, 315, 9507, 29189, 527, 3463, 311, 617, 44853, 505, 279, 8681, 63911, 315, 4892, 5270, 922, 220, 1272, 3610, 1667, 4227, 11, 323, 28520, 73691, 311, 4987, 5270, 922, 2380, 3610, 1667, 4227, 2391, 279, 8681, 3778, 5783, 3455, 13, 3296, 279, 842, 315, 279, 1566, 10054, 4325, 320, 605, 11, 931, 4235, 717, 11, 931, 1667, 4227, 705, 50252, 3447, 1051, 69918, 304, 4892, 5270, 8032, 18, 60, 1666, 315, 220, 1049, 22, 11, 1070, 1051, 927, 8254, 3610, 9507, 29189, 323, 453, 46051, 300, 304, 4987, 5270, 323, 927, 220, 11286, 11, 931, 9507, 29189, 323, 220, 1041, 11, 931, 166, 247, 232, 9468, 99, 247, 453, 46051, 300, 11, 58842, 505, 84360, 12170, 25973, 3389, 304, 279, 220, 508, 339, 9478, 11, 304, 279, 3723, 4273, 323, 7008, 8032, 20, 60, 763, 362, 1631, 5169, 59492, 11, 9507, 29189, 527, 3062, 23837, 13, 578, 88150, 445, 81101, 374, 1071, 311, 7172, 3090, 505, 279, 18435, 323, 4433, 258, 988, 439, 433, 62555, 8032, 21, 60, 10771, 311, 362, 1631, 5169, 1560, 9884, 2508, 11, 9507, 29189, 690, 471, 311, 279, 3090, 42242, 323, 326, 6438, 2439, 1405, 814, 2586, 505, 520, 279, 842, 315, 892, 8032, 21, 60],
      decoded: '<|begin_of_text|>The llama (/\u02c8l\u0251\u02d0m\u0259/; \ud83e\udd99Spanish pronunciation: [\u02c8\u028eama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the Pre-Columbian era. Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5\u20138 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4] The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000\u201312,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000\ua64a\ud83e\udd99 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5] In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology, llamas will return to the water springs and lagoons where they come from at the end of time.[6]',
    },
  },
  "Xenova/TinyLLama-v0": {
    NEWLINES: {
      text: LLAMA_TEST_STRINGS.NEWLINES,
      tokens: ["\u2581ax", "<0x0A>", "####", "<0x0A>", "b", "oo"],
      ids: [1, 9013, 13, 20411, 13, 31842, 2742],
      decoded: "<s> ax\n####\nboo",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u2581", "<0xE7>", "<0x94>", "<0x9F>", "<0xE6>", "<0xB4>", "<0xBB>", "<0xE7>", "<0x9A>", "<0x84>", "<0xE7>", "<0x9C>", "<0x9F>", "<0xE8>", "<0xB0>", "<0x9B>", "<0xE6>", "<0x98>", "<0xAF>"],
      ids: [1, 31822, 234, 151, 162, 233, 183, 190, 234, 157, 135, 234, 159, 162, 235, 179, 158, 233, 155, 178],
      decoded: "<s> \u751f\u6d3b\u7684\u771f\u8c1b\u662f",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["\u2581trailing", "\u2581space", "\u2581", "\u2581", "\u2581"],
      ids: [1, 30174, 2138, 31822, 31822, 31822],
      decoded: "<s> trailing space   ",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["\u2581test", "\u2581$", "1", "\u2581R", "2", "\u2581#", "3", "\u2581\u20ac", "4", "\u2581\u00a3", "5", "\u2581", "<0xC2>", "<0xA5>", "6", "\u2581", "<0xE2>", "<0x82>", "<0xA3>", "7", "\u2581", "<0xE2>", "<0x82>", "<0xB9>", "8", "\u2581", "<0xE2>", "<0x82>", "<0xB1>", "9", "\u2581test"],
      ids: [1, 1397, 569, 31853, 360, 31855, 1257, 31878, 9390, 31882, 3922, 31880, 31822, 197, 168, 31887, 31822, 229, 133, 166, 31888, 31822, 229, 133, 188, 31886, 31822, 229, 133, 180, 31877, 1397],
      decoded: "<s> test $1 R2 #3 \u20ac4 \u00a35 \u00a56 \u20a37 \u20b98 \u20b19 test",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["\u2581you", "\u2026", "\u2581", "\u2581"],
      ids: [1, 365, 31925, 31822, 31822],
      decoded: "<s> you\u2026  ",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["\u2581you", "\u2026", "\u00a0", "\u00a0"],
      ids: [1, 365, 31925, 31963, 31963],
      decoded: "<s> you\u2026\u00a0\u00a0",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["\u2581you", "\u2026", "\u00a0", "\u00a0", "you", "\u2026", "\u00a0", "\u00a0"],
      ids: [1, 365, 31925, 31963, 31963, 7936, 31925, 31963, 31963],
      decoded: "<s> you\u2026\u00a0\u00a0you\u2026\u00a0\u00a0",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["\u2581weird", "\u2581", "<0xEF>", "<0xBD>", "<0x9E>", "\u2581edge", "\u2581", "<0xEF>", "<0xBD>", "<0x9E>", "\u2581case"],
      ids: [1, 9907, 31822, 242, 192, 161, 5991, 31822, 242, 192, 161, 1372],
      decoded: "<s> weird \uff5e edge \uff5e case",
    },
    POPULAR_EMOJIS: {
      text: BASE_TEST_STRINGS.POPULAR_EMOJIS,
      tokens: ["\u2581", "<0xF0>", "<0x9F>", "<0x98>", "<0x82>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0x8D>", "\u2581", "<0xF0>", "<0x9F>", "<0xA4>", "<0xA3>", "\u2581", "<0xF0>", "<0x9F>", "<0x98>", "<0x8D>", "\u2581", "<0xF0>", "<0x9F>", "<0x98>", "<0xAD>", "\u2581", "<0xF0>", "<0x9F>", "<0x8E>", "<0x89>", "\u2581", "<0xF0>", "<0x9F>", "<0x99>", "<0x8F>", "\u2581", "<0xF0>", "<0x9F>", "<0x98>", "<0x8A>", "\u2581", "<0xF0>", "<0x9F>", "<0x94>", "<0xA5>", "\u2581", "<0xF0>", "<0x9F>", "<0x98>", "<0x81>", "\u2581", "<0xF0>", "<0x9F>", "<0x98>", "<0x85>", "\u2581", "<0xF0>", "<0x9F>", "<0xA4>", "<0x97>", "\u2581", "<0xF0>", "<0x9F>", "<0x98>", "<0x86>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0x8F>", "\u2581", "<0xE2>", "<0x9D>", "<0xA4>", "<0xEF>", "<0xB8>", "<0x8F>", "\u2581", "<0xF0>", "<0x9F>", "<0x92>", "<0x9C>", "\u2581", "<0xF0>", "<0x9F>", "<0x92>", "<0x9A>", "\u2581", "<0xF0>", "<0x9F>", "<0x92>", "<0x97>", "\u2581", "<0xF0>", "<0x9F>", "<0x92>", "<0x99>", "\u2581", "<0xF0>", "<0x9F>", "<0x96>", "<0xA4>", "\u2581", "<0xF0>", "<0x9F>", "<0x98>", "<0x8E>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0x8C>", "\u2581", "<0xF0>", "<0x9F>", "<0xA5>", "<0xB3>", "\u2581", "<0xF0>", "<0x9F>", "<0x92>", "<0xAA>", "\u2581", "<0xE2>", "<0x9C>", "<0xA8>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0x89>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0x80>", "\u2581", "<0xF0>", "<0x9F>", "<0x92>", "<0xAF>", "\u2581", "<0xF0>", "<0x9F>", "<0x8E>", "<0x88>", "\u2581", "<0xF0>", "<0x9F>", "<0x99>", "<0x88>", "\u2581", "<0xF0>", "<0x9F>", "<0x99>", "<0x8C>", "\u2581", "<0xF0>", "<0x9F>", "<0x92>", "<0x80>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0x87>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0x8B>", "\u2581", "<0xE2>", "<0x9C>", "<0x85>", "\u2581", "<0xF0>", "<0x9F>", "<0x8E>", "<0x81>", "\u2581", "<0xF0>", "<0x9F>", "<0x8C>", "<0x9E>", "\u2581", "<0xF0>", "<0x9F>", "<0x8C>", "<0xB8>", "\u2581", "<0xF0>", "<0x9F>", "<0x92>", "<0xB0>"],
      ids: [1, 31822, 243, 162, 155, 133, 31822, 243, 162, 148, 144, 31822, 243, 162, 167, 166, 31822, 243, 162, 155, 144, 31822, 243, 162, 155, 176, 31822, 243, 162, 145, 140, 31822, 243, 162, 156, 146, 31822, 243, 162, 155, 141, 31822, 243, 162, 151, 168, 31822, 243, 162, 155, 132, 31822, 243, 162, 155, 136, 31822, 243, 162, 167, 154, 31822, 243, 162, 155, 137, 31822, 243, 162, 148, 146, 31822, 229, 160, 167, 242, 187, 146, 31822, 243, 162, 149, 159, 31822, 243, 162, 149, 157, 31822, 243, 162, 149, 154, 31822, 243, 162, 149, 156, 31822, 243, 162, 153, 167, 31822, 243, 162, 155, 145, 31822, 243, 162, 148, 143, 31822, 243, 162, 168, 182, 31822, 243, 162, 149, 173, 31822, 229, 159, 171, 31822, 243, 162, 148, 140, 31822, 243, 162, 148, 131, 31822, 243, 162, 149, 178, 31822, 243, 162, 145, 139, 31822, 243, 162, 156, 139, 31822, 243, 162, 156, 143, 31822, 243, 162, 149, 131, 31822, 243, 162, 148, 138, 31822, 243, 162, 148, 142, 31822, 229, 159, 136, 31822, 243, 162, 145, 132, 31822, 243, 162, 143, 161, 31822, 243, 162, 143, 187, 31822, 243, 162, 149, 179],
      decoded: "<s> \ud83d\ude02 \ud83d\udc4d \ud83e\udd23 \ud83d\ude0d \ud83d\ude2d \ud83c\udf89 \ud83d\ude4f \ud83d\ude0a \ud83d\udd25 \ud83d\ude01 \ud83d\ude05 \ud83e\udd17 \ud83d\ude06 \ud83d\udc4f \u2764\ufe0f \ud83d\udc9c \ud83d\udc9a \ud83d\udc97 \ud83d\udc99 \ud83d\udda4 \ud83d\ude0e \ud83d\udc4c \ud83e\udd73 \ud83d\udcaa \u2728 \ud83d\udc49 \ud83d\udc40 \ud83d\udcaf \ud83c\udf88 \ud83d\ude48 \ud83d\ude4c \ud83d\udc80 \ud83d\udc47 \ud83d\udc4b \u2705 \ud83c\udf81 \ud83c\udf1e \ud83c\udf38 \ud83d\udcb0",
    },
    MULTIBYTE_EMOJIS: {
      text: BASE_TEST_STRINGS.MULTIBYTE_EMOJIS,
      tokens: ["\u2581", "<0xE2>", "<0x9C>", "<0xA8>", "\u2581", "<0xF0>", "<0x9F>", "<0xA4>", "<0x97>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0x81>", "<0xEF>", "<0xB8>", "<0x8F>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0xB1>", "<0xF0>", "<0x9F>", "<0x8F>", "<0xBB>", "\u2581", "<0xF0>", "<0x9F>", "<0x95>", "<0xB5>", "<0xE2>", "<0x80>", "<0x8D>", "<0xE2>", "<0x99>", "<0x82>", "<0xEF>", "<0xB8>", "<0x8F>", "\u2581", "<0xF0>", "<0x9F>", "<0xA7>", "<0x99>", "<0xF0>", "<0x9F>", "<0x8F>", "<0xBB>", "<0xE2>", "<0x80>", "<0x8D>", "<0xE2>", "<0x99>", "<0x82>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0xA8>", "<0xF0>", "<0x9F>", "<0x8F>", "<0xBB>", "<0xE2>", "<0x80>", "<0x8D>", "<0xF0>", "<0x9F>", "<0x8C>", "<0xBE>", "\u2581", "<0xF0>", "<0x9F>", "<0xA7>", "<0x91>", "<0xE2>", "<0x80>", "<0x8D>", "<0xF0>", "<0x9F>", "<0xA4>", "<0x9D>", "<0xE2>", "<0x80>", "<0x8D>", "<0xF0>", "<0x9F>", "<0xA7>", "<0x91>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0xA9>", "<0xE2>", "<0x80>", "<0x8D>", "<0xE2>", "<0x9D>", "<0xA4>", "<0xE2>", "<0x80>", "<0x8D>", "<0xF0>", "<0x9F>", "<0x92>", "<0x8B>", "<0xE2>", "<0x80>", "<0x8D>", "<0xF0>", "<0x9F>", "<0x91>", "<0xA8>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0xA9>", "<0xE2>", "<0x80>", "<0x8D>", "<0xF0>", "<0x9F>", "<0x91>", "<0xA9>", "<0xE2>", "<0x80>", "<0x8D>", "<0xF0>", "<0x9F>", "<0x91>", "<0xA7>", "<0xE2>", "<0x80>", "<0x8D>", "<0xF0>", "<0x9F>", "<0x91>", "<0xA6>", "\u2581", "<0xF0>", "<0x9F>", "<0xA7>", "<0x91>", "<0xF0>", "<0x9F>", "<0x8F>", "<0xBB>", "<0xE2>", "<0x80>", "<0x8D>", "<0xF0>", "<0x9F>", "<0xA4>", "<0x9D>", "<0xE2>", "<0x80>", "<0x8D>", "<0xF0>", "<0x9F>", "<0xA7>", "<0x91>", "<0xF0>", "<0x9F>", "<0x8F>", "<0xBB>", "\u2581", "<0xF0>", "<0x9F>", "<0x8F>", "<0xB4>", "<0xF3>", "<0xA0>", "<0x81>", "<0xA7>", "<0xF3>", "<0xA0>", "<0x81>", "<0xA2>", "<0xF3>", "<0xA0>", "<0x81>", "<0xA5>", "<0xF3>", "<0xA0>", "<0x81>", "<0xAE>", "<0xF3>", "<0xA0>", "<0x81>", "<0xA7>", "<0xF3>", "<0xA0>", "<0x81>", "<0xBF>", "\u2581", "<0xF0>", "<0x9F>", "<0x91>", "<0xA8>", "<0xF0>", "<0x9F>", "<0x8F>", "<0xBB>", "<0xE2>", "<0x80>", "<0x8D>", "<0xE2>", "<0x9D>", "<0xA4>", "<0xEF>", "<0xB8>", "<0x8F>", "<0xE2>", "<0x80>", "<0x8D>", "<0xF0>", "<0x9F>", "<0x92>", "<0x8B>", "<0xE2>", "<0x80>", "<0x8D>", "<0xF0>", "<0x9F>", "<0x91>", "<0xA8>", "<0xF0>", "<0x9F>", "<0x8F>", "<0xBC>"],
      ids: [1, 31822, 229, 159, 171, 31822, 243, 162, 167, 154, 31822, 243, 162, 148, 132, 242, 187, 146, 31822, 243, 162, 148, 180, 243, 162, 146, 190, 31822, 243, 162, 152, 184, 229, 131, 144, 229, 156, 133, 242, 187, 146, 31822, 243, 162, 170, 156, 243, 162, 146, 190, 229, 131, 144, 229, 156, 133, 31822, 243, 162, 148, 171, 243, 162, 146, 190, 229, 131, 144, 243, 162, 143, 193, 31822, 243, 162, 170, 148, 229, 131, 144, 243, 162, 167, 160, 229, 131, 144, 243, 162, 170, 148, 31822, 243, 162, 148, 172, 229, 131, 144, 229, 160, 167, 229, 131, 144, 243, 162, 149, 142, 229, 131, 144, 243, 162, 148, 171, 31822, 243, 162, 148, 172, 229, 131, 144, 243, 162, 148, 172, 229, 131, 144, 243, 162, 148, 170, 229, 131, 144, 243, 162, 148, 169, 31822, 243, 162, 170, 148, 243, 162, 146, 190, 229, 131, 144, 243, 162, 167, 160, 229, 131, 144, 243, 162, 170, 148, 243, 162, 146, 190, 31822, 243, 162, 146, 183, 246, 163, 132, 170, 246, 163, 132, 165, 246, 163, 132, 168, 246, 163, 132, 177, 246, 163, 132, 170, 246, 163, 132, 194, 31822, 243, 162, 148, 171, 243, 162, 146, 190, 229, 131, 144, 229, 160, 167, 242, 187, 146, 229, 131, 144, 243, 162, 149, 142, 229, 131, 144, 243, 162, 148, 171, 243, 162, 146, 191],
      decoded: "<s> \u2728 \ud83e\udd17 \ud83d\udc41\ufe0f \ud83d\udc71\ud83c\udffb \ud83d\udd75\u200d\u2642\ufe0f \ud83e\uddd9\ud83c\udffb\u200d\u2642 \ud83d\udc68\ud83c\udffb\u200d\ud83c\udf3e \ud83e\uddd1\u200d\ud83e\udd1d\u200d\ud83e\uddd1 \ud83d\udc69\u200d\u2764\u200d\ud83d\udc8b\u200d\ud83d\udc68 \ud83d\udc69\u200d\ud83d\udc69\u200d\ud83d\udc67\u200d\ud83d\udc66 \ud83e\uddd1\ud83c\udffb\u200d\ud83e\udd1d\u200d\ud83e\uddd1\ud83c\udffb \ud83c\udff4\udb40\udc67\udb40\udc62\udb40\udc65\udb40\udc6e\udb40\udc67\udb40\udc7f \ud83d\udc68\ud83c\udffb\u200d\u2764\ufe0f\u200d\ud83d\udc8b\u200d\ud83d\udc68\ud83c\udffc",
    },
    NEWLINE_WITH_LEADING_SPACE: {
      text: LLAMA_TEST_STRINGS.NEWLINE_WITH_LEADING_SPACE,
      tokens: ["\u2581", "\u2581", "<0x0A>"],
      ids: [1, 31822, 31822, 13],
      decoded: "<s>  \n",
    },
    CHINESE_LETTER: {
      text: LLAMA_TEST_STRINGS.CHINESE_LETTER,
      tokens: ["\u2581", "<0xE9>", "<0x95>", "<0x87>"],
      ids: [1, 31822, 236, 152, 138],
      decoded: "<s> \u9547",
    },
    PARAGRAPH: {
      text: LLAMA_TEST_STRINGS.PARAGRAPH,
      tokens: ["\u2581The", "\u2581ll", "ama", "\u2581(", "/", "<0xCB>", "<0x88>", "l", "<0xC9>", "<0x91>", "<0xCB>", "<0x90>", "m", "<0xC9>", "<0x99>", "/", ";", "\u2581", "<0xF0>", "<0x9F>", "<0xA6>", "<0x99>", "Sp", "anish", "\u2581pron", "unciation", ":", "\u2581[", "<0xCB>", "<0x88>", "<0xCA>", "<0x8E>", "ama", "])", "\u2581(", "L", "ama", "\u2581gl", "ama", ")", "\u2581is", "\u2581a", "\u2581domest", "icated", "\u2581South", "\u2581American", "\u2581cam", "el", "id", ",", "\u2581widely", "\u2581used", "\u2581as", "\u2581a", "\u2581meat", "\u2581and", "\u2581pack", "\u2581animal", "\u2581by", "\u2581And", "ean", "\u2581cultures", "\u2581since", "\u2581the", "\u2581Pre", "-", "Col", "umb", "ian", "\u2581era", ".", "\u2581L", "lam", "as", "\u2581are", "\u2581social", "\u2581animals", "\u2581and", "\u2581live", "\u2581with", "\u2581others", "\u2581as", "\u2581a", "\u2581herd", ".", "\u2581Their", "\u2581wool", "\u2581is", "\u2581soft", "\u2581and", "\u2581contains", "\u2581only", "\u2581a", "\u2581small", "\u2581amount", "\u2581of", "\u2581l", "anol", "in", ".[", "2", "]", "\u2581L", "lam", "as", "\u2581can", "\u2581learn", "\u2581simple", "\u2581", "t", "asks", "\u2581after", "\u2581a", "\u2581few", "\u2581repet", "itions", ".", "\u2581When", "\u2581using", "\u2581a", "\u2581pack", ",", "\u2581they", "\u2581can", "\u2581carry", "\u2581about", "\u2581", "2", "5", "\u2581to", "\u2581", "3", "0", "%", "\u2581of", "\u2581their", "\u2581body", "\u2581weight", "\u2581for", "\u2581", "8", "\u2581to", "\u2581", "1", "3", "\u2581km", "\u2581(", "5", "\u2013", "8", "\u2581miles", ").", "[", "3", "]", "\u2581The", "\u2581name", "\u2581ll", "ama", "\u2581(", "in", "\u2581the", "\u2581past", "\u2581also", "\u2581sp", "elled", '\u2581"', "l", "ama", '"', "\u2581or", '\u2581"', "gl", "ama", '")', "\u2581was", "\u2581adopted", "\u2581by", "\u2581European", "\u2581settlers", "\u2581from", "\u2581native", "\u2581Per", "uv", "ians", ".[", "4", "]", "\u2581The", "\u2581ancestors", "\u2581of", "\u2581l", "lam", "as", "\u2581are", "\u2581thought", "\u2581to", "\u2581have", "\u2581originated", "\u2581from", "\u2581the", "\u2581Great", "\u2581Plains", "\u2581of", "\u2581North", "\u2581America", "\u2581about", "\u2581", "4", "0", "\u2581million", "\u2581years", "\u2581ago", ",", "\u2581and", "\u2581subsequently", "\u2581mig", "rated", "\u2581to", "\u2581South", "\u2581America", "\u2581about", "\u2581three", "\u2581million", "\u2581years", "\u2581ago", "\u2581during", "\u2581the", "\u2581Great", "\u2581American", "\u2581Inter", "change", ".", "\u2581By", "\u2581the", "\u2581end", "\u2581of", "\u2581the", "\u2581last", "\u2581ice", "\u2581age", "\u2581(", "1", "0", ",", "0", "0", "0", "\u2013", "1", "2", ",", "0", "0", "0", "\u2581years", "\u2581ago", "),", "\u2581cam", "el", "ids", "\u2581were", "\u2581extinct", "\u2581in", "\u2581North", "\u2581America", ".[", "3", "]", "\u2581As", "\u2581of", "\u2581", "2", "0", "0", "7", ",", "\u2581there", "\u2581were", "\u2581over", "\u2581seven", "\u2581million", "\u2581l", "lam", "as", "\u2581and", "\u2581al", "p", "ac", "as", "\u2581in", "\u2581South", "\u2581America", "\u2581and", "\u2581over", "\u2581", "1", "5", "8", ",", "0", "0", "0", "\u2581l", "lam", "as", "\u2581and", "\u2581", "1", "0", "0", ",", "0", "0", "0", "<0xEA>", "<0x99>", "<0x8A>", "<0xF0>", "<0x9F>", "<0xA6>", "<0x99>", "\u2581al", "p", "ac", "as", ",", "\u2581descended", "\u2581from", "\u2581pro", "gen", "itors", "\u2581imported", "\u2581late", "\u2581in", "\u2581the", "\u2581", "2", "0", "th", "\u2581century", ",", "\u2581in", "\u2581the", "\u2581United", "\u2581States", "\u2581and", "\u2581Canada", ".[", "5", "]", "\u2581In", "\u2581A", "ym", "ara", "\u2581mythology", ",", "\u2581l", "lam", "as", "\u2581are", "\u2581important", "\u2581beings", ".", "\u2581The", "\u2581Heaven", "ly", "\u2581Ll", "ama", "\u2581is", "\u2581said", "\u2581to", "\u2581drink", "\u2581water", "\u2581from", "\u2581the", "\u2581ocean", "\u2581and", "\u2581ur", "inates", "\u2581as", "\u2581it", "\u2581rains", ".[", "6", "]", "\u2581According", "\u2581to", "\u2581A", "ym", "ara", "\u2581es", "chat", "ology", ",", "\u2581l", "lam", "as", "\u2581will", "\u2581return", "\u2581to", "\u2581the", "\u2581water", "\u2581springs", "\u2581and", "\u2581l", "ago", "ons", "\u2581where", "\u2581they", "\u2581come", "\u2581from", "\u2581at", "\u2581the", "\u2581end", "\u2581of", "\u2581time", ".[", "6", "]"],
      ids: [1, 347, 31763, 2269, 352, 31873, 206, 139, 31832, 204, 148, 206, 147, 31836, 204, 156, 31873, 31891, 31822, 243, 162, 169, 156, 8889, 5817, 11155, 26128, 31871, 836, 206, 139, 205, 145, 2269, 9772, 352, 31867, 2269, 1192, 2269, 31861, 322, 260, 27940, 2672, 1897, 1454, 3764, 307, 317, 31844, 7055, 1065, 362, 260, 8659, 291, 2667, 6075, 417, 787, 14083, 10775, 1314, 266, 2345, 31854, 4848, 2234, 620, 5998, 31843, 372, 3082, 295, 397, 1619, 5220, 291, 1983, 351, 1892, 362, 260, 27172, 31843, 4585, 22729, 322, 2647, 291, 5140, 744, 260, 1435, 2399, 287, 309, 18426, 261, 3564, 31855, 31908, 372, 3082, 295, 473, 1977, 3102, 31822, 31824, 5577, 768, 260, 1346, 17042, 1479, 31843, 1408, 1340, 260, 2667, 31844, 526, 473, 3875, 562, 31822, 31855, 31880, 289, 31822, 31878, 31852, 31914, 287, 518, 2108, 4182, 329, 31822, 31886, 289, 31822, 31853, 31878, 6512, 352, 31880, 31906, 31886, 4465, 656, 31907, 31878, 31908, 347, 1382, 31763, 2269, 352, 261, 266, 1646, 615, 612, 5902, 495, 31832, 2269, 31875, 405, 495, 4261, 2269, 4290, 393, 7574, 417, 2821, 23343, 427, 6412, 2083, 10099, 1580, 3564, 31882, 31908, 347, 18294, 287, 309, 3082, 295, 397, 1991, 289, 435, 20355, 427, 266, 3172, 26744, 287, 1975, 2139, 562, 31822, 31882, 31852, 1577, 778, 2236, 31844, 291, 11786, 21052, 3397, 289, 1897, 2139, 562, 1166, 1577, 778, 2236, 1177, 266, 3172, 1454, 3029, 3604, 31843, 1433, 266, 928, 287, 266, 1060, 5707, 2253, 352, 31853, 31852, 31844, 31852, 31852, 31852, 31906, 31853, 31855, 31844, 31852, 31852, 31852, 778, 2236, 698, 3764, 307, 1982, 577, 30610, 288, 1975, 2139, 3564, 31878, 31908, 717, 287, 31822, 31855, 31852, 31852, 31888, 31844, 635, 577, 648, 3931, 1577, 309, 3082, 295, 291, 366, 31837, 380, 295, 288, 1897, 2139, 291, 648, 31822, 31853, 31880, 31886, 31844, 31852, 31852, 31852, 309, 3082, 295, 291, 31822, 31853, 31852, 31852, 31844, 31852, 31852, 31852, 237, 156, 141, 243, 162, 169, 156, 366, 31837, 380, 295, 31844, 27627, 427, 375, 3353, 4705, 17798, 2732, 288, 266, 31822, 31855, 31852, 388, 3373, 31844, 288, 266, 1494, 1769, 291, 3008, 3564, 31880, 31908, 455, 308, 1276, 2776, 24143, 31844, 309, 3082, 295, 397, 1480, 11844, 31843, 347, 15836, 326, 11321, 2269, 322, 664, 289, 5065, 1579, 427, 266, 8622, 291, 4328, 11466, 362, 357, 28738, 3564, 31887, 31908, 3252, 289, 308, 1276, 2776, 1582, 20068, 1058, 31844, 309, 3082, 295, 482, 1199, 289, 266, 1579, 24250, 291, 309, 3405, 680, 804, 526, 1412, 427, 389, 266, 928, 287, 647, 3564, 31887, 31908],
      decoded: '<s> The llama (/\u02c8l\u0251\u02d0m\u0259/; \ud83e\udd99Spanish pronunciation: [\u02c8\u028eama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the Pre-Columbian era. Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5\u20138 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4] The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000\u201312,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000\ua64a\ud83e\udd99 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5] In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology, llamas will return to the water springs and lagoons where they come from at the end of time.[6]',
    },
  },
  "Xenova/deepseek-coder-1.3b-instruct": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      tokens: ["How", "\u0120are", "\u0120you", "\u0120doing", "?"],
      ids: [32013, 2808, 417, 340, 3207, 30],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>How are you doing?",
    },
    SIMPLE_WITH_PUNCTUATION: {
      text: BASE_TEST_STRINGS.SIMPLE_WITH_PUNCTUATION,
      tokens: ["You", "\u0120should", "'", "ve", "\u0120done", "\u0120this"],
      ids: [32013, 2042, 1020, 6, 312, 2359, 437],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>You should've done this",
    },
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "\u0120", "0", "\u0120", "1", "\u0120", "2", "\u0120", "3", "\u0120", "4", "\u0120", "5", "\u0120", "6", "\u0120", "7", "\u0120", "8", "\u0120", "9", "\u0120", "1", "0", "\u0120", "1", "0", "0", "\u0120", "1", "0", "0", "0"],
      ids: [32013, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 207, 15, 207, 16, 207, 17, 207, 18, 207, 19, 207, 20, 207, 21, 207, 22, 207, 23, 207, 24, 207, 16, 15, 207, 16, 15, 15, 207, 16, 15, 15, 15],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["The", "\u0120company", "\u0120was", "\u0120founded", "\u0120in", "\u0120", "2", "0", "1", "6", "."],
      ids: [32013, 546, 2595, 438, 16316, 279, 207, 17, 15, 16, 21, 13],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>The company was founded in 2016.",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["A", "\u010a", "'", "ll", "\u0120!!", "to", "?'", "d", "''", "d", "\u0120of", ",", "\u0120can", "'", "t", "."],
      ids: [32013, 32, 185, 6, 642, 24466, 577, 11665, 67, 4191, 67, 280, 11, 482, 6, 83, 13],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>A\n'll !!to?'d''d of, can't.",
    },
    PYTHON_CODE: {
      text: BASE_TEST_STRINGS.PYTHON_CODE,
      tokens: ["def", "\u0120main", "():", "\u010a", "\u0109", "pass"],
      ids: [32013, 1551, 1959, 10942, 185, 184, 4805],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>def main():\n\tpass",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["let", "\u0120a", "\u0120=", "\u0120obj", ".", "toString", "();", "\u010a", "toString", "();"],
      ids: [32013, 1160, 245, 405, 6528, 13, 12617, 1293, 185, 12617, 1293],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>let a = obj.toString();\ntoString();",
    },
    NEWLINES: {
      text: LLAMA_TEST_STRINGS.NEWLINES,
      tokens: ["ax", "\u010a", "####", "\u010a", "bo", "o"],
      ids: [32013, 1099, 185, 3576, 185, 952, 78],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>ax\n####\nboo",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["UN", "want", "\u00c3\u00a9d", ",", "running"],
      ids: [32013, 4348, 28626, 31898, 11, 22785],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>UNwant\u00e9d,running",
    },
    CONTROL_TOKENS: {
      text: BASE_TEST_STRINGS.CONTROL_TOKENS,
      tokens: ["1", "\u0100", "2", "\u00ef\u00bf\u00bd", "3"],
      ids: [32013, 16, 175, 17, 10006, 18],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>1\u00002\ufffd3",
    },
    HELLO_WORLD_TITLECASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_TITLECASE,
      tokens: ["Hello", "\u0120World"],
      ids: [32013, 17535, 5414],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>Hello World",
    },
    HELLO_WORLD_LOWERCASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_LOWERCASE,
      tokens: ["hello", "\u0120world"],
      ids: [32013, 31702, 1835],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>hello world",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u00e7\u0136\u0141\u00e6\u00b4\u00bb\u00e7\u013c\u0126", "\u00e7\u013e\u0141", "\u00e8\u00b0", "\u013d", "\u00e6\u013a\u00af"],
      ids: [32013, 23393, 2651, 1534, 236, 502],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>\u751f\u6d3b\u7684\u771f\u8c1b\u662f",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      tokens: ["\u0120\u0120", "\u0120leading", "\u0120space"],
      ids: [32013, 243, 5877, 2507],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>   leading space",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["tra", "iling", "\u0120space", "\u0120\u0120\u0120"],
      ids: [32013, 7246, 5964, 2507, 315],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>trailing space   ",
    },
    DOUBLE_SPACE: {
      text: BASE_TEST_STRINGS.DOUBLE_SPACE,
      tokens: ["Hi", "\u0120", "\u0120H", "ello"],
      ids: [32013, 11041, 207, 414, 9489],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>Hi  Hello",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["test", "\u0120$", "1", "\u0120R", "2", "\u0120#", "3", "\u0120", "\u00e2\u0124\u00ac", "4", "\u0120\u00c2\u00a3", "5", "\u0120\u00c2", "\u00a5", "6", "\u0120", "\u00e2\u0124", "\u00a3", "7", "\u0120", "\u00e2\u0124", "\u00b9", "8", "\u0120", "\u00e2\u0124", "\u00b1", "9", "\u0120test"],
      ids: [32013, 2806, 371, 16, 432, 17, 1494, 18, 207, 11010, 19, 8761, 20, 2688, 98, 21, 207, 7935, 96, 22, 207, 7935, 117, 23, 207, 7935, 109, 24, 1719],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>test $1 R2 #3 \u20ac4 \u00a35 \u00a56 \u20a37 \u20b98 \u20b19 test",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["I", "\u0120bought", "\u0120an", "\u0120apple", "\u0120for", "\u0120$", "1", ".", "0", "0", "\u0120at", "\u0120the", "\u0120store", "."],
      ids: [32013, 40, 8942, 274, 15902, 327, 371, 16, 13, 15, 15, 429, 254, 4730, 13],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>I bought an apple for $1.00 at the store.",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u0120\u0120"],
      ids: [32013, 4209, 2484, 243],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>you\u2026  ",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u00c2\u0142\u00c2\u0142"],
      ids: [32013, 4209, 2484, 10447],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>you\u2026\u00a0\u00a0",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u00c2\u0142", "\u00c2\u0142", "you", "\u00e2\u0122\u00a6", "\u00c2\u0142\u00c2\u0142"],
      ids: [32013, 4209, 2484, 1200, 1200, 4209, 2484, 10447],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>you\u2026\u00a0\u00a0you\u2026\u00a0\u00a0",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["we", "ird", "\u0120", "\u00ef", "\u00bd", "\u0140", "\u0120edge", "\u0120", "\u00ef", "\u00bd", "\u0140", "\u0120case"],
      ids: [32013, 828, 2369, 207, 169, 121, 239, 5935, 207, 169, 121, 239, 1452],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>weird \uff5e edge \uff5e case",
    },
    POPULAR_EMOJIS: {
      text: BASE_TEST_STRINGS.POPULAR_EMOJIS,
      tokens: ["\u00f0\u0141", "\u013a", "\u0124", "\u0120\u00f0\u0141", "\u0133", "\u012f", "\u0120\u00f0\u0141", "\u00a4", "\u00a3", "\u0120\u00f0\u0141", "\u013a", "\u012f", "\u0120\u00f0\u0141", "\u013a", "\u0143", "\u0120\u00f0\u0141", "\u0130", "\u012b", "\u0120\u00f0\u0141\u013b", "\u0131", "\u0120\u00f0\u0141", "\u013a", "\u012c", "\u0120\u00f0\u0141", "\u0136", "\u00a5", "\u0120\u00f0\u0141", "\u013a", "\u0123", "\u0120\u00f0\u0141", "\u013a", "\u0127", "\u0120\u00f0\u0141", "\u00a4", "\u0139", "\u0120\u00f0\u0141", "\u013a", "\u0128", "\u0120\u00f0\u0141", "\u0133", "\u0131", "\u0120", "\u00e2", "\u013f", "\u00a4", "\u00ef", "\u00b8", "\u0131", "\u0120\u00f0\u0141", "\u0134", "\u013e", "\u0120\u00f0\u0141", "\u0134", "\u013c", "\u0120\u00f0\u0141", "\u0134", "\u0139", "\u0120\u00f0\u0141", "\u0134", "\u013b", "\u0120\u00f0\u0141", "\u0138", "\u00a4", "\u0120\u00f0\u0141", "\u013a", "\u0130", "\u0120\u00f0\u0141", "\u0133", "\u012e", "\u0120\u00f0\u0141", "\u00a5", "\u00b3", "\u0120\u00f0\u0141", "\u0134", "\u00aa", "\u0120", "\u00e2", "\u013e", "\u00a8", "\u0120\u00f0\u0141", "\u0133", "\u012b", "\u0120\u00f0\u0141", "\u0133", "\u0122", "\u0120\u00f0\u0141", "\u0134", "\u00af", "\u0120\u00f0\u0141", "\u0130", "\u012a", "\u0120\u00f0\u0141\u013b", "\u012a", "\u0120\u00f0\u0141\u013b", "\u012e", "\u0120\u00f0\u0141", "\u0134", "\u0122", "\u0120\u00f0\u0141", "\u0133", "\u0129", "\u0120\u00f0\u0141", "\u0133", "\u012d", "\u0120", "\u00e2", "\u013e", "\u0127", "\u0120\u00f0\u0141", "\u0130", "\u0123", "\u0120\u00f0\u0141", "\u012e", "\u0140", "\u0120\u00f0\u0141", "\u012e", "\u00b8", "\u0120\u00f0\u0141", "\u0134", "\u00b0"],
      ids: [32013, 10047, 233, 211, 12394, 226, 222, 12394, 97, 96, 12394, 233, 222, 12394, 233, 242, 12394, 223, 218, 22709, 224, 12394, 233, 219, 12394, 229, 98, 12394, 233, 210, 12394, 233, 214, 12394, 97, 232, 12394, 233, 215, 12394, 226, 224, 207, 156, 238, 97, 169, 116, 224, 12394, 227, 237, 12394, 227, 235, 12394, 227, 232, 12394, 227, 234, 12394, 231, 97, 12394, 233, 223, 12394, 226, 221, 12394, 98, 111, 12394, 227, 103, 207, 156, 237, 101, 12394, 226, 218, 12394, 226, 209, 12394, 227, 107, 12394, 223, 217, 22709, 217, 22709, 221, 12394, 227, 209, 12394, 226, 216, 12394, 226, 220, 207, 156, 237, 214, 12394, 223, 210, 12394, 221, 239, 12394, 221, 116, 12394, 227, 108],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>\ud83d\ude02 \ud83d\udc4d \ud83e\udd23 \ud83d\ude0d \ud83d\ude2d \ud83c\udf89 \ud83d\ude4f \ud83d\ude0a \ud83d\udd25 \ud83d\ude01 \ud83d\ude05 \ud83e\udd17 \ud83d\ude06 \ud83d\udc4f \u2764\ufe0f \ud83d\udc9c \ud83d\udc9a \ud83d\udc97 \ud83d\udc99 \ud83d\udda4 \ud83d\ude0e \ud83d\udc4c \ud83e\udd73 \ud83d\udcaa \u2728 \ud83d\udc49 \ud83d\udc40 \ud83d\udcaf \ud83c\udf88 \ud83d\ude48 \ud83d\ude4c \ud83d\udc80 \ud83d\udc47 \ud83d\udc4b \u2705 \ud83c\udf81 \ud83c\udf1e \ud83c\udf38 \ud83d\udcb0",
    },
    MULTIBYTE_EMOJIS: {
      text: BASE_TEST_STRINGS.MULTIBYTE_EMOJIS,
      tokens: ["\u00e2", "\u013e", "\u00a8", "\u0120\u00f0\u0141", "\u00a4", "\u0139", "\u0120\u00f0\u0141", "\u0133", "\u0123", "\u00ef", "\u00b8", "\u0131", "\u0120\u00f0\u0141", "\u0133", "\u00b1", "\u00f0\u0141", "\u0131", "\u00bb", "\u0120\u00f0\u0141", "\u0137", "\u00b5", "\u00e2\u0122", "\u012f", "\u00e2", "\u013b", "\u0124", "\u00ef", "\u00b8", "\u0131", "\u0120\u00f0\u0141", "\u00a7", "\u013b", "\u00f0\u0141", "\u0131", "\u00bb", "\u00e2\u0122", "\u012f", "\u00e2", "\u013b", "\u0124", "\u0120\u00f0\u0141", "\u0133", "\u00a8", "\u00f0\u0141", "\u0131", "\u00bb", "\u00e2\u0122", "\u012f", "\u00f0\u0141", "\u012e", "\u00be", "\u0120\u00f0\u0141", "\u00a7", "\u0133", "\u00e2\u0122", "\u012f", "\u00f0\u0141", "\u00a4", "\u013f", "\u00e2\u0122", "\u012f", "\u00f0\u0141", "\u00a7", "\u0133", "\u0120\u00f0\u0141", "\u0133", "\u00a9", "\u00e2\u0122", "\u012f", "\u00e2", "\u013f", "\u00a4", "\u00e2\u0122", "\u012f", "\u00f0\u0141", "\u0134", "\u012d", "\u00e2\u0122", "\u012f", "\u00f0\u0141", "\u0133", "\u00a8", "\u0120\u00f0\u0141", "\u0133", "\u00a9", "\u00e2\u0122", "\u012f", "\u00f0\u0141", "\u0133", "\u00a9", "\u00e2\u0122", "\u012f", "\u00f0\u0141", "\u0133", "\u00a7", "\u00e2\u0122", "\u012f", "\u00f0\u0141", "\u0133", "\u00a6", "\u0120\u00f0\u0141", "\u00a7", "\u0133", "\u00f0\u0141", "\u0131", "\u00bb", "\u00e2\u0122", "\u012f", "\u00f0\u0141", "\u00a4", "\u013f", "\u00e2\u0122", "\u012f", "\u00f0\u0141", "\u00a7", "\u0133", "\u00f0\u0141", "\u0131", "\u00bb", "\u0120\u00f0\u0141", "\u0131", "\u00b4", "\u00f3", "\u0142", "\u0123", "\u00a7", "\u00f3", "\u0142", "\u0123", "\u00a2", "\u00f3", "\u0142", "\u0123", "\u00a5", "\u00f3", "\u0142", "\u0123", "\u00ae", "\u00f3", "\u0142", "\u0123", "\u00a7", "\u00f3", "\u0142", "\u0123", "\u00bf", "\u0120\u00f0\u0141", "\u0133", "\u00a8", "\u00f0\u0141", "\u0131", "\u00bb", "\u00e2\u0122", "\u012f", "\u00e2", "\u013f", "\u00a4", "\u00ef", "\u00b8", "\u0131", "\u00e2\u0122", "\u012f", "\u00f0\u0141", "\u0134", "\u012d", "\u00e2\u0122", "\u012f", "\u00f0\u0141", "\u0133", "\u00a8", "\u00f0\u0141", "\u0131", "\u00bc"],
      ids: [32013, 156, 237, 101, 12394, 97, 232, 12394, 226, 210, 169, 116, 224, 12394, 226, 109, 10047, 224, 119, 12394, 230, 113, 350, 222, 156, 234, 211, 169, 116, 224, 12394, 100, 234, 10047, 224, 119, 350, 222, 156, 234, 211, 12394, 226, 101, 10047, 224, 119, 350, 222, 10047, 221, 122, 12394, 100, 226, 350, 222, 10047, 97, 238, 350, 222, 10047, 100, 226, 12394, 226, 102, 350, 222, 156, 238, 97, 350, 222, 10047, 227, 220, 350, 222, 10047, 226, 101, 12394, 226, 102, 350, 222, 10047, 226, 102, 350, 222, 10047, 226, 100, 350, 222, 10047, 226, 99, 12394, 100, 226, 10047, 224, 119, 350, 222, 10047, 97, 238, 350, 222, 10047, 100, 226, 10047, 224, 119, 12394, 224, 112, 173, 241, 210, 100, 173, 241, 210, 95, 173, 241, 210, 98, 173, 241, 210, 106, 173, 241, 210, 100, 173, 241, 210, 123, 12394, 226, 101, 10047, 224, 119, 350, 222, 156, 238, 97, 169, 116, 224, 350, 222, 10047, 227, 220, 350, 222, 10047, 226, 101, 10047, 224, 120],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>\u2728 \ud83e\udd17 \ud83d\udc41\ufe0f \ud83d\udc71\ud83c\udffb \ud83d\udd75\u200d\u2642\ufe0f \ud83e\uddd9\ud83c\udffb\u200d\u2642 \ud83d\udc68\ud83c\udffb\u200d\ud83c\udf3e \ud83e\uddd1\u200d\ud83e\udd1d\u200d\ud83e\uddd1 \ud83d\udc69\u200d\u2764\u200d\ud83d\udc8b\u200d\ud83d\udc68 \ud83d\udc69\u200d\ud83d\udc69\u200d\ud83d\udc67\u200d\ud83d\udc66 \ud83e\uddd1\ud83c\udffb\u200d\ud83e\udd1d\u200d\ud83e\uddd1\ud83c\udffb \ud83c\udff4\udb40\udc67\udb40\udc62\udb40\udc65\udb40\udc6e\udb40\udc67\udb40\udc7f \ud83d\udc68\ud83c\udffb\u200d\u2764\ufe0f\u200d\ud83d\udc8b\u200d\ud83d\udc68\ud83c\udffc",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      tokens: ["\u00e2\u0138", "\u0123", "This", "\u0120", "\u00e2\u0138", "\u0123", "is", "\u0120", "\u00e2\u0138", "\u0123", "a", "\u0120", "\u00e2\u0138", "\u0123", "test", "\u0120", "\u00e2\u0138", "\u0123", "."],
      ids: [32013, 11028, 210, 1559, 207, 11028, 210, 262, 207, 11028, 210, 64, 207, 11028, 210, 2806, 207, 11028, 210, 13],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>\u2581This \u2581is \u2581a \u2581test \u2581.",
    },
    BPE_SCORES_PRIORITY_1: {
      text: LLAMA_TEST_STRINGS.BPE_SCORES_PRIORITY_1,
      tokens: ["gr", "ab", "bed"],
      ids: [32013, 877, 356, 3861],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>grabbed",
    },
    BPE_SCORES_PRIORITY_2: {
      text: LLAMA_TEST_STRINGS.BPE_SCORES_PRIORITY_2,
      tokens: ["\u0120grab", "bed"],
      ids: [32013, 14596, 3861],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c> grabbed",
    },
    BPE_SCORES_PRIORITY_3: {
      text: LLAMA_TEST_STRINGS.BPE_SCORES_PRIORITY_3,
      tokens: ["\u0120\u0120\u0120\u0120\u0120\u0120\u0120\u0120\u0120\u0120", "\u0120grab", "bed"],
      ids: [32013, 3137, 14596, 3861],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>           grabbed",
    },
    NEWLINE: {
      text: LLAMA_TEST_STRINGS.NEWLINE,
      tokens: ["\u010a"],
      ids: [32013, 185],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>\n",
    },
    NEWLINE_WITH_LEADING_SPACE: {
      text: LLAMA_TEST_STRINGS.NEWLINE_WITH_LEADING_SPACE,
      tokens: ["\u0120", "\u010a"],
      ids: [32013, 207, 185],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c> \n",
    },
    TABS: {
      text: LLAMA_TEST_STRINGS.TABS,
      tokens: ["\u0109", "tabs", "\u0109\u0109\u0109", "\u0109", "out", "\u0120here"],
      ids: [32013, 184, 20611, 1749, 184, 406, 1283],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>\ttabs\t\t\t\tout here",
    },
    NEWLINE_AND_TAB: {
      text: LLAMA_TEST_STRINGS.NEWLINE_AND_TAB,
      tokens: ["\u010a", "\u0109", "\u010a"],
      ids: [32013, 185, 184, 185],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>\n\t\n",
    },
    CHINESE_LETTER: {
      text: LLAMA_TEST_STRINGS.CHINESE_LETTER,
      tokens: ["\u00e9\u0137\u0129"],
      ids: [32013, 6759],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>\u9547",
    },
    EMOJIS_1: {
      text: LLAMA_TEST_STRINGS.EMOJIS_1,
      tokens: ["\u00f0\u0141", "\u00a6", "\u013b"],
      ids: [32013, 10047, 99, 234],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>\ud83e\udd99",
    },
    EMOJIS_2: {
      text: LLAMA_TEST_STRINGS.EMOJIS_2,
      tokens: ["\u00f0\u0141", "\u00a6", "\u013b", "\u00ea", "\u013b", "\u012c"],
      ids: [32013, 10047, 99, 234, 164, 234, 219],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>\ud83e\udd99\ua64a",
    },
    EMOJIS_3: {
      text: LLAMA_TEST_STRINGS.EMOJIS_3,
      tokens: ["\u00ea", "\u013b", "\u012c", "\u00f0\u0141", "\u00a6", "\u013b"],
      ids: [32013, 164, 234, 219, 10047, 99, 234],
      decoded: "<\uff5cbegin\u2581of\u2581sentence\uff5c>\ua64a\ud83e\udd99",
    },
    PARAGRAPH: {
      text: LLAMA_TEST_STRINGS.PARAGRAPH,
      tokens: ["The", "\u0120ll", "ama", "\u0120(/", "\u00cb\u012a", "l", "\u00c9", "\u0133", "\u00cb", "\u0132", "m", "\u00c9\u013b", "/", ";", "\u0120\u00f0\u0141", "\u00a6", "\u013b", "Span", "ish", "\u0120pron", "unciation", ":", "\u0120[", "\u00cb\u012a", "\u00ca", "\u0130", "ama", "])", "\u0120(", "L", "ama", "\u0120gl", "ama", ")", "\u0120is", "\u0120a", "\u0120domestic", "ated", "\u0120South", "\u0120American", "\u0120cam", "el", "id", ",", "\u0120widely", "\u0120used", "\u0120as", "\u0120a", "\u0120meat", "\u0120and", "\u0120pack", "\u0120animal", "\u0120by", "\u0120And", "ean", "\u0120cultures", "\u0120since", "\u0120the", "\u0120Pre", "-", "Col", "umb", "ian", "\u0120era", ".", "\u0120L", "lam", "as", "\u0120are", "\u0120social", "\u0120animals", "\u0120and", "\u0120live", "\u0120with", "\u0120others", "\u0120as", "\u0120a", "\u0120her", "d", ".", "\u0120Their", "\u0120wool", "\u0120is", "\u0120soft", "\u0120and", "\u0120contains", "\u0120only", "\u0120a", "\u0120small", "\u0120amount", "\u0120of", "\u0120lan", "ol", "in", ".[", "2", "]", "\u0120L", "lam", "as", "\u0120can", "\u0120learn", "\u0120simple", "\u0120tasks", "\u0120after", "\u0120a", "\u0120few", "\u0120repet", "itions", ".", "\u0120When", "\u0120using", "\u0120a", "\u0120pack", ",", "\u0120they", "\u0120can", "\u0120carry", "\u0120about", "\u0120", "2", "5", "\u0120to", "\u0120", "3", "0", "%", "\u0120of", "\u0120their", "\u0120body", "\u0120weight", "\u0120for", "\u0120", "8", "\u0120to", "\u0120", "1", "3", "\u0120km", "\u0120(", "5", "\u00e2\u0122\u0135", "8", "\u0120miles", ").", "[", "3", "]", "\u0120The", "\u0120name", "\u0120ll", "ama", "\u0120(", "in", "\u0120the", "\u0120past", "\u0120also", "\u0120sp", "elled", '\u0120"', "l", "ama", '"', "\u0120or", '\u0120"', "gl", "ama", '")', "\u0120was", "\u0120adopted", "\u0120by", "\u0120European", "\u0120sett", "lers", "\u0120from", "\u0120native", "\u0120Per", "uv", "ians", ".[", "4", "]", "\u0120The", "\u0120ancest", "ors", "\u0120of", "\u0120llam", "as", "\u0120are", "\u0120thought", "\u0120to", "\u0120have", "\u0120origin", "ated", "\u0120from", "\u0120the", "\u0120Great", "\u0120Pl", "ains", "\u0120of", "\u0120North", "\u0120America", "\u0120about", "\u0120", "4", "0", "\u0120million", "\u0120years", "\u0120ago", ",", "\u0120and", "\u0120subsequently", "\u0120mig", "rated", "\u0120to", "\u0120South", "\u0120America", "\u0120about", "\u0120three", "\u0120million", "\u0120years", "\u0120ago", "\u0120during", "\u0120the", "\u0120Great", "\u0120American", "\u0120Inter", "change", ".", "\u0120By", "\u0120the", "\u0120end", "\u0120of", "\u0120the", "\u0120last", "\u0120ice", "\u0120age", "\u0120(", "1", "0", ",", "0", "0", "0", "\u00e2\u0122\u0135", "1", "2", ",", "0", "0", "0", "\u0120years", "\u0120ago", "),", "\u0120cam", "el", "ids", "\u0120were", "\u0120ext", "inct", "\u0120in", "\u0120North", "\u0120America", ".[", "3", "]", "\u0120As", "\u0120of", "\u0120", "2", "0", "0", "7", ",", "\u0120there", "\u0120were", "\u0120over", "\u0120seven", "\u0120million", "\u0120llam", "as", "\u0120and", "\u0120al", "p", "ac", "as", "\u0120in", "\u0120South", "\u0120America", "\u0120and", "\u0120over", "\u0120", "1", "5", "8", ",", "0", "0", "0", "\u0120llam", "as", "\u0120and", "\u0120", "1", "0", "0", ",", "0", "0", "0", "\u00ea", "\u013b", "\u012c", "\u00f0\u0141", "\u00a6", "\u013b", "\u0120al", "p", "ac", "as", ",", "\u0120desc", "ended", "\u0120from", "\u0120pro", "gen", "itors", "\u0120imported", "\u0120late", "\u0120in", "\u0120the", "\u0120", "2", "0", "th", "\u0120century", ",", "\u0120in", "\u0120the", "\u0120United", "\u0120States", "\u0120and", "\u0120Canada", ".[", "5", "]", "\u0120In", "\u0120A", "ym", "ara", "\u0120myth", "ology", ",", "\u0120llam", "as", "\u0120are", "\u0120important", "\u0120beings", ".", "\u0120The", "\u0120Heaven", "ly", "\u0120Ll", "ama", "\u0120is", "\u0120said", "\u0120to", "\u0120drink", "\u0120water", "\u0120from", "\u0120the", "\u0120ocean", "\u0120and", "\u0120ur", "in", "ates", "\u0120as", "\u0120it", "\u0120ra", "ins", ".[", "6", "]", "\u0120According", "\u0120to", "\u0120A", "ym", "ara", "\u0120es", "chat", "ology", ",", "\u0120llam", "as", "\u0120will", "\u0120return", "\u0120to", "\u0120the", "\u0120water", "\u0120springs", "\u0120and", "\u0120l", "ago", "ons", "\u0120where", "\u0120they", "\u0120come", "\u0120from", "\u0120at", "\u0120the", "\u0120end", "\u0120of", "\u0120time", ".[", "6", "]"],
      ids: [32013, 546, 1703, 4204, 31905, 31459, 75, 131, 226, 133, 225, 76, 28747, 14, 26, 12394, 99, 234, 20786, 840, 9119, 25307, 25, 821, 31459, 132, 223, 4204, 5589, 334, 43, 4204, 1649, 4204, 8, 317, 245, 13569, 612, 5168, 4115, 4370, 282, 304, 11, 13620, 1219, 372, 245, 12342, 285, 2379, 9542, 457, 1306, 24391, 24783, 1952, 254, 7606, 12, 2608, 4313, 987, 2895, 13, 412, 8265, 281, 417, 3601, 8469, 285, 3516, 365, 3060, 372, 245, 706, 67, 13, 9195, 24547, 317, 2829, 285, 5396, 885, 245, 1752, 3733, 280, 27264, 313, 246, 9469, 17, 60, 412, 8265, 281, 482, 3059, 2966, 9227, 1164, 245, 1853, 15747, 2160, 13, 2463, 1242, 245, 2379, 11, 653, 482, 5642, 782, 207, 17, 20, 276, 207, 18, 15, 4, 280, 699, 3110, 4285, 327, 207, 23, 276, 207, 16, 18, 9004, 334, 20, 887, 23, 6595, 628, 58, 18, 60, 428, 1208, 1703, 4204, 334, 246, 254, 2872, 835, 731, 6679, 440, 75, 4204, 1, 409, 440, 2521, 4204, 2456, 438, 13509, 457, 8717, 6762, 12104, 473, 8118, 3043, 12466, 3091, 9469, 19, 60, 428, 18901, 710, 280, 15410, 281, 417, 2207, 276, 463, 6948, 612, 473, 254, 6984, 2284, 2200, 280, 5216, 6092, 782, 207, 19, 15, 4866, 1547, 4074, 11, 285, 23909, 8290, 9831, 276, 5168, 6092, 782, 1846, 4866, 1547, 4074, 2310, 254, 6984, 4115, 6660, 4865, 13, 3550, 254, 1223, 280, 254, 1554, 9405, 4489, 334, 16, 15, 11, 15, 15, 15, 887, 16, 17, 11, 15, 15, 15, 1547, 4074, 650, 4370, 282, 2929, 773, 1309, 5729, 279, 5216, 6092, 9469, 18, 60, 1725, 280, 207, 17, 15, 15, 22, 11, 741, 773, 851, 7970, 4866, 15410, 281, 285, 360, 79, 305, 281, 279, 5168, 6092, 285, 851, 207, 16, 20, 23, 11, 15, 15, 15, 15410, 281, 285, 207, 16, 15, 15, 11, 15, 15, 15, 164, 234, 219, 10047, 99, 234, 360, 79, 305, 281, 11, 1774, 2611, 473, 381, 4920, 6041, 26357, 5179, 279, 254, 207, 17, 15, 392, 8299, 11, 279, 254, 4783, 5098, 285, 8905, 9469, 20, 60, 680, 338, 1254, 3367, 25157, 2333, 11, 15410, 281, 417, 2364, 22792, 13, 428, 18933, 326, 9140, 4204, 317, 989, 276, 7371, 2345, 473, 254, 15439, 285, 8580, 246, 980, 372, 359, 1809, 1231, 9469, 21, 60, 10068, 276, 338, 1254, 3367, 707, 24570, 2333, 11, 15410, 281, 540, 967, 276, 254, 2345, 30851, 285, 284, 5980, 875, 1064, 653, 1857, 473, 429, 254, 1223, 280, 761, 9469, 21, 60],
      decoded: '<\uff5cbegin\u2581of\u2581sentence\uff5c>The llama (/\u02c8l\u0251\u02d0m\u0259/; \ud83e\udd99Spanish pronunciation: [\u02c8\u028eama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the Pre-Columbian era. Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5\u20138 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4] The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000\u201312,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000\ua64a\ud83e\udd99 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5] In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology, llamas will return to the water springs and lagoons where they come from at the end of time.[6]',
    },
  },
  "Xenova/tamillama_tiny_30m": {
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["\u2581The", "\u2581company", "\u2581was", "\u2581found", "ed", "\u2581in", "\u2581", "2", "0", "1", "6", "."],
      ids: [1, 147, 10984, 139, 949, 78, 198, 31654, 13, 21, 12, 17, 34],
      decoded: "<s> The company was founded in 2016.",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["\u2581A", "\n", "'", "ll", "\u2581", "!", "!", "to", "?", "'", "d", "'", "'", "d", "\u2581of", ",", "\u2581can", "'", "t", "."],
      ids: [1, 231, 5, 31, 370, 31654, 31715, 31715, 5140, 31725, 31, 31679, 31, 31, 31679, 251, 35, 645, 31, 31665, 34],
      decoded: "<s> A\n'll !!to?'d''d of, can't.",
    },
    PYTHON_CODE: {
      text: BASE_TEST_STRINGS.PYTHON_CODE,
      tokens: ["\u2581def", "\u2581main", "(", ")", ":", "\n", "<unk>", "p", "ass"],
      ids: [1, 12849, 17375, 32, 33, 29, 5, 0, 31694, 1917],
      decoded: "<s> def main():\n<unk>pass",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["\u2581let", "\u2581a", "\u2581", "=", "\u2581ob", "j", ".", "to", "St", "ring", "(", ")", ";", "\n", "to", "St", "ring", "(", ")", ";"],
      ids: [1, 1996, 48, 31654, 25, 4083, 31733, 34, 5140, 23417, 6631, 32, 33, 30, 5, 5140, 23417, 6631, 32, 33, 30],
      decoded: "<s> let a = obj.toString();\ntoString();",
    },
    NEWLINES: {
      text: LLAMA_TEST_STRINGS.NEWLINES,
      tokens: ["\u2581ax", "\n", "#", "#", "#", "#", "\n", "boo"],
      ids: [1, 11441, 5, 22, 22, 22, 22, 5, 21260],
      decoded: "<s> ax\n####\nboo",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["\u2581U", "N", "w", "ant", "\u00e9", "d", ",", "r", "un", "ning"],
      ids: [1, 5841, 31748, 31689, 1027, 31771, 31679, 35, 31678, 367, 1855],
      decoded: "<s> UNwant\u00e9d,running",
    },
    CONTROL_TOKENS: {
      text: BASE_TEST_STRINGS.CONTROL_TOKENS,
      tokens: ["\u2581", "1", "<unk>", "2", "<unk>", "3"],
      ids: [1, 31654, 12, 0, 13, 0, 14],
      decoded: "<s> 1<unk>2<unk>3",
    },
    HELLO_WORLD_TITLECASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_TITLECASE,
      tokens: ["\u2581H", "ello", "\u2581World"],
      ids: [1, 207, 3589, 25544],
      decoded: "<s> Hello World",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u2581", "<unk>"],
      ids: [1, 31654, 0],
      decoded: "<s> <unk>",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      tokens: ["\u2581", "\u2581", "\u2581", "\u2581leading", "\u2581space"],
      ids: [1, 31654, 31654, 31654, 7951, 7259],
      decoded: "<s>    leading space",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["\u2581tra", "iling", "\u2581space", "\u2581", "\u2581", "\u2581"],
      ids: [1, 2036, 9850, 7259, 31654, 31654, 31654],
      decoded: "<s> trailing space   ",
    },
    DOUBLE_SPACE: {
      text: BASE_TEST_STRINGS.DOUBLE_SPACE,
      tokens: ["\u2581H", "i", "\u2581", "\u2581H", "ello"],
      ids: [1, 207, 31673, 31654, 207, 3589],
      decoded: "<s> Hi  Hello",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["\u2581test", "\u2581", "$", "1", "\u2581R", "2", "\u2581", "#", "3", "\u2581", "\u20ac", "4", "\u2581", "\u00a3", "5", "\u2581", "<unk>", "6", "\u2581", "<unk>", "7", "\u2581", "\u20b9", "8", "\u2581", "<unk>", "9", "\u2581test"],
      ids: [1, 6370, 31654, 9, 12, 947, 13, 31654, 22, 14, 31654, 31746, 15, 31654, 31792, 16, 31654, 0, 17, 31654, 0, 18, 31654, 31999, 19, 31654, 0, 20, 6370],
      decoded: "<s> test $1 R2 #3 \u20ac4 \u00a35 <unk>6 <unk>7 \u20b98 <unk>9 test",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["\u2581I", "\u2581bought", "\u2581an", "\u2581apple", "\u2581for", "\u2581", "$", "1", ".", "0", "0", "\u2581at", "\u2581the", "\u2581store", "."],
      ids: [1, 320, 4685, 446, 4223, 347, 31654, 9, 12, 34, 21, 21, 586, 70, 2023, 34],
      decoded: "<s> I bought an apple for $1.00 at the store.",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["\u2581you", "<unk>", "\u2581", "\u2581"],
      ids: [1, 356, 0, 31654, 31654],
      decoded: "<s> you<unk>  ",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["\u2581you", "<unk>"],
      ids: [1, 356, 0],
      decoded: "<s> you<unk>",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["\u2581you", "<unk>", "you", "<unk>"],
      ids: [1, 356, 0, 21984, 0],
      decoded: "<s> you<unk>you<unk>",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["\u2581weird", "\u2581", "<unk>", "\u2581edge", "\u2581", "<unk>", "\u2581case"],
      ids: [1, 7865, 31654, 0, 11148, 31654, 0, 10143],
      decoded: "<s> weird <unk> edge <unk> case",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      tokens: ["\u2581", "\u2581This", "\u2581", "\u2581is", "\u2581", "\u2581a", "\u2581", "\u2581test", "\u2581", "\u2581", "."],
      ids: [1, 31654, 3827, 31654, 344, 31654, 48, 31654, 6370, 31654, 31654, 34],
      decoded: "<s>  This  is  a  test  .",
    },
    POPULAR_EMOJIS: {
      text: BASE_TEST_STRINGS.POPULAR_EMOJIS,
      tokens: ["\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>"],
      ids: [1, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0],
      decoded: "<s> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>",
    },
    MULTIBYTE_EMOJIS: {
      text: BASE_TEST_STRINGS.MULTIBYTE_EMOJIS,
      tokens: ["\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u200d", "<unk>", "\u2581", "<unk>", "\u200d", "<unk>", "\u2581", "<unk>", "\u200d", "<unk>", "\u2581", "<unk>", "\u200d", "<unk>", "\u200d", "<unk>", "\u2581", "<unk>", "\u200d", "<unk>", "\u200d", "<unk>", "\u200d", "<unk>", "\u2581", "<unk>", "\u200d", "<unk>", "\u200d", "<unk>", "\u200d", "<unk>", "\u2581", "<unk>", "\u200d", "<unk>", "\u200d", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u200d", "<unk>", "\u200d", "<unk>", "\u200d", "<unk>"],
      ids: [1, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31654, 0, 31928, 0, 31654, 0, 31928, 0, 31654, 0, 31928, 0, 31654, 0, 31928, 0, 31928, 0, 31654, 0, 31928, 0, 31928, 0, 31928, 0, 31654, 0, 31928, 0, 31928, 0, 31928, 0, 31654, 0, 31928, 0, 31928, 0, 31654, 0, 31654, 0, 31928, 0, 31928, 0, 31928, 0],
      decoded: "<s> <unk> <unk> <unk> <unk> <unk>\u200d<unk> <unk>\u200d<unk> <unk>\u200d<unk> <unk>\u200d<unk>\u200d<unk> <unk>\u200d<unk>\u200d<unk>\u200d<unk> <unk>\u200d<unk>\u200d<unk>\u200d<unk> <unk>\u200d<unk>\u200d<unk> <unk> <unk>\u200d<unk>\u200d<unk>\u200d<unk>",
    },
    BPE_SCORES_PRIORITY_1: {
      text: LLAMA_TEST_STRINGS.BPE_SCORES_PRIORITY_1,
      tokens: ["\u2581grabbed"],
      ids: [1, 3618],
      decoded: "<s> grabbed",
    },
    BPE_SCORES_PRIORITY_2: {
      text: LLAMA_TEST_STRINGS.BPE_SCORES_PRIORITY_2,
      tokens: ["\u2581", "\u2581grabbed"],
      ids: [1, 31654, 3618],
      decoded: "<s>  grabbed",
    },
    BPE_SCORES_PRIORITY_3: {
      text: LLAMA_TEST_STRINGS.BPE_SCORES_PRIORITY_3,
      tokens: ["\u2581", "\u2581", "\u2581", "\u2581", "\u2581", "\u2581", "\u2581", "\u2581", "\u2581", "\u2581", "\u2581", "\u2581grabbed"],
      ids: [1, 31654, 31654, 31654, 31654, 31654, 31654, 31654, 31654, 31654, 31654, 31654, 3618],
      decoded: "<s>            grabbed",
    },
    NEWLINE: {
      text: LLAMA_TEST_STRINGS.NEWLINE,
      tokens: ["\u2581", "\n"],
      ids: [1, 31654, 5],
      decoded: "<s> \n",
    },
    NEWLINE_WITH_LEADING_SPACE: {
      text: LLAMA_TEST_STRINGS.NEWLINE_WITH_LEADING_SPACE,
      tokens: ["\u2581", "\u2581", "\n"],
      ids: [1, 31654, 31654, 5],
      decoded: "<s>  \n",
    },
    TABS: {
      text: LLAMA_TEST_STRINGS.TABS,
      tokens: ["\u2581", "<unk>", "t", "ab", "s", "<unk>", "out", "\u2581here"],
      ids: [1, 31654, 0, 31665, 878, 31675, 0, 415, 3278],
      decoded: "<s> <unk>tabs<unk>out here",
    },
    NEWLINE_AND_TAB: {
      text: LLAMA_TEST_STRINGS.NEWLINE_AND_TAB,
      tokens: ["\u2581", "\n", "<unk>", "\n"],
      ids: [1, 31654, 5, 0, 5],
      decoded: "<s> \n<unk>\n",
    },
    CHINESE_LETTER: {
      text: LLAMA_TEST_STRINGS.CHINESE_LETTER,
      tokens: ["\u2581", "<unk>"],
      ids: [1, 31654, 0],
      decoded: "<s> <unk>",
    },
    EMOJIS_1: {
      text: LLAMA_TEST_STRINGS.EMOJIS_1,
      tokens: ["\u2581", "<unk>"],
      ids: [1, 31654, 0],
      decoded: "<s> <unk>",
    },
    EMOJIS_2: {
      text: LLAMA_TEST_STRINGS.EMOJIS_2,
      tokens: ["\u2581", "<unk>"],
      ids: [1, 31654, 0],
      decoded: "<s> <unk>",
    },
    EMOJIS_3: {
      text: LLAMA_TEST_STRINGS.EMOJIS_3,
      tokens: ["\u2581", "<unk>"],
      ids: [1, 31654, 0],
      decoded: "<s> <unk>",
    },
    PARAGRAPH: {
      text: LLAMA_TEST_STRINGS.PARAGRAPH,
      tokens: ["\u2581The", "\u2581l", "l", "ama", "\u2581", "(", "/", "\u02c8", "l", "\u0251", "\u02d0", "m", "\u0259", "/", ";", "\u2581", "<unk>", "Sp", "an", "ish", "\u2581pr", "on", "un", "ci", "ation", ":", "\u2581", "[", "\u02c8", "<unk>", "ama", "]", ")", "\u2581", "(", "L", "ama", "\u2581gl", "ama", ")", "\u2581is", "\u2581a", "\u2581d", "om", "est", "ic", "ated", "\u2581South", "\u2581American", "\u2581cam", "el", "id", ",", "\u2581wid", "ely", "\u2581used", "\u2581as", "\u2581a", "\u2581meat", "\u2581and", "\u2581pack", "\u2581animal", "\u2581by", "\u2581And", "e", "an", "\u2581c", "ult", "ures", "\u2581since", "\u2581the", "\u2581P", "re", "-", "C", "ol", "umb", "ian", "\u2581", "era", ".", "\u2581L", "l", "am", "as", "\u2581are", "\u2581social", "\u2581animals", "\u2581and", "\u2581live", "\u2581with", "\u2581others", "\u2581as", "\u2581a", "\u2581her", "d", ".", "\u2581Their", "\u2581wool", "\u2581is", "\u2581soft", "\u2581and", "\u2581contains", "\u2581only", "\u2581a", "\u2581small", "\u2581amount", "\u2581of", "\u2581l", "an", "ol", "in", ".", "[", "2", "]", "\u2581L", "l", "am", "as", "\u2581can", "\u2581learn", "\u2581simple", "\u2581tasks", "\u2581after", "\u2581a", "\u2581few", "\u2581rep", "et", "itions", ".", "\u2581When", "\u2581using", "\u2581a", "\u2581pack", ",", "\u2581they", "\u2581can", "\u2581carry", "\u2581about", "\u2581", "2", "5", "\u2581to", "\u2581", "3", "0", "%", "\u2581of", "\u2581their", "\u2581body", "\u2581weight", "\u2581for", "\u2581", "8", "\u2581to", "\u2581", "1", "3", "\u2581km", "\u2581", "(", "5", "\u2013", "8", "\u2581miles", ")", ".", "[", "3", "]", "\u2581The", "\u2581name", "\u2581l", "l", "ama", "\u2581", "(", "in", "\u2581the", "\u2581past", "\u2581also", "\u2581spell", "ed", '\u2581"', "l", "ama", '"', "\u2581or", '\u2581"', "gl", "ama", '"', ")", "\u2581was", "\u2581adop", "ted", "\u2581by", "\u2581E", "urope", "an", "\u2581sett", "l", "ers", "\u2581from", "\u2581n", "ative", "\u2581Per", "u", "v", "ians", ".", "[", "4", "]", "\u2581The", "\u2581an", "c", "est", "ors", "\u2581of", "\u2581l", "l", "am", "as", "\u2581are", "\u2581thought", "\u2581to", "\u2581have", "\u2581origin", "ated", "\u2581from", "\u2581the", "\u2581Great", "\u2581Pl", "ain", "s", "\u2581of", "\u2581North", "\u2581America", "\u2581about", "\u2581", "4", "0", "\u2581million", "\u2581years", "\u2581ago", ",", "\u2581and", "\u2581sub", "sequ", "ently", "\u2581m", "ig", "r", "ated", "\u2581to", "\u2581South", "\u2581America", "\u2581about", "\u2581three", "\u2581million", "\u2581years", "\u2581ago", "\u2581during", "\u2581the", "\u2581Great", "\u2581American", "\u2581Int", "er", "ch", "ange", ".", "\u2581By", "\u2581the", "\u2581end", "\u2581of", "\u2581the", "\u2581last", "\u2581ice", "\u2581age", "\u2581", "(", "1", "0", ",", "0", "0", "0", "\u2013", "1", "2", ",", "0", "0", "0", "\u2581years", "\u2581ago", ")", ",", "\u2581cam", "el", "ids", "\u2581were", "\u2581ext", "inct", "\u2581in", "\u2581North", "\u2581America", ".", "[", "3", "]", "\u2581As", "\u2581of", "\u2581", "2", "0", "0", "7", ",", "\u2581there", "\u2581were", "\u2581over", "\u2581seven", "\u2581million", "\u2581l", "l", "am", "as", "\u2581and", "\u2581al", "p", "ac", "as", "\u2581in", "\u2581South", "\u2581America", "\u2581and", "\u2581over", "\u2581", "1", "5", "8", ",", "0", "0", "0", "\u2581l", "l", "am", "as", "\u2581and", "\u2581", "1", "0", "0", ",", "0", "0", "0", "<unk>", "\u2581al", "p", "ac", "as", ",", "\u2581des", "ce", "nd", "ed", "\u2581from", "\u2581pro", "gen", "it", "ors", "\u2581import", "ed", "\u2581late", "\u2581in", "\u2581the", "\u2581", "2", "0", "th", "\u2581cent", "ury", ",", "\u2581in", "\u2581the", "\u2581United", "\u2581States", "\u2581and", "\u2581Can", "ada", ".", "[", "5", "]", "\u2581In", "\u2581A", "ym", "ara", "\u2581my", "th", "ology", ",", "\u2581l", "l", "am", "as", "\u2581are", "\u2581important", "\u2581be", "ings", ".", "\u2581The", "\u2581He", "aven", "ly", "\u2581L", "l", "ama", "\u2581is", "\u2581said", "\u2581to", "\u2581drink", "\u2581water", "\u2581from", "\u2581the", "\u2581ocean", "\u2581and", "\u2581ur", "in", "ates", "\u2581as", "\u2581it", "\u2581rains", ".", "[", "6", "]", "\u2581Acc", "ord", "ing", "\u2581to", "\u2581A", "ym", "ara", "\u2581es", "ch", "at", "ology", ",", "\u2581l", "l", "am", "as", "\u2581will", "\u2581return", "\u2581to", "\u2581the", "\u2581water", "\u2581spr", "ings", "\u2581and", "\u2581l", "ag", "oons", "\u2581where", "\u2581they", "\u2581come", "\u2581from", "\u2581at", "\u2581the", "\u2581end", "\u2581of", "\u2581time", ".", "[", "6", "]"],
      ids: [1, 147, 105, 31683, 4464, 31654, 32, 31753, 31774, 31683, 31813, 31779, 31687, 31781, 31753, 30, 31654, 0, 30106, 142, 531, 1823, 111, 367, 8762, 633, 29, 31654, 31778, 31774, 0, 4464, 31780, 33, 31654, 32, 31717, 4464, 1861, 4464, 33, 344, 48, 108, 120, 504, 515, 3062, 29052, 18424, 8829, 256, 153, 35, 20517, 2001, 2680, 488, 48, 9910, 83, 4314, 1448, 1015, 1736, 31660, 142, 103, 3441, 605, 13397, 70, 1629, 86, 7, 31739, 819, 4618, 1685, 31654, 7129, 34, 218, 31683, 235, 691, 617, 23632, 1707, 83, 5860, 249, 2905, 488, 48, 192, 31679, 34, 5290, 11964, 344, 3077, 83, 12959, 2859, 48, 1388, 7238, 251, 105, 142, 819, 81, 34, 31778, 13, 31780, 218, 31683, 235, 691, 645, 907, 16188, 22936, 1609, 48, 4505, 4706, 183, 29049, 34, 1354, 5247, 48, 4314, 35, 338, 645, 4923, 1096, 31654, 13, 16, 84, 31654, 14, 21, 10, 251, 626, 6011, 9152, 347, 31654, 19, 84, 31654, 12, 14, 29496, 31654, 32, 16, 31760, 19, 7843, 33, 34, 31778, 14, 31780, 147, 3516, 105, 31683, 4464, 31654, 32, 81, 70, 4829, 2320, 9948, 78, 245, 31683, 4464, 31690, 1187, 245, 686, 4464, 31690, 33, 139, 25228, 2490, 1015, 465, 25799, 142, 16405, 31683, 983, 825, 152, 12724, 24466, 31688, 31711, 26361, 34, 31778, 15, 31780, 147, 446, 31692, 504, 4166, 251, 105, 31683, 235, 691, 617, 1302, 84, 649, 7206, 3062, 825, 70, 27718, 12966, 588, 31675, 251, 26698, 27393, 1096, 31654, 15, 21, 23109, 3514, 17246, 35, 83, 5097, 17541, 19560, 114, 258, 31678, 3062, 84, 29052, 27393, 1096, 2765, 23109, 3514, 17246, 5823, 70, 27718, 18424, 25473, 98, 345, 3292, 34, 15498, 70, 1645, 251, 70, 6103, 2802, 13463, 31654, 32, 12, 21, 35, 21, 21, 21, 31760, 12, 13, 35, 21, 21, 21, 3514, 17246, 33, 35, 8829, 256, 16185, 579, 7522, 21465, 198, 26698, 27393, 34, 31778, 14, 31780, 1822, 251, 31654, 13, 21, 21, 18, 35, 478, 579, 1407, 20358, 23109, 105, 31683, 235, 691, 83, 789, 31694, 1324, 691, 198, 29052, 27393, 83, 1407, 31654, 12, 16, 19, 35, 21, 21, 21, 105, 31683, 235, 691, 83, 31654, 12, 21, 21, 35, 21, 21, 21, 0, 789, 31694, 1324, 691, 35, 3601, 215, 65, 78, 825, 2482, 8170, 93, 4166, 1777, 78, 5359, 198, 70, 31654, 13, 21, 1671, 11823, 11325, 35, 198, 70, 17562, 18843, 83, 3226, 19507, 34, 31778, 16, 31780, 2266, 231, 10586, 1362, 1286, 1671, 25316, 35, 105, 31683, 235, 691, 617, 2288, 233, 826, 34, 147, 264, 21794, 321, 218, 31683, 4464, 344, 309, 84, 4057, 1357, 825, 70, 5187, 83, 9947, 81, 4897, 488, 182, 24761, 34, 31778, 17, 31780, 28616, 4173, 127, 84, 231, 10586, 1362, 4469, 345, 122, 25316, 35, 105, 31683, 235, 691, 1214, 3520, 84, 70, 1357, 12312, 826, 83, 105, 762, 31431, 1930, 338, 1909, 825, 586, 70, 1645, 251, 470, 34, 31778, 17, 31780],
      decoded: '<s> The llama (/\u02c8l\u0251\u02d0m\u0259/; <unk>Spanish pronunciation: [\u02c8<unk>ama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the Pre-Columbian era. Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5\u20138 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4] The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000\u201312,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000<unk> alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5] In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology, llamas will return to the water springs and lagoons where they come from at the end of time.[6]',
    },
  },
};

const MAX_EXECUTION_TIME = 10_000;
export const CUSTOM_TESTS = () => {
  // Tests to ensure that no matter what, the correct tokenization is returned.
  // This is necessary since there are sometimes bugs in the transformers library.
  describe("hard-coded", () => {
    const TESTS = {
      "Xenova/llama-tokenizer": [
        // Test legacy compatibility
        {
          // legacy unset => legacy=true
          // NOTE: While incorrect, it is necessary to match legacy behaviour
          data: {
            "<s>\n": [1, 29871, 13],
          },
          legacy: null,
        },
        {
          // override legacy=true (same results as above)
          data: {
            "<s>\n": [1, 29871, 13],
          },
          legacy: true,
        },
        {
          // override legacy=false (fixed results)
          data: {
            "<s>\n": [1, 13],
          },
          legacy: false,
        },
      ],

      "Xenova/llama-tokenizer_new": [
        // legacy=false
        {
          data: {
            " </s> 1  2   3    4   ": [259, 2, 29871, 29896, 259, 29906, 1678, 29941, 268, 29946, 1678],
            "<s>\n": [1, 13],
            "</s>test</s>": [2, 1688, 2],
            " </s> test </s> ": [259, 2, 1243, 29871, 2, 29871],
            "A\n'll": [319, 13, 29915, 645],
            "Hey </s>. how are you": [18637, 29871, 2, 29889, 920, 526, 366],
            "  Hi  Hello  ": [259, 6324, 29871, 15043, 259],
          },
          reversible: true,
          legacy: null,
        },
        {
          // override legacy=true (incorrect results, but necessary to match legacy behaviour)
          data: {
            "<s>\n": [1, 29871, 13],
          },
          legacy: true,
        },
      ],

      // new serialization format (tokenizers >= 0.20.0)
      // BPE merges are now [string, string][] instead of string[]
      "Xenova/Llama-3.2-Tokenizer": [
        {
          data: {
            "hello world": [15339, 1917],
            " belirtilen": [120909],
          },
          reversible: true,
        },

        // Test ignore_merges=false
        {
          data: {
            "hello world": [15339, 1917],
            " belirtilen": [101664, 1678, 268],
          },
          reversible: true,
          override: (tokenizer) => {
            tokenizer.model.ignore_merges = false;
          },
        },
      ],
    };

    // Re-use the same tests for the llama2 tokenizer
    TESTS["Xenova/llama2-tokenizer"] = TESTS["Xenova/llama-tokenizer_new"];

    for (const [tokenizerName, test_data] of Object.entries(TESTS)) {
      it(
        tokenizerName,
        async () => {
          for (const { data, reversible, legacy, override } of test_data) {
            const tokenizer = await LlamaTokenizer.from_pretrained(tokenizerName, { legacy });
            if (override) {
              override(tokenizer);
            }
            for (const [text, expected] of Object.entries(data)) {
              const token_ids = tokenizer.encode(text, { add_special_tokens: false });
              expect(token_ids).toEqual(expected);

              // If reversible, test that decoding produces the original text
              if (reversible) {
                const decoded = tokenizer.decode(token_ids);
                expect(decoded).toEqual(text);
              }
            }
          }
        },
        MAX_EXECUTION_TIME,
      );
    }
  });
};
