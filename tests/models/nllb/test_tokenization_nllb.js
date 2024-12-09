import { NllbTokenizer } from "../../../src/tokenizers.js";
import { BASE_TEST_STRINGS } from "../test_strings.js";

export const TOKENIZER_CLASS = NllbTokenizer;
export const TEST_CONFIG = {
  "Xenova/nllb-200-distilled-600M": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      tokens: ["\u2581How", "\u2581are", "\u2581you", "\u2581doing", "?"],
      ids: [256047, 13374, 2442, 1259, 34512, 248130, 2],
      decoded: "eng_Latn How are you doing?</s>",
    },
    SIMPLE_WITH_PUNCTUATION: {
      text: BASE_TEST_STRINGS.SIMPLE_WITH_PUNCTUATION,
      tokens: ["\u2581You", "\u2581should", "'", "ve", "\u2581done", "\u2581this"],
      ids: [256047, 3555, 12516, 248116, 279, 27236, 3423, 2],
      decoded: "eng_Latn You should've done this</s>",
    },
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["\u25810", "123", "45", "67", "89", "\u25810", "\u25811", "\u25812", "\u25813", "\u25814", "\u25815", "\u25816", "\u25817", "\u25818", "\u25819", "\u258110", "\u2581100", "\u25811000"],
      ids: [256047, 4097, 232903, 25497, 37462, 42763, 4097, 94, 140, 315, 436, 481, 617, 757, 799, 855, 772, 3037, 18041, 2],
      decoded: "eng_Latn 0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000</s>",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["\u2581The", "\u2581company", "\u2581was", "\u2581found", "ed", "\u2581in", "\u25812016."],
      ids: [256047, 1617, 32796, 1398, 26710, 76, 108, 31889, 2],
      decoded: "eng_Latn The company was founded in 2016.</s>",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["\u2581A", "\u2581'", "ll", "\u2581!!", "to", "?'", "d", "'", "'", "d", "\u2581of", ",", "\u2581can", "'", "t", "."],
      ids: [256047, 70, 238, 1015, 12434, 208, 7358, 248072, 248116, 248116, 248072, 452, 248079, 2125, 248116, 248065, 248075, 2],
      decoded: "eng_Latn A 'll!!to?'d''d of, can't.</s>",
    },
    PYTHON_CODE: {
      text: BASE_TEST_STRINGS.PYTHON_CODE,
      tokens: ["\u2581def", "\u2581main", "(", "):", "\u2581pass"],
      ids: [256047, 9274, 8385, 248168, 9481, 5800, 2],
      decoded: "eng_Latn def main(): pass</s>",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["\u2581let", "\u2581a", "\u2581=", "\u2581ob", "j", ".", "to", "Str", "ing", "(", ");", "\u2581to", "Str", "ing", "(", ");"],
      ids: [256047, 3190, 9, 5636, 859, 248086, 248075, 208, 134293, 87, 248168, 12387, 202, 134293, 87, 248168, 12387, 2],
      decoded: "eng_Latn let a = obj.toString(); toString();</s>",
    },
    NEWLINES: {
      text: BASE_TEST_STRINGS.NEWLINES,
      tokens: ["\u2581This", "\u2581is", "\u2581a", "\u2581test", "."],
      ids: [256047, 9680, 248, 9, 7356, 248075, 2],
      decoded: "eng_Latn This is a test.</s>",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["\u2581UN", "want", "\u00e9d", ",", "run", "ning"],
      ids: [256047, 16297, 41691, 11317, 248079, 8464, 888, 2],
      decoded: "eng_Latn UNwant\u00e9d,running</s>",
    },
    CONTROL_TOKENS: {
      text: BASE_TEST_STRINGS.CONTROL_TOKENS,
      tokens: ["\u25811", "<unk>", "2", "\u25813"],
      ids: [256047, 94, 3, 248147, 315, 2],
      decoded: "eng_Latn 1<unk>2 3</s>",
    },
    HELLO_WORLD_TITLECASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_TITLECASE,
      tokens: ["\u2581Hello", "\u2581World"],
      ids: [256047, 94124, 13855, 2],
      decoded: "eng_Latn Hello World</s>",
    },
    HELLO_WORLD_LOWERCASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_LOWERCASE,
      tokens: ["\u2581hello", "\u2581world"],
      ids: [256047, 133863, 15697, 2],
      decoded: "eng_Latn hello world</s>",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u2581\u751f\u6d3b", "\u7684", "\u771f", "<unk>", "\u662f"],
      ids: [256047, 182892, 248506, 249573, 3, 249221, 2],
      decoded: "eng_Latn \u751f\u6d3b\u7684\u771f<unk>\u662f</s>",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      tokens: ["\u2581leading", "\u2581space"],
      ids: [256047, 151175, 72147, 2],
      decoded: "eng_Latn leading space</s>",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["\u2581tra", "iling", "\u2581space", "\u2581"],
      ids: [256047, 1372, 21263, 72147, 248059, 2],
      decoded: "eng_Latn trailing space </s>",
    },
    DOUBLE_SPACE: {
      text: BASE_TEST_STRINGS.DOUBLE_SPACE,
      tokens: ["\u2581Hi", "\u2581Hello"],
      ids: [256047, 2867, 94124, 2],
      decoded: "eng_Latn Hi Hello</s>",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["\u2581test", "\u2581$1", "\u2581R", "2", "\u2581#3", "\u2581\u20ac", "4", "\u2581\u00a3", "5", "\u2581", "\u00a5", "6", "\u2581", "<unk>", "7", "\u2581", "\u20b9", "8", "\u2581", "<unk>", "9", "\u2581test"],
      ids: [256047, 7356, 68462, 250, 248147, 186447, 22935, 248215, 25400, 248210, 248059, 252351, 248262, 248059, 3, 248283, 248059, 254867, 248268, 248059, 3, 248212, 7356, 2],
      decoded: "eng_Latn test $1 R2 #3 \u20ac4 \u00a35 \u00a56 <unk>7 \u20b98 <unk>9 test</s>",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["\u2581I", "\u2581bought", "\u2581an", "\u2581apple", "\u2581for", "\u2581$", "1.", "00", "\u2581at", "\u2581the", "\u2581store", "."],
      ids: [256047, 117, 177233, 111, 203152, 351, 4589, 3044, 460, 230, 349, 21087, 248075, 2],
      decoded: "eng_Latn I bought an apple for $1.00 at the store.</s>",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["\u2581you", "...", "\u2581"],
      ids: [256047, 1259, 284, 248059, 2],
      decoded: "eng_Latn you... </s>",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["\u2581you", "...", "\u2581"],
      ids: [256047, 1259, 284, 248059, 2],
      decoded: "eng_Latn you... </s>",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["\u2581you", "...", "\u2581you", "...", "\u2581"],
      ids: [256047, 1259, 284, 1259, 284, 248059, 2],
      decoded: "eng_Latn you... you... </s>",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["\u2581weird", "\u2581", "<unk>", "\u2581ed", "ge", "\u2581", "<unk>", "\u2581case"],
      ids: [256047, 197348, 248059, 3, 1074, 479, 248059, 3, 23555, 2],
      decoded: "eng_Latn weird <unk> edge <unk> case</s>",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      tokens: ["\u2581This", "\u2581is", "\u2581a", "\u2581test", "\u2581."],
      ids: [256047, 9680, 248, 9, 7356, 81, 2],
      decoded: "eng_Latn This is a test.</s>",
    },
    POPULAR_EMOJIS: {
      text: BASE_TEST_STRINGS.POPULAR_EMOJIS,
      tokens: ["\u2581\ud83d\ude02", "\u2581", "\ud83d\udc4d", "\u2581", "\ud83e\udd23", "\u2581\ud83d\ude0d", "\u2581", "\ud83d\ude2d", "\u2581", "<unk>", "\u2581", "\ud83d\ude4f", "\u2581", "\ud83d\ude0a", "\u2581", "\ud83d\udd25", "\u2581", "\ud83d\ude01", "\u2581", "\ud83d\ude05", "\u2581", "\ud83e\udd17", "\u2581\ud83d\ude06", "\u2581", "\ud83d\udc4f", "\u2581\u2764", "\ufe0f", "\u2581", "\ud83d\udc9c", "\u2581", "\ud83d\udc9a", "\u2581", "<unk>", "\u2581", "\ud83d\udc99", "\u2581", "<unk>", "\u2581", "\ud83d\ude0e", "\u2581", "\ud83d\udc4c", "\u2581", "<unk>", "\u2581", "\ud83d\udcaa", "\u2581", "\u2728", "\u2581", "\ud83d\udc49", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "\ud83d\ude48", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "\ud83d\udc47", "\u2581", "<unk>", "\u2581", "\u2705", "\u2581", "\ud83c\udf81", "\u2581", "<unk>", "\u2581", "\ud83c\udf38", "\u2581", "<unk>"],
      ids: [256047, 104709, 248059, 253416, 248059, 253516, 241830, 248059, 253476, 248059, 3, 248059, 253443, 248059, 253515, 248059, 254402, 248059, 253288, 248059, 253776, 248059, 255232, 147677, 248059, 255420, 82495, 251759, 248059, 255742, 248059, 255949, 248059, 3, 248059, 255649, 248059, 3, 248059, 254297, 248059, 254723, 248059, 3, 248059, 255515, 248059, 254957, 248059, 253985, 248059, 3, 248059, 3, 248059, 3, 248059, 255855, 248059, 3, 248059, 3, 248059, 255354, 248059, 3, 248059, 254268, 248059, 255879, 248059, 3, 248059, 255952, 248059, 3, 2],
      decoded: "eng_Latn \ud83d\ude02 \ud83d\udc4d \ud83e\udd23 \ud83d\ude0d \ud83d\ude2d <unk> \ud83d\ude4f \ud83d\ude0a \ud83d\udd25 \ud83d\ude01 \ud83d\ude05 \ud83e\udd17 \ud83d\ude06 \ud83d\udc4f \u2764\ufe0f \ud83d\udc9c \ud83d\udc9a <unk> \ud83d\udc99 <unk> \ud83d\ude0e \ud83d\udc4c <unk> \ud83d\udcaa \u2728 \ud83d\udc49 <unk> <unk> <unk> \ud83d\ude48 <unk> <unk> \ud83d\udc47 <unk> \u2705 \ud83c\udf81 <unk> \ud83c\udf38 <unk></s>",
    },
    MULTIBYTE_EMOJIS: {
      text: BASE_TEST_STRINGS.MULTIBYTE_EMOJIS,
      tokens: ["\u2581", "\u2728", "\u2581", "\ud83e\udd17", "\u2581", "\ud83d\udc41", "\ufe0f", "\u2581", "<unk>", "\ud83c\udffb", "\u2581", "<unk>", "\u2581", "\u2642", "\ufe0f", "\u2581", "<unk>", "\ud83c\udffb", "\u2581", "\u2642", "\u2581", "<unk>", "\ud83c\udffb", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581\u2764", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\u2581", "<unk>", "\ud83c\udffb", "\u2581", "<unk>", "\u2581", "<unk>", "\ud83c\udffb", "\u2581", "<unk>", "\u2581", "<unk>", "\ud83c\udffb", "\u2581\u2764", "\ufe0f", "\u2581", "<unk>", "\u2581", "<unk>", "\ud83c\udffc"],
      ids: [256047, 248059, 254957, 248059, 255232, 248059, 255123, 251759, 248059, 3, 254422, 248059, 3, 248059, 255331, 251759, 248059, 3, 254422, 248059, 255331, 248059, 3, 254422, 248059, 3, 248059, 3, 248059, 3, 248059, 3, 248059, 3, 82495, 248059, 3, 248059, 3, 248059, 3, 248059, 3, 248059, 3, 248059, 3, 248059, 3, 254422, 248059, 3, 248059, 3, 254422, 248059, 3, 248059, 3, 254422, 82495, 251759, 248059, 3, 248059, 3, 255832, 2],
      decoded: "eng_Latn \u2728 \ud83e\udd17 \ud83d\udc41\ufe0f <unk>\ud83c\udffb <unk> \u2642\ufe0f <unk>\ud83c\udffb \u2642 <unk>\ud83c\udffb <unk> <unk> <unk> <unk> <unk> \u2764 <unk> <unk> <unk> <unk> <unk> <unk> <unk>\ud83c\udffb <unk> <unk>\ud83c\udffb <unk> <unk>\ud83c\udffb \u2764\ufe0f <unk> <unk>\ud83c\udffc</s>",
    },
  },
};
