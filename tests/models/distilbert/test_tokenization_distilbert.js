import { DistilBertTokenizer } from "../../../src/tokenizers.js";
import { BASE_TEST_STRINGS, BERT_TEST_STRINGS } from "../test_strings.js";

export const TOKENIZER_CLASS = DistilBertTokenizer;
export const TEST_CONFIG = {
  "Xenova/distilbert-base-cased-distilled-squad": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      tokens: ["How", "are", "you", "doing", "?"],
      ids: [101, 1731, 1132, 1128, 1833, 136, 102],
      decoded: "[CLS] How are you doing? [SEP]",
    },
    SIMPLE_WITH_PUNCTUATION: {
      text: BASE_TEST_STRINGS.SIMPLE_WITH_PUNCTUATION,
      tokens: ["You", "should", "'", "ve", "done", "this"],
      ids: [101, 1192, 1431, 112, 1396, 1694, 1142, 102],
      decoded: "[CLS] You should've done this [SEP]",
    },
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["01", "##23", "##45", "##6", "##7", "##8", "##9", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "100", "1000"],
      ids: [101, 5187, 22737, 21336, 1545, 1559, 1604, 1580, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 1275, 1620, 6087, 102],
      decoded: "[CLS] 0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000 [SEP]",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["The", "company", "was", "founded", "in", "2016", "."],
      ids: [101, 1109, 1419, 1108, 1771, 1107, 1446, 119, 102],
      decoded: "[CLS] The company was founded in 2016. [SEP]",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["A", "'", "ll", "!", "!", "to", "?", "'", "d", "'", "'", "d", "of", ",", "can", "'", "t", "."],
      ids: [101, 138, 112, 1325, 106, 106, 1106, 136, 112, 173, 112, 112, 173, 1104, 117, 1169, 112, 189, 119, 102],
      decoded: "[CLS] A'll!! to?'d'' d of, can't. [SEP]",
    },
    PYTHON_CODE: {
      text: BASE_TEST_STRINGS.PYTHON_CODE,
      tokens: ["def", "main", "(", ")", ":", "pass"],
      ids: [101, 19353, 1514, 113, 114, 131, 2789, 102],
      decoded: "[CLS] def main ( ) : pass [SEP]",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["let", "a", "=", "o", "##b", "##j", ".", "to", "##S", "##tring", "(", ")", ";", "to", "##S", "##tring", "(", ")", ";"],
      ids: [101, 1519, 170, 134, 184, 1830, 3361, 119, 1106, 1708, 28108, 113, 114, 132, 1106, 1708, 28108, 113, 114, 132, 102],
      decoded: "[CLS] let a = obj. toString ( ) ; toString ( ) ; [SEP]",
    },
    NEWLINES: {
      text: BASE_TEST_STRINGS.NEWLINES,
      tokens: ["This", "is", "a", "test", "."],
      ids: [101, 1188, 1110, 170, 2774, 119, 102],
      decoded: "[CLS] This is a test. [SEP]",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["UN", "##wan", "##t\u00e9", "##d", ",", "running"],
      ids: [101, 7414, 5491, 14608, 1181, 117, 1919, 102],
      decoded: "[CLS] UNwant\u00e9d, running [SEP]",
    },
    CONTROL_TOKENS: {
      text: BASE_TEST_STRINGS.CONTROL_TOKENS,
      tokens: ["123"],
      ids: [101, 13414, 102],
      decoded: "[CLS] 123 [SEP]",
    },
    HELLO_WORLD_TITLECASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_TITLECASE,
      tokens: ["Hello", "World"],
      ids: [101, 8667, 1291, 102],
      decoded: "[CLS] Hello World [SEP]",
    },
    HELLO_WORLD_LOWERCASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_LOWERCASE,
      tokens: ["hello", "world"],
      ids: [101, 19082, 1362, 102],
      decoded: "[CLS] hello world [SEP]",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u751f", "[UNK]", "[UNK]", "\u771f", "[UNK]", "[UNK]"],
      ids: [101, 1056, 100, 100, 1061, 100, 100, 102],
      decoded: "[CLS] \u751f [UNK] [UNK] \u771f [UNK] [UNK] [SEP]",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      tokens: ["leading", "space"],
      ids: [101, 2020, 2000, 102],
      decoded: "[CLS] leading space [SEP]",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["trailing", "space"],
      ids: [101, 13161, 2000, 102],
      decoded: "[CLS] trailing space [SEP]",
    },
    DOUBLE_SPACE: {
      text: BASE_TEST_STRINGS.DOUBLE_SPACE,
      tokens: ["Hi", "Hello"],
      ids: [101, 8790, 8667, 102],
      decoded: "[CLS] Hi Hello [SEP]",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["test", "$", "1", "R", "##2", "#", "3", "\u20ac", "##4", "\u00a3", "##5", "\u00a5", "##6", "[UNK]", "\u20b9", "##8", "\u20b1", "##9", "test"],
      ids: [101, 2774, 109, 122, 155, 1477, 108, 124, 836, 1527, 202, 1571, 203, 1545, 100, 838, 1604, 837, 1580, 2774, 102],
      decoded: "[CLS] test $ 1 R2 # 3 \u20ac4 \u00a35 \u00a56 [UNK] \u20b98 \u20b19 test [SEP]",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["I", "bought", "an", "apple", "for", "$", "1", ".", "00", "at", "the", "store", "."],
      ids: [101, 146, 3306, 1126, 12075, 1111, 109, 122, 119, 3135, 1120, 1103, 2984, 119, 102],
      decoded: "[CLS] I bought an apple for $ 1. 00 at the store. [SEP]",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["you", "\u2026"],
      ids: [101, 1128, 795, 102],
      decoded: "[CLS] you \u2026 [SEP]",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["you", "\u2026"],
      ids: [101, 1128, 795, 102],
      decoded: "[CLS] you \u2026 [SEP]",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["you", "\u2026", "you", "\u2026"],
      ids: [101, 1128, 795, 1128, 795, 102],
      decoded: "[CLS] you \u2026 you \u2026 [SEP]",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["weird", "[UNK]", "edge", "[UNK]", "case"],
      ids: [101, 6994, 100, 2652, 100, 1692, 102],
      decoded: "[CLS] weird [UNK] edge [UNK] case [SEP]",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      tokens: ["[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "."],
      ids: [101, 100, 100, 100, 100, 100, 119, 102],
      decoded: "[CLS] [UNK] [UNK] [UNK] [UNK] [UNK]. [SEP]",
    },
    POPULAR_EMOJIS: {
      text: BASE_TEST_STRINGS.POPULAR_EMOJIS,
      tokens: ["[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]"],
      ids: [101, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 102],
      decoded: "[CLS] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [SEP]",
    },
    MULTIBYTE_EMOJIS: {
      text: BASE_TEST_STRINGS.MULTIBYTE_EMOJIS,
      tokens: ["[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]"],
      ids: [101, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 102],
      decoded: "[CLS] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [SEP]",
    },
  },
  "Xenova/distilbert-base-uncased-finetuned-sst-2-english": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      tokens: ["how", "are", "you", "doing", "?"],
      ids: [101, 2129, 2024, 2017, 2725, 1029, 102],
      decoded: "[CLS] how are you doing? [SEP]",
    },
    SIMPLE_WITH_PUNCTUATION: {
      text: BASE_TEST_STRINGS.SIMPLE_WITH_PUNCTUATION,
      tokens: ["you", "should", "'", "ve", "done", "this"],
      ids: [101, 2017, 2323, 1005, 2310, 2589, 2023, 102],
      decoded: "[CLS] you should've done this [SEP]",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["the", "company", "was", "founded", "in", "2016", "."],
      ids: [101, 1996, 2194, 2001, 2631, 1999, 2355, 1012, 102],
      decoded: "[CLS] the company was founded in 2016. [SEP]",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["a", "'", "ll", "!", "!", "to", "?", "'", "d", "'", "'", "d", "of", ",", "can", "'", "t", "."],
      ids: [101, 1037, 1005, 2222, 999, 999, 2000, 1029, 1005, 1040, 1005, 1005, 1040, 1997, 1010, 2064, 1005, 1056, 1012, 102],
      decoded: "[CLS] a'll!! to?'d'' d of, can't. [SEP]",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["let", "a", "=", "ob", "##j", ".", "to", "##st", "##ring", "(", ")", ";", "to", "##st", "##ring", "(", ")", ";"],
      ids: [101, 2292, 1037, 1027, 27885, 3501, 1012, 2000, 3367, 4892, 1006, 1007, 1025, 2000, 3367, 4892, 1006, 1007, 1025, 102],
      decoded: "[CLS] let a = obj. tostring ( ) ; tostring ( ) ; [SEP]",
    },
    NEWLINES: {
      text: BASE_TEST_STRINGS.NEWLINES,
      tokens: ["this", "is", "a", "test", "."],
      ids: [101, 2023, 2003, 1037, 3231, 1012, 102],
      decoded: "[CLS] this is a test. [SEP]",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["unwanted", ",", "running"],
      ids: [101, 18162, 1010, 2770, 102],
      decoded: "[CLS] unwanted, running [SEP]",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u751f", "[UNK]", "\u7684", "\u771f", "[UNK]", "[UNK]"],
      ids: [101, 1910, 100, 1916, 1921, 100, 100, 102],
      decoded: "[CLS] \u751f [UNK] \u7684 \u771f [UNK] [UNK] [SEP]",
    },
    DOUBLE_SPACE: {
      text: BASE_TEST_STRINGS.DOUBLE_SPACE,
      tokens: ["hi", "hello"],
      ids: [101, 7632, 7592, 102],
      decoded: "[CLS] hi hello [SEP]",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["test", "$", "1", "r", "##2", "#", "3", "\u20ac", "##4", "\u00a35", "\u00a5", "##6", "[UNK]", "\u20b9", "##8", "\u20b1", "##9", "test"],
      ids: [101, 3231, 1002, 1015, 1054, 2475, 1001, 1017, 1574, 2549, 27813, 1071, 2575, 100, 1576, 2620, 1575, 2683, 3231, 102],
      decoded: "[CLS] test $ 1 r2 # 3 \u20ac4 \u00a35 \u00a56 [UNK] \u20b98 \u20b19 test [SEP]",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["i", "bought", "an", "apple", "for", "$", "1", ".", "00", "at", "the", "store", "."],
      ids: [101, 1045, 4149, 2019, 6207, 2005, 1002, 1015, 1012, 4002, 2012, 1996, 3573, 1012, 102],
      decoded: "[CLS] i bought an apple for $ 1. 00 at the store. [SEP]",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["weird", "\uff5e", "edge", "\uff5e", "case"],
      ids: [101, 6881, 1995, 3341, 1995, 2553, 102],
      decoded: "[CLS] weird \uff5e edge \uff5e case [SEP]",
    },
  },
  "Xenova/distiluse-base-multilingual-cased-v2": {
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["012", "##34", "##5", "##6", "##7", "##8", "##9", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "100", "1000"],
      ids: [101, 69878, 78301, 11166, 11211, 11305, 11396, 11373, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 10150, 10407, 12186, 102],
      decoded: "[CLS] 0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000 [SEP]",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["let", "a", "=", "ob", "##j", ".", "to", "##S", "##trin", "##g", "(", ")", ";", "to", "##S", "##trin", "##g", "(", ")", ";"],
      ids: [101, 13595, 169, 134, 17339, 10418, 119, 10114, 10731, 109163, 10240, 113, 114, 132, 10114, 10731, 109163, 10240, 113, 114, 132, 102],
      decoded: "[CLS] let a = obj. toString ( ) ; toString ( ) ; [SEP]",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["UN", "##want", "##\u00e9d", ",", "running"],
      ids: [101, 26578, 104216, 84193, 117, 18020, 102],
      decoded: "[CLS] UNwant\u00e9d, running [SEP]",
    },
    HELLO_WORLD_LOWERCASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_LOWERCASE,
      tokens: ["hell", "##o", "world"],
      ids: [101, 61694, 10133, 11356, 102],
      decoded: "[CLS] hello world [SEP]",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u751f", "\u6d3b", "\u7684", "\u771f", "\u8c1b", "\u662f"],
      ids: [101, 5600, 4978, 5718, 5769, 7378, 4380, 102],
      decoded: "[CLS] \u751f \u6d3b \u7684 \u771f \u8c1b \u662f [SEP]",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["trail", "##ing", "space"],
      ids: [101, 56559, 10230, 16199, 102],
      decoded: "[CLS] trailing space [SEP]",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["test", "$", "1", "R2", "#", "3", "\u20ac", "##4", "\u00a3", "##5", "\u00a5", "##6", "[UNK]", "\u20b9", "##8", "[UNK]", "test"],
      ids: [101, 15839, 109, 122, 94000, 108, 124, 1775, 11011, 201, 11166, 202, 11211, 100, 1776, 11396, 100, 15839, 102],
      decoded: "[CLS] test $ 1 R2 # 3 \u20ac4 \u00a35 \u00a56 [UNK] \u20b98 [UNK] test [SEP]",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["I", "bought", "an", "app", "##le", "for", "$", "1", ".", "00", "at", "the", "store", "."],
      ids: [101, 146, 28870, 10151, 72894, 10284, 10142, 109, 122, 119, 11025, 10160, 10105, 13708, 119, 102],
      decoded: "[CLS] I bought an apple for $ 1. 00 at the store. [SEP]",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["you", "[UNK]"],
      ids: [101, 13028, 100, 102],
      decoded: "[CLS] you [UNK] [SEP]",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["you", "[UNK]"],
      ids: [101, 13028, 100, 102],
      decoded: "[CLS] you [UNK] [SEP]",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["you", "[UNK]", "you", "[UNK]"],
      ids: [101, 13028, 100, 13028, 100, 102],
      decoded: "[CLS] you [UNK] you [UNK] [SEP]",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["wei", "##rd", "\uff5e", "edge", "\uff5e", "case"],
      ids: [101, 86981, 12023, 10096, 30599, 10096, 13474, 102],
      decoded: "[CLS] weird \uff5e edge \uff5e case [SEP]",
    },
  },
  // `model.type` field missing in tokenizer.json
  "distilbert/distilbert-base-multilingual-cased": {
    CHINESE_LATIN_MIXED: {
      text: BERT_TEST_STRINGS.CHINESE_LATIN_MIXED,
      tokens: ["ah", "\u535a", "\u63a8", "z", "##z"],
      ids: [101, 69863, 2684, 4163, 194, 10305, 102],
      decoded: "[CLS] ah \u535a \u63a8 zz [SEP]",
    },
  },
};
