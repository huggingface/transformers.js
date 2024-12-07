import { T5Tokenizer } from "../../../src/tokenizers.js";
import { BASE_TEST_STRINGS, SENTENCEPIECE_TEST_STRINGS, T5_TEST_STRINGS } from "../test_strings.js";

export const TOKENIZER_CLASS = T5Tokenizer;
export const TEST_CONFIG = {
  "Xenova/t5-small": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      tokens: ["\u2581How", "\u2581are", "\u2581you", "\u2581doing", "?"],
      ids: [571, 33, 25, 692, 58, 1],
      decoded: "How are you doing?</s>",
    },
    SIMPLE_WITH_PUNCTUATION: {
      text: BASE_TEST_STRINGS.SIMPLE_WITH_PUNCTUATION,
      tokens: ["\u2581You", "\u2581should", "'", "ve", "\u2581done", "\u2581this"],
      ids: [148, 225, 31, 162, 612, 48, 1],
      decoded: "You should've done this</s>",
    },
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["\u258101", "23", "45", "67", "89", "\u2581", "0", "\u25811", "\u25812", "\u25813", "\u25814", "\u25815", "\u25816", "\u25817", "\u25818", "\u25819", "\u258110", "\u2581100", "\u25811000"],
      ids: [7088, 2773, 2128, 3708, 3914, 3, 632, 209, 204, 220, 314, 305, 431, 489, 505, 668, 335, 910, 5580, 1],
      decoded: "0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000</s>",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["\u2581The", "\u2581company", "\u2581was", "\u2581founded", "\u2581in", "\u25812016."],
      ids: [37, 349, 47, 5710, 16, 4619, 1],
      decoded: "The company was founded in 2016.</s>",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["\u2581A", "\u2581", "'", "ll", "\u2581", "!!", "to", "?", "'", "d", "'", "'", "d", "\u2581of", ",", "\u2581can", "'", "t", "."],
      ids: [71, 3, 31, 195, 3, 1603, 235, 58, 31, 26, 31, 31, 26, 13, 6, 54, 31, 17, 5, 1],
      decoded: "A 'll!!to?'d''d of, can't.</s>",
    },
    PYTHON_CODE: {
      text: BASE_TEST_STRINGS.PYTHON_CODE,
      tokens: ["\u2581de", "f", "\u2581main", "()", ":", "\u2581pass"],
      ids: [20, 89, 711, 9960, 10, 1903, 1],
      decoded: "def main(): pass</s>",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["\u2581let", "\u2581", "a", "\u2581=", "\u2581", "o", "b", "j", ".", "to", "Str", "ing", "()", ";", "\u2581to", "Str", "ing", "()", ";"],
      ids: [752, 3, 9, 3274, 3, 32, 115, 354, 5, 235, 11500, 53, 9960, 117, 12, 11500, 53, 9960, 117, 1],
      decoded: "let a = obj.toString(); toString();</s>",
    },
    NEWLINES: {
      text: BASE_TEST_STRINGS.NEWLINES,
      tokens: ["\u2581This", "\u2581is", "\u2581", "a", "\u2581test", "."],
      ids: [100, 19, 3, 9, 794, 5, 1],
      decoded: "This is a test.</s>",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["\u2581UN", "wan", "t\u00e9", "d", ",", "running"],
      ids: [4417, 3877, 2229, 26, 6, 24549, 1],
      decoded: "UNwant\u00e9d,running</s>",
    },
    CONTROL_TOKENS: {
      text: BASE_TEST_STRINGS.CONTROL_TOKENS,
      tokens: ["\u25811", "\u0000", "2", "\u25813"],
      ids: [209, 2, 357, 220, 1],
      decoded: "1<unk>2 3</s>",
    },
    HELLO_WORLD_TITLECASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_TITLECASE,
      tokens: ["\u2581Hello", "\u2581World"],
      ids: [8774, 1150, 1],
      decoded: "Hello World</s>",
    },
    HELLO_WORLD_LOWERCASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_LOWERCASE,
      tokens: ["\u2581hello", "\u2581world"],
      ids: [21820, 296, 1],
      decoded: "hello world</s>",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u2581", "\u751f\u6d3b\u7684\u771f\u8c1b\u662f"],
      ids: [3, 2, 1],
      decoded: "<unk></s>",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      tokens: ["\u2581leading", "\u2581space"],
      ids: [1374, 628, 1],
      decoded: "leading space</s>",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["\u2581trail", "ing", "\u2581space"],
      ids: [5032, 53, 628, 1],
      decoded: "trailing space</s>",
    },
    DOUBLE_SPACE: {
      text: BASE_TEST_STRINGS.DOUBLE_SPACE,
      tokens: ["\u2581Hi", "\u2581Hello"],
      ids: [2018, 8774, 1],
      decoded: "Hi Hello</s>",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["\u2581test", "\u2581$1", "\u2581R", "2", "\u2581#3", "\u2581\u20ac", "4", "\u2581\u00a35", "\u2581", "\u00a5", "6", "\u2581", "\u20a3", "7", "\u2581", "\u20b9", "8", "\u2581", "\u20b1", "9", "\u2581test"],
      ids: [794, 1970, 391, 357, 20206, 3416, 591, 23978, 3, 2, 948, 3, 2, 940, 3, 2, 927, 3, 2, 1298, 794, 1],
      decoded: "test $1 R2 #3 \u20ac4 \u00a35 <unk>6 <unk>7 <unk>8 <unk>9 test</s>",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["\u2581I", "\u2581bought", "\u2581an", "\u2581apple", "\u2581for", "\u2581$1", ".00", "\u2581at", "\u2581the", "\u2581store", "."],
      ids: [27, 2944, 46, 8947, 21, 1970, 4200, 44, 8, 1078, 5, 1],
      decoded: "I bought an apple for $1.00 at the store.</s>",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["\u2581you", "..."],
      ids: [25, 233, 1],
      decoded: "you...</s>",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["\u2581you", "..."],
      ids: [25, 233, 1],
      decoded: "you...</s>",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["\u2581you", "...", "\u2581you", "..."],
      ids: [25, 233, 25, 233, 1],
      decoded: "you... you...</s>",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["\u2581weird", "\u2581", "\uff5e", "\u2581edge", "\u2581", "\uff5e", "\u2581case"],
      ids: [10088, 3, 2, 3023, 3, 2, 495, 1],
      decoded: "weird <unk> edge <unk> case</s>",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      tokens: ["\u2581This", "\u2581is", "\u2581", "a", "\u2581test", "\u2581", "."],
      ids: [100, 19, 3, 9, 794, 3, 5, 1],
      decoded: "This is a test.</s>",
    },
    POPULAR_EMOJIS: {
      text: BASE_TEST_STRINGS.POPULAR_EMOJIS,
      tokens: ["\u2581", "\ud83d\ude02", "\u2581", "\ud83d\udc4d", "\u2581", "\ud83e\udd23", "\u2581", "\ud83d\ude0d", "\u2581", "\ud83d\ude2d", "\u2581", "\ud83c\udf89", "\u2581", "\ud83d\ude4f", "\u2581", "\ud83d\ude0a", "\u2581", "\ud83d\udd25", "\u2581", "\ud83d\ude01", "\u2581", "\ud83d\ude05", "\u2581", "\ud83e\udd17", "\u2581", "\ud83d\ude06", "\u2581", "\ud83d\udc4f", "\u2581", "\u2764\ufe0f", "\u2581", "\ud83d\udc9c", "\u2581", "\ud83d\udc9a", "\u2581", "\ud83d\udc97", "\u2581", "\ud83d\udc99", "\u2581", "\ud83d\udda4", "\u2581", "\ud83d\ude0e", "\u2581", "\ud83d\udc4c", "\u2581", "\ud83e\udd73", "\u2581", "\ud83d\udcaa", "\u2581", "\u2728", "\u2581", "\ud83d\udc49", "\u2581", "\ud83d\udc40", "\u2581", "\ud83d\udcaf", "\u2581", "\ud83c\udf88", "\u2581", "\ud83d\ude48", "\u2581", "\ud83d\ude4c", "\u2581", "\ud83d\udc80", "\u2581", "\ud83d\udc47", "\u2581", "\ud83d\udc4b", "\u2581", "\u2705", "\u2581", "\ud83c\udf81", "\u2581", "\ud83c\udf1e", "\u2581", "\ud83c\udf38", "\u2581", "\ud83d\udcb0"],
      ids: [3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 1],
      decoded: "<unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk></s>",
    },
    MULTIBYTE_EMOJIS: {
      text: BASE_TEST_STRINGS.MULTIBYTE_EMOJIS,
      tokens: ["\u2581", "\u2728", "\u2581", "\ud83e\udd17", "\u2581", "\ud83d\udc41\ufe0f", "\u2581", "\ud83d\udc71\ud83c\udffb", "\u2581", "\ud83d\udd75", "\u2581", "\u2642\ufe0f", "\u2581", "\ud83e\uddd9\ud83c\udffb", "\u2581", "\u2642", "\u2581", "\ud83d\udc68\ud83c\udffb", "\u2581", "\ud83c\udf3e", "\u2581", "\ud83e\uddd1", "\u2581", "\ud83e\udd1d", "\u2581", "\ud83e\uddd1", "\u2581", "\ud83d\udc69", "\u2581", "\u2764", "\u2581", "\ud83d\udc8b", "\u2581", "\ud83d\udc68", "\u2581", "\ud83d\udc69", "\u2581", "\ud83d\udc69", "\u2581", "\ud83d\udc67", "\u2581", "\ud83d\udc66", "\u2581", "\ud83e\uddd1\ud83c\udffb", "\u2581", "\ud83e\udd1d", "\u2581", "\ud83e\uddd1\ud83c\udffb", "\u2581", "\ud83c\udff4\udb40\udc67\udb40\udc62\udb40\udc65\udb40\udc6e\udb40\udc67\udb40\udc7f", "\u2581", "\ud83d\udc68\ud83c\udffb", "\u2581", "\u2764\ufe0f", "\u2581", "\ud83d\udc8b", "\u2581", "\ud83d\udc68\ud83c\udffc"],
      ids: [3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 1],
      decoded: "<unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk></s>",
    },
    SPECIAL_WITH_TRAILING_WHITESPACE: {
      text: SENTENCEPIECE_TEST_STRINGS.SPECIAL_WITH_TRAILING_WHITESPACE,
      tokens: ["\u2581", "<", "s", ">"],
      ids: [3, 2, 7, 3155, 1],
      decoded: "<unk>s></s>",
    },
    SPECIAL_SURROUNDED_BY_WHITESPACE: {
      text: SENTENCEPIECE_TEST_STRINGS.SPECIAL_SURROUNDED_BY_WHITESPACE,
      tokens: ["</s>", "\u2581test", "</s>"],
      ids: [1, 794, 1, 1],
      decoded: "</s> test</s></s>",
    },
    SPECIAL_NO_WHITESPACE: {
      text: SENTENCEPIECE_TEST_STRINGS.SPECIAL_NO_WHITESPACE,
      tokens: ["</s>", "\u2581test", "</s>"],
      ids: [1, 794, 1, 1],
      decoded: "</s> test</s></s>",
    },
    PREPEND_SCHEME: {
      text: T5_TEST_STRINGS.PREPEND_SCHEME,
      tokens: ["\u2581Hey", "</s>", "\u2581", ".", "\u2581how", "\u2581are", "\u2581you"],
      ids: [9459, 1, 3, 5, 149, 33, 25, 1],
      decoded: "Hey</s>. how are you</s>",
    },
  },
  "Xenova/t5-tokenizer-new": {
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["\u2581trail", "ing", "\u2581space", "\u2581"],
      ids: [5032, 53, 628, 3, 1],
      decoded: "trailing space </s>",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["\u2581you", "...", "\u2581"],
      ids: [25, 233, 3, 1],
      decoded: "you... </s>",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["\u2581you", "...", "\u2581"],
      ids: [25, 233, 3, 1],
      decoded: "you... </s>",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["\u2581you", "...", "\u2581you", "...", "\u2581"],
      ids: [25, 233, 25, 233, 3, 1],
      decoded: "you... you... </s>",
    },
    SPECIAL_WITH_TRAILING_WHITESPACE: {
      text: SENTENCEPIECE_TEST_STRINGS.SPECIAL_WITH_TRAILING_WHITESPACE,
      tokens: ["\u2581", "<", "s", ">", "\u2581"],
      ids: [3, 2, 7, 3155, 3, 1],
      decoded: "<unk>s> </s>",
    },
    SPECIAL_SURROUNDED_BY_WHITESPACE: {
      text: SENTENCEPIECE_TEST_STRINGS.SPECIAL_SURROUNDED_BY_WHITESPACE,
      tokens: ["\u2581", "</s>", "\u2581test", "\u2581", "</s>", "\u2581"],
      ids: [3, 1, 794, 3, 1, 3, 1],
      decoded: "</s> test </s> </s>",
    },
    SPECIAL_NO_WHITESPACE: {
      text: SENTENCEPIECE_TEST_STRINGS.SPECIAL_NO_WHITESPACE,
      tokens: ["</s>", "test", "</s>"],
      ids: [1, 4377, 1, 1],
      decoded: "</s>test</s></s>",
    },
    PREPEND_SCHEME: {
      text: T5_TEST_STRINGS.PREPEND_SCHEME,
      tokens: ["\u2581Hey", "\u2581", "</s>", ".", "\u2581how", "\u2581are", "\u2581you"],
      ids: [9459, 3, 1, 5, 149, 33, 25, 1],
      decoded: "Hey </s>. how are you</s>",
    },
  },
  "Xenova/LaMini-Flan-T5-783M": {
    PREPEND_SCHEME: {
      text: T5_TEST_STRINGS.PREPEND_SCHEME,
      tokens: ["\u2581Hey", "\u2581", "</s>", "\u2581", ".", "\u2581how", "\u2581are", "\u2581you"],
      ids: [9459, 3, 1, 3, 5, 149, 33, 25, 1],
      decoded: "Hey </s>. how are you</s>",
    },
  },
};

// Test that tokenizer type can be inferred (`type: "Unigram"` is missing)
TEST_CONFIG["google-t5/t5-small"] = TEST_CONFIG["Xenova/t5-small"];

const MAX_EXECUTION_TIME = 10_000;
export const CUSTOM_TESTS = () => {
  // Tests to ensure that no matter what, the correct tokenization is returned.
  // This is necessary since there are sometimes bugs in the transformers library.
  describe("hard-coded", () => {
    const TESTS = {
      // legacy=false
      "Xenova/t5-tokenizer-new": [
        {
          data: {
            // https://github.com/huggingface/transformers/pull/26678
            // ['▁Hey', '▁', '</s>', '.', '▁how', '▁are', '▁you']
            "Hey </s>. how are you": [9459, 3, 1, 5, 149, 33, 25],
          },
          reversible: true,
          legacy: null,
        },
        {
          data: {
            "</s>\n": [1, 3],
            "A\n'll": [71, 3, 31, 195],
          },
          reversible: false,
          legacy: null,
        },
      ],
    };

    for (const [tokenizerName, test_data] of Object.entries(TESTS)) {
      it(
        tokenizerName,
        async () => {
          for (const { data, reversible, legacy } of test_data) {
            const tokenizer = await T5Tokenizer.from_pretrained(tokenizerName, { legacy });

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
