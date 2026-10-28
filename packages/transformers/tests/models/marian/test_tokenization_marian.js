import { MarianTokenizer } from "../../../src/models/tokenizers.js";
import { BASE_TEST_STRINGS, MARIAN_TEST_STRINGS } from "../test_strings.js";

export const TOKENIZER_CLASS = MarianTokenizer;
export const TEST_CONFIG = {
  "Xenova/opus-mt-en-ROMANCE": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      tokens: ["\u2581How", "\u2581are", "\u2581you", "\u2581doing", "?"],
      ids: [621, 63, 33, 951, 19, 0],
      decoded: "How are you doing?</s>",
    },
    LANGUAGE_CODE: {
      text: MARIAN_TEST_STRINGS.LANGUAGE_CODE,
      tokens: [">>fr<<", "\u2581The", "\u2581weather", "\u2581is", "\u2581nice", "\u2581today", "."],
      ids: [14, 46, 7032, 28, 2922, 1071, 3, 0],
      decoded: ">>fr<< The weather is nice today.</s>",
    },
    LANGUAGE_CODE_WITH_REGION: {
      text: MARIAN_TEST_STRINGS.LANGUAGE_CODE_WITH_REGION,
      tokens: [">>es_MX<<", "\u2581Hello", "\u2581world"],
      ids: [51471, 4708, 468, 0],
      decoded: ">>es_MX<< Hello world</s>",
    },
    LANGUAGE_CODE_WITHOUT_SPACE: {
      text: MARIAN_TEST_STRINGS.LANGUAGE_CODE_WITHOUT_SPACE,
      tokens: [">>fr<<", "\u2581Hello"],
      ids: [14, 4708, 0],
      decoded: ">>fr<< Hello</s>",
    },
    LANGUAGE_CODE_NOT_AT_START: {
      text: MARIAN_TEST_STRINGS.LANGUAGE_CODE_NOT_AT_START,
      tokens: ["\u2581Hello", "\u2581>", ">", "fr", "<", "<", "\u2581world"],
      ids: [4708, 658, 11862, 8844, 22829, 22829, 468, 0],
      decoded: "Hello >>fr<< world</s>",
    },
    UNKNOWN_LANGUAGE_CODE: {
      text: MARIAN_TEST_STRINGS.UNKNOWN_LANGUAGE_CODE,
      tokens: [">>xx<<", "\u2581Hello"],
      ids: [1, 4708, 0],
    },
  },
};

const MAX_EXECUTION_TIME = 10_000;
export const CUSTOM_TESTS = () => {
  const model_id = "Xenova/opus-mt-en-ROMANCE";

  it(
    "should support every language code in the vocabulary",
    async () => {
      const tokenizer = await MarianTokenizer.from_pretrained(model_id);
      const language_codes = Array.from(tokenizer.get_vocab().keys()).filter((x) => x.startsWith(">>") && x.endsWith("<<"));
      expect(language_codes).toHaveLength(47);
      expect(tokenizer.supported_language_codes).toEqual(language_codes);
    },
    MAX_EXECUTION_TIME,
  );

  it(
    "should keep the language code when padding a batch",
    async () => {
      const tokenizer = await MarianTokenizer.from_pretrained(model_id);
      const { input_ids, attention_mask } = tokenizer([">>fr<< Hello world", ">>es<< Hello"], { padding: true, return_tensor: false });
      expect(input_ids).toEqual([
        [14, 4708, 468, 0],
        [16, 4708, 0, 65000],
      ]);
      expect(attention_mask).toEqual([
        [1, 1, 1, 1],
        [1, 1, 1, 0],
      ]);
    },
    MAX_EXECUTION_TIME,
  );
};
