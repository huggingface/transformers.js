export const BASE_TEST_STRINGS = {
  SIMPLE: "How are you doing?",
  SIMPLE_WITH_PUNCTUATION: "You should've done this",
  NUMBERS: "0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000",
  TEXT_WITH_NUMBERS: "The company was founded in 2016.",
  PUNCTUATION: "A\n'll !!to?'d''d of, can't.",
  PYTHON_CODE: "def main():\n\tpass",
  JAVASCRIPT_CODE: "let a = obj.toString();\ntoString();",
  NEWLINES: "This\n\nis\na\ntest.",
  BASIC: "UNwant\u00e9d,running",
  CONTROL_TOKENS: "1\u00002\uFFFD3",
  HELLO_WORLD_TITLECASE: "Hello World",
  HELLO_WORLD_LOWERCASE: "hello world",
  CHINESE_ONLY: "生活的真谛是",
  LEADING_SPACE: "   leading space",
  TRAILING_SPACE: "trailing space   ",
  SURROUNDING_SPACE: "   surrounding space   ",
  DOUBLE_SPACE: "Hi  Hello",
  CURRENCY: "test $1 R2 #3 €4 £5 ¥6 ₣7 ₹8 ₱9 test",
  CURRENCY_WITH_DECIMALS: "I bought an apple for $1.00 at the store.",
  ELLIPSIS: "you…  ",
  TEXT_WITH_ESCAPE_CHARACTERS: "\u0079\u006F\u0075\u2026\u00A0\u00A0",
  TEXT_WITH_ESCAPE_CHARACTERS_2: "\u0079\u006F\u0075\u2026\u00A0\u00A0\u0079\u006F\u0075\u2026\u00A0\u00A0",
  TILDE_NORMALIZATION: "weird \uFF5E edge \uFF5E case",
  SPIECE_UNDERSCORE: "▁This ▁is ▁a ▁test ▁.",
  POPULAR_EMOJIS: "😂 👍 🤣 😍 😭 🎉 🙏 😊 🔥 😁 😅 🤗 😆 👏 ❤️ 💜 💚 💗 💙 🖤 😎 👌 🥳 💪 ✨ 👉 👀 💯 🎈 🙈 🙌 💀 👇 👋 ✅ 🎁 🌞 🌸 💰",
  MULTIBYTE_EMOJIS: "✨ 🤗 👁️ 👱🏻 🕵‍♂️ 🧙🏻‍♂ 👨🏻‍🌾 🧑‍🤝‍🧑 👩‍❤‍💋‍👨 👩‍👩‍👧‍👦 🧑🏻‍🤝‍🧑🏻 🏴󠁧󠁢󠁥󠁮󠁧󠁿 👨🏻‍❤️‍💋‍👨🏼", // 1 2 3 4 5 6 7 8 10 11 12 14 15
  ONLY_WHITESPACE: " \t\n",
};

export const BERT_TEST_STRINGS = {
  CHINESE_LATIN_MIXED: "ah\u535a\u63a8zz",
  SIMPLE_WITH_ACCENTS: "H\u00e9llo",
  MIXED_CASE_WITHOUT_ACCENTS: " \tHeLLo!how  \n Are yoU?  ",
  MIXED_CASE_WITH_ACCENTS: " \tHäLLo!how  \n Are yoU?  ",
};

// SentencePiece-specific test cases
export const SENTENCEPIECE_TEST_STRINGS = {
  SPECIAL_WITH_TRAILING_WHITESPACE: "<s>\n",
  SPECIAL_SURROUNDED_BY_WHITESPACE: " </s> test </s> ",
  SPECIAL_NO_WHITESPACE: "</s>test</s>",
};

// Additional test-cases for the Llama tokenizer, adapted from
// https://github.com/belladoreai/llama-tokenizer-js/blob/master/llama-tokenizer.js#L381-L452
export const LLAMA_TEST_STRINGS = {
  BPE_SCORES_PRIORITY_1: "grabbed",
  BPE_SCORES_PRIORITY_2: " grabbed",
  BPE_SCORES_PRIORITY_3: "           grabbed",
  NEWLINE: "\n",
  NEWLINES: "ax\n####\nboo",
  NEWLINE_WITH_LEADING_SPACE: " \n",
  TABS: "	tabs				out here",
  NEWLINE_AND_TAB: "\n\t\n",
  CHINESE_LETTER: "镇",
  EMOJIS_1: "🦙",
  EMOJIS_2: "🦙Ꙋ",
  EMOJIS_3: "Ꙋ🦙",
  PARAGRAPH: 'The llama (/ˈlɑːmə/; 🦙Spanish pronunciation: [ˈʎama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the Pre-Columbian era. Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4] The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000Ꙋ🦙 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5] In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology, llamas will return to the water springs and lagoons where they come from at the end of time.[6]',
  IGNORE_MERGES: "Ne için gittiğimi falan bilmiyordum, Washington'da belirtilen bir yere rapor vermem gerekiyordu.",
};

export const VITS_TEST_STRINGS = {
  BASIC: "abcdefghijklmnopqrstuvwxyz01234567890",
  // Special treatment of characters in certain language
  SPECIAL_CHARACTERS: "ț ţ",
};

export const QWEN_TEST_STRINGS = {
  PUNCTUATION_SPLIT: "i'm i'M i've i've i'Ve i'vE i'VE",
};

export const WHISPER_TEST_STRINGS = {
  SPECIAL_TOKENS: "   <|startoftranscript|> <|en|>   ", // Tests lstrip+rstrip
};

export const BLENDERBOT_SMALL_TEST_STRINGS = {
  SPECIAL_TOKENS: "__start__hello world__end__",
  // The original (python) tokenizer simply joins by spaces (regardless of special tokens or not)
  WHITESPACE_1: "__start__ hey __end__", // --> ... --> "__start__ hey __end__"
  WHITESPACE_2: "__start__hey __end__", // --> ... --> "__start__ hey __end__"
};

export const T5_TEST_STRINGS = {
  // Tests the new T5 tokenizer, which uses a different prepend_scheme for its pre_tokenizer:
  // tokenizer._tokenizer.pre_tokenizer = Metaspace(add_prefix_space = True, replacement = "▁", prepend_scheme = "first")
  // See https://github.com/huggingface/transformers/pull/26678 for more information.
  //  - Old (incorrect): ['▁Hey', '▁', '</s>', '▁', '.', '▁how', '▁are', '▁you']
  //  - New (correct):   ['▁Hey', '▁', '</s>', '.', '▁how', '▁are', '▁you']
  PREPEND_SCHEME: "Hey </s>. how are you",
};

export const FALCON_TEST_STRINGS = {
  // Special case for splitting on 3 numbers
  NUMBERS_SPLIT: "12 and 123 and 1234",
};

export const ESM_TEST_STRINGS = {
  // Special tokens
  SPECIAL_TOKENS: "<unk><pad><mask><cls><eos><bos>",
  // Actual protein sequences
  PROTEIN_SEQUENCES_1: "ATTCCGATTCCGATTCCG",
  PROTEIN_SEQUENCES_2: "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT",
};

export const BLOOM_TEST_STRINGS = {
  END_OF_SENTENCE_PUNCTUATION: "test. test, test! test? test… test。 test， test、 test। test۔ test، test",
};

export const M2M_100_TEST_STRINGS = {
  TRANSLATION_INPUTS: "__en__ hello world</s>",
  HIDNI_TEXT: "जीवन एक चॉकलेट बॉक्स की तरह है।",
  CHINESE_TEXT: "生活就像一盒巧克力。",
};

// Test strings adapted from https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String/normalize
export const NORMALIZATION_TEST_STRINGS = {
  DEFAULT_EXAMPLE: "\u0041\u006d\u00e9\u006c\u0069\u0065 | \u0041\u006d\u0065\u0301\u006c\u0069\u0065",
  CANONICAL_EQUIVALENCE_NORMALIZATION: "\u00F1 | \u006E\u0303",
  COMPATIBILITY_NORMALIZATION: "\uFB00 | \u0066\u0066",

  // Original | NFC | NFD | NFKC | NFKD
  COMBINED_EXAMPLE: "\u1E9B\u0323 | \u1E9B\u0323 | \u017F\u0323\u0307 | \u1E69 | \u0073\u0323\u0307",
};
