import { PreTrainedTokenizer } from "../../../src/tokenization_utils.js";

// Small offline Unigram vocabulary with the relevant converted-normalizer rules.
// This exercises real tokenizer normalization/decoding, not a mocked encode().
const letters = Array.from({ length: 26 }, (_, i) => String.fromCharCode(97 + i));
export const TOKENIZER_JSON = {
  version: "1.0",
  truncation: null,
  padding: null,
  added_tokens: [{ id: 0, content: "<unk>", single_word: false, lstrip: false, rstrip: false, normalized: false, special: true }],
  normalizer: {
    type: "Sequence",
    normalizers: [
      { type: "Replace", pattern: { Regex: "[»«‹›“”„‟″‶⹂❝❞❠〝〞〟＂]" }, content: '"' },
      { type: "Replace", pattern: { Regex: "[‘’‚‛′‵❛❜❟＇ʼʻʹ´`]" }, content: "'" },
      { type: "Replace", pattern: { Regex: "[‐‑‒―]" }, content: "-" },
      // Same literal Unicode ranges as the pinned Hub tokenizer.
      { type: "Replace", pattern: { Regex: "[–*—\\[\\]{}•‣◦·→←↑↓➡➜🀀-🫿☀-➿⬀-⯿️‍\\p{Cc}\\s]+" }, content: " " },
      { type: "Replace", pattern: { String: "…" }, content: "." },
      { type: "Replace", pattern: { Regex: "(?<!\\S):|:(?!\\S)|[;()]" }, content: ", " },
      { type: "Replace", pattern: { Regex: "(?:\\s|[!()*,\\-./:;<=>?\\[\\\\\\]^_`{|}~]|(?<![0-9])\\$)+(?=[.,])" }, content: "" },
      { type: "Replace", pattern: { Regex: "\\s+" }, content: " " },
      { type: "Strip", strip_left: true, strip_right: true },
      { type: "Replace", pattern: { Regex: '"+$' }, content: "" },
      { type: "Replace", pattern: { Regex: "(?<=[.!?,;])'$" }, content: "" },
      { type: "Strip", strip_left: true, strip_right: true },
      ...letters.map((letter) => ({ type: "Replace", pattern: { Regex: `^${letter}` }, content: letter.toUpperCase() })),
      { type: "Replace", pattern: { Regex: "(?<=[\\p{Alphabetic}\\p{N}])$" }, content: "." },
      { type: "Prepend", prepend: "▁" },
    ],
  },
  pre_tokenizer: { type: "Metaspace", replacement: "▁", prepend_scheme: "never", split: false },
  post_processor: null,
  decoder: { type: "Sequence", decoders: [{ type: "ByteFallback" }, { type: "Replace", pattern: { String: "▁" }, content: " " }, { type: "Fuse" }, { type: "Strip", content: " ", start: 1, stop: 0 }] },
  model: { type: "Unigram", unk_id: 0, byte_fallback: true, vocab: [["<unk>", 0], ["▁", -1], ...Array.from({ length: 95 }, (_, i) => [String.fromCharCode(32 + i), -1]), ...Array.from({ length: 256 }, (_, i) => [`<0x${i.toString(16).toUpperCase().padStart(2, "0")}>`, -10])] },
};

export const AUDIO_TOKENS = ["<|audio_bos|>", "<|audio_prompt|>", "<|audio_latent|>"];
export const AUDIO_TOKEN_IDS = AUDIO_TOKENS.map((_, i) => TOKENIZER_JSON.model.vocab.length + i);
TOKENIZER_JSON.added_tokens.push(...AUDIO_TOKENS.map((content, i) => ({ id: AUDIO_TOKEN_IDS[i], content, single_word: false, lstrip: false, rstrip: false, normalized: false, special: true })));
export const CHAT_TEMPLATE = "{{ '<|audio_prompt|>' if messages[0]['content'] is not string else '' }}{{ messages[0]['content'][1]['text'] if messages[0]['content'] is not string else messages[0]['content'] }}{{ '<|audio_bos|>' if add_generation_prompt else '' }}";

export function create_tokenizer() {
  return new PreTrainedTokenizer(TOKENIZER_JSON, { tokenizer_class: "PreTrainedTokenizer", unk_token: "<unk>", clean_up_tokenization_spaces: false });
}
