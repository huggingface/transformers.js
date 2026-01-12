from transformers.convert_slow_tokenizer import Converter
from tokenizers import Tokenizer, pre_tokenizers, processors
from tokenizers.models import WordPiece


class EsmConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, continuing_subword_prefix='', max_input_chars_per_word=int(
            1e10), unk_token=str(self.original_tokenizer.unk_token)))

        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

        cls = str(self.original_tokenizer.cls_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        # No sep token in ESM vocabulary
        sep = str(self.original_tokenizer.eos_token)
        sep_token_id = self.original_tokenizer.eos_token_id

        if sep_token_id is None:
            tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{cls}:0 $A:0",
                special_tokens=[
                    (cls, cls_token_id),
                ],
            )
        else:
            tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{cls}:0 $A:0 {sep}:0",
                pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
                special_tokens=[
                    (cls, cls_token_id),
                    (sep, sep_token_id),
                ],
            )

        # For some reason, all tokens are added: none of them are special, but they all need special splitting.
        # See https://github.com/huggingface/transformers/blob/df5c5c62ae253055336f5bb0828ca8e3e15ab6bd/src/transformers/models/esm/tokenization_esm.py#L79-L80
        special_tokens = []
        other_tokens = []
        for token, token_id in vocab.items():
            if token[0] == '<' and token[-1] == '>' and token_id <= 3:
                special_tokens.append(token)
            else:
                other_tokens.append(token)

        tokenizer.add_special_tokens(special_tokens)
        tokenizer.add_tokens(other_tokens)
        return tokenizer


def generate_fast_tokenizer(tokenizer):
    tokenizer.vocab = tokenizer._token_to_id
    return EsmConverter(tokenizer).converted()
