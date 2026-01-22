
def generate_tokenizer_json(tokenizer):
    vocab = tokenizer.vocab

    special_tokens_vocab = vocab
    if "<pad>" not in tokenizer.vocab:
        # For MMS tokenizers, the vocab is of the form:
        # {
        #   language_id: { language_vocab }
        # }
        # So, to get the list of special tokens, we just get the english vocab
        special_tokens_vocab = vocab['eng']

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": v,
                "content": k,
                "single_word": False,
                "lstrip": True,
                "rstrip": True,
                "normalized": False,
                "special": True
            }
            for k, v in special_tokens_vocab.items()
            if k.startswith('<') and k.endswith('>')
        ],
        "normalizer": {
            "type": "Replace",
            "pattern": {
                "String": " "
            },
            "content": "|"
        },
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {
                "Regex": ""
            },
            "behavior": "Isolated",
            "invert": False
        },
        "post_processor": None,
        "decoder": {
            "type": "CTC",
            "pad_token": "<pad>",
            "word_delimiter_token": "|",
            "cleanup": True
        },
        "model": {
            "vocab": vocab
        }
    }

    return tokenizer_json
