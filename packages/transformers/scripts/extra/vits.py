

def generate_tokenizer_json(tokenizer):
    vocab = tokenizer.get_vocab()

    normalizers = []

    if tokenizer.normalize:
        # Lowercase the input string
        normalizers.append({
            "type": "Lowercase",
        })

    if tokenizer.language == 'ron':
        # Replace diacritics
        normalizers.append({
            "type": "Replace",
            "pattern": {
                "String": "ț",
            },
            "content": "ţ",
        })

    if tokenizer.phonemize:
        raise NotImplementedError("Phonemization is not implemented yet")

    elif tokenizer.normalize:
        # strip any chars outside of the vocab (punctuation)
        chars = ''.join(x for x in vocab if len(x) == 1)
        escaped = chars.replace('-', r'\-').replace(']', r'\]')
        normalizers.append({
            "type": "Replace",
            "pattern": {
                "Regex": f"[^{escaped}]",
            },
            "content": "",
        })
        normalizers.append({
            "type": "Strip",
            "strip_left": True,
            "strip_right": True,
        })

    if tokenizer.add_blank:
        # add pad token between each char
        normalizers.append({
            "type": "Replace",
            "pattern": {
                # Add a blank token between each char, except when blank (then do nothing)
                "Regex": "(?=.)|(?<!^)$",
            },
            "content": tokenizer.pad_token,
        })

    if len(normalizers) == 0:
        normalizer = None
    elif len(normalizers) == 1:
        normalizer = normalizers[0]
    else:
        normalizer = {
            "type": "Sequence",
            "normalizers": normalizers,
        }

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": vocab[token],
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            }
            for token in vocab

            # `tokenizer.pad_token` should not be considered an added token
            if token in (tokenizer.unk_token, )
        ],
        "normalizer": normalizer,
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {
                "Regex": ""
            },
            "behavior": "Isolated",
            "invert": False
        },
        "post_processor": None,
        "decoder": None,  # Custom decoder implemented in JS
        "model": {
            "vocab": vocab
        },
    }

    return tokenizer_json
