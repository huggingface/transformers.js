import json


def generate_tokenizer_json(tokenizer):
    vocab = tokenizer.get_vocab()

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
            if token.startswith('<') and token.endswith('>')
        ],

        "normalizer": {
            "type": "Precompiled",
            "precompiled_charsmap": None
        },
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "WhitespaceSplit"
                },
                {
                    "type": "Metaspace",
                    "replacement": "▁",
                    "add_prefix_space": True
                },
                {
                    "type": "Split",
                    "pattern": {
                        "Regex": ""
                    },
                    "behavior": "Isolated",
                    "invert": False
                }
            ]
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {
                    "Sequence": {
                        "id": "A",
                        "type_id": 0
                    }
                },
                {
                    "SpecialToken": {
                        "id": "</s>",
                        "type_id": 0
                    }
                }
            ],
            "pair": [
                {
                    "Sequence": {
                        "id": "A",
                        "type_id": 0
                    }
                },
                {
                    "SpecialToken": {
                        "id": "</s>",
                        "type_id": 0
                    }
                },
                {
                    "Sequence": {
                        "id": "B",
                        "type_id": 0
                    }
                },
                {
                    "SpecialToken": {
                        "id": "</s>",
                        "type_id": 0
                    }
                }
            ],
            "special_tokens": {
                "</s>": {
                    "id": "</s>",
                    "ids": [
                        2
                    ],
                    "tokens": [
                        "</s>"
                    ]
                }
            }
        },
        "decoder": {
            "type": "Metaspace",
            "replacement": "▁",
            "add_prefix_space": True
        },
        'model': {
            # 'type': 'Char',
            'unk_id': 2,
            "vocab": vocab
        }
    }

    return tokenizer_json
