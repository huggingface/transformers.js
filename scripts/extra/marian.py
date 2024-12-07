import json
from transformers.utils import cached_file


def generate_tokenizer_json(model_path, tokenizer):
    # Marian models use two separate tokenizers for source and target languages.
    # So, we merge them into a single tokenizer.

    vocab_file = cached_file(model_path, 'vocab.json')
    with open(vocab_file) as fp:
        vocab = json.load(fp)

    added_tokens = [
        dict(
            id=vocab.get(x),
            special=True,
            content=x,
            single_word=False,
            lstrip=False,
            rstrip=False,
            normalized=False
        )
        for x in tokenizer.all_special_tokens
    ]

    tokenizer_json = {
        'version': '1.0',
        'truncation': None,
        'padding': None,
        'added_tokens': added_tokens,
        'normalizer': {
            'type': 'Precompiled',
            'precompiled_charsmap': None  # TODO add this
        },
        'pre_tokenizer': {
            'type': 'Sequence',
            'pretokenizers': [
                {
                    'type': 'WhitespaceSplit'
                },
                {
                    'type': 'Metaspace',
                    'replacement': '\u2581',
                    'add_prefix_space': True
                }
            ]
        },
        'post_processor': {
            'type': 'TemplateProcessing', 'single': [
                {'Sequence': {'id': 'A', 'type_id': 0}},
                {'SpecialToken': {'id': tokenizer.eos_token, 'type_id': 0}}
            ],
            'pair': [
                {'Sequence': {'id': 'A', 'type_id': 0}},
                {'SpecialToken': {'id': tokenizer.eos_token, 'type_id': 0}},
                {'Sequence': {'id': 'B', 'type_id': 0}},
                {'SpecialToken': {'id': tokenizer.eos_token, 'type_id': 0}}
            ],
            'special_tokens': {
                tokenizer.eos_token: {
                    'id': tokenizer.eos_token,
                    'ids': [tokenizer.eos_token_id],
                    'tokens': [tokenizer.eos_token]
                }
            }
        },
        'decoder': {
            'type': 'Metaspace',
            'replacement': '\u2581',
            'add_prefix_space': True
        },
        'model': {
            'type': 'Unigram',
            'unk_id': 2,
        }
    }

    # NOTE: Must have sentencepiece installed
    spm_source = tokenizer.spm_source
    spm_target = tokenizer.spm_target

    src_vocab_dict = {
        spm_source.IdToPiece(i): spm_source.GetScore(i)
        for i in range(spm_source.GetPieceSize())
    }
    tgt_vocab_dict = {
        spm_target.IdToPiece(i): spm_target.GetScore(i)
        for i in range(spm_target.GetPieceSize())
    }

    tokenizer_json['model']['vocab'] = [
        [
            k,
            0.0 if k in tokenizer.all_special_tokens else max(
                src_vocab_dict.get(k, -100), tgt_vocab_dict.get(k, -100))
        ]
        for k in vocab
    ]

    return tokenizer_json
