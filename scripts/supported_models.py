from .extra.marian import SUPPORTED_HELSINKI_NLP_MODELS


SUPPORTED_MODELS = {
    # NOTE: keys of `SUPPORTED_MODELS` are subsets of https://github.com/huggingface/optimum/blob/7f8e606689365931300ef5e6d3b20cb88771cb08/optimum/exporters/tasks.py#L281-L965
    'albert': {
        # Masked language modelling
        'fill-mask': [
            'albert-base-v2',
            'albert-large-v2',
        ],

        # Feature extraction
        'feature-extraction': [
            'sentence-transformers/paraphrase-albert-small-v2',
            'sentence-transformers/paraphrase-albert-base-v2',
        ],
    },
    'audio-spectrogram-transformer': {
        # Audio classification
        'audio-classification': {
            'MIT/ast-finetuned-audioset-10-10-0.4593',
            'MIT/ast-finetuned-audioset-16-16-0.442',
            'MIT/ast-finetuned-speech-commands-v2',
            'mtg-upf/discogs-maest-30s-pw-73e-ts',
        }
    },
    'bart': {
        # Summarization
        'summarization': [
            'sshleifer/distilbart-xsum-12-1',
            'sshleifer/distilbart-xsum-6-6',
            'sshleifer/distilbart-xsum-12-3',
            'sshleifer/distilbart-xsum-9-6',
            'sshleifer/distilbart-xsum-12-6',
            'sshleifer/distilbart-cnn-12-3',
            'sshleifer/distilbart-cnn-12-6',
            'sshleifer/distilbart-cnn-6-6',
            'facebook/bart-large-cnn',
            'facebook/bart-large-xsum',
        ],
        # Zero-shot classification
        'zero-shot-classification': {
            'facebook/bart-large-mnli',
        },
    },
    'beit': {
        # Image classification
        'image-classification': [
            'microsoft/beit-base-patch16-224',
            'microsoft/beit-base-patch16-224-pt22k',
            'microsoft/beit-base-patch16-384',
            'microsoft/beit-base-patch16-224-pt22k-ft22k',
            'microsoft/beit-large-patch16-224',
            'microsoft/beit-large-patch16-224-pt22k',
            'microsoft/beit-large-patch16-512',
            'microsoft/beit-large-patch16-224-pt22k-ft22k',
            'microsoft/beit-large-patch16-384',
            'microsoft/dit-base-finetuned-rvlcdip',
            'microsoft/dit-large-finetuned-rvlcdip',
        ],
    },
    'bert': {
        # Feature extraction
        'feature-extraction': [
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/all-MiniLM-L12-v2',
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'sentence-transformers/paraphrase-MiniLM-L6-v2',
            'sentence-transformers/paraphrase-MiniLM-L3-v2',
            'sentence-transformers/bert-base-nli-mean-tokens',
            'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
            'sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens',
            'sentence-transformers/LaBSE',
            'deepset/sentence_bert',
            'intfloat/e5-small',
            'intfloat/e5-small-v2',
            'intfloat/e5-base',
            'intfloat/e5-base-v2',
            'intfloat/e5-large',
            'intfloat/e5-large-v2',
            'intfloat/multilingual-e5-base',
            'thenlper/gte-small',
            'thenlper/gte-base',
            'thenlper/gte-large',
            'BAAI/bge-small-en',
            'BAAI/bge-base-en',
            'BAAI/bge-large-en',
            'BAAI/bge-large-en-v1.5',
            'BAAI/bge-base-en-v1.5',
            'BAAI/bge-small-en-v1.5',
            'BAAI/bge-large-zh-v1.5',
            'BAAI/bge-base-zh-v1.5',
            'BAAI/bge-small-zh-v1.5',
            'allenai/scibert_scivocab_uncased',
            'SpanBERT/spanbert-large-cased',
            'SpanBERT/spanbert-base-cased',
            'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
            'indobenchmark/indobert-base-p1',
            'GanjinZero/UMLSBert_ENG',
            'DeepPavlov/rubert-base-cased',
            'monologg/kobert',
        ],

        # Text classification
        'text-classification': [
            'nlptown/bert-base-multilingual-uncased-sentiment',
            'ProsusAI/finbert',
            'unitary/toxic-bert',
            'BAAI/bge-reranker-large',
            'BAAI/bge-reranker-base',
            'cross-encoder/ms-marco-TinyBERT-L-2-v2',
            'cross-encoder/ms-marco-MiniLM-L-2-v2',
            'cross-encoder/ms-marco-MiniLM-L-4-v2',
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'cross-encoder/ms-marco-MiniLM-L-12-v2',
        ],

        # Token classification
        'token-classification': [
            'Davlan/bert-base-multilingual-cased-ner-hrl',
            'ckiplab/bert-base-chinese-ner',
            'ckiplab/bert-base-chinese-ws',
            'ckiplab/bert-base-chinese-pos',
            'dslim/bert-base-NER',
            'dslim/bert-base-NER-uncased',
        ],

        # Masked language modelling
        'fill-mask': [
            'bert-base-uncased',
            'bert-base-cased',
            'bert-base-multilingual-uncased',
            'bert-base-multilingual-cased',
            'bert-base-chinese',
            'emilyalsentzer/Bio_ClinicalBERT',
        ],
    },
    'blenderbot': {
        # Text-to-text (TODO add conversational)
        'text2text-generation': [
            'facebook/blenderbot-400M-distill',
            # 'facebook/blenderbot-1B-distill',
        ],
    },
    'blenderbot-small': {
        # Text-to-text (TODO add conversational)
        'text2text-generation': [
            # 'facebook/blenderbot-90M',  # DEPRECATED
            'facebook/blenderbot_small-90M',
        ],
    },
    'bloom': {
        # Text generation
        'text-generation': [
            'bigscience/bloom-560m',
            'bigscience/bloomz-560m',
        ],
    },

    'camembert': {
        # Feature extraction
        'feature-extraction': [
            'dangvantuan/sentence-camembert-large',
        ],

        # Token classification
        'token-classification': [
            'Jean-Baptiste/camembert-ner',
            'Jean-Baptiste/camembert-ner-with-dates',
            'pythainlp/thainer-corpus-v2-base-model',
            'gilf/french-camembert-postag-model',
        ],

        # Masked language modelling
        'fill-mask': [
            'camembert-base',
            'airesearch/wangchanberta-base-att-spm-uncased',
        ],
    },
    'clap': {
        # Zero-shot audio classification and feature extraction
        # (with and without `--split_modalities`)
        'zero-shot-audio-classification': {
            'laion/clap-htsat-unfused',
            # TODO add 'laion/clap-htsat-fused',
            'laion/larger_clap_general',
            'laion/larger_clap_music_and_speech',
            # 'Xenova/tiny-random-ClapModel',
        }
    },
    'chinese_clip': {
        # Zero-shot image classification
        # TODO: Add `--split_modalities` option
        'zero-shot-image-classification': [
            'OFA-Sys/chinese-clip-vit-base-patch16',
            'OFA-Sys/chinese-clip-vit-large-patch14',
            'OFA-Sys/chinese-clip-vit-large-patch14-336px',
            # 'OFA-Sys/chinese-clip-vit-huge-patch14', # TODO add
        ],
    },
    'clip': {
        # Zero-shot image classification (and feature extraction)
        # (with and without `--split_modalities`)
        'zero-shot-image-classification': [
            'openai/clip-vit-base-patch16',
            'openai/clip-vit-base-patch32',
            'openai/clip-vit-large-patch14',
            'openai/clip-vit-large-patch14-336',
        ],
    },
    'clipseg': {
        # Image segmentation
        'image-segmentation': [
            'CIDAS/clipseg-rd64-refined',
            'CIDAS/clipseg-rd64',
            'CIDAS/clipseg-rd16',
        ],
    },
    'codegen': {
        # Text generation
        'text-generation': [
            'Salesforce/codegen-350M-mono',
            'Salesforce/codegen-350M-multi',
            'Salesforce/codegen-350M-nl',
        ],
    },
    'convbert': {
        # Feature extraction
        'feature-extraction': [
            'YituTech/conv-bert-small',
            'YituTech/conv-bert-medium-small',
            'YituTech/conv-bert-base',
        ],
    },
    'convnext': {
        # Image classification
        'image-classification': [
            'facebook/convnext-tiny-224',
            'facebook/convnext-small-224',
            'facebook/convnext-base-224',
            'facebook/convnext-base-224-22k',
            'facebook/convnext-base-224-22k-1k',
            'facebook/convnext-base-384',
            'facebook/convnext-base-384-22k-1k',
            'facebook/convnext-large-224',
            'facebook/convnext-large-224-22k',
            'facebook/convnext-large-224-22k-1k',
            'facebook/convnext-large-384',
            'facebook/convnext-large-384-22k-1k',
            'facebook/convnext-xlarge-224-22k',
            'facebook/convnext-xlarge-224-22k-1k',
            'facebook/convnext-xlarge-384-22k-1k',
        ],
    },
    'convnextv2': {
        # Image classification
        'image-classification': [
            'facebook/convnextv2-atto-1k-224',
            'facebook/convnextv2-femto-1k-224',
            'facebook/convnextv2-pico-1k-224',
            'facebook/convnextv2-tiny-1k-224',
            'facebook/convnextv2-tiny-22k-384',
            'facebook/convnextv2-tiny-22k-224',
            'facebook/convnextv2-nano-1k-224',
            'facebook/convnextv2-nano-22k-384',
            'facebook/convnextv2-base-22k-224',
            'facebook/convnextv2-base-1k-224',
            'facebook/convnextv2-base-22k-384',
            'facebook/convnextv2-large-22k-224',
            'facebook/convnextv2-large-1k-224',
            'facebook/convnextv2-large-22k-384',
            # 'facebook/convnextv2-huge-22k-512',
            # 'facebook/convnextv2-huge-1k-224',
            # 'facebook/convnextv2-huge-22k-384',
            # 'facebook/convnextv2-nano-22k-224',
        ],
    },
    'deberta': {
        # Zero-shot classification
        'zero-shot-classification': [
            'cross-encoder/nli-deberta-base',
            'Narsil/deberta-large-mnli-zero-cls',
        ],
    },
    'deberta-v2': {
        # Zero-shot classification
        'zero-shot-classification': [
            'cross-encoder/nli-deberta-v3-xsmall',
            'cross-encoder/nli-deberta-v3-small',
            'cross-encoder/nli-deberta-v3-base',
            'cross-encoder/nli-deberta-v3-large',
            'MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary',
            'MoritzLaurer/DeBERTa-v3-base-mnli',
            'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
            'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
            'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
            'sileod/deberta-v3-base-tasksource-nli',
            'sileod/deberta-v3-large-tasksource-nli',
        ],
    },
    # TODO: Add back in v3
    # 'decision-transformer': {
    #     # Reinforcement learning
    #     'reinforcement-learning': [
    #         'edbeeching/decision-transformer-gym-hopper-expert',
    #         'edbeeching/decision-transformer-gym-hopper-medium',
    #         'edbeeching/decision-transformer-gym-hopper-medium-replay',
    #         'edbeeching/decision-transformer-gym-hopper-expert-new',
    #         'edbeeching/decision-transformer-gym-halfcheetah-expert',
    #         'edbeeching/decision-transformer-gym-halfcheetah-medium',
    #         'edbeeching/decision-transformer-gym-halfcheetah-medium-replay',
    #         'edbeeching/decision-transformer-gym-walker2d-expert',
    #         'edbeeching/decision-transformer-gym-walker2d-medium',
    #         'edbeeching/decision-transformer-gym-walker2d-medium-replay',
    #     ],
    # },
    'deit': {
        # Image classification
        'image-classification': [
            'facebook/deit-tiny-distilled-patch16-224',
            'facebook/deit-small-distilled-patch16-224',
            'facebook/deit-base-distilled-patch16-224',
            'facebook/deit-base-distilled-patch16-384',
        ],
    },
    'detr': {
        # Object detection
        'object-detection': [
            'facebook/detr-resnet-50',
            'facebook/detr-resnet-101',
        ],

        # Image segmentation
        'image-segmentation': [
            'facebook/detr-resnet-50-panoptic',
        ],
    },
    'dinov2': {
        # Feature extraction
        'feature-extraction': [
            'facebook/dinov2-small',
            'facebook/dinov2-base',
            'facebook/dinov2-large',
            # 'facebook/dinov2-giant',  # TODO add
        ],

        # Image classification
        'image-classification': [
            'facebook/dinov2-small-imagenet1k-1-layer',
            'facebook/dinov2-base-imagenet1k-1-layer',
            'facebook/dinov2-large-imagenet1k-1-layer',
            # 'facebook/dinov2-giant-imagenet1k-1-layer',  # TODO add
        ],
    },
    'distilbert': {
        # Feature extraction
        'feature-extraction': [
            'sentence-transformers/multi-qa-distilbert-cos-v1',
            'sentence-transformers/distiluse-base-multilingual-cased-v1',
            'sentence-transformers/distiluse-base-multilingual-cased-v2',
            'sentence-transformers/distilbert-base-nli-mean-tokens',
            'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
            'sentence-transformers/msmarco-distilbert-base-v4',
        ],

        # Text classification
        'text-classification': [
            'distilbert-base-uncased-finetuned-sst-2-english',
        ],

        # Question answering
        'question-answering': [
            'distilbert-base-uncased-distilled-squad',
            'distilbert-base-cased-distilled-squad',
        ],

        # Zero-shot classification
        'zero-shot-classification': [
            'typeform/distilbert-base-uncased-mnli',
        ],

        # Token classification
        'token-classification': [
            'Davlan/distilbert-base-multilingual-cased-ner-hrl',
        ],

        # Masked language modelling
        'fill-mask': [
            'distilbert-base-uncased',
            'distilbert-base-cased',
        ],
    },
    'dit': {  # NOTE: DiT has the same architecture as BEiT.
        # Feature extraction
        # NOTE: requires --task feature-extraction
        'feature-extraction': [
            'microsoft/dit-base',
            'microsoft/dit-large',
        ],

        # Image classification
        'image-classification': [
            'microsoft/dit-base-finetuned-rvlcdip',
            'microsoft/dit-large-finetuned-rvlcdip',
        ],
    },
    'donut': {  # NOTE: also a `vision-encoder-decoder`
        # Image-to-text
        'image-to-text': [
            'naver-clova-ix/donut-base-finetuned-cord-v2',
            'naver-clova-ix/donut-base-finetuned-zhtrainticket',
        ],

        # Document Question Answering
        'document-question-answering': [
            'naver-clova-ix/donut-base-finetuned-docvqa',
        ],
    },
    'dpt': {
        # Depth estimation
        'depth-estimation': [
            'Intel/dpt-hybrid-midas',
            'Intel/dpt-large',
        ],
    },
    'depth_anything': {
        # Depth estimation
        # NOTE: requires --task depth-estimation
        'depth-estimation': [
            'LiheYoung/depth-anything-small-hf',
            'LiheYoung/depth-anything-base-hf',
            'LiheYoung/depth-anything-large-hf',
        ],
    },
    'electra': {
        # Feature extraction
        'feature-extraction': [
            # NOTE: requires --task feature-extraction
            'google/electra-small-discriminator',
            'google/electra-base-discriminator',
        ],
    },
    'esm': {
        # Masked language modelling
        'fill-mask': [
            # with and without --task feature-extraction
            'InstaDeepAI/nucleotide-transformer-500m-human-ref',
            'InstaDeepAI/nucleotide-transformer-500m-1000g',

            # NOTE: requires --opset 12
            'facebook/esm2_t6_8M_UR50D',
            'facebook/esm2_t12_35M_UR50D',
            'facebook/esm2_t30_150M_UR50D',
            'facebook/esm2_t33_650M_UR50D',
        ],

        # Token classification
        'token-classification': [
            'AmelieSchreiber/esm2_t6_8M_UR50D_rna_binding_site_predictor',
        ],

        # Zero-shot classification
        'zero-shot-classification': [
            'AmelieSchreiber/esm2_t6_8M_UR50D_sequence_classifier_v1',
        ],
    },
    'falcon': {
        # Text generation
        'text-generation': [
            'Rocketknight1/tiny-random-falcon-7b',
            'fxmarty/really-tiny-falcon-testing',
        ],
    },
    'fastvit': {
        # Image classification
        'image-classification': [
            # NOTE: Supported by timm, but not by transformers
            # 'timm/fastvit_t8.apple_in1k',
            # 'timm/fastvit_t8.apple_dist_in1k',
            # 'timm/fastvit_t12.apple_in1k',
            # 'timm/fastvit_t12.apple_dist_in1k',
            # 'timm/fastvit_s12.apple_in1k',
            # 'timm/fastvit_s12.apple_dist_in1k',
            # 'timm/fastvit_sa12.apple_in1k',
            # 'timm/fastvit_sa12.apple_dist_in1k',
            # 'timm/fastvit_sa24.apple_in1k',
            # 'timm/fastvit_sa24.apple_dist_in1k',
            # 'timm/fastvit_sa36.apple_in1k',
            # 'timm/fastvit_sa36.apple_dist_in1k',
            # 'timm/fastvit_ma36.apple_in1k',
            # 'timm/fastvit_ma36.apple_dist_in1k',
        ],
    },
    'glpn': {
        # Depth estimation
        'depth-estimation': [
            'vinvino02/glpn-kitti',
            'vinvino02/glpn-nyu',
        ],
    },
    'gpt_neo': {
        # Text generation
        'text-generation': [
            'EleutherAI/gpt-neo-125M',
            'MBZUAI/LaMini-Neo-125M',
            # 'MBZUAI/LaMini-Neo-1.3B', # TODO add
            'iliemihai/gpt-neo-romanian-125m',
        ],
    },
    'gpt_neox': {
        # Text generation
        'text-generation': [
            'EleutherAI/pythia-14m',
            'EleutherAI/pythia-31m',
            'EleutherAI/pythia-70m',
            'EleutherAI/pythia-70m-deduped',
            'EleutherAI/pythia-160m',
            'EleutherAI/pythia-160m-deduped',
            'EleutherAI/pythia-410m',
            'EleutherAI/pythia-410m-deduped',
        ],
    },
    'gpt2': {
        # Text generation
        'text-generation': [
            'gpt2',
            'distilgpt2',
            'MBZUAI/LaMini-Cerebras-111M',
            'MBZUAI/LaMini-Cerebras-256M',
            'MBZUAI/LaMini-Cerebras-590M',
            # 'MBZUAI/LaMini-Cerebras-1.3B', # TODO add
            'MBZUAI/LaMini-GPT-124M',
            'MBZUAI/LaMini-GPT-774M',
            # 'MBZUAI/LaMini-GPT-1.5B', # TODO add
            'aisquared/dlite-v2-774m',
            'Locutusque/gpt2-large-conversational',
        ],
    },
    'gpt_bigcode': {
        # Text generation
        'text-generation': [
            'bigcode/tiny_starcoder_py',
            'abacaj/starcoderbase-1b-sft',
            # 'bigcode/starcoderbase-1b', # NOTE: This model is gated, so we ignore it when testing
        ],
    },
    'gptj': {
        # Text generation
        'text-generation': [
            'TabbyML/J-350M',
            'Milos/slovak-gpt-j-405M',
            'heegyu/kogpt-j-350m',
        ],
    },
    'herbert': {
        # Feature extraction
        'feature-extraction': [
            'allegro/herbert-base-cased',
            'allegro/herbert-large-cased',
        ],
    },
    'hubert': {
        # Feature extraction
        'feature-extraction': [
            'facebook/hubert-base-ls960',
        ],

        # Audio classification
        'audio-classification': [
            'superb/hubert-base-superb-ks',
        ],

        # Automatic speech recognition
        'automatic-speech-recognition': [
            'facebook/hubert-large-ls960-ft',
        ],
    },
    'llama': {
        # Text generation
        'text-generation': [
            'Xenova/llama2.c-stories15M',
            'Xenova/llama2.c-stories42M',
            'Xenova/llama2.c-stories110M',
            'RajuKandasamy/tamillama_tiny_30m',
            'JackFram/llama-68m',
            'JackFram/llama-160m',
        ],
    },
    'longt5': {
        # Text-to-text
        'text2text-generation': [
            'google/long-t5-local-base',
            'google/long-t5-tglobal-base',
            # 'google/long-t5-tglobal-xl', # too large
            # 'google/long-t5-tglobal-large', # too large
            # 'google/long-t5-local-large', # too large
        ],

        # Summarization
        'summarization': [
            'pszemraj/long-t5-tglobal-base-16384-book-summary',
        ],

        # Feature extraction
        'feature-extraction': [
            # NOTE: requires --task feature-extraction
            'voidful/long-t5-encodec-tglobal-base',
        ],
    },
    'm2m_100': {
        # Translation
        'translation': [
            'facebook/nllb-200-distilled-600M',
            'facebook/m2m100_418M',
        ],
    },
    'marian': {
        # Translation
        'translation': [
            f'Helsinki-NLP/opus-mt-{x}'
            for x in SUPPORTED_HELSINKI_NLP_MODELS
        ],
    },
    'mbart': {
        # Translation
        'translation': [
            'facebook/mbart-large-50-many-to-many-mmt',
            'facebook/mbart-large-50-many-to-one-mmt',
            'facebook/mbart-large-50',
        ],
    },
    'mistral': {
        # Text generation
        'text-generation': [
            'echarlaix/tiny-random-mistral',
        ],
    },
    'mobilebert': {
        # Zero-shot classification
        'zero-shot-classification': [
            'typeform/mobilebert-uncased-mnli',

            # TODO:
            # https://github.com/huggingface/optimum/issues/1027
            # 'google/mobilebert-uncased',
        ],
    },
    'mobilevit': {
        # Image classification
        'image-classification': [
            'apple/mobilevit-small',
            'apple/mobilevit-x-small',
            'apple/mobilevit-xx-small',
        ],

        # TODO: Image segmentation
        # 'image-segmentation': [
        #     'apple/deeplabv3-mobilevit-small',
        #     'apple/deeplabv3-mobilevit-x-small',
        #     'apple/deeplabv3-mobilevit-xx-small',
        # ],
    },
    'mobilevitv2': {
        # Image classification
        'image-classification': [
            'apple/mobilevitv2-1.0-imagenet1k-256',
        ],

        # TODO: Image segmentation
        # 'image-segmentation': [
        #     'apple/mobilevitv2-1.0-voc-deeplabv3',
        # ],
    },
    'mpt': {
        # Text generation
        'text-generation': [
            'efederici/ipt-350m',
        ],
    },
    'mpnet': {
        # Feature extraction
        'feature-extraction': [
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/nli-mpnet-base-v2',
            'sentence-transformers/paraphrase-mpnet-base-v2',
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            'sentence-transformers/multi-qa-mpnet-base-cos-v1',
            'sentence-transformers/multi-qa-mpnet-base-dot-v1',
        ],
    },
    'mt5': {
        # Text-to-text
        'text2text-generation': [
            'google/mt5-small',
            'google/mt5-base',
        ],
    },
    'nougat': {
        # Image-to-text
        'image-to-text': [
            'facebook/nougat-small',
            'facebook/nougat-base',
        ],
    },
    'opt': {
        # Text generation
        'text-generation': [
            # Text generation
            'facebook/opt-125m',
            'facebook/opt-350m',
            # (TODO conversational)
            'PygmalionAI/pygmalion-350m',
        ],
    },
    'owlv2': {
        # Object detection (Zero-shot object detection)
        # NOTE: Exported with --batch_size 1
        'zero-shot-object-detection': [
            'google/owlv2-base-patch16',
            'google/owlv2-base-patch16-finetuned',
            'google/owlv2-base-patch16-ensemble',
            # TODO: add
            # 'google/owlv2-large-patch14',
            # 'google/owlv2-large-patch14-finetuned',
            # 'google/owlv2-large-patch14-ensemble',
        ],
    },
    'owlvit': {
        # Object detection (Zero-shot object detection)
        # NOTE: Exported with --batch_size 1
        'zero-shot-object-detection': [
            'google/owlvit-base-patch32',
            'google/owlvit-base-patch16',
            'google/owlvit-large-patch14',
        ],
    },
    'resnet': {
        # Image classification
        'image-classification': [
            'microsoft/resnet-18',
            'microsoft/resnet-26',
            'microsoft/resnet-34',
            'microsoft/resnet-50',
            'microsoft/resnet-101',
            'microsoft/resnet-152',
        ],
    },
    'roformer': {
        # Feature extraction
        'feature-extraction': [
            'hf-tiny-model-private/tiny-random-RoFormerModel',
        ],

        # Text classification
        'text-classification': [
            'hf-tiny-model-private/tiny-random-RoFormerForSequenceClassification',
        ],

        # Token classification
        'token-classification': [
            'hf-tiny-model-private/tiny-random-RoFormerForTokenClassification',
        ],

        # TODO
        # # Text generation
        # 'text-generation': [
        #     'hf-tiny-model-private/tiny-random-RoFormerForCausalLM',
        # ],

        # Masked language modelling
        'fill-mask': [
            'alchemab/antiberta2',
            'hf-tiny-model-private/tiny-random-RoFormerForMaskedLM',
        ],

        # Question answering
        'question-answering': [
            'hf-tiny-model-private/tiny-random-RoFormerForQuestionAnswering',
        ],

        # Multiple choice
        'multiple-choice': [
            'hf-tiny-model-private/tiny-random-RoFormerForMultipleChoice',
        ],
    },
    'phi': {
        # Text generation
        'text-generation': [
            'hf-internal-testing/tiny-random-PhiForCausalLM',
            'susnato/phi-1_5_dev',
        ],
    },
    'qwen2': {
        # Text generation
        'text-generation': [
            'Qwen/Qwen1.5-0.5B',
            'Qwen/Qwen1.5-0.5B-Chat',
            'Qwen/Qwen1.5-1.8B',
            'Qwen/Qwen1.5-1.8B-Chat',
        ],
    },
    'roberta': {
        # Feature extraction
        'feature-extraction': [
            'sentence-transformers/all-distilroberta-v1',
            'sentence-transformers/all-roberta-large-v1',
        ],

        # Text classification
        'text-classification': [
            'roberta-large-mnli',
        ],

        # Token classification
        'token-classification': [
            'julien-c/EsperBERTo-small-pos',
        ],

        # Masked language modelling
        'fill-mask': [
            'roberta-base',
            'distilroberta-base',
        ],
    },
    'sam': {
        # Mask generation
        'mask-generation': [
            # SAM
            'facebook/sam-vit-base',
            'facebook/sam-vit-large',
            'facebook/sam-vit-huge',
            'wanglab/medsam-vit-base',

            # SlimSAM
            'nielsr/slimsam-50-uniform',
            'nielsr/slimsam-77-uniform',
        ],
    },
    'segformer': {
        # Image segmentation
        'image-segmentation': [
            'mattmdjaga/segformer_b0_clothes',
            'mattmdjaga/segformer_b2_clothes',
            'jonathandinu/face-parsing',

            'nvidia/segformer-b0-finetuned-cityscapes-768-768',
            'nvidia/segformer-b0-finetuned-cityscapes-512-1024',
            'nvidia/segformer-b0-finetuned-cityscapes-640-1280',
            'nvidia/segformer-b0-finetuned-cityscapes-1024-1024',
            'nvidia/segformer-b1-finetuned-cityscapes-1024-1024',
            'nvidia/segformer-b2-finetuned-cityscapes-1024-1024',
            'nvidia/segformer-b3-finetuned-cityscapes-1024-1024',
            'nvidia/segformer-b4-finetuned-cityscapes-1024-1024',
            'nvidia/segformer-b5-finetuned-cityscapes-1024-1024',
            'nvidia/segformer-b0-finetuned-ade-512-512',
            'nvidia/segformer-b1-finetuned-ade-512-512',
            'nvidia/segformer-b2-finetuned-ade-512-512',
            'nvidia/segformer-b3-finetuned-ade-512-512',
            'nvidia/segformer-b4-finetuned-ade-512-512',
            'nvidia/segformer-b5-finetuned-ade-640-640',
        ],

        # Image classification
        'image-classification': [
            'nvidia/mit-b0',
            'nvidia/mit-b1',
            'nvidia/mit-b2',
            'nvidia/mit-b3',
            'nvidia/mit-b4',
            'nvidia/mit-b5',
        ],
    },
    'siglip': {
        # Zero-shot image classification and feature extraction
        # (with and without `--split_modalities`)
        # NOTE: requires --opset 13
        'zero-shot-image-classification': [
            'nielsr/siglip-base-patch16-224',
        ],
    },
    'speecht5': {
        # Text-to-audio/Text-to-speech
        'text-to-audio': [
            'microsoft/speecht5_tts',
        ],
    },
    'stablelm': {
        # Text generation
        'text-generation': [
            'hf-internal-testing/tiny-random-StableLmForCausalLM',
            'stabilityai/stablelm-2-1_6b',
            'stabilityai/stablelm-2-zephyr-1_6b',
        ],
    },
    'squeezebert': {
        # Feature extraction
        'feature-extraction': [
            'squeezebert/squeezebert-uncased',
            'squeezebert/squeezebert-mnli',
        ],
    },
    'starcoder2': {
        # Text generation
        'text-generation': [
            'hf-internal-testing/tiny-random-Starcoder2ForCausalLM',
        ],
    },
    'swin': {
        # Image classification
        'image-classification': [
            'microsoft/swin-tiny-patch4-window7-224',
            'microsoft/swin-base-patch4-window7-224',
            'microsoft/swin-large-patch4-window12-384-in22k',
            'microsoft/swin-base-patch4-window7-224-in22k',
            'microsoft/swin-base-patch4-window12-384-in22k',
            'microsoft/swin-base-patch4-window12-384',
            'microsoft/swin-large-patch4-window7-224',
            'microsoft/swin-small-patch4-window7-224',
            'microsoft/swin-large-patch4-window7-224-in22k',
            'microsoft/swin-large-patch4-window12-384',
        ],
    },
    'swin2sr': {
        # Image-to-image (Super-resolution)
        'image-to-image': [
            'caidas/swin2SR-classical-sr-x2-64',
            'caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr',
            'caidas/swin2SR-classical-sr-x4-64',
            'caidas/swin2SR-compressed-sr-x4-48',
            'caidas/swin2SR-lightweight-x2-64',
        ],

        # Feature extraction
        'feature-extraction': [
            'hf-tiny-model-private/tiny-random-Swin2SRModel',
        ],
    },
    't5': {
        # Translation/Summarization
        ('translation', 'summarization'): [
            't5-small',
            't5-base',
            'google/t5-v1_1-small',
            'google/t5-v1_1-base',
            'google/flan-t5-small',
            'google/flan-t5-base',
        ],

        # Text-to-text
        'text2text-generation': [
            'MBZUAI/LaMini-Flan-T5-77M',
            'MBZUAI/LaMini-Flan-T5-248M',
            'MBZUAI/LaMini-Flan-T5-783M',
            'MBZUAI/LaMini-T5-61M',
            'MBZUAI/LaMini-T5-223M',
            'MBZUAI/LaMini-T5-738M',
            'declare-lab/flan-alpaca-base',
            'declare-lab/flan-alpaca-large',
        ],

        # Feature extraction
        'feature-extraction': [
            'sentence-transformers/sentence-t5-large',
            'hkunlp/instructor-base',
            'hkunlp/instructor-large',
        ],
    },
    'table-transformer': {
        # Object detection
        'object-detection': [
            'microsoft/table-transformer-detection',
            'microsoft/table-transformer-structure-recognition',
            'microsoft/table-transformer-structure-recognition-v1.1-all',
            'microsoft/table-transformer-structure-recognition-v1.1-fin',
            'microsoft/table-transformer-structure-recognition-v1.1-pub',
        ],
    },
    'trocr': {  # NOTE: also a `vision-encoder-decoder`
        # Text-to-image
        'text-to-image': [
            'microsoft/trocr-small-printed',
            'microsoft/trocr-base-printed',
            'microsoft/trocr-small-handwritten',
            'microsoft/trocr-base-handwritten',
        ],
    },
    'unispeech': {
        # Feature extraction
        'feature-extraction': [
            # Requires --task feature-extraction
            'microsoft/unispeech-large-1500h-cv',
        ],
        # TODO: add support for
        # # Automatic speech recognition
        # 'automatic-speech-recognition': [
        #     'microsoft/unispeech-1350-en-353-fr-ft-1h',
        #     'microsoft/unispeech-1350-en-17h-ky-ft-1h',
        #     'microsoft/unispeech-1350-en-90-it-ft-1h',
        #     'microsoft/unispeech-1350-en-168-es-ft-1h',
        # ],
    },
    'unispeech-sat': {
        # Feature extraction
        'feature-extraction': [
            # Requires --task feature-extraction
            'microsoft/unispeech-sat-base',
        ],

        # Audio XVector (e.g., for speaker verification)
        'audio-xvector': [
            'microsoft/unispeech-sat-base-plus-sv',
            'microsoft/unispeech-sat-base-sv',
            'microsoft/unispeech-sat-large-sv',
        ],

        # Audio frame classification
        'audio-frame-classification': [
            'microsoft/unispeech-sat-base-plus-sd',
        ],

        # Automatic speech recognition
        'automatic-speech-recognition': [
            'microsoft/unispeech-sat-base-100h-libri-ft',
        ],
    },
    'vision-encoder-decoder': {
        # Image-to-text
        'image-to-text': [
            'nlpconnect/vit-gpt2-image-captioning',
        ],
    },
    'vit': {
        # Feature extraction
        'feature-extraction': [
            'google/vit-base-patch16-224-in21k',
            'facebook/dino-vitb16',
            'facebook/dino-vits8',
            'facebook/dino-vitb8',
            'facebook/dino-vits16',
        ],
        # Image classification
        'image-classification': [
            'google/vit-base-patch16-224',
        ],
    },
    'vitmatte': {
        # Image matting
        'image-matting': [
            'hustvl/vitmatte-small-distinctions-646',
            'hustvl/vitmatte-base-distinctions-646',
            'hustvl/vitmatte-small-composition-1k',
            'hustvl/vitmatte-base-composition-1k',
        ],
    },
    'vits': {
        # Text-to-audio/Text-to-speech/Text-to-waveform
        'text-to-waveform': {
            # NOTE: requires --task text-to-waveform --skip_validation
            'echarlaix/tiny-random-vits',
            'facebook/mms-tts-eng',
            'facebook/mms-tts-rus',
            'facebook/mms-tts-hin',
            'facebook/mms-tts-yor',
            'facebook/mms-tts-spa',
            'facebook/mms-tts-fra',
            'facebook/mms-tts-ara',
            'facebook/mms-tts-ron',
            'facebook/mms-tts-vie',
            'facebook/mms-tts-deu',
            'facebook/mms-tts-kor',
            'facebook/mms-tts-por',
            # TODO add more checkpoints from
            # https://huggingface.co/models?other=vits&sort=trending&search=facebook-tts
        }
    },
    'wav2vec2': {
        # Feature extraction # NOTE: requires --task feature-extraction
        'feature-extraction': [
            'facebook/mms-300m',
            'facebook/mms-1b',
        ],

        # Audio classification
        'audio-classification': [
            'alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech',
            'superb/wav2vec2-base-superb-ks',
            'facebook/mms-lid-126',
            'facebook/mms-lid-256',
            'facebook/mms-lid-512',
            'facebook/mms-lid-1024',
            'facebook/mms-lid-2048',
            'facebook/mms-lid-4017',
        ],

        # Audio frame classification
        'audio-frame-classification': [
            'anton-l/wav2vec2-base-superb-sd',
        ],

        # Automatic speech recognition
        'automatic-speech-recognition': [
            'jonatasgrosman/wav2vec2-large-xlsr-53-english',
            'facebook/wav2vec2-base-960h',
            'facebook/mms-1b-l1107',
            'facebook/mms-1b-all',
            'facebook/mms-1b-fl102',
        ],
    },
    'wav2vec2-bert': {
        'feature-extraction': [
            'facebook/w2v-bert-2.0',
        ],

        # Automatic speech recognition
        'automatic-speech-recognition': [
            'hf-audio/wav2vec2-bert-CV16-en',
        ],
    },
    'wavlm': {
        # Feature extraction
        'feature-extraction': [
            'microsoft/wavlm-base',
            'microsoft/wavlm-base-plus',
            'microsoft/wavlm-large',
        ],

        # Audio frame classification
        'audio-frame-classification': [
            'anton-l/wav2vec2-base-superb-sd',
            'microsoft/wavlm-base-plus-sd',
        ],

        # Audio XVector (e.g., for speaker verification)
        'audio-xvector': [
            'microsoft/wavlm-base-plus-sv',
            'microsoft/wavlm-base-sv',
        ],
    },
    'whisper': {
        # Automatic speech recognition
        'automatic-speech-recognition': [
            'openai/whisper-tiny',
            'openai/whisper-tiny.en',
            'openai/whisper-base',
            'openai/whisper-base.en',
            'openai/whisper-small',
            'openai/whisper-small.en',
            'openai/whisper-medium',
            'openai/whisper-medium.en',
            'openai/whisper-large',
            'openai/whisper-large-v2',
            'NbAiLab/nb-whisper-tiny-beta',
            'NbAiLab/nb-whisper-base-beta',
            'NbAiLab/nb-whisper-small-beta',
            'NbAiLab/nb-whisper-medium-beta',
            'NbAiLab/nb-whisper-large-beta',
        ],
    },
    'xlm': {
        # Masked language modelling
        'fill-mask': [
            'xlm-clm-ende-1024',
            'xlm-mlm-ende-1024',
            'xlm-clm-enfr-1024',
            'xlm-mlm-enfr-1024',
            'xlm-mlm-17-1280',
            'xlm-mlm-100-1280',
            'xlm-mlm-en-2048',
            'xlm-mlm-enro-1024',
            'xlm-mlm-tlm-xnli15-1024',
            'xlm-mlm-xnli15-1024',
        ],
    },
    'xlm-roberta': {
        # Masked language modelling
        'fill-mask': [
            'xlm-roberta-base'
        ],
    },
    'yolos': {
        # Object detection
        'object-detection': [
            # Object detection
            'hustvl/yolos-tiny',
            'hustvl/yolos-small',
            'hustvl/yolos-base',
            'hustvl/yolos-small-dwr',
            'hustvl/yolos-small-300',
        ],
    },
}


def main():
    for model_type, tasks in SUPPORTED_MODELS.items():
        for task, model_ids in tasks.items():
            print(f'# {model_type:=^80}')
            for model_id in model_ids:
                print(
                    f'python -m scripts.convert --quantize --model_id {model_id}')
            print()


if __name__ == '__main__':
    main()
