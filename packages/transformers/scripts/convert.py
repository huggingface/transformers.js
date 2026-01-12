
import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser
)

import onnxslim
from optimum.exporters.onnx import main_export, export_models
from optimum.onnx.graph_transformations import check_and_save_model
from optimum.exporters.tasks import TasksManager

from .quantize import QuantizationArguments, quantize

NO_PER_CHANNEL_REDUCE_RANGE_MODELS = {
    # Decoder-only models
    'codegen',
    'gpt2',
    'gpt_bigcode',
    'gptj',
    'gpt-neo',
    'gpt-neox',
    'mpt',
    'bloom',
    'llama',
    'gemma',
    'opt',
    'mistral',
    'falcon',
    'phi',
    'phi3',
    'qwen2',
    'stablelm',
    'starcoder2',
    'openelm',
    'mobilellm',
    'olmo',

    # Encoder-decoder models
    'whisper',
    'vision-encoder-decoder',

    # Encoder-only models
    'owlv2',
    'wavlm',
    'wav2vec2',
    'unispeech',
    'unispeech-sat',
}

MODELS_WITHOUT_TOKENIZERS = [
    'wav2vec2',
    'wav2vec2-bert',
    'wavlm',
    'hubert',
    'unispeech',
    'unispeech-sat',
]


class QuantMode(Enum):
    # F32 = 'fp32'
    FP16 = 'fp16'
    Q8 = 'q8'
    QI8 = 'int8'
    QU8 = 'uint8'
    Q4 = 'q4'
    BNB4 = 'bnb4'


@dataclass
class ConversionArguments:
    """
    Arguments used for converting HuggingFace models to onnx.
    """

    model_id: str = field(
        metadata={
            "help": "Model identifier"
        }
    )
    tokenizer_id: str = field(
        default=None,
        metadata={
            "help": "Tokenizer identifier (if different to `model_id`)"
        }
    )
    quantize: bool = field(
        default=False,
        metadata={
            "help": "Whether to quantize the model."
        }
    )
    output_parent_dir: str = field(
        default='./models/',
        metadata={
            "help": "Path where the converted model will be saved to."
        }
    )

    task: Optional[str] = field(
        default='auto',
        metadata={
            "help": (
                "The task to export the model for. If not specified, the task will be auto-inferred based on the model. Available tasks depend on the model, but are among:"
                f" {str(TasksManager.get_all_tasks())}. For decoder models, use `xxx-with-past` to export the model using past key values in the decoder."
            )
        }
    )
    library_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The library name to use for the export. If not specified, the library name will be auto-inferred based on the model."
            )
        }
    )


    variant: Optional[str] = field(
        default='default',
        metadata={
            "help": "The variant of the ONNX export to use."
        }
    )
    opset: int = field(
        default=None,
        metadata={
            "help": (
                "If specified, ONNX opset version to export the model with. Otherwise, the default opset will be used."
            )
        }
    )

    device: str = field(
        default='cpu',
        metadata={
            "help": 'The device to use to do the export.'
        }
    )
    skip_validation: bool = field(
        default=False,
        metadata={
            "help": "Whether to skip validation of the converted model"
        }
    )

    output_attentions: bool = field(
        default=False,
        metadata={
            "help": "Whether to output attentions from the model. NOTE: This is only supported for whisper models right now."
        }
    )

    split_modalities: bool = field(
        default=False,
        metadata={
            "help": "Whether to split multimodal models. NOTE: This is only supported for CLIP models right now."
        }
    )

    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Allows to use custom code for the modeling hosted in the model repository. This option should only be set for repositories"
            "you trust and in which you have read the code, as it will execute on your local machine arbitrary code present in the model repository."
        }
    )

    custom_onnx_configs: str = field(
        default=None,
        metadata={
            "help": "Experimental usage: override the default ONNX config used for the given model. This argument may be useful for advanced users "
            "that desire a finer-grained control on the export."
        }
    )
    skip_onnxslim: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to skip onnxslim."
        }
    )

def main():

    parser = HfArgumentParser(
        (ConversionArguments, QuantizationArguments)
    )
    conv_args, quantization_args = parser.parse_args_into_dataclasses()

    model_id = conv_args.model_id
    tokenizer_id = conv_args.tokenizer_id or model_id

    output_model_folder = os.path.join(conv_args.output_parent_dir, model_id)

    # Create output folder
    os.makedirs(output_model_folder, exist_ok=True)

    from_pretrained_kwargs = dict(
        trust_remote_code=conv_args.trust_remote_code,
    )

    # Saving the model config
    config = AutoConfig.from_pretrained(model_id, **from_pretrained_kwargs)

    custom_kwargs = {}
    if conv_args.custom_onnx_configs is not None:
        if conv_args.task == 'auto':
            raise Exception(
                '`--task` must be set when exporting with `--custom_onnx_configs`')
        custom_onnx_configs = json.loads(conv_args.custom_onnx_configs)

        for key in custom_onnx_configs:
            onnx_configs = TasksManager._SUPPORTED_MODEL_TYPE[custom_onnx_configs[key]]['onnx']
            mapping = onnx_configs[conv_args.task]
            new_kwargs = {}
            if conv_args.task.startswith('text-generation'):
                new_kwargs['use_past_in_inputs'] = True

            custom_onnx_configs[key] = mapping.func(
                config, **mapping.keywords, **new_kwargs)

        custom_kwargs['custom_onnx_configs'] = custom_onnx_configs

    tokenizer = None
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **from_pretrained_kwargs)

    except KeyError:
        pass  # No Tokenizer

    except Exception as e:
        if config.model_type not in MODELS_WITHOUT_TOKENIZERS:
            raise e

    core_export_kwargs = dict(
        opset=conv_args.opset,
        device=conv_args.device,
        trust_remote_code=conv_args.trust_remote_code,
        **custom_kwargs,
    )

    export_kwargs = dict(
        model_name_or_path=model_id,
        output=output_model_folder,
        task=conv_args.task,
        do_validation=not conv_args.skip_validation,
        _variant=conv_args.variant,
        library_name=conv_args.library_name,
        **core_export_kwargs,
    )

    # Handle special cases
    if config.model_type == 'marian':
        from .extra.marian import generate_tokenizer_json
        tokenizer_json = generate_tokenizer_json(model_id, tokenizer)

        with open(os.path.join(output_model_folder, 'tokenizer.json'), 'w', encoding='utf-8') as fp:
            json.dump(tokenizer_json, fp, indent=4)

    elif config.model_type == 'esm':
        from .extra.esm import generate_fast_tokenizer
        fast_tokenizer = generate_fast_tokenizer(tokenizer)
        fast_tokenizer.save(os.path.join(
            output_model_folder, 'tokenizer.json'))

    elif config.model_type == 'whisper':
        if conv_args.output_attentions:
            from .extra.whisper import get_main_export_kwargs

            export_kwargs.update(
                **get_main_export_kwargs(config, "automatic-speech-recognition")
            )

    elif config.model_type in ('wav2vec2', 'wav2vec2-bert', 'hubert', 'unispeech', 'unispeech-sat'):
        if tokenizer is not None:
            from .extra.wav2vec2 import generate_tokenizer_json
            tokenizer_json = generate_tokenizer_json(tokenizer)

            with open(os.path.join(output_model_folder, 'tokenizer.json'), 'w', encoding='utf-8') as fp:
                json.dump(tokenizer_json, fp, indent=4)

    elif config.model_type == 'vits':
        if tokenizer is not None:
            from .extra.vits import generate_tokenizer_json
            tokenizer_json = generate_tokenizer_json(tokenizer)

            with open(os.path.join(output_model_folder, 'tokenizer.json'), 'w', encoding='utf-8') as fp:
                json.dump(tokenizer_json, fp, indent=4)

    elif config.model_type == 'speecht5':
        # TODO allow user to specify vocoder path
        export_kwargs["model_kwargs"] = {
            "vocoder": "microsoft/speecht5_hifigan"}

        if tokenizer is not None:
            from .extra.speecht5 import generate_tokenizer_json
            tokenizer_json = generate_tokenizer_json(tokenizer)

            with open(os.path.join(output_model_folder, 'tokenizer.json'), 'w', encoding='utf-8') as fp:
                json.dump(tokenizer_json, fp, indent=4)

    elif config.model_type in ('owlvit', 'owlv2'):
        # Override default batch size to 1, needed because non-maximum suppression is performed for exporting.
        # For more information, see https://github.com/huggingface/optimum/blob/e3b7efb1257c011db907ef40ab340e795cc5684c/optimum/exporters/onnx/model_configs.py#L1028-L1032
        export_kwargs['batch_size'] = 1

    elif config.model_type == 'openelm':
        from .extra.openelm import OpenElmOnnxConfig

        config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=conv_args.trust_remote_code)

        onnx_config = OpenElmOnnxConfig(
            config=config,
            task="text-generation",
            use_past=True,
            use_past_in_inputs=True,
        )

        custom_onnx_configs = {
            "model": onnx_config,
        }

        export_kwargs['task'] = "text-generation-with-past"
        export_kwargs['custom_onnx_configs'] = custom_onnx_configs

    else:
        pass  # TODO

    # Step 1. convert huggingface model to onnx
    if not conv_args.split_modalities:
        main_export(**export_kwargs)
    else:
        custom_export_kwargs = dict(
            output_dir=output_model_folder,
            **core_export_kwargs,
        )

        if config.model_type == 'clip':
            # Handle special case for exporting text and vision models separately
            from .extra.clip import CLIPTextModelWithProjectionOnnxConfig, CLIPVisionModelWithProjectionOnnxConfig
            from transformers.models.clip import CLIPTextModelWithProjection, CLIPVisionModelWithProjection

            text_model = CLIPTextModelWithProjection.from_pretrained(
                model_id, **from_pretrained_kwargs)
            vision_model = CLIPVisionModelWithProjection.from_pretrained(
                model_id, **from_pretrained_kwargs)

            export_models(
                models_and_onnx_configs={
                    "text_model": (text_model, CLIPTextModelWithProjectionOnnxConfig(text_model.config)),
                    "vision_model": (vision_model, CLIPVisionModelWithProjectionOnnxConfig(vision_model.config)),
                },
                **custom_export_kwargs,
            )

        elif config.model_type == 'siglip':
            # Handle special case for exporting text and vision models separately
            from .extra.siglip import SiglipTextModelOnnxConfig, SiglipVisionModelOnnxConfig
            from transformers.models.siglip import SiglipTextModel, SiglipVisionModel

            text_model = SiglipTextModel.from_pretrained(
                model_id, **from_pretrained_kwargs)
            vision_model = SiglipVisionModel.from_pretrained(
                model_id, **from_pretrained_kwargs)

            export_models(
                models_and_onnx_configs={
                    "text_model": (text_model, SiglipTextModelOnnxConfig(text_model.config)),
                    "vision_model": (vision_model, SiglipVisionModelOnnxConfig(vision_model.config)),
                },
                **custom_export_kwargs,
            )

        # TODO: Enable once https://github.com/huggingface/optimum/pull/1552 is merged
        # elif config.model_type == 'clap':
        #     # Handle special case for exporting text and audio models separately
        #     from .extra.clap import ClapTextModelWithProjectionOnnxConfig, ClapAudioModelWithProjectionOnnxConfig
        #     from transformers.models.clap import ClapTextModelWithProjection, ClapAudioModelWithProjection

        #     text_model = ClapTextModelWithProjection.from_pretrained(model_id, **from_pretrained_kwargs)
        #     audio_model = ClapAudioModelWithProjection.from_pretrained(model_id, **from_pretrained_kwargs)

        #     export_models(
        #         models_and_onnx_configs={
        #             "text_model": (text_model, ClapTextModelWithProjectionOnnxConfig(text_model.config)),
        #             "audio_model": (audio_model, ClapAudioModelWithProjectionOnnxConfig(audio_model.config)),
        #         },
        #         **custom_export_kwargs,
        #     )
        else:
            raise Exception(
                f'Unable to export {config.model_type} model with `--split_modalities`.')

    os.makedirs(os.path.join(output_model_folder, 'onnx'), exist_ok=True)

    if not conv_args.skip_onnxslim:
        onnx_models = [os.path.join(output_model_folder, x)
                    for x in os.listdir(output_model_folder) if x.endswith('.onnx')]

        for model in onnx_models:
            try:
                slimmed_model = onnxslim.slim(model)
                check_and_save_model(slimmed_model, model)
            except Exception as e:
                print(f"Failed to slim {model}: {e}")

    # Step 2. (optional, recommended) quantize the converted model for fast inference and to reduce model size.
    if conv_args.quantize:

        # Possibly update quantize config with model specific defaults
        use_per_channel_reduce_range = config.model_type not in NO_PER_CHANNEL_REDUCE_RANGE_MODELS

        if quantization_args.per_channel is None:
            quantization_args.per_channel = use_per_channel_reduce_range
        if quantization_args.reduce_range is None:
            quantization_args.reduce_range = use_per_channel_reduce_range

        quantize(
            output_model_folder,
            os.path.join(output_model_folder, 'onnx'),
            quantization_args,
        )
        with open(os.path.join(output_model_folder, 'quantize_config.json'), 'w') as fp:
            json.dump(asdict(quantization_args), fp, indent=4)

    # Step 3. Move .onnx files to the 'onnx' subfolder
    for file in os.listdir(output_model_folder):
        if file.endswith(('.onnx', '.onnx_data')):
            shutil.move(os.path.join(output_model_folder, file),
                        os.path.join(output_model_folder, 'onnx', file))

    # Step 4. Update the generation config if necessary
    if config.model_type == 'whisper':
        from transformers import GenerationConfig
        from .extra.whisper import get_alignment_heads

        generation_config = GenerationConfig.from_pretrained(
            model_id, **from_pretrained_kwargs)
        alignment_heads = get_alignment_heads(config)
        if alignment_heads:
            generation_config.alignment_heads = alignment_heads
        generation_config.save_pretrained(output_model_folder)


if __name__ == '__main__':
    main()
