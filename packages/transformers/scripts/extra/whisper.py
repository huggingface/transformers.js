from optimum.exporters.onnx.model_configs import WhisperOnnxConfig

from optimum.exporters.onnx.base import ConfigBehavior
from typing import Dict
import logging

logger = logging.getLogger(__name__)

# List of [layer, head] pairs that select the cross-attention heads that are highly correlated to word-level timing.
# Source: https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a
ALIGNMENT_HEADS_MAPPING = {
    'whisper-tiny.en': [[1, 0], [2, 0], [2, 5], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4]],
    'whisper-tiny': [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]],
    'whisper-base.en': [[3, 3], [4, 7], [5, 1], [5, 5], [5, 7]],
    'whisper-base': [[3, 1], [4, 2], [4, 3], [4, 7], [5, 1], [5, 2], [5, 4], [5, 6]],
    'whisper-small.en': [[6, 6], [7, 0], [7, 3], [7, 8], [8, 2], [8, 5], [8, 7], [9, 0], [9, 4], [9, 8], [9, 10], [10, 0], [10, 1], [10, 2], [10, 3], [10, 6], [10, 11], [11, 2], [11, 4]],
    'whisper-small': [[5, 3], [5, 9], [8, 0], [8, 4], [8, 7], [8, 8], [9, 0], [9, 7], [9, 9], [10, 5]],
    'whisper-medium.en': [[11, 4], [14, 1], [14, 12], [14, 14], [15, 4], [16, 0], [16, 4], [16, 9], [17, 12], [17, 14], [18, 7], [18, 10], [18, 15], [20, 0], [20, 3], [20, 9], [20, 14], [21, 12]],
    'whisper-medium': [[13, 15], [15, 4], [15, 15], [16, 1], [20, 0], [23, 4]],
    'whisper-large-v3-turbo': [[2, 4], [2, 11], [3, 3], [3, 6], [3, 11], [3, 14]],
    'whisper-large-v2': [[10, 12], [13, 17], [16, 11], [16, 12], [16, 13], [17, 15], [17, 16], [18, 4], [18, 11], [18, 19], [19, 11], [21, 2], [21, 3], [22, 3], [22, 9], [22, 12], [23, 5], [23, 7], [23, 13], [25, 5], [26, 1], [26, 12], [27, 15]],
    'whisper-large': [[9, 19], [11, 2], [11, 4], [11, 17], [22, 7], [22, 11], [22, 17], [23, 2], [23, 15]],
}


class CustomWhisperOnnxConfig(WhisperOnnxConfig):
    """
    Custom ONNX config for Whisper models to output cross attentions.
    Needed to compute token-level timestamps.
    """
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = super().outputs

        if self._behavior is ConfigBehavior.DECODER:
            for i in range(self._config.decoder_layers):
                common_outputs[f"cross_attentions.{i}"] = {
                    0: "batch_size",
                    2: "decoder_sequence_length",
                    3: "encoder_sequence_length_out"
                }
        return common_outputs

def get_main_export_kwargs(config, task):

    # See https://github.com/huggingface/optimum/blob/a39b1f5637af9725c0c788b86ca1fdf71ad3dcc2/docs/source/exporters/onnx/usage_guides/export_a_model.mdx#L264
    custom_config = CustomWhisperOnnxConfig(config=config, task=task)

    custom_onnx_configs = dict(
        encoder_model=custom_config.with_behavior("encoder"),
        decoder_model=custom_config.with_behavior("decoder", use_past=True, use_past_in_inputs=False),
        decoder_with_past_model=custom_config.with_behavior("decoder", use_past=True, use_past_in_inputs=True),
    )

    return dict(
        model_kwargs={"output_attentions": True},
        custom_onnx_configs=custom_onnx_configs,
    )


def get_alignment_heads(config):
    if getattr(config, '_name_or_path', None) is None:
        logger.warning(
            "Unable to determine model type from config. Please specify `_name_or_path` in the config.")
        return None

    for model_name, heads in ALIGNMENT_HEADS_MAPPING.items():
        if model_name in config._name_or_path:
            return heads

    logger.warning(f"Unknown model type: {config._name_or_path}. Please add one of the following model types to `_name_or_path` in the config file: {list(ALIGNMENT_HEADS_MAPPING.keys())}")
    return None
