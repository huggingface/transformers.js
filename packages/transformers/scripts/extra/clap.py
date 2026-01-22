# TODO: Enable once https://github.com/huggingface/optimum/pull/1552 is merged

# # Support exporting vision and text models separately:
# # Adapted from https://github.com/huggingface/optimum/issues/1186#issuecomment-1637641760

# from optimum.exporters.onnx.model_configs import CLAPTextWithProjectionOnnxConfig, AudioOnnxConfig
# from optimum.utils.normalized_config import NormalizedAudioConfig
# from optimum.utils.input_generators import DummyAudioInputGenerator
# from typing import Dict


# class ClapAudioModelWithProjectionOnnxConfig(AudioOnnxConfig):
#     NORMALIZED_CONFIG_CLASS = NormalizedAudioConfig
#     DUMMY_INPUT_GENERATOR_CLASSES = (DummyAudioInputGenerator, )

#     @property
#     def inputs(self) -> Dict[str, Dict[int, str]]:
#         return {
#             "input_features": {0: "audio_batch_size", 1: "num_channels", 2: "height", 3: "width"}, # As described in modeling_clap.py
#         }

#     @property
#     def outputs(self) -> Dict[str, Dict[int, str]]:
#         return {
#             "audio_embeds": {0: "batch_size"},
#         }

# class ClapTextModelWithProjectionOnnxConfig(CLAPTextWithProjectionOnnxConfig):
#     @property
#     def outputs(self) -> Dict[str, Dict[int, str]]:
#         return {
#             "text_embeds": {0: "batch_size"},
#         }

#     def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
#         dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)
#         if framework == "pt":
#             import torch
#             dummy_inputs["input_ids"] = dummy_inputs["input_ids"].to(dtype=torch.int64)
#         return dummy_inputs
