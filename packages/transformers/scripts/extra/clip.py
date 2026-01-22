# Support exporting vision and text models separately:
# Adapted from https://github.com/huggingface/optimum/issues/1186#issuecomment-1637641760

from optimum.exporters.onnx.model_configs import CLIPTextOnnxConfig, ViTOnnxConfig
from typing import Dict


class CLIPVisionOnnxConfig(ViTOnnxConfig):
    pass


class CLIPTextModelWithProjectionOnnxConfig(CLIPTextOnnxConfig):
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "text_embeds": {0: "batch_size"},
        }

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)
        if framework == "pt":
            import torch
            dummy_inputs["input_ids"] = dummy_inputs["input_ids"].to(dtype=torch.int64)
        return dummy_inputs

class CLIPVisionModelWithProjectionOnnxConfig(CLIPVisionOnnxConfig):
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "image_embeds": {0: "batch_size"},
        }
