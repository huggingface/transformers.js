# Support exporting vision and text models separately:
# Adapted from https://github.com/huggingface/optimum/issues/1186#issuecomment-1637641760

from optimum.exporters.onnx.model_configs import SiglipTextOnnxConfig, ViTOnnxConfig
from typing import Dict


class SiglipVisionOnnxConfig(ViTOnnxConfig):
    pass


class SiglipTextModelOnnxConfig(SiglipTextOnnxConfig):
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
            "pooler_output": {0: "batch_size"},
        }

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)
        if framework == "pt":
            import torch
            dummy_inputs["input_ids"] = dummy_inputs["input_ids"].to(dtype=torch.int64)
        return dummy_inputs

class SiglipVisionModelOnnxConfig(SiglipVisionOnnxConfig):
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "last_hidden_state": {0: "batch_size"},
            "pooler_output": {0: "batch_size"},
        }
