import random
from typing import Optional, Tuple

from optimum.exporters.onnx.config import TextDecoderOnnxConfig
from optimum.utils import NormalizedTextConfig, DummyInputGenerator, DEFAULT_DUMMY_SHAPES, DummyTextInputGenerator, NormalizedConfig

class OpenElmDummyPastKeyValuesGenerator(DummyInputGenerator):

    SUPPORTED_INPUT_NAMES = ("past_key_values", )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        self.num_layers = normalized_config.num_layers
        self.num_kv_heads = normalized_config.num_kv_heads
        self.num_query_heads = normalized_config.num_query_heads
        self.head_dim = normalized_config.head_dim

        self.hidden_size = normalized_config.model_dim
        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size
        if random_sequence_length_range:
            low, high = random_sequence_length_range
            self.sequence_length = random.randint(low, high)
        else:
            self.sequence_length = sequence_length

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        data = []
        for i in range(self.num_layers):
            kv_shape = (
                self.batch_size,
                self.num_kv_heads[i],
                self.sequence_length,
                self.head_dim,
            )
            data.append((
                self.random_float_tensor(kv_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(kv_shape, framework=framework, dtype=float_dtype),
            ))
        return data


class OpenElmOnnxConfig(TextDecoderOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, OpenElmDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = OpenElmDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_kv_heads="num_kv_heads",
        num_query_heads="num_query_heads",
        num_layers="num_transformer_layers",
        allow_new=True,
    )
