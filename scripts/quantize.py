from enum import Enum

from tqdm import tqdm
from typing import Set, List, Optional
import onnx
import os

from dataclasses import dataclass, field

from transformers import HfArgumentParser
from optimum.onnx.graph_transformations import check_and_save_model

from onnxruntime.quantization import QuantType, QuantizationMode
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.registry import IntegerOpsRegistry
from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer
from onnxruntime.quantization.matmul_bnb4_quantizer import MatMulBnb4Quantizer
from onnxconverter_common import float16
import onnx_graphsurgeon as gs


class QuantMode(Enum):
    # F32 = 'fp32'
    FP16 = "fp16"
    Q8 = "q8"
    QI8 = "int8"
    QU8 = "uint8"
    Q4 = "q4"
    Q4F16 = "q4f16"
    BNB4 = "bnb4"


QUANTIZE_SUFFIX_MAPPING = {
    QuantMode.Q8: "quantized",
}

QUANTIZE_OPTIONS = tuple(x.value for x in QuantMode)


@dataclass
class IOArguments:
    """
    Arguments to specify input and output folders
    """
    input_folder: str = field(
        metadata={
            "help": "Path of the input folder containing the .onnx models to quantize"
        }
    )
    output_folder: str = field(
        metadata={
            "help": "Path of the output folder where the quantized .onnx models will be saved"
        }
    )

@dataclass
class QuantizationArguments:
    """
    Arguments for quantizing ONNX models
    """

    modes: QuantMode = field(
        default=QUANTIZE_OPTIONS,
        metadata={
            "help": "Quantization mode to use.",
            "choices": QUANTIZE_OPTIONS,
            "nargs": "+",
        },
    )

    # 8-bit quantization
    per_channel: bool = field(
        default=None, metadata={"help": "Whether to quantize weights per channel"}
    )
    reduce_range: bool = field(
        default=None,
        metadata={
            "help": "Whether to quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode"
        },
    )

    # 4-bit quantization
    block_size: int = field(
        default=None,
        metadata={
            "help": "Block size for blockwise quantization. Note: bnb.nn.Linear4bit only uses block_size=64"
        },
    )

    # MatMul4BitsQuantizer
    is_symmetric: bool = field(
        default=True,
        metadata={"help": "Indicate whether to quantize the model symmetrically"},
    )
    accuracy_level: int = field(
        default=None,
        metadata={
            "help": "Accuracy level of the 4-bit quantized MatMul computation. "
            "Refer to the MatMulNBits contrib op's 'accuracy_level' attribute for details "
            "(https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftmatmulnbits)."
        },
    )

    # MatMulBnb4Quantizer
    quant_type: int = field(
        default=MatMulBnb4Quantizer.NF4,
        metadata={
            "help": "Quantization data type. 0: FP4, 1: NF4",
            "choices": [MatMulBnb4Quantizer.FP4, MatMulBnb4Quantizer.NF4],
        },
    )

    op_block_list: List[str] = field(
        default=None,
        metadata={
            "help": "List of operators to exclude from quantization."
            "Can be any standard ONNX operator (see https://onnx.ai/onnx/operators/)"
            "or your custom implemented operators.",
            "nargs": "+",
        },
    )


def get_operators(model: onnx.ModelProto) -> Set[str]:
    operators = set()

    def traverse_graph(graph):
        for node in graph.node:
            operators.add(node.op_type)
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    traverse_graph(attr.g)

    traverse_graph(model.graph)
    return operators


def quantize_q8(
    model: onnx.ModelProto,
    save_path: str,
    per_channel: bool,
    reduce_range: bool,
    weight_type: QuantType,
    op_block_list: Optional[List[str]]
):
    """
    Quantize the weights of the model from float32 to int8/uint8

    Uses unsigned ints for activation values, signed ints for weights, per
    https://onnxruntime.ai/docs/performance/quantization.html#data-type-selection
    it is faster on most CPU architectures
    """

    op_types_to_quantize = set(IntegerOpsRegistry.keys())
    if op_block_list is not None:
        op_types_to_quantize.difference_update(op_block_list)

    quantizer = ONNXQuantizer(
        model,
        per_channel,
        reduce_range,
        mode=QuantizationMode.IntegerOps,
        static=False,
        weight_qType=weight_type,
        activation_qType=QuantType.QUInt8,  # dynamic activation only supports uint8
        tensors_range=None,
        nodes_to_quantize=[],
        nodes_to_exclude=[],
        op_types_to_quantize=op_types_to_quantize,
        extra_options=dict(
            EnableSubgraph=True,
            MatMulConstBOnly=True,
        ),
    )

    quantizer.quantize_model()
    check_and_save_model(quantizer.model.model, save_path)


def quantize_fp16(
    model: onnx.ModelProto,
    save_path: str,
    op_block_list: Optional[List[str]]
):
    """
    Quantize the weights of the model from float32 to float16
    """

    # Check whether we should disable shape infer:
    # ValueError: Message onnx.ModelProto exceeds maximum protobuf size of 2GB: 2338583841
    disable_shape_infer = model.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF

    blocked_ops = set(float16.DEFAULT_OP_BLOCK_LIST)
    if op_block_list is not None:
        blocked_ops.update(op_block_list)

    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=disable_shape_infer,
        op_block_list=blocked_ops,
    )
    graph = gs.import_onnx(model_fp16)
    graph.toposort()
    model_fp16 = gs.export_onnx(graph)
    check_and_save_model(model_fp16, save_path)


def quantize_q4(
    model: onnx.ModelProto,
    save_path: str | None,
    block_size: int,
    is_symmetric: bool,
    accuracy_level: int,
):
    """
    Quantize the weights of the model from float32 to 4-bit int
    """

    quantizer = MatMul4BitsQuantizer(
        model=model,
        block_size=block_size,
        is_symmetric=is_symmetric,
        accuracy_level=accuracy_level,
    )
    quantizer.process()
    if save_path:
        check_and_save_model(quantizer.model.model, save_path)
    return quantizer.model.model


def quantize_bnb4(
    model: onnx.ModelProto,
    save_path: str,
    block_size: int,
    quant_type: int,
):
    """
    Quantize the weights of the model from float32 to 4-bit int using MatMulBnb4Quantizer
    """

    quantizer = MatMulBnb4Quantizer(
        model=model,
        block_size=block_size,
        quant_type=quant_type,
    )
    quantizer.process()
    check_and_save_model(quantizer.model.model, save_path)
    return quantizer.model.model


def quantize(input_folder, output_folder, quantization_args: QuantizationArguments):

    # (Step 1) Validate the arguments
    if not quantization_args.modes:
        raise ValueError("At least one quantization mode must be specified")

    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder {input_folder} does not exist")

    model_names_or_paths = [
        os.path.join(input_folder, file)
        for file in os.listdir(input_folder)
        if file.endswith(".onnx")
    ]
    if not model_names_or_paths:
        raise ValueError(f"No .onnx models found in {input_folder}")

    os.makedirs(output_folder, exist_ok=True)

    # (Step 2) Quantize the models
    for model_path in (progress_models := tqdm(model_names_or_paths)):
        progress_models.set_description(f"Processing {model_path}")

        file_name_without_extension = os.path.splitext(os.path.basename(model_path))[0]

        for mode in (progress := tqdm(quantization_args.modes)):
            progress.set_description(f" - Quantizing to {mode}")
            mode = QuantMode(mode)
            suffix = QUANTIZE_SUFFIX_MAPPING.get(mode, mode.value)
            save_path = os.path.join(
                output_folder,
                f"{file_name_without_extension}_{suffix}.onnx",
            )

            # NOTE: Unfortunately, we need to reload the model for each quantization mode,
            # which is memory inefficient. This is because the quantization functions
            # modify the model in-place, and we need to keep the original model for each mode.
            model = onnx.load_model(model_path)

            if mode == QuantMode.FP16:
                quantize_fp16(
                    model,
                    save_path,
                    quantization_args.op_block_list
                )

            elif mode in (QuantMode.Q4, QuantMode.Q4F16):
                block_size = quantization_args.block_size or 32

                q4_model = quantize_q4(
                    model,
                    save_path=None if mode == QuantMode.Q4F16 else save_path,
                    block_size=block_size,
                    is_symmetric=quantization_args.is_symmetric,
                    accuracy_level=quantization_args.accuracy_level,
                )
                if mode == QuantMode.Q4F16:
                    quantize_fp16(
                        q4_model,
                        save_path,
                        quantization_args.op_block_list,
                    )

            elif mode == QuantMode.BNB4:
                quantize_bnb4(
                    model,
                    save_path,
                    block_size=quantization_args.block_size or 64,
                    quant_type=(
                        quantization_args.quant_type
                        if quantization_args.quant_type is not None
                        else MatMulBnb4Quantizer.NF4
                    ),
                )

            elif mode in (QuantMode.Q8, QuantMode.QI8, QuantMode.QU8):
                if mode == QuantMode.Q8:
                    # NOTE:
                    # As of 2024/06/28, the current latest version of onnxruntime-web is 1.18.0, and does not support INT8 weights for Conv layers.
                    # If you attempt to run a model with INT8 weights for Conv layers, you will get an error like:
                    # `Can't create a session. ERROR_CODE: 9, ERROR_MESSAGE: Could not find an implementation for ConvInteger(10) node with name '/.../Conv_quant'`
                    #
                    # For this reason, we choose model weight types to ensure compatibility with onnxruntime-web.
                    #
                    # As per docs, signed weight type (QInt8) is faster on most CPUs, so, we use that unless the model contains a Conv layer.
                    # For more information, see:
                    #  - https://github.com/microsoft/onnxruntime/issues/3130#issuecomment-1105200621
                    #  - https://github.com/microsoft/onnxruntime/issues/2339
                    op_types = get_operators(model)
                    weight_type = (
                        QuantType.QUInt8 if "Conv" in op_types else QuantType.QInt8
                    )

                elif mode == QuantMode.QI8:
                    weight_type = QuantType.QInt8

                else:  # mode == QuantMode.QU8:
                    weight_type = QuantType.QUInt8

                quantize_q8(
                    model,
                    save_path,
                    per_channel=quantization_args.per_channel,
                    reduce_range=quantization_args.reduce_range,
                    weight_type=weight_type,
                    op_block_list=quantization_args.op_block_list,
                )


def main():
    parser = HfArgumentParser((IOArguments, QuantizationArguments))
    io_args, quantization_args = parser.parse_args_into_dataclasses()
    input_folder = io_args.input_folder
    output_folder = io_args.output_folder
    quantize(input_folder, output_folder, quantization_args)

if __name__ == "__main__":
    main()
