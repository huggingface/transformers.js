import onnx
from typing import Optional, Union
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)


# https://github.com/onnx/onnx/pull/6556
MAXIMUM_PROTOBUF = 2147483648  # 2GiB


def strict_check_model(model_or_path: Union[onnx.ModelProto, str, Path]):
    try:
        onnx.checker.check_model(model_or_path, full_check=True)
    except Exception as e:
        if "No Op registered for" in str(e):
            pass
        else:
            raise e


def check_and_save_model(model: onnx.ModelProto, save_path: Optional[Union[str, Path]]):
    # for large models, a path must be provided instead of a ModelProto:
    # https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#checking-a-large-onnx-model-2gb
    if model.ByteSize() < MAXIMUM_PROTOBUF:
        # For the try catch, refer to https://github.com/microsoft/onnxruntime/issues/14768
        strict_check_model(model)
        if save_path:
            # Overwrite.
            save_path = Path(save_path).as_posix()
            external_file_name = os.path.basename(save_path) + "_data"
            # path/to/model.onnx_data
            external_path = os.path.join(os.path.dirname(save_path), external_file_name)

            if save_path.endswith(".onnx") and os.path.isfile(save_path):
                os.remove(save_path)
            if os.path.isfile(external_path):
                # The new model may be below the maximum protobuf size, overwritting a model that was larger. Hence this os.remove.
                os.remove(external_path)

            onnx.save(
                model,
                save_path,
                convert_attribute=True,
            )
    elif save_path is not None:
        # path/to/model.onnx
        save_path = Path(save_path).as_posix()

        external_file_name = os.path.basename(save_path) + "_data"
        # path/to/model.onnx_data
        external_path = os.path.join(os.path.dirname(save_path), external_file_name)

        if save_path.endswith(".onnx") and os.path.isfile(save_path):
            os.remove(save_path)
        if os.path.isfile(external_path):
            os.remove(external_path)

        onnx.save(
            model,
            save_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_file_name,
            convert_attribute=True,
        )

    else:
        logger.info(
            "Merged ONNX model exceeds 2GB, the model will not be checked without `save_path` given."
        )
