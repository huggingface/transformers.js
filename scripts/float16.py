# MIT License
#
# Copyright (c) Microsoft Corporation, Hugging Face. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Optional
import itertools
import numpy as np
import onnx
import packaging.version as pv
import warnings
from onnx import helper, numpy_helper
from onnx import onnx_pb as onnx_proto
import onnxslim.third_party.onnx_graphsurgeon as gs


FLOAT32 = 1
FLOAT16 = 10


def _npfloat16_to_int(np_list):
    """
    Convert numpy float16 to python int.

    :param np_list: numpy float16 list
    :return int_list: python int list
    """
    return [int(bin(_.view("H"))[2:].zfill(16), 2) for _ in np_list]


def convert_np_to_float16(np_array, min_positive_val=1e-7, max_finite_val=1e4):
    """
    Convert float32 numpy array to float16 without changing sign or finiteness.
    Positive values less than min_positive_val are mapped to min_positive_val.
    Positive finite values greater than max_finite_val are mapped to max_finite_val.
    Similar for negative values. NaN, 0, inf, and -inf are unchanged.
    """

    def between(a, b, c):
        return np.logical_and(a < b, b < c)

    positive_values = np_array[np.where(np_array > 0)]
    if positive_values.shape[0] > 0:
        pos_max = positive_values.max()
        pos_min = positive_values.min()

        if pos_max >= max_finite_val:
            warnings.warn(
                "the float32 number {} will be truncated to {}".format(
                    pos_max, max_finite_val
                )
            )

        if pos_min <= min_positive_val:
            warnings.warn(
                "the float32 number {} will be truncated to {}".format(
                    pos_min, min_positive_val
                )
            )

    negative_values = np_array[np.where(np_array < 0)]
    if negative_values.shape[0] > 0:
        neg_max = negative_values.max()
        neg_min = negative_values.min()

        if neg_min <= -max_finite_val:
            warnings.warn(
                "the float32 number {} will be truncated to {}".format(
                    neg_min, -max_finite_val
                )
            )

        if neg_max >= -min_positive_val:
            warnings.warn(
                "the float32 number {} will be truncated to {}".format(
                    neg_max, -min_positive_val
                )
            )

    np_array = np.where(
        between(0, np_array, min_positive_val), min_positive_val, np_array
    )
    np_array = np.where(
        between(-min_positive_val, np_array, 0), -min_positive_val, np_array
    )
    np_array = np.where(
        between(max_finite_val, np_array, float("inf")), max_finite_val, np_array
    )
    np_array = np.where(
        between(float("-inf"), np_array, -max_finite_val), -max_finite_val, np_array
    )
    return np.float16(np_array)


def convert_tensor_float_to_float16(tensor, min_positive_val=1e-7, max_finite_val=1e4):
    """
    Convert tensor float to float16.

    :param tensor: TensorProto object
    :return tensor_float16: converted TensorProto object
    """
    if not isinstance(tensor, onnx_proto.TensorProto):
        raise ValueError(
            "Expected input type is an ONNX TensorProto but got %s" % type(tensor)
        )

    if tensor.data_type == onnx_proto.TensorProto.FLOAT:
        tensor.data_type = onnx_proto.TensorProto.FLOAT16
        # convert float_data (float type) to float16 and write to int32_data
        if tensor.float_data:
            float16_data = convert_np_to_float16(
                np.array(tensor.float_data), min_positive_val, max_finite_val
            )
            int_list = _npfloat16_to_int(float16_data)
            tensor.int32_data[:] = int_list
            tensor.float_data[:] = []
        # convert raw_data (bytes type)
        if tensor.raw_data:
            # convert n.raw_data to float
            float32_list = np.fromstring(tensor.raw_data, dtype="float32")
            # convert float to float16
            float16_list = convert_np_to_float16(
                float32_list, min_positive_val, max_finite_val
            )
            # convert float16 to bytes and write back to raw_data
            tensor.raw_data = float16_list.tostring()
    return tensor


def make_value_info_from_tensor(tensor):
    shape = numpy_helper.to_array(tensor).shape
    return helper.make_tensor_value_info(tensor.name, tensor.data_type, shape)


DEFAULT_OP_BLOCK_LIST = [
    "ArrayFeatureExtractor",
    "Binarizer",
    "CastMap",
    "CategoryMapper",
    "DictVectorizer",
    "FeatureVectorizer",
    "Imputer",
    "LabelEncoder",
    "LinearClassifier",
    "LinearRegressor",
    "Normalizer",
    "OneHotEncoder",
    "RandomUniformLike",
    "SVMClassifier",
    "SVMRegressor",
    "Scaler",
    "TreeEnsembleClassifier",
    "TreeEnsembleRegressor",
    "ZipMap",
    "NonMaxSuppression",
    "TopK",
    "RoiAlign",
    "Resize",
    # 'Range',
    "CumSum",
    "Min",
    "Max",
    "Upsample",
    # NEW:
    "RandomNormalLike",
    # TODO: Ideally, "Cast" nodes should not be here, for the following reasons:
    #  - It breaks the semantics that the default list contains "ops that are not supported for float16 in ONNX Runtime".
    #  - When fp32 casts already exist in the model (e.g., for rotary embeddings), this script will insert redundant casts around it.
    # However, without it, the graphs produced are invalid. Eventually, we will resolve this.
    "Cast",
]


def initial_checking(model, disable_shape_infer):
    func_infer_shape = None
    if not disable_shape_infer and pv.Version(onnx.__version__) >= pv.Version("1.2"):
        try:
            from onnx.shape_inference import infer_shapes

            func_infer_shape = infer_shapes
        finally:
            pass

    if not isinstance(model, onnx_proto.ModelProto):
        raise ValueError(
            "Expected model type is an ONNX ModelProto but got %s" % type(model)
        )

    if func_infer_shape is not None:
        model = func_infer_shape(model)

    is_fp16_ready_flag = check_if_fp16_ready(model.graph)

    return model, func_infer_shape, is_fp16_ready_flag


def convert_float_to_float16(
    model,
    min_positive_val=1e-7,
    max_finite_val=1e4,
    keep_io_types=False,
    disable_shape_infer=False,
    op_block_list=None,
    node_block_list=None,
    check_fp16_ready=True,
):

    # create blocklists
    if op_block_list is None:
        op_block_list = DEFAULT_OP_BLOCK_LIST
    if node_block_list is None:
        node_block_list = []
    op_block_list = set(op_block_list)
    node_block_list = set(node_block_list)

    global_input_name_dict = (
        {}
    )  # key: input name, value: new output name after Cast node
    # basic checking, including shape inference
    model, func_infer_shape, is_fp16_ready_flag = initial_checking(
        model, disable_shape_infer
    )
    if is_fp16_ready_flag and check_fp16_ready:
        raise ValueError(
            "The model is already converted to float16, if convert again, the model might be wrong. \n If you are sure to convert again, please set check_fp16_ready=False."
        )

    graph_stack = [model.graph]

    is_top_level = True
    while graph_stack:
        next_level = []
        for curr_graph in graph_stack:
            process_graph_input(
                curr_graph, is_top_level, keep_io_types, global_input_name_dict
            )
            value_info_block_list = process_tensor_in_node(
                curr_graph,
                op_block_list,
                node_block_list,
                min_positive_val,
                max_finite_val,
            )
            process_value_info(curr_graph, value_info_block_list)
            process_node_in_block_list(
                curr_graph, global_input_name_dict, op_block_list, node_block_list
            )
            process_initializers(
                curr_graph,
                op_block_list,
                node_block_list,
                min_positive_val,
                max_finite_val,
            )
            process_graph_output(curr_graph, is_top_level, keep_io_types)
            sub_graph_list = get_next_level_graph(
                curr_graph, op_block_list, node_block_list
            )
            if len(sub_graph_list) > 0:
                next_level.extend(sub_graph_list)

            if not is_top_level:
                process_node_input_output(curr_graph, global_input_name_dict)
            is_top_level = False  # Going to process sub-graph
        graph_stack = next_level

    remove_unnecessary_cast_node(model.graph)

    # Topologically sort the graph
    # NOTE: We do not perform another round of optimization as the model is already optimized
    graph = gs.import_onnx(model)
    graph.toposort()
    model = gs.export_onnx(graph)

    return model


# Change the input/output of the node to the new output name after Cast node for sub-graph
# Because there have NO value_info start from
def process_node_input_output(
    graph: onnx_proto.GraphProto, global_input_name_dict: dict
):
    for node in graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in global_input_name_dict:
                node.input[i] = global_input_name_dict[input_name]
        for i, output_name in enumerate(node.output):
            if output_name in global_input_name_dict:
                node.output[i] = global_input_name_dict[output_name]


def process_graph_input(
    graph: onnx_proto.GraphProto,
    is_top_level: bool,
    is_io_fp32: bool,
    global_input_name_dict: dict,
):
    # The input dtype is float32, need to cast to fp16
    if is_top_level and is_io_fp32:
        for graph_input in graph.input:  # n_input is ValueInfoProto
            if graph_input.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                downstream_nodes = find_downstream_node_by_input_name(
                    graph, graph_input.name
                )
                for d_node in downstream_nodes:
                    # More than one node may consume the model input, so we only create
                    # a single cast node, and then reuse this node when needed.
                    cast_exists = graph_input.name in global_input_name_dict
                    if cast_exists:
                        cast_node_output_name = global_input_name_dict[graph_input.name]
                    else:
                        cast_node_output_name = graph_input.name + "_fp16"
                        add_cast_node(
                            graph,
                            [graph_input.name],
                            [cast_node_output_name],
                            cast_node_output_name,  # Set node name same as output name
                            FLOAT16,
                        )
                        add_new_value_info(
                            graph,
                            graph_input,
                            cast_node_output_name,
                            onnx_proto.TensorProto.FLOAT16,
                        )
                    for i, input_name in enumerate(d_node.input):
                        if input_name == graph_input.name:
                            d_node.input[i] = (
                                cast_node_output_name  # Change the input of the second node
                            )
                            global_input_name_dict[graph_input.name] = (
                                cast_node_output_name
                            )

    # For the sub-graph, don't do cast
    else:  # Change the input dtype to fp16 without any cast
        for graph_input in graph.input:
            if graph_input.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                graph_input.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16


def process_graph_output(
    graph: onnx_proto.GraphProto, is_top_level: bool, is_io_fp32: bool
):
    if is_top_level and is_io_fp32:  # the output dtype is float32, need to cast to fp16
        for i, graph_output in enumerate(graph.output):
            if graph_output.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                new_producer_name = graph_output.name + "_fp16"
                original_name = graph_output.name  # The correct output name

                # Get the node(s) that produce the model output
                # These will most likely be fp16, but could be fp32 if the previous node is in block_list
                upstream_nodes = find_upstream_node_by_output_name(graph, original_name)
                assert len(upstream_nodes) == 1  # Should be only one node

                producer_node = upstream_nodes[0]

                for i, output_name in enumerate(producer_node.output):
                    if output_name == original_name:
                        producer_node.output[i] = new_producer_name

                cast_node_name = new_producer_name + "_input_cast" + str(i)
                add_cast_node(
                    graph,
                    [new_producer_name],
                    [original_name],
                    cast_node_name,
                    onnx_proto.TensorProto.FLOAT,
                )
                for value_info in graph.value_info:
                    if original_name == value_info.name:
                        value_info.type.tensor_type.elem_type = (
                            onnx_proto.TensorProto.FLOAT
                        )

                # Get the node(s) that consume the model output
                downstream_nodes = find_downstream_node_by_input_name(
                    graph,
                    original_name,
                    include_subgraphs=False,
                )

                # It is possible that the producer node is also input to downstream nodes
                # So, we update the inputs of these downstream nodes
                for d_node in downstream_nodes:
                    for i, input_name in enumerate(d_node.input):
                        if input_name == original_name:
                            d_node.input[i] = new_producer_name

    else:  # change the output dtype to fp16 in tensor
        for graph_output in graph.output:
            if graph_output.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                graph_output.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16


def process_node_in_block_list(
    graph: onnx_proto.GraphProto,
    global_input_name_dict: dict,
    op_block_list: list,
    node_block_list: list,
):
    # NB: Important to create a copy of the nodes in the graph to avoid modifying
    # the graph in-place while iterating (causing an infinite loop)
    for node in list(graph.node):
        if (node.op_type in op_block_list) or (node.name in node_block_list):
            insert_cast32_before_node(graph, node, global_input_name_dict)
            insert_cast16_after_node(graph, node, global_input_name_dict)


# Todo: global_input_name_dict still not fill value
def insert_cast32_before_node(
    graph: onnx_proto.GraphProto, node: onnx_proto.NodeProto, global_input_name_dict
):
    for i, input_name in enumerate(node.input):
        for value_info in itertools.chain(graph.value_info, graph.input):
            if input_name == value_info.name:
                if (
                    value_info.type.tensor_type.elem_type
                    != onnx_proto.TensorProto.FLOAT16
                ):
                    break
                cast_output_name = node.name + "_input_cast_" + str(i)
                add_new_value_info(
                    graph, value_info, cast_output_name, onnx_proto.TensorProto.FLOAT
                )
                cast_node_name = node.name + "_input_cast" + str(i)
                add_cast_node(
                    graph,
                    [input_name],
                    [cast_output_name],
                    cast_node_name,
                    onnx_proto.TensorProto.FLOAT,
                )
                node.input[i] = cast_output_name
                break


# Todo: global_input_name_dict still not fill value
def insert_cast16_after_node(
    graph: onnx_proto.GraphProto, node: onnx_proto.NodeProto, global_input_name_dict
):
    for i, output_name in enumerate(node.output):
        for value_info in itertools.chain(graph.value_info, graph.output):
            if output_name == value_info.name:
                if (
                    value_info.type.tensor_type.elem_type
                    != onnx_proto.TensorProto.FLOAT
                ):
                    break
                cast_input_name = node.name + "_output_cast_" + str(i)
                add_new_value_info(
                    graph, value_info, cast_input_name, onnx_proto.TensorProto.FLOAT
                )
                value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                cast_node_name = node.name + "_output_cast" + str(i)
                add_cast_node(
                    graph,
                    [cast_input_name],
                    [output_name],
                    cast_node_name,
                    onnx_proto.TensorProto.FLOAT16,
                )
                node.output[i] = cast_input_name
                break


# Process tensor data in attribute of the node
def process_tensor_in_node(
    graph: onnx_proto.GraphProto,
    op_block_list: list,
    node_block_list: list,
    min_positive_val,
    max_finite_val,
):
    value_info_block_list = set()  # This is for later use, not in this step
    for node in graph.node:
        # NOTE: "Cast" operation cannot change its output type because it is strongly typed.
        if (
            (node.op_type in op_block_list)
            or (node.name in node_block_list)
            or (node.op_type == "Cast")
        ):
            # if (node.op_type in op_block_list) or (node.name in node_block_list):
            # Only need to block the output value_info changing
            for output_name in node.output:
                value_info_block_list.add(output_name)
        else:
            for attr in node.attribute:
                # one tensor
                if attr.t.data_type == onnx_proto.TensorProto.FLOAT:
                    attr.t.CopyFrom(
                        convert_tensor_float_to_float16(
                            attr.t, min_positive_val, max_finite_val
                        )
                    )
                # list of tensor
                for t in attr.tensors:
                    if t.data_type == onnx_proto.TensorProto.FLOAT:
                        t.CopyFrom(
                            convert_tensor_float_to_float16(
                                t, min_positive_val, max_finite_val
                            )
                        )
    return value_info_block_list


# Change all the value info type from float32 to float16 if not in block list
def process_value_info(graph: onnx_proto.GraphProto, value_info_block_list: list):
    for value_info in graph.value_info:
        if value_info.name in value_info_block_list:
            continue
        else:
            if value_info.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16


# Initializer is 'edge' type, so doesn't have value_info
def process_initializers(
    graph: onnx_proto.GraphProto,
    op_block_list,
    node_block_list,
    min_positive_val,
    max_finite_val,
):
    # Find the input of the block node, don't need to change this kind of initializer
    initializer_block_list = set()
    for node in graph.node:
        if (node.op_type in op_block_list) or (node.name in node_block_list):
            for (
                input_name
            ) in (
                node.input
            ):  # some is initializer, some is value_info, can't distinguish but doesn't matter
                initializer_block_list.add(input_name)
    # Process initializers
    for initializer in graph.initializer:
        if initializer.name not in initializer_block_list:
            if initializer.data_type == onnx_proto.TensorProto.FLOAT:
                convert_tensor_float_to_float16(
                    initializer, min_positive_val, max_finite_val
                )


def get_next_level_graph(
    graph: onnx_proto.GraphProto, op_block_list: list, node_block_list: list
):
    sub_graph_list = []
    for node in graph.node:
        if node.op_type in op_block_list or node.name in node_block_list:
            continue
        for attr in node.attribute:
            # Check if sub-graph exist
            if len(attr.g.node) > 0:  # single sub-graph
                sub_graph_list.append(attr.g)
            for g in attr.graphs:
                if len(g.node) > 0:  # multiple sub-graphs
                    sub_graph_list.append(g)
    return sub_graph_list


def add_cast_node(
    graph: onnx_proto.GraphProto,
    inputs: list,
    outputs: list,
    node_name: str,
    to_type: int,
):
    new_node = [helper.make_node("Cast", inputs, outputs, to=to_type, name=node_name)]
    graph.node.extend(new_node)


def add_new_value_info(
    graph: onnx_proto.GraphProto,
    exist_value_info: onnx_proto.ValueInfoProto,
    name: str,
    dtype: int,
):
    new_value_info = graph.value_info.add()
    new_value_info.CopyFrom(exist_value_info)
    new_value_info.name = name
    new_value_info.type.tensor_type.elem_type = dtype


# Find the node that has the specified output name
def find_upstream_node_by_output_name(graph: onnx_proto.GraphProto, output_name: str):
    nodes = []
    for node in graph.node:
        if output_name in node.output:
            nodes.append(node)
    assert len(nodes) <= 1  # Suppose there is less than one node found
    return nodes


# Find the node that has the specified input name, including in subgraphs
def find_downstream_node_by_input_name(
    graph: onnx_proto.GraphProto, input_name: str, include_subgraphs=True
):
    nodes = []

    # Check nodes in current graph
    for node in graph.node:
        if input_name in node.input:
            nodes.append(node)

        if not include_subgraphs:
            continue

        # Recursively check subgraphs in node attributes
        for attr in node.attribute:
            if attr.type == onnx_proto.AttributeProto.GRAPH:
                # Single subgraph
                if len(attr.g.node) > 0:
                    nodes.extend(find_downstream_node_by_input_name(attr.g, input_name))

            # Multiple subgraphs
            if attr.type == onnx_proto.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    if len(g.node) > 0:
                        nodes.extend(find_downstream_node_by_input_name(g, input_name))

    return nodes


# Remove identity node
def remove_identity_node_from_model(model: onnx_proto.ModelProto):
    remove_identity_node_from_graph(model.graph)
    try:
        from onnx.shape_inference import infer_shapes

        func_infer_shape = infer_shapes
        model = func_infer_shape(model)
        return model
    finally:
        pass


# Remove identity node
def remove_identity_node_from_graph(graph: onnx_proto.GraphProto):
    for curr_node in graph.node:
        if curr_node.op_type == "Identity":
            for input_name in curr_node.input:
                upstream_nodes = find_upstream_node_by_output_name(graph, input_name)
                for u_node in upstream_nodes:
                    if u_node is not None:
                        u_node.output[0] = curr_node.output[0]
                        graph.node.remove(curr_node)


def convert_float_to_float16_model_path(
    model_path, min_positive_val=1e-7, max_finite_val=1e4, keep_io_types=False
):
    """
    Convert tensor float type in the ONNX Model to tensor float16.
    *It is to fix an issue that infer_shapes func cannot be used to infer >2GB models.
    *But this function can be applied to all model sizes.
    :param model_path: ONNX Model path
    :return: converted ONNX ModelProto object
    Examples
    ::
        #Convert to ONNX ModelProto object and save model binary file:
        from onnxmltools.utils.float16_converter import convert_float_to_float16_model_path
        new_onnx_model = convert_float_to_float16_model_path('model.onnx')
        onnx.save(new_onnx_model, 'new_model.onnx')
    """

    disable_shape_infer = False
    if pv.Version(onnx.__version__) >= pv.Version("1.8"):
        try:
            # infer_shapes_path can be applied to all model sizes
            from onnx.shape_inference import infer_shapes_path
            import tempfile
            import os

            # shape_infer_model_path should be in the same folder of model_path
            with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(model_path)
            ) as tmpfile:
                shape_infer_model_path = tmpfile.name
                infer_shapes_path(model_path, shape_infer_model_path)
                model = onnx.load(shape_infer_model_path)
                disable_shape_infer = True
        finally:
            pass
    if not disable_shape_infer:
        model = onnx.load(model_path)
    return convert_float_to_float16(
        model, min_positive_val, max_finite_val, keep_io_types, disable_shape_infer
    )


def remove_unnecessary_cast_node(graph_proto: onnx_proto.GraphProto):
    # 1. find all cast nodes in the graph
    cast_node_list = []
    input_name_to_cast_node_dict = {}
    output_name_to_cast_node_dict = {}
    # using name as key to point to a node. because node object cannot be key
    name_to_node_dict = {}
    for node in graph_proto.node:
        if node.op_type == "Cast":
            # if node.name not in ["graph_input_cast0", "graph_output_cast0"]:
            cast_node_list.append(node)

            name_to_node_dict[node.name] = node
            for input_name in node.input:
                input_name_to_cast_node_dict[input_name] = node
            for output_name in node.output:
                output_name_to_cast_node_dict[output_name] = node

    # 2. find upstream and downstream node of the cast node
    cast_node_upstream_dict = {}  # mapping cast node(name) to its upstream node
    cast_node_downstream_dict = {}  # mapping cast node(name) to its downstream node
    for current_node in graph_proto.node:
        # find the downstream node(s)
        for input_name in current_node.input:
            if input_name in output_name_to_cast_node_dict:
                # found the downstream node of the cast node, might be multiple
                cast_node = output_name_to_cast_node_dict[input_name]
                if cast_node.name not in cast_node_downstream_dict:
                    cast_node_downstream_dict[cast_node.name] = current_node
                else:  # already exists one downstream node, make it a list
                    existing_downstream_nodes = cast_node_downstream_dict[
                        cast_node.name
                    ]
                    if isinstance(existing_downstream_nodes, list):
                        existing_downstream_nodes.append(current_node)
                    else:  # make a list
                        existing_downstream_nodes = [
                            existing_downstream_nodes,
                            current_node,
                        ]
                        cast_node_downstream_dict[cast_node.name] = (
                            existing_downstream_nodes
                        )
        # find the upstream node
        for output_name in current_node.output:
            if output_name in input_name_to_cast_node_dict:
                # found the upstream node of the cast node, should be unique
                cast_node = input_name_to_cast_node_dict[output_name]
                cast_node_upstream_dict[cast_node.name] = current_node

    # 3. remove the cast node which upstream is 'Constant'
    for cast_node_name, upstream_node in cast_node_upstream_dict.items():
        cast_node = name_to_node_dict[cast_node_name]
        if upstream_node.op_type == "Constant":
            cast_node_list.remove(cast_node)

    # 4. find (cast_to_fp16, cast_to_fp32) pairs where --fp32--> cast_to_fp16 --fp16--> cast_to_fp32.
    remove_candidate = []

    name_to_value_info = {
        value_info.name: value_info
        for value_info in itertools.chain(graph_proto.value_info, graph_proto.input)
    }

    def get_type(name: str) -> Optional[int]:
        if name in name_to_value_info:
            return name_to_value_info[name].type
        else:
            # `name` has no value info.
            return None

    for cast_node_name, downstream_node in cast_node_downstream_dict.items():
        cast_node = name_to_node_dict[cast_node_name]
        if len(cast_node.input) != 1:
            raise RuntimeError(
                f"Cast node {cast_node_name} should have only one input, but has {len(cast_node.input)}."
            )

        input_type = get_type(cast_node.input[0])
        if input_type != onnx_proto.TensorProto.FLOAT:
            continue
        if isinstance(downstream_node, list):
            for dn in downstream_node:
                if (
                    dn.op_type == "Cast"
                    and dn.attribute[0].i == 32
                    and cast_node.attribute[0].i == 16
                    and dn in cast_node_list
                    and cast_node in cast_node_list
                ):
                    remove_candidate.append((cast_node, dn))
        else:
            if (
                downstream_node.op_type == "Cast"
                and cast_node.attribute[0].i == FLOAT16
                and downstream_node.attribute[0].i == FLOAT32
                and downstream_node in cast_node_list
                and cast_node in cast_node_list
            ):
                remove_candidate.append((cast_node, downstream_node))

    # 5. change "upstream --fp32--> cast_to_fp16 --fp16--> cast_to_fp32 --fp32--> downstream" to
    # "upstream --fp32--> downstream".
    for cast_node_pair in remove_candidate:
        first_cast_node = cast_node_pair[0]
        second_cast_node = cast_node_pair[1]
        upstream_node = cast_node_upstream_dict.get(first_cast_node.name)
        downstream_node = cast_node_downstream_dict.get(second_cast_node.name)
        if upstream_node is None and downstream_node is not None:
            # The upstream_node should be graph input
            out = first_cast_node.input[0]
            for i, input_name in enumerate(downstream_node.input):
                for output_name in second_cast_node.output:
                    if input_name == output_name:
                        # change the input as the upstream node's output
                        downstream_node.input[i] = out
        elif upstream_node is not None and downstream_node is None:
            raise ValueError(
                "The downstream node of the second cast node should be graph output"
            )
        else:
            # find the upstream node's output to first_cast_node
            out = None
            for output_name in upstream_node.output:
                if output_name == first_cast_node.input[0]:
                    out = output_name
                    break
            # find the downstream node's input as second_cast_node's output
            for i, input_name in enumerate(downstream_node.input):
                for output_name in second_cast_node.output:
                    if input_name == output_name:
                        # change the input as the upstream node's output
                        downstream_node.input[i] = out

    # 6. remove the cast node pair
    for cast_node_pair in remove_candidate:
        graph_proto.node.remove(cast_node_pair[0])
        graph_proto.node.remove(cast_node_pair[1])


# Check if the model is already converted to float16
def check_if_fp16_ready(graph_proto):
    # Check graph input and ouput
    is_value_info_fp16 = False
    for value_info in itertools.chain(
        graph_proto.output, graph_proto.input, graph_proto.value_info
    ):
        if value_info.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT16:
            is_value_info_fp16 = True
            break

    # Check initializer
    is_initializer_fp16 = False
    for initializer in graph_proto.initializer:
        if initializer.data_type == onnx_proto.TensorProto.FLOAT16:
            is_initializer_fp16 = True
            break

    # Check cast node
    has_cast_node_fp16 = False
    for node in graph_proto.node:
        if node.op_type == "Cast" and node.attribute[0].i == FLOAT16:
            has_cast_node_fp16 = True
            break

    # Any of above flags is True, return True
    if is_value_info_fp16 or is_initializer_fp16 or has_cast_node_fp16:
        return True  # already converted to float16
    else:
        return False  # not converted to float16 yet
