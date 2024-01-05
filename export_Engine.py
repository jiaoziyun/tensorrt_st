import tensorrt as trt
import ctypes
import numpy as np
onnx_path = "model_gs.onnx"
engine_path = "model.engine"

logger  = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser  = trt.OnnxParser(network, logger)
config  = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)

trt.init_libnvinfer_plugins(logger, "")
plugin_library_path = "/home/ubuntu/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so"
try:
    ctypes.CDLL(plugin_library_path, mode=ctypes.RTLD_GLOBAL)
except OSError as e:
    print(f"Failed to load plugin library ({plugin_library_path}): {e}")
    raise
plugin_registry = trt.get_plugin_registry()
plugin_creator = plugin_registry.get_plugin_creator("GridSample3D", "1", "")
if plugin_creator is None:
    raise RuntimeError("Failed to get the custom plugin creator.")

interpolation_mode_value = 0
padding_mode_value =  0
align_corners_value = 0

plugin_fields = [
    trt.PluginField("interpolation_mode", np.array([interpolation_mode_value], dtype=np.int32), trt.PluginFieldType.INT32),
    trt.PluginField("padding_mode", np.array([padding_mode_value], dtype=np.int32), trt.PluginFieldType.INT32),
    trt.PluginField("align_corners", np.array([align_corners_value], dtype=np.int32), trt.PluginFieldType.INT32)
]
plugin_field_collection = trt.PluginFieldCollection(plugin_fields)
custom_plugin = plugin_creator.create_plugin("CustomPluginInstanceName", plugin_field_collection)

with open(onnx_path, "rb") as f:
    parser.parse(f.read())
for i in range(network.num_layers):
    layer = network.get_layer(i)
    if layer.name == "GridSample3D":  # 检查层名称
        input_tensors = [layer.get_input(j) for j in range(layer.num_inputs)]
        output_tensors = [layer.get_output(j) for j in range(layer.num_outputs)]
        network.unmark_output(output_tensors[0])
        network.remove_layer(layer)
        custom_plugin_layer = network.add_plugin_v2(inputs=input_tensors, plugin=custom_plugin)
        if i == network.num_layers - 1:
            network.mark_output(custom_plugin_layer.get_output(0))
        break

        
engineString = builder.build_serialized_network(network, config)
with open(engine_path, "wb") as f:
    f.write(engineString)
