import onnx
import onnx_graphsurgeon as gs

model = onnx.load("model.onnx")
graph = gs.import_onnx(model)
for node in graph.nodes:
    if "GridSample" in node.name:
        node.attrs = {"name": "GridSample3D", "version": 1, "namespace": ""}
        node.op = "GridSample3D"

onnx.save(gs.export_onnx(graph), "./model_gs.onnx")
