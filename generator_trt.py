import tensorrt as trt
import ctypes
import numpy as np
#import pycuda.driver as cuda
#import pycuda.autoinit
from cuda import cudart

# class OutputAllocator(trt.IOutputAllocator):
#     def __init__(self):
#         trt.IOutputAllocator.__init__(self)
#         self.buffers = {}
#         self.shapes = {}
#
#     def reallocate_output(self, tensor_name, memory, size, alignment):
#         ptr = cudart.cudaMalloc(size)[1]
#         self.buffers[tensor_name] = ptr
#         return ptr
#
#     def notify_shape(self, tensor_name, shape):
#         self.shapes[tensor_name] = tuple(shape)
#

def load_engine(engine_path,logger):
    with open(engine_path, "rb") as f:
        enginestring = f.read()

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(enginestring)
    context = engine.create_execution_context()
    return engine,context


def inference(engine, context, inputs):

    input1 = np.ascontiguousarray(inputs["source_image"].cpu().numpy())
    #print(input1.shape)
    _,d_input1 = cudart.cudaMalloc(1 * input1.size * input1.dtype.itemsize)
    cudart.cudaMemcpy(d_input1, input1.ctypes.data, input1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.set_input_shape("source_image", input1.shape)
    context.set_tensor_address("source_image", d_input1)

    input2 = np.ascontiguousarray(inputs["kp_driving"].cpu().numpy())
    #print(input2.shape)
    _,d_input2 = cudart.cudaMalloc(1 * input2.size * input2.dtype.itemsize)
    cudart.cudaMemcpy(d_input2, input2.ctypes.data, input2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.set_input_shape("kp_driving", input2.shape)
    context.set_tensor_address("kp_driving", d_input2)

    input3 = np.ascontiguousarray(inputs["kp_source"].cpu().numpy())
    #print(input3.shape)
    _,d_input3 = cudart.cudaMalloc(1 * input3.size * input3.dtype.itemsize)
    cudart.cudaMemcpy(d_input3, input3.ctypes.data, input3.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.set_input_shape("kp_source", input3.shape)
    context.set_tensor_address("kp_source", d_input3)

    output1 = np.empty((2, 3, 256, 256), dtype=np.float32)
    _,d_output1 = cudart.cudaMalloc(1 * output1.size * output1.dtype.itemsize)
    context.set_tensor_address("prediction",d_output1)

    output2 = np.empty((2, 16, 16, 64, 64), dtype=np.float32)
    _,d_output2 = cudart.cudaMalloc(1 * output2.size * output2.dtype.itemsize)
    context.set_tensor_address("mask",d_output2)

    output3 = np.empty((2, 1, 64, 64), dtype=np.float32)
    _,d_output3 = cudart.cudaMalloc(1 * output3.size * output3.dtype.itemsize)
    context.set_tensor_address("occlusion_map",d_output3)

    bindings = [int(d_input1),int(d_input2),int(d_input3), int(d_output1),int(d_output2),int(d_output3)]
    #stream = cudart.cudaStreamCreate()
    context.execute_async_v2(bindings=bindings,stream_handle=0)
    #context.execute_async(batch_size=1, bindings=bindings,stream_handle=0)
    #cudart.cudaStreamSynchronize(stream)
    # 将预测结果从从缓冲区取出
    cudart.cudaMemcpy(output1.ctypes.data, d_output1, output1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    cudart.cudaMemcpy(output2.ctypes.data, d_output2, output2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    cudart.cudaMemcpy(output3.ctypes.data, d_output3, output3.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    cudart.cudaFree(d_input1)
    cudart.cudaFree(d_input2)
    cudart.cudaFree(d_input3)
    cudart.cudaFree(d_output1)
    cudart.cudaFree(d_output2)
    cudart.cudaFree(d_output3)

    return output1,output2,output3

class OcclusionAwareSPADEGenerator:
    def __init__(self, engine_path: str, plugin_path: str):
        logger  = trt.Logger(trt.Logger.VERBOSE)
        success = ctypes.CDLL(plugin_path, mode = ctypes.RTLD_GLOBAL)
        if not success:
            print("load grid_sample_3d plugin error")
            raise Exception()

        trt.init_libnvinfer_plugins(logger, "")
        self.engine, self.context = load_engine(engine_path, logger)

    def __call__(self, source_image, kp_driving, kp_source):

        # kp_driving_jacobian = kp_driving["jacobian"] if "jacobian" in kp_driving else None
        # kp_source_jacobian = kp_source["jacobian"] if "jacobian" in kp_source else None
        # kp_driving = kp_driving["value"]
        # kp_source = kp_source["value"]

        inputs = {
            "source_image": source_image,
            "kp_driving": kp_driving,
            "kp_source": kp_source,
            # "kp_driving_jacobian": kp_driving_jacobian,
            # "kp_source_jacobian": kp_source_jacobian
        }
        # print(source_image.shape)
        # print(kp_driving.shape)
        # print(kp_source.shape)
        output1, output2,output3 = inference(self.engine, self.context, inputs)
        #print(output1,output2,output3)
        return output1,output2,output3
