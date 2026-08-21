import torch
import numpy as np
import subprocess

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt


class FramePreprocess:
    def __init__(self, wfsParams, centers):
        self.reference = 1.0
        self.normalization = 1.0

        self.centers = centers

        self.Nres = wfsParams["Nres"]
        self.Substract_reference = wfsParams["Substract_Reference"]
        self.Bin_factor = wfsParams["Bin_factor"]
        self.Extract_pupils_pad = (wfsParams["Extract_pupils_pad"] * wfsParams["Bin_factor"])
        self.Ncrop = self.Nres + self.Extract_pupils_pad

        self.yy, self.xx = np.meshgrid(np.arange(self.Ncrop), np.arange(self.Ncrop), indexing="ij")

        self.yy = self.yy[None]
        self.xx = self.xx[None]

        centers = np.copy(centers)

        self.yy = (self.yy + centers[:, 0][..., None, None] - self.Ncrop // 2)  # [B,C,N,N]
        self.xx = (self.xx + centers[:, 1][..., None, None] - self.Ncrop // 2)  # [B,C,N,N]

    def ProcessReference(self, reference_frame):

        frame = np.copy(reference_frame)

        pupils = self.GetPupils(frame)

        self.normalization = np.std(pupils, axis=(-2, -1), keepdims=True)
        self.reference = pupils

    def ProcessFrame(self, input_frame):

        frame = self.GetPupils(input_frame)

        if self.Substract_reference:
            frame = frame - self.reference
            frame = frame / self.normalization
        else:
            frame = frame - frame.mean(axis=(-2, -1), keepdim=True)
            frame = frame / frame.std(axis=(-2, -1), keepdim=True)

        return frame

    def GetPupils(self, images=None):
        patches = images[self.yy, self.xx]
        return patches



def MakeTRTModel(NNModel, size, device, output_file_name, output_file_directory):
    example_input = torch.randn(size, device=device)

    ONNX_PATH = output_file_directory + output_file_name + ".onnx"
    TRT_PATH = output_file_directory + output_file_name + ".model"

    TRT_COMMAND = f"/usr/src/tensorrt/bin/trtexec --onnx={ONNX_PATH} --saveEngine={TRT_PATH} --builderOptimizationLevel=5 --noTF32 --useSpinWait --verbose"

    torch.onnx.export(NNModel, example_input, ONNX_PATH, opset_version=18)

    subprocess.run(TRT_COMMAND, shell=True, check=True)


class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

    def allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            size = trt.volume(engine.get_tensor_shape(tensor_name))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer address to device bindings
            bindings.append(int(device_mem))

            # Append to the appropiate input/output list
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        input_data = np.ascontiguousarray(input_data, dtype=np.float32)
        # Transfer input data to device
        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)
        # cuda.memcpy_htod_async(self.inputs[0].device, input_data, self.stream)

        # Set tensor address
        # for i in range(self.engine.num_io_tensors):
        #     self.context.set_tensor_address(
        #         self.engine.get_tensor_name(i), self.bindings[i]
        #     )

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer predictions back
        cuda.memcpy_dtoh_async(
            self.outputs[0].host, self.outputs[0].device, self.stream
        )

        # Synchronize the stream
        self.stream.synchronize()
        # cuda.Context.synchronize()

        return self.outputs[0].host


