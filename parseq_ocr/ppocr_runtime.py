"""Torch-free PP price TensorRT with per-shape managed workspace and CUDAgraphs.
Apache-2.0 derivative; see _vendor/ppocr/NOTICE. Static128×320, batch1..32.
"""
import pathlib, json, math
import numpy as np, cv2
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image
from cuda.bindings import runtime as rt


def checked(result):
    if result[0] != rt.cudaError_t.cudaSuccess:
        raise RuntimeError(str(result[0]))
    return result[1] if len(result) > 1 else None


class PPTRTManaged:
    def __init__(self, folder, max_batch=32):
        folder = pathlib.Path(folder)
        self.meta = json.loads((folder / "metadata.json").read_text())
        self.max_batch = max_batch
        self.height = self.meta["height"]
        self.width = self.meta["width"]
        cuda.init()
        self.cuda_context = cuda.Device(0).make_context()
        logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            (folder / "model.engine").read_bytes()
        )
        assert self.engine is not None
        self.max_batch = min(
            max_batch, self.engine.get_tensor_profile_shape("pixels", 0)[2][0]
        )
        ctx = self.engine.create_execution_context(
            trt.ExecutionContextAllocationStrategy.USER_MANAGED
        )
        ctx.set_input_shape("pixels", (self.max_batch, 3, self.height, self.width))
        shape = tuple(ctx.get_tensor_shape("ids"))
        self.h_input = cuda.pagelocked_empty(
            (self.max_batch, 3, self.height, self.width), np.float32
        )
        self.h_output = cuda.pagelocked_empty(shape, np.int32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream = cuda.Stream()
        self.graphs = {}
        self.cuda_context.pop()

    def capture(self, n):
        ctx = self.engine.create_execution_context(
            trt.ExecutionContextAllocationStrategy.USER_MANAGED
        )
        ctx.set_input_shape("pixels", (n, 3, self.height, self.width))
        required = ctx.update_device_memory_size_for_shapes()
        workspace = cuda.mem_alloc(max(required, 256))
        ctx.set_device_memory(int(workspace), max(required, 256))
        ctx.set_tensor_address("pixels", int(self.d_input))
        ctx.set_tensor_address("ids", int(self.d_output))
        cuda.memcpy_htod_async(self.d_input, self.h_input[:n], self.stream)
        ctx.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        stream = rt.cudaStream_t(self.stream.handle)
        checked(
            rt.cudaStreamBeginCapture(
                stream, rt.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal
            )
        )
        cuda.memcpy_htod_async(self.d_input, self.h_input[:n], self.stream)
        assert ctx.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        graph = checked(rt.cudaStreamEndCapture(stream))
        ex = checked(rt.cudaGraphInstantiate(graph, 0))
        self.graphs[n] = (ctx, graph, ex, workspace, required)

    def infer(self, images):
        n = len(images)
        assert 0 < n <= self.max_batch
        for i, im in enumerate(images):
            if isinstance(im, Image.Image):
                im = np.asarray(im.convert("RGB"))[:, :, ::-1]
            else:
                im = im[:, :, ::-1]
            if self.meta["resize"] == "stretch":
                self.h_input[i] = (
                    cv2.resize(im, (self.width, self.height))
                    .transpose(2, 0, 1)
                    .astype(np.float32)
                    / 127.5
                    - 1
                )
            else:
                h, w = im.shape[:2]
                nw = min(self.width, math.ceil(w / h * self.height))
                self.h_input[i].fill(0)
                self.h_input[i, :, :, :nw] = (
                    cv2.resize(im, (nw, self.height))
                    .transpose(2, 0, 1)
                    .astype(np.float32)
                    / 127.5
                    - 1
                )
        self.cuda_context.push()
        try:
            if n not in self.graphs:
                self.capture(n)
            checked(
                rt.cudaGraphLaunch(
                    self.graphs[n][2], rt.cudaStream_t(self.stream.handle)
                )
            )
            self.stream.synchronize()
            texts = []
            for ids in self.h_output[:n].tolist():
                prev = -1
                out = []
                for x in ids:
                    if x and x != prev:
                        out.append(self.meta["vocabulary"][x])
                    prev = x
                texts.append("".join(out))
            return texts
        finally:
            self.cuda_context.pop()

    def close(self):
        if getattr(self, "cuda_context", None):
            self.cuda_context.push()
            workspaces = [v[3] for v in self.graphs.values()]
            for ctx, graph, ex, workspace, required in self.graphs.values():
                checked(rt.cudaGraphExecDestroy(ex))
                checked(rt.cudaGraphDestroy(graph))
            self.graphs.clear()
            ctx = None
            for workspace in workspaces:
                workspace.free()
            workspaces.clear()
            workspace = None
            self.engine = None
            self.runtime = None
            self.d_input.free()
            self.d_output.free()
            self.stream = None
            self.cuda_context.pop()
            self.cuda_context.detach()
            self.cuda_context = None


class TensorRTRecognizer(PPTRTManaged):
    def infer(self, images):
        images = list(images)
        out = []
        for at in range(0, len(images), self.max_batch):
            out.extend(super().infer(images[at : at + self.max_batch]))
        return out

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
