import os
from abc import ABC, abstractmethod
from time import time_ns

import iree.runtime as iree_rt
import onnxruntime as ort
import numpy as np


class InferenceRunner(ABC):

    def __init__(
        self,
        model_path: str | os.PathLike,
    ):
        self._model_path = model_path
        self._infer_time_ms: float = 0.0

    @property
    def infer_time_ms(self) -> float:
        return self._infer_time_ms

    @abstractmethod
    def _infer(self, inputs: list[np.ndarray] | dict[str, np.ndarray]) -> list[np.ndarray]:
        ...

    def infer(self, inputs: list[np.ndarray] | dict[str, np.ndarray]) -> list[np.ndarray]:
        st = time_ns()
        results = self._infer(inputs)
        self._infer_time_ms = (time_ns() - st) / 1e6
        return results


class ORTInferenceRunner(InferenceRunner):

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        n_threads: int | None = None
    ):
        super().__init__(model_path)

        self._opts = ort.SessionOptions()
        if n_threads is not None:
            self._opts.intra_op_num_threads = n_threads
            self._opts.inter_op_num_threads = n_threads
        self._sess = ort.InferenceSession(self._model_path, self._opts, providers=['CPUExecutionProvider'])

    def _infer(self, inputs: list[np.ndarray] | dict[str, np.ndarray]) -> list[np.ndarray]:
        return [np.asarray(o) for o in self._sess.run(None, inputs)]


class IREEInferenceRunner(InferenceRunner):

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        function: str = "main_graph",
        device_uri: str = "local-task",
    ):
        super().__init__(model_path)

        module = iree_rt.load_vm_flatbuffer_file(self._model_path, driver=device_uri)
        if function not in module.vm_module.function_names:
            raise ValueError(f"Function '{function}' not found in graph @ '{model_path}'")
        self._invoker = module[function]

    def _infer(self, inputs: list[np.ndarray] | dict[str, np.ndarray]) -> list[np.ndarray]:
        if isinstance(inputs, dict):
            inputs = list(inputs.values())
        result: iree_rt.DeviceArray | tuple[iree_rt.DeviceArray] = self._invoker(*inputs)
        if isinstance(result, tuple):
            return [r.to_host() for r in result]
        return [result.to_host()]


if __name__ == "__main__":
    pass
