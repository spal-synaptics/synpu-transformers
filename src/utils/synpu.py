import subprocess
import tempfile
from pathlib import Path
from shutil import copy2

import iree.compiler.tools as iree_c


def export_onnx_to_mlir(
    onnx_model: str | Path, mlir_model: str | Path, opset: int = 17
):
    if not Path(onnx_model).exists():
        raise FileNotFoundError(f"ONNX model '{onnx_model}' not found")
    
    try:
        subprocess.check_output(
            [
                "iree-import-onnx",
                str(onnx_model),
                "-o", str(mlir_model),
                "--opset-version", str(opset),
                "--data-prop",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to export ONNX model via '{' '.join(e.cmd)}':\n    "
            + "\n    ".join(e.output.strip().splitlines())
        ) from None


def compile_mlir_for_vm(
    mlir_model: str | Path,
    output_model: str | Path,
):
    compiled_bytes = iree_c.compile_file(
        str(mlir_model),
        target_backends=["llvm-cpu"],
        extra_args=[
            "--iree-hal-target-device=local",
            "--iree-llvmcpu-target-cpu=host",
        ],
    )
    with open(output_model, "wb") as f:
        f.write(compiled_bytes)


def export_iree(
    onnx_model: str | Path,
    output_dir: str | Path,
    save_mlir: bool = True,
    opset: int = 17,
):
    onnx_model = Path(onnx_model)
    output_dir = Path(output_dir)
    model_name = onnx_model.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        mlir_model = Path(temp_dir) / f"{model_name}.mlir"
        export_onnx_to_mlir(onnx_model, mlir_model, opset)
        if save_mlir:
            copy2(mlir_model, output_dir / f"{model_name}.mlir")
        compile_mlir_for_vm(mlir_model, output_dir / f"{model_name}.vmfb")


if __name__ == "__main__":
    pass