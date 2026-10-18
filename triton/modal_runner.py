"""Run a local kernel script on Modal: modal run modal_runner.py --script 01_vadd.py."""

import os
from pathlib import Path
import shlex
import subprocess
import sys

import modal

LOCAL_DIR = Path(__file__).resolve().parent
REMOTE_DIR = Path('/root/kernels')
GPU = os.environ.get('MODAL_GPU', 'L4')

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.9.1', index_url='https://download.pytorch.org/whl/cu128')
    .pip_install('numpy==2.2.6', 'matplotlib==3.10.3', 'pandas==2.2.3', 'pytest')
    .add_local_dir(
        LOCAL_DIR,
        remote_path=str(REMOTE_DIR),
        ignore=['.venv', '__pycache__', '.git', '*.pyc'],
    )
)
app = modal.App('triton-kernels')


def resolve_script(root: Path, script: str) -> Path:
    path = (root / script).resolve()
    if not path.is_relative_to(root.resolve()) or path.suffix != '.py':
        raise ValueError('Choose a Python script inside the triton directory.')
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.name == 'modal_runner.py':
        raise ValueError('Choose a kernel script, not the Modal runner.')
    return path


@app.function(image=image, gpu=GPU, timeout=600)
def run_kernel(script: str, args: list[str]):
    import torch
    import triton

    path = resolve_script(REMOTE_DIR, script)
    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    print(f'Torch: {torch.__version__}; Triton: {triton.__version__}; CUDA: {torch.version.cuda}', flush=True)
    subprocess.run([sys.executable, '-u', str(path), *args], cwd=REMOTE_DIR, check=True)


@app.local_entrypoint()
def main(script: str = '01_vadd.py', args: str = ''):
    path = resolve_script(LOCAL_DIR, script)
    run_kernel.remote(str(path.relative_to(LOCAL_DIR)), shlex.split(args))
