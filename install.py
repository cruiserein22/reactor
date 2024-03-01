import subprocess
import os
import sys
from pathlib import Path
import urllib.request
from tqdm import tqdm
from packaging import version as packaging_version
import pkg_resources

# Attempt to import models_path from different modules
try:
    from modules.paths_internal import models_path
    models_path = Path(models_path)  # Ensure models_path is a Path object
except ImportError:
    try:
        from modules.paths import models_path
        models_path = Path(models_path)  # Ensure models_path is a Path object
    except ImportError:
        models_path = Path.cwd() / "models"

BASE_PATH = Path(__file__).parent
req_file = BASE_PATH / "requirements.txt"
models_dir = models_path / "insightface"
model_url = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx"
model_name = model_url.split('/')[-1]
model_path = models_dir / model_name

def run_command(*args):
    """Executes a given command using subprocess."""
    subprocess.run(args, check=True)

def pip_install(package: str):
    run_command(sys.executable, "-m", "pip", "install", "-U", package)

def pip_uninstall(package: str):
    run_command(sys.executable, "-m", "pip", "uninstall", "-y", package)

def is_installed(package: str, min_version: str = None) -> bool:
    """Checks if a package is installed, optionally verifying the minimum version."""
    try:
        installed_version = packaging_version.parse(pkg_resources.get_distribution(package).version)
        if min_version:
            return installed_version >= packaging_version.parse(min_version)
        return True
    except pkg_resources.DistributionNotFound:
        return False

def download(url, path):
    """Downloads a file from a specified URL to a given path."""
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc='Downloading...', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

# Ensure models directory exists
models_dir.mkdir(parents=True, exist_ok=True)

# Download model if it doesn't exist
if not model_path.exists():
    download(model_url, str(model_path))

# Determine the appropriate ONNX Runtime (ORT) version and install it
def install_ort():
    import torch
    ort = "onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime"
    min_version = "1.17.0" if torch.cuda.is_available() and float(torch.version.cuda) >= 12 else "1.16.1"
    if not is_installed(ort, min_version):
        pip_install(f"{ort}=={min_version}")

# Install requirements from the requirements file
def install_requirements():
    with open(req_file) as file:
        for package in file:
            package = package.strip()
            pip_install(package)

if __name__ == "__main__":
    install_ort()
    install_requirements()
    print("Installation complete. Please, restart the server if necessary.")
