import os
from pathlib import Path

try:
    from modules.paths_internal import models_path
except:
    try:
        from modules.paths import models_path
    except:
        models_path = os.path.abspath("models")

IS_RUN: bool = False
BASE_PATH = os.path.join(Path(__file__).parents[1])
DEVICE_LIST: list = ["CPU", "CUDA"]

MODELS_PATH = models_path
REACTOR_MODELS_PATH = os.path.join(models_path, "reactor")
FACE_MODELS_PATH = os.path.join(REACTOR_MODELS_PATH, "faces")

if not os.path.exists(REACTOR_MODELS_PATH):
    os.makedirs(REACTOR_MODELS_PATH)
    if not os.path.exists(FACE_MODELS_PATH):
        os.makedirs(FACE_MODELS_PATH)

def updateDevice():
    device = "CUDA"
    try:
        LAST_DEVICE_PATH = os.path.join(BASE_PATH, "last_device.txt")
        with open(LAST_DEVICE_PATH) as f:
            for el in f:
                device = el.strip()
    except:
        device = "CPU"
    return device

DEVICE = updateDevice()
