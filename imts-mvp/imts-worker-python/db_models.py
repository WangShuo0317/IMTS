import os

# Model local paths on remote GPU server
MODEL_PATHS = {
    "Qwen3-8B": "/home/user/workspace/wangshuo/Models/Qwen3-8B",
    "Qwen3-0.6B": "/home/user/workspace/wangshuo/Models/Qwen/Qwen3-0.6B",
    "Qwen2.5-1.5B": "/home/user/workspace/wangshuo/Models/Qwen2.5-1.5B",
}


def get_model_path(model_name):
    return MODEL_PATHS.get(model_name, MODEL_PATHS["Qwen3-8B"])