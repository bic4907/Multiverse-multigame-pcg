import os
from os.path import abspath, join

def get_log_dir(root_dir, epoch, sub_dir) -> str:
    path = abspath(join(root_dir, f"epoch_{epoch:03d}", sub_dir))
    os.makedirs(path, exist_ok=True)
    return path
