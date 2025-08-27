import os
import platform
import subprocess
import tensorflow as tf
import numpy as np
import pandas as pd

def get_nvidia_smi():
    try:
        result = subprocess.check_output(["nvidia-smi"], encoding='utf-8')
        return result
    except Exception as e:
        return f"NVIDIA-SMI 실행 실패: {e}"

def main():
    print("=== [시스템 정보] ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python 버전: {platform.python_version()}")
    print()

    print("=== [GPU 및 드라이버 정보] ===")
    print(get_nvidia_smi())
    print()

    print("=== [TensorFlow 및 CUDA 환경] ===")
    print(f"TensorFlow 버전: {tf.__version__}")
    print(f"CUDA 사용 가능 여부: {tf.test.is_built_with_cuda()}")
    print(f"GPU 디바이스 수: {len(tf.config.list_physical_devices('GPU'))}")
    print()

    print("=== [라이브러리 버전] ===")
    print(f"NumPy 버전: {np.__version__}")
    print(f"Pandas 버전: {pd.__version__}")
    print()

    print("=== [분산 전략 정보] ===")
    strategy = tf.distribute.MirroredStrategy()
    print(f"사용 중인 GPU 수: {strategy.num_replicas_in_sync}")

if __name__ == "__main__":
    main()