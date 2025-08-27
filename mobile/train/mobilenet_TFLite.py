# mobilenet_train_0814_4_TFLite.py
#  - mobilenet_7 체크포인트를 로드해 확률 출력 모델로 래핑 후 TFLite로 내보냄
#  - 기본 실행: python mobilenet_train_0814_4_TFLite.py
#  - 양자화 예시: python mobilenet_train_0814_4_TFLite.py --quant float16

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers, constraints
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# ========= Args =========
p = argparse.ArgumentParser(description="Export mobilenet_{model_num} to TFLite (prob output)")
p.add_argument('--model_num', type=str, default='8', help='불러올 모델 번호 (기본 7)')
p.add_argument('--checkpoint_path', type=str, default='/home/user/chae_project/model_save/')
p.add_argument('--dataset_path', type=str, default='/home/user/chae_project/vgg_data/0805_224/')
p.add_argument('--alpha', type=float, default=1.0)
p.add_argument('--quant', type=str, default='float32', choices=['float32','dynamic','float16','int8'],
               help='TFLite 양자화 모드')
p.add_argument('--rep_samples', type=int, default=128, help='int8 대표 샘플 개수')
p.add_argument('--gpu_num', type=str, default='0')  # CPU로 돌려도 OK
args = p.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
tf.keras.backend.set_floatx('float32')

CLASS_LIST = ['disease1','disease2','disease6','disease8','disease11','disease13','disease19','disease21']
NUM_CLASSES = len(CLASS_LIST)

# ========= 경로 유틸 =========
def get_model_dir():
    return os.path.join(args.checkpoint_path, f'mobilenet_{args.model_num}')

def find_ckpt_prefix():
    """학습 규칙: best_model 우선, 없으면 saved_weights 사용."""
    md = get_model_dir()
    best = os.path.join(md, 'best_model')
    saved = os.path.join(md, 'saved_weights')
    if os.path.exists(best + '.index') or os.path.exists(best + '.h5'):
        return best
    if os.path.exists(saved + '.index') or os.path.exists(saved + '.h5'):
        return saved
    raise FileNotFoundError(f'체크포인트 없음: {best} / {saved}')

# ========= 모델 빌드(학습 구조 동일, logits 출력) =========
def build_model_logits(alpha=1.0):
    """
    입력: [0..1] float32, (224,224,3)
    내부: *255 -> preprocess_input -> MobileNetV2 -> Dense stack -> logits
    """
    backbone = MobileNetV2(include_top=False, input_shape=(224,224,3),
                           weights=None, alpha=alpha)
    x_in = layers.Input(shape=(224,224,3))
    z = preprocess_input(x_in * 255.0)
    z = backbone(z, training=False)
    z = layers.GlobalAveragePooling2D()(z)
    z = layers.Dense(1024, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-5),
                     kernel_constraint=constraints.MaxNorm(3))(z)
    z = layers.Dropout(0.4)(z)
    z = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-5),
                     kernel_constraint=constraints.MaxNorm(3))(z)
    z = layers.Dropout(0.4)(z)
    z = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-5),
                     kernel_constraint=constraints.MaxNorm(3))(z)
    out_logits = layers.Dense(NUM_CLASSES, activation=None,
                              kernel_constraint=constraints.MaxNorm(3),
                              name='logits')(z)
    return Model(x_in, out_logits, name='mobilenet_asl')

# ========= 학습 래퍼(체크포인트 네임스페이스 맞추기용) =========
class ASLModel(tf.keras.Model):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
    def call(self, x, training=False):
        return self.base(x, training=training)  # logits

# ========= 대표데이터 (INT8) =========
def representative_dataset_gen():
    """
    INT8 양자화용. valid/train/test.npy 중 존재하는 것 사용.
    입력 스케일은 학습과 동일: [0..1] → (내부에서 *255, preprocess_input)
    """
    npy_candidates = [
        os.path.join(args.dataset_path, 'valid.npy'),
        os.path.join(args.dataset_path, 'train.npy'),
        os.path.join(args.dataset_path, 'test.npy'),
    ]
    x = None
    for pth in npy_candidates:
        if os.path.exists(pth):
            x = np.load(pth, mmap_mode='r')
            break
    if x is None:
        raise FileNotFoundError("대표데이터용 npy 파일(valid/train/test.npy)이 필요합니다.")

    total = min(args.rep_samples, len(x))
    step = max(1, len(x)//total)
    cnt = 0
    for i in range(0, len(x), step):
        if cnt >= total: break
        sample = x[i].astype(np.float32) / 255.0           # [H,W,3] 0..1
        # 224 보장. 이미 224면 resize는 무부담.
        sample = tf.image.resize(sample, (224,224)).numpy()
        sample = np.expand_dims(sample, 0)                 # [1,224,224,3]
        yield [sample]
        cnt += 1
    print(f"[INFO] Representative samples: {cnt}")

# ========= 변환 =========
def convert_to_tflite(export_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)

    if args.quant == 'float32':
        pass  # 최적화 없이 그대로

    elif args.quant == 'dynamic':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 가중치 8bit, I/O float32

    elif args.quant == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]  # 가중치 fp16, I/O float32

    elif args.quant == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # I/O는 float32 유지 → 앱 수정 최소화
        # I/O까지 int8로 바꾸려면 아래 두 줄 활성화
        # converter.inference_input_type  = tf.int8
        # converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out_dir = get_model_dir()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"mobilenet_{args.model_num}_{args.quant}.tflite")
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(out_path) / (1024*1024)
    print(f"[INFO] TFLite saved: {out_path}  ({size_mb:.2f} MB)")
    return out_path

def main():
    # 1) logits 모델 구성 (학습 구조 동일)
    base_logits = build_model_logits(alpha=args.alpha)

    # 2) 래퍼로 가중치 로드(체크포인트 네임스페이스 맞춤)
    wrapper = ASLModel(base_model=base_logits)
    ckpt = find_ckpt_prefix()
    status = wrapper.load_weights(ckpt)
    try:
        status.expect_partial()  # optimizer 등 누락 경고 무시
    except Exception:
        pass
    print(f"[INFO] Weights loaded from: {ckpt}")

    # 3) 내보낼 모델: Sigmoid 확률 출력으로 래핑
    prob_out = layers.Activation('sigmoid', name='prob')(base_logits.output)
    export_model = Model(inputs=base_logits.input, outputs=prob_out, name='mobilenet_asl_export')
    export_model.summary()

    # 4) TFLite 변환
    convert_to_tflite(export_model)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
