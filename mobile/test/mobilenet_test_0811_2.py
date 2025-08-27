
import os
import argparse
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers, constraints
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, multilabel_confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# ==================== Args ====================
parser = argparse.ArgumentParser(description='MobileNetV2 다중질병 테스트 (ASL logits 학습 호환)')
parser.add_argument('--model_num', type=str, default='8')
parser.add_argument('--checkpoint_path', type=str, default='/home/user/chae_project/model_save/')
parser.add_argument('--dataset_path', type=str, default='/home/user/chae_project/vgg_data/0805_224/')
parser.add_argument('--gpu_num', type=str, default='0,1,2,3')
parser.add_argument('--batch_size', type=int, default=128)   # 전역 배치
parser.add_argument('--top_k', type=int, default=0)          # top-K 제한(0/None이면 끔)
parser.add_argument('--samples', type=int, default=10)       # 샘플 프린트 개수

# === 추가 옵션 ===
parser.add_argument('--d21_min_thr', type=float, default=None)    # d21 임계값 하한 강제(예: 0.62)
parser.add_argument('--top_k_override', type=int, default=None)   # 테스트 시 top-k 강제
parser.add_argument('--use_temp', action='store_true', default=True)  # temp_scale.npy 있으면 사용
parser.add_argument('--try_topk1', action='store_true', default=False) # top_k=1도 추가로 평가

# === d21 배타 옵션 ===
parser.add_argument('--d21_exclusive', action='store_true', default=False,
                    help='d21을 단일(배타)로 후처리: d21이 확실하면 d21만 1, 나머지 0')
parser.add_argument('--d21_margin', type=float, default=0.06,
                    help='배타 적용 마진(2위보다 이만큼 높을 때만 배타)')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
tf.keras.backend.set_floatx('float32')

# ==================== Const ====================
CLASS_LIST = ['disease1','disease2','disease6','disease8','disease11','disease13','disease19','disease21']
NUM_CLASSES = len(CLASS_LIST)
EPS = 1e-7

# ==================== Utils ====================
def get_model_dir():
    return os.path.join(args.checkpoint_path, f'mobilenet_{args.model_num}')

def get_checkpoint_prefix():  # 학습에서 ModelCheckpoint(..., 'best_model', save_weights_only=True)
    return os.path.join(get_model_dir(), 'best_model')

def get_saved_weights_prefix():
    return os.path.join(get_model_dir(), 'saved_weights')

def get_threshold_path():
    return os.path.join(get_model_dir(), 'best_threshold.npy')

def get_temp_scale_path():
    return os.path.join(get_model_dir(), 'temp_scale.npy')

def get_checkpoint_size_mb(prefix_path):
    total = 0
    for ext in [".data-00000-of-00001", ".index"]:
        p = prefix_path + ext
        if os.path.exists(p):
            total += os.path.getsize(p)
    return total / (1024 * 1024)

def normalize_01(x):
    if x.dtype != np.float32:
        x = x.astype('float32')
    x /= 255.0
    return x

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def binarize_with_threshold_and_topk(y_prob, thresholds, top_k):
    # thresholds: scalar or per-class shape [C]
    if np.isscalar(thresholds):
        thr = np.full((y_prob.shape[1],), float(thresholds), dtype=np.float32)
    else:
        thr = thresholds.astype(np.float32)
    y_bin = (y_prob > thr[None, :]).astype(np.int32)
    if not top_k or top_k <= 0:
        return y_bin
    out = np.zeros_like(y_bin)
    for i in range(y_prob.shape[0]):
        idx = np.argsort(-y_prob[i])[:top_k]
        out[i, idx] = 1
    return (y_bin & out).astype(np.int32)

def ensure_per_class_thresholds(thresholds, num_classes):
    if np.isscalar(thresholds):
        return np.full((num_classes,), float(thresholds), dtype=np.float32)
    return thresholds.astype(np.float32)

def load_thresholds():
    thr_path = get_threshold_path()
    if os.path.exists(thr_path):
        arr = np.load(thr_path)
        if arr.ndim == 0:
            thr = float(arr)
            print(f"[INFO] 임계값 로드(스칼라): {thr:.2f}")
            return thr
        elif arr.ndim == 1 and arr.shape[0] == NUM_CLASSES:
            print(f"[INFO] 임계값 로드(클래스별): {np.round(arr,2)}")
            return arr.astype(np.float32)
    print("[INFO] 임계값 파일 없음 → 기본 0.5 사용")
    return 0.5

def load_temp_scale():
    p = get_temp_scale_path()
    if os.path.exists(p):
        T = np.load(p).astype(np.float32)
        if T.ndim == 1 and T.shape[0] == NUM_CLASSES:
            print(f"[INFO] temp_scale 로드: {np.round(T,2)}")
            return T
    print("[INFO] temp_scale 없음 → 미사용")
    return None

# --- d21 배타 후처리 ---
def apply_d21_exclusive(y_prob, y_bin, thr_vec, margin=0.06):
    """
    d21(인덱스 7)이 임계값 이상이고, 2위보다 margin 이상 높으면 -> d21만 1로, 나머지는 0으로.
    """
    second_best = np.partition(y_prob, -2, axis=1)[:, -2]
    cond = (y_prob[:, 7] >= thr_vec[7]) & ((y_prob[:, 7] - second_best) >= float(margin))
    if np.any(cond):
        y_bin[cond, :] = 0
        y_bin[cond, 7] = 1
    return y_bin

# ==================== Inference Model (학습 구조와 동일) ====================
def build_base_model_for_inference(class_bias=None, alpha=1.0):
    backbone = MobileNetV2(include_top=False, input_shape=(224, 224, 3),
                           weights=None, alpha=alpha)
    x_in = layers.Input(shape=(224, 224, 3))
    z = preprocess_input(x_in * 255.0)     # 학습과 동일
    z = backbone(z, training=False)        # BN 통계 고정
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
    bias_init = tf.keras.initializers.Constant(class_bias) if class_bias is not None else 'zeros'
    out = layers.Dense(NUM_CLASSES, activation=None, bias_initializer=bias_init,
                       kernel_constraint=constraints.MaxNorm(3))(z)  # logits
    base = Model(inputs=x_in, outputs=out, name="mobilenet_asl")
    return base

class ASLModel(tf.keras.Model):
    """학습 때와 동일한 래퍼(이름/구조 일치시켜서 가중치 로딩 안정화)"""
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
    def call(self, x, training=False):
        return self.base(x, training=training)  # logits

def load_model_from_checkpoint(alpha=1.0):
    base = build_base_model_for_inference(class_bias=np.zeros((NUM_CLASSES,), np.float32), alpha=alpha)
    model = ASLModel(base_model=base)

    ckpt_prefix = get_checkpoint_prefix()
    alt_prefix = get_saved_weights_prefix()

    if os.path.exists(ckpt_prefix + ".index"):
        model.load_weights(ckpt_prefix)
        used = ckpt_prefix
    elif os.path.exists(alt_prefix + ".index"):
        model.load_weights(alt_prefix)
        used = alt_prefix
    else:
        raise FileNotFoundError(f"[ERROR] 체크포인트 없음: {ckpt_prefix} 또는 {alt_prefix}")

    print(f"[INFO] 체크포인트 로드 완료: {used}")
    return model, used

# ==================== Data ====================
def load_test_data():
    csv_path = os.path.join(args.dataset_path, 'test.csv')
    npy_path = os.path.join(args.dataset_path, 'test.npy')
    df = pd.read_csv(csv_path)
    x = np.load(npy_path)
    x = normalize_01(x)   # 학습 파이프라인과 일치(0..1 후 모델 내부에서 *255→preprocess_input)
    y = df[CLASS_LIST].values.astype(np.int32)
    filenames = df['file_name'].values
    return x, y, filenames

# ==================== Eval ====================
def evaluate(model, thresholds, x, y_true, filenames, batch_size,
             temp_scale=None, top_k=None, print_samples=True, tag=""):
    start = time.time()
    # logits 예측
    y_logits = model.predict(x, batch_size=batch_size, verbose=1)
    elapsed = time.time() - start

    # 온도 보정(있으면)
    if temp_scale is not None:
        temp_scale = np.clip(temp_scale, 1e-3, 10.0).astype(np.float32)
        y_logits = y_logits / temp_scale[None, :]

    # 확률로 변환
    y_prob = sigmoid(y_logits)
    y_prob = np.clip(y_prob, 1.0e-6, 1.0 - 1.0e-6)

    tk = args.top_k if top_k is None else top_k
    y_bin = binarize_with_threshold_and_topk(y_prob, thresholds, tk)

    # --- d21 배타 규칙(옵션) ---
    thr_vec = ensure_per_class_thresholds(thresholds, NUM_CLASSES)
    if args.d21_exclusive:
        y_bin = apply_d21_exclusive(y_prob, y_bin, thr_vec, margin=args.d21_margin)

    # 안전한 문자열 조합(중첩 f-string 피함)
    excl = 'on' if args.d21_exclusive else 'off'
    tag_str = f"[{tag}, exclusive={excl}]" if tag else f"[exclusive={excl}]"
    print(f"\n[INFO] 예측 완료 (총 소요 시간: {elapsed:.2f}초) {tag_str}")
    print(f"예측 확률 평균: {np.round(np.mean(y_prob, axis=0), 4)}")
    print(f"예측 확률 분포: {np.min(y_prob):.4f} ~ {np.max(y_prob):.4f}")
    print(f"클래스별 1 예측 비율: {np.round(np.mean(y_bin, axis=0), 4)}\n")

    # === d21(인덱스 7) 임계값/과임계 비율 확인 ===
    d21_mean = float(np.mean(y_prob[:, 7]))
    d21_posrate = float(np.mean(y_bin[:, 7]))
    d21_above_thr = float(np.mean(y_prob[:, 7] > thr_vec[7]))
    print(f"[d21] mean_prob={d21_mean:.4f} / thr={thr_vec[7]:.2f} / "
          f"rate_above_thr={d21_above_thr:.4f} / predicted_pos={d21_posrate:.4f}")

    # F1
    f1_micro = f1_score(y_true, y_bin, average='micro', zero_division=0)
    f1_round = f1_score(y_true, np.round(y_prob), average='micro', zero_division=0)
    print(f"F1 Score (micro, binarized):  {f1_micro * 100:.2f}%")
    print(f"F1 Score (legacy, round):     {f1_round * 100:.2f}%")

    # Precision / Recall
    prec = precision_score(y_true, y_bin, average='micro', zero_division=0)
    rec_sklearn = recall_score(y_true, y_bin, average='micro', zero_division=0)

    cm = multilabel_confusion_matrix(y_true, y_bin)
    TP_sum = sum(cm[i][1,1] for i in range(len(CLASS_LIST)))
    FN_sum = sum(cm[i][1,0] for i in range(len(CLASS_LIST)))
    rec_manual = TP_sum / (TP_sum + FN_sum + 1e-8)

    print(f"Precision (micro):            {prec * 100:.2f}%")
    print(f"Recall (sklearn):             {rec_sklearn * 100:.2f}%")
    print(f"Recall (manual):              {rec_manual * 100:.2f}%")

    # Specificity
    spec_list = []
    for i in range(len(CLASS_LIST)):
        tn = cm[i][0,0]
        fp = cm[i][0,1]
        spec = tn / (tn + fp + 1e-8)
        spec_list.append(spec)
    print(f"Specificity (avg):            {np.mean(spec_list) * 100:.2f}%")

    # AUPRC (per-class → avg)
    auprcs = [average_precision_score(y_true[:, i], y_prob[:, i]) for i in range(len(CLASS_LIST))]
    print(f"AUPRC (avg):                  {np.mean(auprcs) * 100:.2f}%")

    # Inference Time
    print(f"\nInference Time/Image:         {elapsed / len(x):.4f} sec\n")

    # 체크포인트 크기
    used_prefix = get_checkpoint_prefix() if os.path.exists(get_checkpoint_prefix()+".index") else get_saved_weights_prefix()
    print(f"Model Size (checkpoint):      {get_checkpoint_size_mb(used_prefix):.2f} MB\n")

    # 샘플 예측
    if print_samples:
        print(f"========== Sample Predictions ==========")
        sample_idxs = random.sample(range(len(y_bin)), min(args.samples, len(y_bin)))
        for idx in sample_idxs:
            gt = [CLASS_LIST[i] for i, v in enumerate(y_true[idx]) if v == 1]
            pred = [CLASS_LIST[i] for i, v in enumerate(y_bin[idx]) if v == 1]
            print(f"[{idx:>5}] File: {filenames[idx]}")
            print(f"     GT      : {gt}")
            print(f"     Predict : {pred}\n")

# ==================== Run ====================
if __name__ == '__main__':
    # 멀티 GPU 정렬
    strategy = tf.distribute.MirroredStrategy()
    replicas = strategy.num_replicas_in_sync
    print(f"[INFO] Using {replicas} GPUs for inference")

    orig_bs = args.batch_size
    global_bs = int(np.ceil(orig_bs / replicas) * replicas)
    if global_bs != orig_bs:
        print(f"[WARN] batch_size {orig_bs} -> {global_bs} (x{replicas} GPUs)")
    args.batch_size = global_bs

    with strategy.scope():
        model, used_prefix = load_model_from_checkpoint(alpha=1.0)

    thresholds = load_thresholds()
    thresholds = ensure_per_class_thresholds(thresholds, NUM_CLASSES)

    # d21 임계값 하한 강제(옵션)
    if args.d21_min_thr is not None:
        old = thresholds[7]
        thresholds[7] = max(thresholds[7], float(args.d21_min_thr))
        if thresholds[7] != old:
            print(f"[INFO] d21 thr 상승: {old:.2f} -> {thresholds[7]:.2f}")

    # top-k 오버라이드(옵션)
    if args.top_k_override is not None:
        print(f"[INFO] top_k override: {args.top_k} -> {args.top_k_override}")
        args.top_k = int(args.top_k_override)

    # 온도보정 로드(있으면 사용)
    temp_scale = load_temp_scale() if args.use_temp else None

    x_test, y_test, filenames = load_test_data()

    # 기본 평가
    evaluate(model, thresholds, x_test, y_test, filenames,
             batch_size=args.batch_size, temp_scale=temp_scale, top_k=args.top_k,
             tag=f"top_k={args.top_k}, temp={'on' if temp_scale is not None else 'off'}")

    # 요청 시 top_k=1도 추가 비교
    if args.try_topk1 and (args.top_k != 1):
        evaluate(model, thresholds, x_test, y_test, filenames,
                 batch_size=args.batch_size, temp_scale=temp_scale, top_k=1,
                 print_samples=False, tag="top_k=1")
